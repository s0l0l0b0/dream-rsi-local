"""Tests for the untrusted-policy runner (``src/meta/policy_runner.py``).

The runner validates candidate policy files against the mutation contract and
replays them offline.  Timing-out tests (and anything that spawns a subprocess)
are marked ``sandbox``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.evaluator import evaluator_fingerprint
from src.core.tree import CandidateOutcome, DiscoveryTree
from src.meta import policy_runner as pr
from src.policy.base import ACTION_NAMES, ActionSpace, ExplorationPolicy, context_from_record

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_POLICY = REPO_ROOT / "src" / "policy" / "search_policy.py"

#: A module that passes the static gate but never defines ``SearchPolicy``.
NO_POLICY_CLASS = '''"""No policy class here."""

VALUE = 1
'''

#: A contract-valid class that forgets the ``POLICY_ID`` literal.
MISSING_POLICY_ID = '''"""A class without POLICY_ID."""
from src.policy.base import ExplorationPolicy

PARAMS = {"w": 0.0}


class SearchPolicy(ExplorationPolicy):
    @classmethod
    def param_space(cls):
        return {"w": (-1.0, 1.0)}

    def select(self, ctx):
        return ctx.candidates[0]
'''

#: ``PARAMS`` declares a knob that ``param_space()`` does not bound.
PARAMS_NOT_IN_SPACE = '''"""param_space() does not cover every PARAMS entry."""
from src.policy.base import ExplorationPolicy

POLICY_ID = "candidate_missing_bound"
PARAMS = {"alpha": 0.0, "beta": 0.0}


class SearchPolicy(ExplorationPolicy):
    @classmethod
    def param_space(cls):
        return {"alpha": (-1.0, 1.0)}

    def select(self, ctx):
        return ctx.candidates[0]
'''

#: An empty ``PARAMS`` dict.
EMPTY_PARAMS = '''"""PARAMS is empty."""
from src.policy.base import ExplorationPolicy

POLICY_ID = "candidate_empty_params"
PARAMS = {}


class SearchPolicy(ExplorationPolicy):
    @classmethod
    def param_space(cls):
        return {}

    def select(self, ctx):
        return ctx.candidates[0]
'''

#: The minimum contract-valid surface, with a ``select`` that never returns.
#: Nothing else is defined so validation must accept the module and let the
#: runner's timeout guard trip inside ``smoke_test``.
HANGING_POLICY = '''"""A contract-valid policy whose select() hangs."""
from src.policy.base import ExplorationPolicy

POLICY_ID = "candidate_hang"
PARAMS = {"w": 0.0}


class SearchPolicy(ExplorationPolicy):
    @classmethod
    def param_space(cls):
        return {"w": (-1.0, 1.0)}

    def select(self, ctx):
        while True:
            pass
'''


def write_module(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


# ======================================================================================
# Static gate
# ======================================================================================
def test_policy_static_gate_accepts_the_real_policy() -> None:
    assert pr.policy_static_gate(REAL_POLICY.read_text(encoding="utf-8")) == ""


@pytest.mark.parametrize(
    ("source", "hint"),
    [
        ("import os\n", "os"),
        ("import subprocess\n", "subprocess"),
        ("from socket import socket\n", "socket"),
        ('open("/etc/passwd")\n', "open"),
        ('eval("1")\n', "eval"),
    ],
)
def test_policy_static_gate_rejects_unsafe_sources(source: str, hint: str) -> None:
    violation = pr.policy_static_gate(source)
    assert violation, f"gate accepted {source!r}"
    assert hint in violation, f"{hint!r} missing from {violation!r}"


def test_load_policy_module_refuses_unsafe_source(tmp_path: Path) -> None:
    path = write_module(tmp_path, "unsafe_policy", "import os\n\nPOLICY_ID = 'x'\n")
    with pytest.raises(pr.PolicyContractError):
        pr.load_policy_module(path)


# ======================================================================================
# Contract validation
# ======================================================================================
def test_load_and_validate_accept_the_real_policy() -> None:
    module = pr.load_policy_module(REAL_POLICY)
    assert pr.validate_module(module) == []

    policy = pr.instantiate(module, rng_seed=0)
    # ``load_policy_module`` execs the file, so the class object is *not* the one
    # imported elsewhere; assert on the contract surface instead.
    assert type(policy).__name__ == "SearchPolicy"
    assert isinstance(policy, ExplorationPolicy)
    assert isinstance(policy.policy_id, str) and policy.policy_id
    assert set(policy.params) == set(policy.param_space())


def test_validate_reports_a_missing_policy_class(tmp_path: Path) -> None:
    module = pr.load_policy_module(write_module(tmp_path, "no_policy", NO_POLICY_CLASS))
    assert pr.validate_module(module) == ["module does not define SearchPolicy"]


def test_validate_reports_a_missing_policy_id(tmp_path: Path) -> None:
    module = pr.load_policy_module(write_module(tmp_path, "no_policy_id", MISSING_POLICY_ID))
    problems = pr.validate_module(module)
    assert any("POLICY_ID" in problem for problem in problems), problems


def test_validate_reports_params_missing_from_param_space(tmp_path: Path) -> None:
    module = pr.load_policy_module(
        write_module(tmp_path, "missing_bound", PARAMS_NOT_IN_SPACE)
    )
    problems = pr.validate_module(module)
    assert any("missing from param_space()" in problem for problem in problems), problems
    assert any("beta" in problem for problem in problems), problems


def test_validate_reports_empty_params(tmp_path: Path) -> None:
    module = pr.load_policy_module(write_module(tmp_path, "empty_params", EMPTY_PARAMS))
    problems = pr.validate_module(module)
    assert any("PARAMS must be a non-empty dict" in problem for problem in problems), problems


def test_smoke_test_on_the_real_policy_is_deterministic() -> None:
    module = pr.load_policy_module(REAL_POLICY, name="smoke_test_real")
    report = pr.smoke_test(pr.instantiate(module, rng_seed=0))

    assert report["deterministic"] is True
    assert report["chosen"] in ACTION_NAMES
    assert report["slate_size"] > 0
    assert report["proposed"]


# ======================================================================================
# Timeout guard
# ======================================================================================
@pytest.mark.sandbox
def test_hanging_policy_times_out_without_killing_the_batch(tmp_path: Path) -> None:
    path = write_module(tmp_path, "hanging", HANGING_POLICY)

    payload = pr.evaluate_policy_file(path, [], per_candidate_timeout=1.0)

    assert payload["ok"] is False
    assert "timeout" in payload["error"].lower(), payload["error"]
    assert payload["label"] == "hanging"
    assert payload["policy_path"] == str(path)


# ======================================================================================
# Offline replay
# ======================================================================================
def build_synthetic_tree(path: Path) -> DiscoveryTree:
    """A one-decision tree with three measured candidates and one chosen action."""
    space = ActionSpace()
    tree = DiscoveryTree.create(
        "synthetic_replay",
        policy_id="recorder",
        task_meta={
            "id": "synthetic_replay",
            "suite": "humaneval",
            "entry_point": "f",
            "signature": "def f(x):",
            "prompt": "Return x.",
            "difficulty": 0.5,
            "tags": ["unit"],
        },
        root_state={"solution": "", "explanation": "root"},
        tree_id="tree_synthetic",
    )
    tree.metadata["evaluator_sha256"] = evaluator_fingerprint()

    slate = [
        CandidateOutcome(action=space.build("direct"), score=0.375, tests_passed=2, tests_total=4),
        CandidateOutcome(action=space.build("chain_of_thought"), score=0.75, tests_passed=3, tests_total=4),
        CandidateOutcome(action=space.build("optimize"), score=1.0, passed=True, tests_passed=4, tests_total=4),
    ]
    assert tree.root_id is not None
    tree.record_slate(tree.root_id, slate, None)
    tree.mark_chosen(tree.root_id, slate[1].action)
    tree.save(path)
    return tree


def test_replay_batch_scores_what_the_candidate_policy_picks(tmp_path: Path) -> None:
    tree_path = tmp_path / "tree_synthetic.json"
    tree = build_synthetic_tree(tree_path)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"label": "synthetic", "tree_paths": [str(tree_path)]}), encoding="utf-8"
    )
    trees = pr.load_manifest(manifest)
    assert len(trees) == 1
    assert trees[0].tree_id == tree.tree_id

    # A second, byte-identical candidate file checks "one result per policy path".
    copy_dir = tmp_path / "candidate_copy"
    copy_dir.mkdir()
    copied = copy_dir / "search_policy.py"
    copied.write_text(REAL_POLICY.read_text(encoding="utf-8"), encoding="utf-8")

    payload = pr.replay_batch([REAL_POLICY, copied], trees)
    assert payload["ok"] is True
    assert payload["n_policies"] == 2
    assert payload["n_trees"] == 1
    assert payload["n_decisions"] == 1
    assert len(payload["results"]) == 2
    assert payload["evaluator_sha256"] == evaluator_fingerprint()

    # Compute the expectation from the recorded tree itself: the real policy's
    # pick, looked up in the recorded rewards.  The expectation policy is loaded
    # from the same file the runner reads, so the comparison cannot drift.
    records = trees[0].decisions()
    assert len(records) == 1
    record = records[0]
    expectation_module = pr.load_policy_module(REAL_POLICY, name="expectation_policy")
    expectation_policy = pr.instantiate(expectation_module, rng_seed=0)
    chosen = expectation_policy.select(context_from_record(record))
    expected = record.reward_of(chosen)
    assert expected is not None, "the real policy picked an action outside the recorded slate"

    for result in payload["results"]:
        assert result["ok"] is True, result.get("error")
        assert result["policy_id"] == expectation_policy.policy_id
        assert result["replay"]["n_decisions"] == 1
        assert result["replay"]["mean_reward"] == pytest.approx(expected), (
            f"{result['label']}: replay mean {result['replay']['mean_reward']} != "
            f"reward {expected} of the picked action {chosen.name!r}"
        )


@pytest.mark.parametrize("payload", [{}, {"tree_paths": []}, {"trees": []}])
def test_load_manifest_raises_on_an_empty_manifest(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        pr.load_manifest(manifest)


def test_rewritten_policy_is_reloaded_without_stale_bytecode(tmp_path):
    """Regression: a promotion must take effect immediately.

    CPython validates __pycache__ entries with second-truncated mtimes.  Rewriting
    a policy within the same second it was last loaded, with an unchanged file size
    (``"prior_repair": 0.0`` -> ``0.5``), used to make the loader return the stale
    module: the Dream Engine saw the new policy (fresh subprocess) while the live
    loop kept running the old one.
    """
    from src.meta.optimizer import apply_params, read_params_literal

    source = (Path(__file__).resolve().parents[1] / "src" / "policy" / "search_policy.py").read_text(
        encoding="utf-8"
    )
    params = read_params_literal(source)
    path = tmp_path / "policy.py"

    path.write_text(apply_params(source, {**params, "prior_repair": 0.0}, "policy_vA"), encoding="utf-8")
    first = pr.load_policy_module(path, name="stale_probe")
    assert first.SearchPolicy().param("prior_repair") == pytest.approx(0.0)

    # Same size, same second: exactly what promote() does.
    path.write_text(apply_params(source, {**params, "prior_repair": 0.5}, "policy_vB"), encoding="utf-8")
    second = pr.load_policy_module(path, name="stale_probe")

    assert second.SearchPolicy().param("prior_repair") == pytest.approx(0.5)
    assert second.POLICY_ID == "policy_vB"
    assert second is not first
