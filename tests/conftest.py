"""Shared pytest fixtures for the Dream-RSI test suite.

Every fixture is deterministic and offline: the suite must run with no network,
no API key and no LLM endpoint.  Tests that spawn sandboxed subprocesses are
marked ``sandbox``; tests that run a whole RSI cycle are marked ``integration``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.evaluator import evaluator_fingerprint, load_tasks, task_meta  # noqa: E402
from src.core.llm_client import MockLLMClient  # noqa: E402
from src.core.tree import Action, CandidateOutcome, DiscoveryTree  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tasks() -> list[dict[str, Any]]:
    """The full shipped task suite."""
    return load_tasks(REPO_ROOT / "benchmarks" / "tasks.jsonl")


@pytest.fixture(scope="session")
def task_index(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(t["id"]): t for t in tasks}


@pytest.fixture(scope="session")
def small_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two tasks per suite -- enough to exercise splits without long runs."""
    picked: list[dict[str, Any]] = []
    for suite in ("humaneval", "math", "kernelbench"):
        picked.extend([t for t in tasks if t.get("suite") == suite][:2])
    return picked


@pytest.fixture
def mock_client(task_index: Mapping[str, Mapping[str, Any]]) -> MockLLMClient:
    """Deterministic offline solver simulator (no cache wrapper)."""
    return MockLLMClient(task_index, seed=11)


@pytest.fixture
def make_outcomes():
    """Build a slate of :class:`CandidateOutcome` values from (name, score) pairs."""

    def factory(pairs: Sequence[tuple[str, float]]) -> list[CandidateOutcome]:
        return [
            CandidateOutcome(
                action=Action(name=name, params={}),
                score=float(score),
                passed=float(score) >= 1.0,
                tests_passed=int(round(float(score) * 4)),
                tests_total=4,
            )
            for name, score in pairs
        ]

    return factory


@pytest.fixture
def make_chain_tree(make_outcomes):
    """Build a deterministic discovery tree along one branch with known rewards.

    ``slates`` is a list of slates, one per decision point, each a sequence of
    ``(action_name, score)`` pairs.  The best-scoring candidate is always chosen,
    so expectations can be computed by hand:

    * ``n_decisions == len(slates)``
    * ``oracle_mean == mean(max(score) per slate)``
    * a policy that always picks the best candidate scores exactly ``oracle_mean``
    """

    def factory(
        slates: Sequence[Sequence[tuple[str, float]]],
        *,
        task_id: str = "task_a",
        suite: str = "humaneval",
        tree_id: str = "tree_a",
        difficulty: float = 0.5,
        reference_solution: str = "def f():\n    return 1\n",
        fingerprint: bool = True,
        choose: str | None = None,
    ) -> DiscoveryTree:
        tree = DiscoveryTree.create(
            task_id,
            policy_id="test_policy",
            task_meta={
                "id": task_id,
                "suite": suite,
                "entry_point": "f",
                "signature": "def f():",
                "prompt": "Return 1.",
                "difficulty": difficulty,
                "tags": ["unit"],
                "reference_solution": reference_solution,
                "features": {
                    "difficulty": float(difficulty),
                    "prompt_len": 9.0,
                    "n_tags": 1.0,
                    **{f"suite_{name}": 1.0 if suite == name else 0.0
                       for name in ("humaneval", "math", "kernelbench")},
                },
            },
            tree_id=tree_id,
            root_state={"solution": "", "explanation": "root"},
        )
        if fingerprint:
            tree.metadata["evaluator_sha256"] = evaluator_fingerprint()
        node = tree.root
        for index, slate in enumerate(slates):
            outcomes = make_outcomes(slate)
            tree.record_slate(node.id, outcomes, None)
            if choose is not None:
                chosen = next(o for o in outcomes if o.action.name == choose)
            else:
                chosen = max(outcomes, key=lambda o: o.score)
            tree.mark_chosen(node.id, chosen.action)
            node = tree.add_node(
                node.id,
                chosen.action,
                {"solution": f"def f():\n    return {index}\n", "explanation": f"depth {index}"},
                chosen.score,
                feedback="" if chosen.passed else "some tests failed",
                passed=chosen.passed,
                policy_id="test_policy",
            )
        return tree

    return factory


@pytest.fixture
def tiny_manifest(tmp_path):
    """Write trees to disk and return ``(trees, manifest_path)``."""

    def factory(trees: Sequence[DiscoveryTree], name: str = "manifest.json") -> tuple[list[DiscoveryTree], Path]:
        paths = []
        for tree in trees:
            path = tmp_path / f"{tree.tree_id}.json"
            tree.save(path)
            paths.append(str(path))
        manifest = tmp_path / name
        manifest.write_text(
            '{"label": "test", "tree_paths": ' + repr(paths).replace("'", '"') + "}",
            encoding="utf-8",
        )
        return list(trees), manifest

    return factory


@pytest.fixture(scope="session")
def untuned_policy_source() -> str:
    """Source of the *untuned* baseline, whatever version is currently live.

    A run of the RSI loop legitimately promotes a tuned policy into
    ``src/policy/search_policy.py``, so tests that need the naive starting point
    (all weights zero) must reconstruct it instead of assuming the live file is
    ``policy_v0``.  The reconstruction only rewrites the PARAMS literal, which is
    exactly what the optimizer itself does.
    """
    from src.meta.optimizer import apply_params, read_params_literal, read_policy_id_literal

    source = (REPO_ROOT / "src" / "policy" / "search_policy.py").read_text(encoding="utf-8")
    if read_policy_id_literal(source) == "policy_v0" and all(
        value == 0.0
        for key, value in read_params_literal(source).items()
        if key not in {"value_decay", "stop_at"}
    ):
        return source
    params = {key: 0.0 for key in read_params_literal(source)}
    params.update({"value_decay": 0.7, "stop_at": 1.0})
    return apply_params(source, params, "policy_v0")


def expected_slate_stats(slates: Sequence[Sequence[tuple[str, float]]]) -> dict[str, float]:
    """Hand-computable expectations for a chain tree built from ``slates``."""
    oracle = [max(score for _, score in slate) for slate in slates]
    first = [slate[0][1] for slate in slates]
    return {
        "n_decisions": float(len(slates)),
        "oracle_mean": sum(oracle) / len(oracle) if oracle else 0.0,
        "first_mean": sum(first) / len(first) if first else 0.0,
    }
