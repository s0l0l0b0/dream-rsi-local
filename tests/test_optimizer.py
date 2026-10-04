"""Tests for the meta-optimizer: profiling, mutation, gating and promotion.

These tests pin down the parts that decide whether the system improves *honestly*:
the parameter-only source mutation, the deterministic task split, the paired
statistical comparison, and every branch of the promotion gate.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from benchmarks.evaluator import evaluator_fingerprint
from src.core.tree import Action, DiscoveryTree
from src.engine.dreamer import ReplayConfig, ReplayResult
from src.meta.optimizer import (
    Candidate,
    FailureProfiler,
    MetaOptimizer,
    OfflineMutator,
    OptimizerConfig,
    _render_params_block,
    _result_from_dict,
    apply_params,
    paired_comparison,
    read_params_literal,
    read_policy_id_literal,
)
from src.policy.search_policy import PARAMS, SearchPolicy, param_space

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "src" / "policy" / "search_policy.py"


# ======================================================================================
# Source mutation
# ======================================================================================
def test_read_params_and_policy_id_literals():
    """Reading the literals must not execute the file, and must work for any version.

    Deliberately does *not* assert ``POLICY_ID == "policy_v0"``: a run of the RSI
    loop legitimately promotes a new version into the tracked source, and the test
    suite must stay green afterwards.
    """
    source = POLICY_PATH.read_text(encoding="utf-8")
    params = read_params_literal(source)
    assert set(params) == set(PARAMS)
    assert read_policy_id_literal(source).startswith("policy_")

    # The untuned baseline is reconstructible from any version of the file.
    untuned = {k: 0.0 for k in params}
    untuned.update({"value_decay": 0.7, "stop_at": 1.0})
    baseline = apply_params(source, untuned, "policy_v0")
    assert all(v == 0.0 for k, v in read_params_literal(baseline).items()
               if k not in {"value_decay", "stop_at"})
    assert read_policy_id_literal(baseline) == "policy_v0"


def test_apply_params_changes_only_the_params_block():
    source = POLICY_PATH.read_text(encoding="utf-8")
    updated = apply_params(source, {**PARAMS, "repair_bonus": 1.5}, "policy_v9")

    # Every changed line must lie inside the PARAMS or POLICY_ID assignment.
    def literal_spans(text: str) -> set[int]:
        spans: set[int] = set()
        for node in ast.parse(text).body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                target = names[0] if names else None
            if target in {"PARAMS", "POLICY_ID"}:
                spans.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        return spans

    def remainder(text: str) -> list[str]:
        spans = literal_spans(text)
        return [
            line for number, line in enumerate(text.splitlines(), start=1)
            if number not in spans
        ]

    # Dropping the two literals from both versions must leave identical code.
    assert remainder(source) == remainder(updated)
    assert read_params_literal(updated)["repair_bonus"] == 1.5
    assert read_policy_id_literal(updated) == "policy_v9"
    ast.parse(updated)

    # Re-applying the same params is idempotent.
    assert apply_params(updated, read_params_literal(updated), "policy_v9") == updated


def test_render_params_block_is_the_format_apply_params_produces():
    rendered = _render_params_block({"a": 0.0, "b": 1.5})
    assert rendered == 'PARAMS: dict[str, float] = {\n    \'a\': 0.0,\n    \'b\': 1.5,\n}'


def test_apply_params_rejects_source_without_params():
    from src.meta.policy_runner import PolicyContractError

    with pytest.raises(PolicyContractError):
        apply_params("X = 1\n", {"a": 0.0})


def test_mutated_source_still_satisfies_the_contract(tmp_path):
    """A parameter-only rewrite must remain a loadable, valid policy."""
    from src.meta.policy_runner import instantiate, load_policy_module, validate_module

    source = POLICY_PATH.read_text(encoding="utf-8")
    mutated = apply_params(source, {**PARAMS, "prior_restart": 0.5, "w_suite_value": 1.0}, "policy_vX")
    path = tmp_path / "policy_vX.py"
    path.write_text(mutated, encoding="utf-8")

    module = load_policy_module(path)
    assert validate_module(module) == []
    policy = instantiate(module)
    assert policy.param("prior_restart") == pytest.approx(0.5)
    assert policy.param("w_suite_value") == pytest.approx(1.0)
    assert policy.policy_id == "policy_vX"


# ======================================================================================
# Offline mutator
# ======================================================================================
def test_mutator_clips_to_the_declared_bounds():
    space = {"a": (0.0, 1.0), "b": (-2.0, 2.0)}
    mutator = OfflineMutator(space, seed=1)
    clipped = mutator.clip({"a": 5.0, "b": -9.0})
    assert clipped == {"a": 1.0, "b": -2.0}


def test_coordinate_probes_cover_every_parameter_even_when_truncated():
    space = {f"p{i}": (0.0, 1.0) for i in range(20)}
    mutator = OfflineMutator(space, seed=3)
    probes = mutator.coordinate_probes({"p0": 0.0}, limit=len(space))
    touched = {key for probe in probes for key in probe if probe[key] != 0.0}
    # A naive truncation of a per-parameter grouping would only cover the first
    # few keys; round-robin interleaving must touch all of them.
    assert touched == set(space)


def test_uniform_samples_and_perturbations_stay_in_bounds():
    space = dict(param_space())
    mutator = OfflineMutator(space, seed=5)
    base = dict(PARAMS)
    for candidate in mutator.uniform_samples(10) + mutator.sparse_perturbations(base, 10):
        for key, value in candidate.items():
            low, high = space[key]
            assert low - 1e-9 <= value <= high + 1e-9


def test_mutator_is_seed_deterministic():
    space = dict(param_space())
    a = OfflineMutator(space, seed=7).uniform_samples(5)
    b = OfflineMutator(space, seed=7).uniform_samples(5)
    assert a == b


# ======================================================================================
# Task split
# ======================================================================================
def test_split_is_deterministic_disjoint_and_suite_stratified(tmp_path):
    optimizer = MetaOptimizer(storage_dir=tmp_path, trees_dir=tmp_path, config=OptimizerConfig(train_ratio=0.5))
    ids = [f"t{i}" for i in range(10)]
    suites = {f"t{i}": ("math" if i < 4 else "humaneval") for i in range(10)}

    train, holdout = optimizer.split_task_ids(ids, suites)
    assert train & holdout == set()
    assert train | holdout == set(ids)
    assert optimizer.split_task_ids(ids, suites) == (train, holdout)  # deterministic
    assert any(suites[t] == "math" for t in holdout)
    assert any(suites[t] == "humaneval" for t in holdout)


def test_split_puts_a_lone_suite_task_in_train(tmp_path):
    optimizer = MetaOptimizer(storage_dir=tmp_path, trees_dir=tmp_path)
    train, holdout = optimizer.split_task_ids(["only"], {"only": "math"})
    assert train == {"only"}
    assert holdout == set()


def test_split_tasks_keeps_every_tree_of_a_task_on_one_side(make_chain_tree, tmp_path):
    optimizer = MetaOptimizer(storage_dir=tmp_path, trees_dir=tmp_path, config=OptimizerConfig(train_ratio=0.5))
    trees = []
    for i in range(6):
        suite = "math" if i % 2 else "humaneval"
        trees.append(make_chain_tree([[("direct", 0.5)]], task_id=f"t{i}", suite=suite, tree_id=f"tree{i}a"))
        trees.append(make_chain_tree([[("direct", 0.9)]], task_id=f"t{i}", suite=suite, tree_id=f"tree{i}b"))

    train, holdout = optimizer.split_tasks(trees)
    train_tasks = {t.task_id for t in train}
    holdout_tasks = {t.task_id for t in holdout}
    assert train_tasks & holdout_tasks == set()
    assert len(train) + len(holdout) == len(trees)


# ======================================================================================
# Paired statistics
# ======================================================================================
def test_paired_comparison_is_exact_on_identical_rewards():
    rewards = [0.2, 0.5, 1.0, 0.0]
    result = paired_comparison(rewards, rewards, seed=0, bootstrap=50)
    assert result["n"] == 4
    assert result["n_changed"] == 0
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["z"] == pytest.approx(0.0)


def test_paired_comparison_detects_a_consistent_improvement():
    incumbent = [0.0] * 20
    champion = [0.25] * 20
    result = paired_comparison(incumbent, champion, seed=0, bootstrap=100)
    assert result["mean_delta"] == pytest.approx(0.25)
    assert result["n_changed"] == 20
    assert result["z"] > 5  # zero variance -> enormous z
    assert result["p_value"] < 0.01
    assert result["ci90"][0] > 0


def test_paired_comparison_handles_mismatched_lengths_and_empties():
    result = paired_comparison([0.0, 1.0], [1.0], seed=0, bootstrap=20)
    assert result["n"] == 1
    assert result["mean_delta"] == pytest.approx(1.0)
    assert paired_comparison([], [], seed=0)["n"] == 0


# ======================================================================================
# Profiler
# ======================================================================================
def test_profiler_measures_action_rewards_and_hypotheses(make_chain_tree):
    tree = make_chain_tree(
        [
            [("brute_force", 0.0), ("repair", 0.5), ("restart", 0.9)],
            [("brute_force", 0.1), ("repair", 1.0), ("restart", 0.8)],
            [("brute_force", 0.0), ("repair", 0.6), ("restart", 1.0)],
        ],
        choose="brute_force",
    )
    profiler = FailureProfiler()
    profile = profiler.profile([tree])

    assert profile["n_decisions"] == 3
    assert profile["actions"]["brute_force"]["n"] == 3
    assert profile["actions"]["brute_force"]["mean_reward"] == pytest.approx(0.1 / 3, abs=1e-6)
    assert profile["actions"]["restart"]["mean_reward"] == pytest.approx(0.9)
    # Only the third slate's restart=1.0 counts as a pass.
    assert profile["per_suite"]["humaneval"]["restart"]["pass_rate"] == pytest.approx(1 / 3)
    # `repair` only becomes "partial-conditional" evidence at the second decision
    # point, where the incoming artefact already scored above zero.
    assert profile["conditional"]["repair_when_partial"]["n"] == 1
    assert "restart" in profiler.report(profile)

    hypotheses = profiler.hypotheses(profile)
    assert hypotheses, "the profiler must produce at least one testable hypothesis"
    # The best measured action should be proposed for a positive prior.
    assert any("prior_restart" in patch for patch in hypotheses)


def test_profiler_handles_an_empty_corpus():
    profile = FailureProfiler().profile([])
    assert profile["n_decisions"] == 0
    assert profile["actions"] == {}
    assert FailureProfiler().hypotheses(profile) == []


# ======================================================================================
# Gate
# ======================================================================================
def _result(rewards, *, solved=0, n_trees=1, off_slate=0, policy_id="p", error=""):
    result = ReplayResult(policy_id=policy_id, n_trees=n_trees, n_decisions=len(rewards))
    result.rewards = list(rewards)
    result.mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    result.n_solved_trees = solved
    result.off_slate_picks = off_slate
    result.coverage = (len(rewards) - off_slate) / len(rewards) if rewards else 1.0
    result.task_solve_rate = solved / n_trees if n_trees else 0.0
    result.oracle_mean = max(rewards) if rewards else 0.0
    result.efficiency = (
        result.mean_reward / result.oracle_mean if result.oracle_mean else 0.0
    )
    result.error = error
    return result


def _optimizer(tmp_path, **overrides):
    return MetaOptimizer(
        storage_dir=tmp_path,
        trees_dir=tmp_path,
        config=OptimizerConfig(**overrides),
    )


def test_gate_accepts_a_consistent_improvement(tmp_path):
    optimizer = _optimizer(tmp_path)
    incumbent = _result([0.0] * 40, solved=1, n_trees=3)
    champion = _result([0.25] * 40, solved=2, n_trees=3)
    accepted, reason, paired = optimizer.gate(incumbent, champion)
    assert accepted, reason
    assert paired["mean_delta"] == pytest.approx(0.25)


def test_gate_rejects_a_small_gain(tmp_path):
    optimizer = _optimizer(tmp_path, min_holdout_improvement=0.05)
    incumbent = _result([0.0] * 20, solved=1)
    champion = _result([0.01] * 20, solved=1)
    accepted, reason, _ = optimizer.gate(incumbent, champion)
    assert not accepted
    assert "below threshold" in reason


def test_gate_rejects_an_insignificant_gain(tmp_path):
    """A positive but noisy advantage must not be promoted."""
    optimizer = _optimizer(tmp_path, min_holdout_improvement=0.005, min_z=1.0)
    incumbent = _result([0.0] * 20, solved=1)
    # Net +0.0175 per decision, but the wins and losses almost cancel.
    champion = _result([0.2] * 4 + [-0.15] * 3 + [0.0] * 13, solved=1)
    assert champion.mean_reward > 0.005
    accepted, reason, paired = optimizer.gate(incumbent, champion)
    assert not accepted
    assert "z-score" in reason
    assert paired["n_changed"] == 7


def test_gate_rejects_off_slate_picks(tmp_path):
    optimizer = _optimizer(tmp_path)
    incumbent = _result([0.0] * 30, solved=1)
    champion = _result([1.0] * 30, solved=3, off_slate=5)
    accepted, reason, _ = optimizer.gate(incumbent, champion)
    assert not accepted
    assert "off-slate" in reason


def test_gate_rejects_losing_too_many_solved_trees(tmp_path):
    """Trading away solved tasks for partial credit must be refused."""
    optimizer = _optimizer(tmp_path, max_solve_loss_tasks=1)
    incumbent = _result([1.0] * 4 + [0.0] * 26, solved=4, n_trees=4)
    champion = _result([0.9] * 4 + [0.5] * 26, solved=1, n_trees=4)
    assert champion.mean_reward > incumbent.mean_reward  # +0.42 per decision
    accepted, reason, _ = optimizer.gate(incumbent, champion)
    assert not accepted
    assert "solved trees" in reason


def test_gate_rejects_a_champion_with_a_replay_error(tmp_path):
    optimizer = _optimizer(tmp_path)
    incumbent = _result([0.0] * 10, solved=1)
    champion = _result([1.0] * 10, error="boom")
    accepted, reason, _ = optimizer.gate(incumbent, champion)
    assert not accepted
    assert "replay error" in reason


def test_gate_needs_holdout_decisions(tmp_path):
    optimizer = _optimizer(tmp_path)
    accepted, reason, _ = optimizer.gate(_result([]), _result([]))
    assert not accepted
    assert "no holdout decisions" in reason


# ======================================================================================
# Archive
# ======================================================================================
def test_archive_and_registry(tmp_path, make_chain_tree):
    storage = tmp_path / "storage"
    optimizer = MetaOptimizer(storage_dir=storage, trees_dir=storage / "trees")
    tree = make_chain_tree([[("direct", 0.5)]], task_id="t0", suite="math", tree_id="tree0")

    path = optimizer.archive_policy(
        POLICY_PATH.read_text(encoding="utf-8"),
        "policy_vTest",
        generation=3,
        train={"mean_reward": 0.5},
        holdout={"mean_reward": 0.4},
        accepted=True,
        extra={"origin": "unit-test"},
    )
    assert (path / "search_policy.py").exists()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["accepted"] is True
    assert metadata["holdout"]["mean_reward"] == 0.4
    assert metadata["origin"] == "unit-test"

    registry = json.loads((storage / "policies" / "registry.json").read_text(encoding="utf-8"))
    assert registry["current"] == "policy_vTest"
    assert [entry["policy_id"] for entry in registry["policies"]] == ["policy_vTest"]


def test_ensure_incumbent_archived_is_idempotent_and_measures(tmp_path, make_chain_tree):
    storage = tmp_path / "storage"
    optimizer = MetaOptimizer(storage_dir=storage, trees_dir=storage / "trees")
    trees = [
        make_chain_tree([[("direct", 0.5)]], task_id="t0", suite="math", tree_id="tree0"),
        make_chain_tree([[("direct", 1.0)]], task_id="t1", suite="math", tree_id="tree1"),
    ]
    first = optimizer.ensure_incumbent_archived(trees)
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["train"]["mean_reward"] is not None
    assert metadata["params"]["prior_restart"] == 0.0

    # Second call must not re-measure or move anything.
    again = optimizer.ensure_incumbent_archived(trees)
    assert again == first


def test_promote_writes_the_source_and_keeps_a_backup(tmp_path):
    """Promotion is the one place the tracker-ed policy file is rewritten."""
    policy_copy = tmp_path / "search_policy.py"
    shutil.copy2(POLICY_PATH, policy_copy)
    optimizer = MetaOptimizer(
        policy_path=policy_copy, storage_dir=tmp_path / "storage", trees_dir=tmp_path / "trees"
    )
    source = policy_copy.read_text(encoding="utf-8")
    candidate = Candidate(
        label="cand",
        params={**PARAMS, "prior_restart": 0.5},
        source=apply_params(source, {**PARAMS, "prior_restart": 0.5}, "policy_v1"),
        policy_id="policy_v1",
    )

    optimizer.promote(candidate)

    assert read_policy_id_literal(policy_copy.read_text(encoding="utf-8")) == "policy_v1"
    assert read_params_literal(policy_copy.read_text(encoding="utf-8"))["prior_restart"] == 0.5
    assert (tmp_path / "search_policy.py.bak").exists()


def test_result_from_dict_tolerates_a_minimal_payload():
    result = _result_from_dict({"n_trees": 2, "rewards": [1.0, 0.0]})
    assert result.n_trees == 2
    assert result.rewards == [1.0, 0.0]
    assert result.policy_id == ""


def test_gate_requires_a_minimum_number_of_holdout_decisions(tmp_path):
    """One lucky decision is not evidence.

    A live run with a tiny budget can produce a handful of holdout decisions where
    a single flipped pick yields a *perfect* paired z-score; the gate must refuse
    to promote on that.
    """
    optimizer = _optimizer(tmp_path, min_holdout_decisions=6)
    incumbent = _result([0.0, 0.0], solved=1)
    champion = _result([1.0, 1.0], solved=2)
    accepted, reason, paired = optimizer.gate(incumbent, champion)
    assert not accepted
    assert "at least 6" in reason
    assert paired["mean_delta"] == pytest.approx(1.0)  # the gain is real, just unproven

    # ... and the same gain on enough decisions is promotable.
    sufficient = _optimizer(tmp_path, min_holdout_decisions=6)
    accepted, reason, _ = sufficient.gate(
        _result([0.0] * 8, solved=1), _result([1.0] * 8, solved=2)
    )
    assert accepted, reason

