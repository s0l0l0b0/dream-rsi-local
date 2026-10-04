"""Tests for the Dream Engine: offline replay at zero inference cost.

The point of these tests is that replay is *arithmetically checkable*.  The
fixtures build trees with hand-chosen slates and rewards, so every metric
(mean reward, oracle, regret, efficiency, coverage) can be computed in the test
itself rather than trusted.
"""

from __future__ import annotations

import pytest

from benchmarks.evaluator import evaluator_fingerprint
from src.core.llm_client import NullLLMClient
from src.core.tree import Action, CandidateOutcome, DecisionRecord, DiscoveryTree
from src.engine.dreamer import (
    Dreamer,
    FirstCandidatePolicy,
    GlobalBestActionPolicy,
    ReplayConfig,
    UniformRandomPolicy,
)
from src.policy.base import ExplorationPolicy, PolicyContext
from tests.conftest import expected_slate_stats

SLATES = [
    [("direct", 0.2), ("repair", 1.0), ("restart", 0.4)],
    [("chain_of_thought", 0.5), ("test_driven", 0.25)],
    [("repair", 1.0), ("brute_force", 0.0), ("decompose", 0.75)],
]


# ======================================================================================
# Metric arithmetic
# ======================================================================================
def test_replay_metrics_match_hand_computation(make_chain_tree):
    tree = make_chain_tree(SLATES)
    expected = expected_slate_stats(SLATES)
    dreamer = Dreamer([tree], config=ReplayConfig(rng_seed=0))

    result = dreamer.replay(FirstCandidatePolicy(), label="first")

    assert result.n_decisions == int(expected["n_decisions"])
    assert result.n_trees == 1
    assert result.mean_reward == pytest.approx(expected["first_mean"])
    assert result.oracle_mean == pytest.approx(expected["oracle_mean"])
    assert result.regret == pytest.approx(expected["oracle_mean"] - expected["first_mean"])
    assert result.efficiency == pytest.approx(expected["first_mean"] / expected["oracle_mean"])
    assert result.coverage == pytest.approx(1.0)
    assert result.off_slate_picks == 0
    assert result.mean_slate_size == pytest.approx((3 + 2 + 3) / 3)
    assert result.rewards == pytest.approx([0.2, 0.5, 1.0])


def test_oracle_selector_attains_the_oracle_upper_bound(make_chain_tree):
    tree = make_chain_tree(SLATES)
    dreamer = Dreamer([tree], config=ReplayConfig(rng_seed=0))

    result = dreamer.replay(selector=dreamer.oracle_selector(), label="oracle")

    assert result.mean_reward == pytest.approx(result.oracle_mean)
    assert result.regret == pytest.approx(0.0)
    assert result.efficiency == pytest.approx(1.0)
    assert result.best_pick_rate == pytest.approx(1.0)
    assert result.pass_rate == pytest.approx(2 / 3)  # two slates contain a 1.0
    assert result.task_solve_rate == pytest.approx(1.0)


def test_task_solve_rate_requires_a_full_pass(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.5), ("repair", 0.75)]])
    dreamer = Dreamer([tree], config=ReplayConfig(rng_seed=0))
    result = dreamer.replay(selector=dreamer.oracle_selector(), label="oracle")
    assert result.n_solved_trees == 0
    assert result.task_solve_rate == pytest.approx(0.0)


def test_per_suite_and_per_action_breakdown(make_chain_tree):
    trees = [
        make_chain_tree(SLATES, task_id="t_he", suite="humaneval", tree_id="tree_he"),
        make_chain_tree([[("optimize", 1.0), ("direct", 0.0)]], task_id="t_kb", suite="kernelbench", tree_id="tree_kb"),
    ]
    result = Dreamer(trees, config=ReplayConfig(rng_seed=0)).replay(
        FirstCandidatePolicy(), label="first"
    )
    assert set(result.per_suite) == {"humaneval", "kernelbench"}
    assert result.per_suite["kernelbench"]["mean_reward"] == pytest.approx(1.0)
    assert sum(entry["n"] for entry in result.per_action.values()) == result.n_decisions


# ======================================================================================
# Coverage and off-slate picks
# ======================================================================================
def test_off_slate_pick_is_scored_zero_and_reported(make_chain_tree):
    tree = make_chain_tree(SLATES)

    class OffSlatePolicy(ExplorationPolicy):
        POLICY_ID = "off_slate"

        def select(self, ctx: PolicyContext) -> Action:
            return Action(name="never_attempted", params={})

    dreamer = Dreamer([tree], config=ReplayConfig(rng_seed=0))
    result = dreamer.replay(OffSlatePolicy(), label="off_slate")

    assert result.off_slate_picks == result.n_decisions
    assert result.coverage == pytest.approx(0.0)
    assert result.mean_reward == pytest.approx(0.0)


def test_max_decisions_truncates_the_replay(make_chain_tree):
    tree = make_chain_tree(SLATES)
    dreamer = Dreamer([tree], config=ReplayConfig(rng_seed=0, max_decisions=2))
    result = dreamer.replay(FirstCandidatePolicy())
    assert result.n_decisions == 2


def test_suite_and_task_filters(make_chain_tree):
    trees = [
        make_chain_tree([[("direct", 1.0)]], task_id="a", suite="math", tree_id="ta"),
        make_chain_tree([[("direct", 0.0)]], task_id="b", suite="humaneval", tree_id="tb"),
    ]
    by_suite = Dreamer(trees, config=ReplayConfig(suites=("math",)))
    assert by_suite.replay(FirstCandidatePolicy()).mean_reward == pytest.approx(1.0)

    by_task = Dreamer(trees, config=ReplayConfig(task_ids=("b",)))
    assert by_task.replay(FirstCandidatePolicy()).mean_reward == pytest.approx(0.0)


# ======================================================================================
# Zero-inference guarantee
# ======================================================================================
def test_replay_never_calls_a_model(make_chain_tree):
    tree = make_chain_tree(SLATES)
    null = NullLLMClient()
    dreamer = Dreamer([tree], config=ReplayConfig(rng_seed=0), llm=null)
    result = dreamer.replay(FirstCandidatePolicy())
    assert null.calls == 0
    assert result.llm_calls == 0
    assert isinstance(dreamer.llm, NullLLMClient)


def test_null_client_raises_if_used():
    from src.core.llm_client import GenerationRequest, LLMError

    request = GenerationRequest(
        task_id="t", suite="math", entry_point="f", prompt="p", signature="def f():",
        action=Action(name="direct"),
    )
    with pytest.raises(LLMError):
        NullLLMClient().generate(request)


# ======================================================================================
# Faithful learning signal
# ======================================================================================
def test_observe_receives_the_recorded_slate(make_chain_tree):
    """Replay must hand a policy the same slate observations it had live."""
    tree = make_chain_tree(SLATES)
    observed: list[list[tuple[str, float]]] = []

    class SpyPolicy(ExplorationPolicy):
        POLICY_ID = "spy"

        def select(self, ctx: PolicyContext) -> Action:
            return ctx.candidates[0]

        def observe(self, ctx, chosen, outcome, slate):
            observed.append(sorted((c.action.name, round(c.score, 6)) for c in slate))
            super().observe(ctx, chosen, outcome, slate)

    Dreamer([tree], config=ReplayConfig(rng_seed=0)).replay(SpyPolicy())

    assert observed == [sorted((name, score) for name, score in slate) for slate in SLATES]


def test_policy_is_shared_across_trees_like_live_exploration(make_chain_tree):
    """One policy instance sees every tree, mirroring a live multi-task run."""
    trees = [
        make_chain_tree([[(f"act{i}", 1.0)]], task_id=f"t{i}", tree_id=f"tree{i}")
        for i in range(3)
    ]

    class CountingPolicy(ExplorationPolicy):
        POLICY_ID = "counting"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seen = 0

        def select(self, ctx: PolicyContext) -> Action:
            self.seen += 1
            return ctx.candidates[0]

    policy = CountingPolicy()
    result = Dreamer(trees, config=ReplayConfig(rng_seed=0)).replay(policy)
    assert result.n_decisions == 3
    assert policy.seen == 3  # one instance, not one per tree


def test_broken_select_aborts_cleanly(make_chain_tree):
    tree = make_chain_tree(SLATES)

    class ExplodingPolicy(ExplorationPolicy):
        POLICY_ID = "boom"

        def select(self, ctx: PolicyContext) -> Action:
            raise ValueError("nope")

    result = Dreamer([tree], config=ReplayConfig(rng_seed=0)).replay(ExplodingPolicy())
    assert "nope" in result.error
    assert result.n_decisions == 0


# ======================================================================================
# Baselines and deduplication
# ======================================================================================
def test_baselines_cover_the_reference_points(make_chain_tree):
    tree = make_chain_tree(SLATES)
    baselines = Dreamer([tree], config=ReplayConfig(rng_seed=0)).baselines(uniform_seeds=4)
    assert set(baselines) == {"oracle", "uniform", "first_candidate", "global_best"}
    assert baselines["oracle"].mean_reward >= baselines["global_best"].mean_reward - 1e-9
    assert baselines["uniform"].metadata["n_seeds"] == 4
    assert len(baselines["uniform"].metadata["per_seed_mean_reward"]) == 4


def test_global_best_baseline_picks_the_best_observed_action(make_chain_tree):
    tree = make_chain_tree([[("good", 0.9), ("bad", 0.1)], [("bad", 0.2), ("good", 1.0)]])
    result = Dreamer([tree], config=ReplayConfig(rng_seed=0)).replay(GlobalBestActionPolicy())
    # First decision ties (nothing observed) -> first candidate; then `good` is known best.
    assert result.mean_reward == pytest.approx((0.9 + 1.0) / 2)


def test_uniform_baseline_is_seed_deterministic(make_chain_tree):
    tree = make_chain_tree(SLATES)
    a = Dreamer([tree], config=ReplayConfig(rng_seed=3)).replay(UniformRandomPolicy(rng_seed=3))
    b = Dreamer([tree], config=ReplayConfig(rng_seed=3)).replay(UniformRandomPolicy(rng_seed=3))
    assert a.rewards == b.rewards


def test_duplicate_decision_points_are_collapsed(make_chain_tree):
    a = make_chain_tree(SLATES, tree_id="tree_a")
    b = make_chain_tree(SLATES, tree_id="tree_b")  # identical evidence

    deduped = Dreamer([a, b], config=ReplayConfig(rng_seed=0))
    result = deduped.replay(FirstCandidatePolicy())
    assert result.n_decisions == len(SLATES)
    assert deduped.duplicates_skipped == len(SLATES)

    raw = Dreamer([a, b], config=ReplayConfig(rng_seed=0, deduplicate=False))
    assert raw.replay(FirstCandidatePolicy()).n_decisions == 2 * len(SLATES)


def test_different_rewards_are_not_collapsed(make_chain_tree):
    a = make_chain_tree([[("direct", 0.2)]], tree_id="tree_a")
    b = make_chain_tree([[("direct", 0.8)]], tree_id="tree_b")
    result = Dreamer([a, b], config=ReplayConfig(rng_seed=0)).replay(FirstCandidatePolicy())
    assert result.n_decisions == 2
    assert result.mean_reward == pytest.approx(0.5)


# ======================================================================================
# Frozen-oracle guard
# ======================================================================================
def test_replay_refuses_a_tree_recorded_by_another_oracle(make_chain_tree):
    tree = make_chain_tree(SLATES, fingerprint=False)
    tree.metadata["evaluator_sha256"] = "deadbeef-000000000000"
    with pytest.raises(RuntimeError, match="recorded with evaluator"):
        Dreamer([tree], config=ReplayConfig(verify_oracle=True))

    # Opting out is possible for exploratory analysis.
    assert Dreamer([tree], config=ReplayConfig(verify_oracle=False)).decision_count() > 0


def test_replay_result_roundtrips_through_dict(make_chain_tree):
    from src.meta.optimizer import _result_from_dict

    tree = make_chain_tree(SLATES)
    result = Dreamer([tree], config=ReplayConfig(rng_seed=0)).replay(FirstCandidatePolicy())
    restored = _result_from_dict(result.to_dict())
    assert restored.n_decisions == result.n_decisions
    assert restored.rewards == pytest.approx(result.rewards)
    assert restored.n_solved_trees == result.n_solved_trees
    assert restored.row() == result.row()
