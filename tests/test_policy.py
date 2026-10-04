"""Tests for the policy SDK (``src/policy/base.py``) and the evolving policy.

Contexts are built directly from :class:`~src.policy.base.PolicyContext` and
:func:`~src.core.tree.state_features`, so nothing here touches the filesystem or
spawns a process.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Sequence

import pytest

from src.core.tree import (
    Action,
    CandidateOutcome,
    DecisionRecord,
    state_features,
)
from src.policy.base import (
    ACTION_NAMES,
    ACTION_SPECS,
    SUITES,
    ActionSpace,
    PolicyContext,
    context_from_record,
)
from src.policy.search_policy import PARAMS, SearchPolicy, param_space

# ======================================================================================
# Helpers
# ======================================================================================
def zero_params() -> dict[str, float]:
    """The all-zero weight vector -- ``policy_v0``'s "take the first option" regime."""
    return {name: 0.0 for name in param_space()}


def make_ctx(
    candidates: Sequence[Action] = (),
    *,
    suite: str = "humaneval",
    best_score: float = 0.0,
    depth: int = 0,
    solution: str = "",
    feedback: str = "",
    difficulty: float = 0.5,
    history: Sequence[Mapping[str, Any]] = (),
    action_counts: Mapping[str, int] | None = None,
) -> PolicyContext:
    features = state_features(
        {"solution": solution, "feedback": feedback},
        depth=depth,
        history=list(history),
        action_counts=action_counts or {},
        extra={"difficulty": difficulty},
    )
    features["best_score"] = float(best_score)
    return PolicyContext(
        task_id="task_a",
        suite=suite,
        entry_point="f",
        signature="def f(x):",
        prompt="Return x.",
        difficulty=difficulty,
        current_state={"solution": solution},
        feedback=feedback,
        history=[dict(item) for item in history],
        features=features,
        candidates=list(candidates),
    )


def with_candidates(ctx: PolicyContext, candidates: Sequence[Action]) -> PolicyContext:
    return PolicyContext(**{**ctx.__dict__, "candidates": list(candidates)})


def make_record(*, suite: str = "humaneval") -> DecisionRecord:
    slate = [ActionSpace().build(name) for name in ("direct", "repair", "optimize")]
    rewards = {action.signature(): value for action, value in zip(slate, (0.123456, 0.654321, 0.111111))}
    return DecisionRecord(
        decision_id="tree_x:node_1",
        tree_id="tree_x",
        task_id="he_001",
        node_id="node_1",
        depth=2,
        suite=suite,
        features={"depth": 2.0, "best_score": 0.5},
        history=[
            {"node_id": "root", "depth": 0, "action": "root", "score": 0.0, "feedback": ""},
            {"node_id": "node_0", "depth": 1, "action": "direct", "score": 0.5, "feedback": "1/2"},
        ],
        slate=slate,
        rewards=rewards,
        chosen_action_id=slate[0].signature(),
        chosen_score=0.123456,
        current_state={"solution": "def f(x):\n    return 0\n"},
        feedback="1/2 tests passed",
        task={
            "id": "he_001",
            "suite": suite,
            "entry_point": "solve",
            "signature": "def solve(n):",
            "prompt": "Solve it.",
            "difficulty": 0.42,
            "tags": ["algebra", "numbers"],
        },
    )


# ======================================================================================
# PARAMS / param_space
# ======================================================================================
def test_params_and_param_space_cover_the_same_keys() -> None:
    space = param_space()
    assert set(space) == set(PARAMS)
    assert set(SearchPolicy.param_space()) == set(PARAMS)
    assert set(SearchPolicy().params) == set(PARAMS)

    assert {f"prior_{name}" for name in ACTION_NAMES} <= set(PARAMS)
    assert {f"sp_{suite}_{name}" for suite in SUITES for name in ACTION_NAMES} <= set(PARAMS)
    for name in ACTION_NAMES:
        assert name in ACTION_SPECS


def test_param_space_bounds_are_ordered_and_contain_the_defaults() -> None:
    space = param_space()
    for name, bounds in space.items():
        assert isinstance(bounds, (tuple, list)) and len(bounds) == 2, name
        low, high = float(bounds[0]), float(bounds[1])
        assert low < high, f"{name}: empty range ({low}, {high})"
        default = float(PARAMS[name])
        assert low <= default <= high, f"{name}: default {default} outside ({low}, {high})"


# ======================================================================================
# SearchPolicy.select
# ======================================================================================
@pytest.mark.parametrize("name", ["direct", "repair", "optimize"])
def test_select_returns_a_candidate_action(name: str) -> None:
    space = ActionSpace()
    base = make_ctx(best_score=0.5, depth=1, solution="def f(x):\n    return 0\n", feedback="1/2")
    slate = space.slate(base, 4)
    slate = [space.build(name), *slate]
    ctx = with_candidates(base, slate)

    chosen = SearchPolicy().select(ctx)
    assert isinstance(chosen, Action)
    assert chosen.signature() in {action.signature() for action in slate}


def test_select_is_deterministic_for_an_identical_context() -> None:
    space = ActionSpace()
    base = make_ctx(best_score=0.5, depth=1, solution="def f(x):\n    return 0\n")
    ctx = with_candidates(base, space.slate(base, 4))

    policy = SearchPolicy()
    assert policy.select(ctx).signature() == policy.select(ctx).signature()

    fresh = SearchPolicy()
    assert fresh.select(ctx).signature() == policy.select(ctx).signature()


def test_all_zero_weights_pick_the_first_candidate() -> None:
    space = ActionSpace()
    base = make_ctx()
    slate = space.slate(base, 6)
    assert len(slate) > 1
    ctx = with_candidates(base, slate)

    chosen = SearchPolicy(zero_params()).select(ctx)
    assert chosen.signature() == slate[0].signature(), (
        f"zero weights must tie and take the first candidate, got {chosen.name}"
    )


def test_prior_makes_a_candidate_win() -> None:
    space = ActionSpace()
    slate = [space.build(name) for name in ("direct", "optimize", "decompose", "brute_force")]
    params = zero_params()
    params["prior_optimize"] = 1.0
    ctx = make_ctx(slate)

    chosen = SearchPolicy(params).select(ctx)
    assert chosen.name == "optimize"
    assert chosen.name != slate[0].name


def test_repair_bonus_wins_when_a_partial_artefact_exists() -> None:
    space = ActionSpace()
    slate = [space.build(name) for name in ("direct", "optimize", "repair")]
    params = zero_params()
    params["repair_bonus"] = 3.0
    policy = SearchPolicy(params)

    partial = make_ctx(slate, best_score=0.5, solution="def f(x):\n    return 0\n", feedback="1/2")
    assert partial.has_partial_solution()
    assert policy.select(partial).name == "repair"

    # Control: with no partial artefact ``repair`` earns no bonus, so the tie
    # breaks on slate order.
    empty = make_ctx(slate, best_score=0.0, solution="")
    assert not empty.has_partial_solution()
    assert policy.select(empty).name == slate[0].name


# ======================================================================================
# SearchPolicy.observe / online statistics
# ======================================================================================
def test_observe_updates_statistics_with_the_configured_decay() -> None:
    space = ActionSpace()
    policy = SearchPolicy()
    decay = policy.param("value_decay")
    ctx = make_ctx(suite="math")

    first = [
        CandidateOutcome(action=space.build("direct"), score=0.4),
        CandidateOutcome(action=space.build("chain_of_thought"), score=0.9),
    ]
    policy.observe(ctx, first[1], first[1], first)

    assert policy._observations == 1
    assert policy.counts == {"direct": 1, "chain_of_thought": 1}
    assert policy.suite_values["math"]["direct"] == pytest.approx(0.4)
    assert policy.suite_values["math"]["chain_of_thought"] == pytest.approx(0.9)
    assert policy.global_values["direct"] == pytest.approx(0.4)
    assert set(policy.suite_values) == {"math"}

    second = [CandidateOutcome(action=space.build("direct"), score=1.0)]
    policy.observe(ctx, second[0], second[0], second)

    expected = 0.4 * decay + 1.0 * (1 - decay)
    assert policy._observations == 2
    assert policy.counts["direct"] == 2
    assert policy.suite_values["math"]["direct"] == pytest.approx(expected)
    assert policy.global_values["direct"] == pytest.approx(expected)


def test_high_suite_value_steers_toward_the_historically_better_action() -> None:
    space = ActionSpace()
    observed_ctx = make_ctx(suite="humaneval")
    observed = [
        CandidateOutcome(action=space.build("direct"), score=1.0),
        CandidateOutcome(action=space.build("optimize"), score=0.0),
    ]

    slate = [space.build("optimize"), space.build("direct")]
    decision_ctx = make_ctx(slate, suite="humaneval")

    learning = SearchPolicy({**zero_params(), "w_suite_value": 3.0})
    learning.observe(observed_ctx, observed[0], observed[0], observed)
    assert learning.select(decision_ctx).name == "direct"

    # Same observations, no weight -> the slate order decides, proving the
    # steering above came from the accumulated statistic.
    inert = SearchPolicy(zero_params())
    inert.observe(observed_ctx, observed[0], observed[0], observed)
    assert inert.select(decision_ctx).name == "optimize"


# ======================================================================================
# SearchPolicy.should_stop
# ======================================================================================
def test_should_stop_respects_stop_at() -> None:
    policy = SearchPolicy()
    stop_at = policy.param("stop_at")
    assert stop_at == PARAMS["stop_at"]

    assert policy.should_stop(make_ctx(best_score=stop_at)) is True
    assert policy.should_stop(make_ctx(best_score=stop_at - 0.01)) is False

    early = SearchPolicy({"stop_at": 0.5})
    assert early.should_stop(make_ctx(best_score=0.5)) is True
    assert early.should_stop(make_ctx(best_score=0.49)) is False


# ======================================================================================
# PolicyContext is reward-blind
# ======================================================================================
def test_policy_context_exposes_no_slate_rewards() -> None:
    record = make_record()
    ctx = context_from_record(record)

    for forbidden in ("rewards", "scores", "slate_rewards", "candidate_scores", "oracle_score"):
        assert not hasattr(ctx, forbidden), f"PolicyContext must not expose {forbidden!r}"

    for field in dataclasses.fields(PolicyContext):
        assert "reward" not in field.name, field.name
        assert "score" not in field.name, field.name

    assert ctx.candidates, "the recorded slate should round-trip"
    for candidate in ctx.candidates:
        assert isinstance(candidate, Action)
        assert not hasattr(candidate, "score")

    blob = json.dumps({k: v for k, v in ctx.__dict__.items() if k != "candidates"}, default=str)
    for reward in record.rewards.values():
        assert repr(reward) not in blob, f"recorded reward {reward} leaked into the context"


# ======================================================================================
# ActionSpace.slate
# ======================================================================================
def test_slate_is_deterministic_and_respects_the_budget() -> None:
    space = ActionSpace()
    base = make_ctx(best_score=0.4, depth=1, solution="def f(x):\n    return 0\n")

    first = space.slate(base, 3)
    second = space.slate(base, 3)
    assert [action.signature() for action in first] == [action.signature() for action in second]
    assert len(first) == 3
    assert len({action.name for action in first}) == 3

    applicable = len(space.applicable(base))
    for budget in (1, 2, 3, 5):
        picked = space.slate(base, budget)
        assert len(picked) == min(budget, applicable), f"budget {budget}"
        assert [action.name for action in picked] == [
            action.name for action in space.slate(base, budget)
        ]


def test_slate_only_proposes_applicable_actions() -> None:
    space = ActionSpace()

    root = make_ctx(best_score=0.0, solution="", depth=0)
    assert not root.has_partial_solution()
    root_names = [action.name for action in space.slate(root, 99)]
    assert root_names == space.applicable(root)
    assert "repair" not in root_names, "repair needs a partial artefact"
    assert "restart" not in root_names, "restart needs history"
    assert set(root_names) <= set(ACTION_SPECS)

    deep = make_ctx(
        best_score=0.5,
        solution="def f(x):\n    return 0\n",
        depth=2,
        history=[
            {"node_id": "root", "depth": 0, "action": "root", "score": 0.0, "feedback": ""},
            {"node_id": "n1", "depth": 1, "action": "direct", "score": 0.5, "feedback": "1/2"},
        ],
    )
    deep_names = [action.name for action in space.slate(deep, 99)]
    assert "repair" in deep_names
    assert "restart" in deep_names


def test_slate_returns_every_applicable_action_when_budget_is_large() -> None:
    space = ActionSpace()
    # depth >= 1 makes ``restart`` legal, while the empty artefact keeps
    # ``repair`` illegal -- so the applicable set is a strict subset.
    ctx = make_ctx(best_score=0.0, solution="", depth=1)
    applicable = space.applicable(ctx)
    assert 0 < len(applicable) < len(ACTION_SPECS), "fixture should leave some actions inapplicable"
    assert "repair" not in applicable

    for budget in (len(applicable), len(applicable) + 1, len(ACTION_SPECS) * 3):
        assert [action.name for action in space.slate(ctx, budget)] == applicable


# ======================================================================================
# context_from_record
# ======================================================================================
def test_context_from_record_round_trips_the_recorded_view() -> None:
    record = make_record()
    ctx = context_from_record(record, policy_id="candidate_7", rng_seed=11)

    assert ctx.task_id == record.task_id
    assert ctx.suite == "humaneval"
    assert ctx.entry_point == "solve"
    assert ctx.signature == "def solve(n):"
    assert ctx.prompt == "Solve it."
    assert ctx.difficulty == pytest.approx(0.42)
    assert ctx.history == record.history
    assert ctx.features == record.features
    assert ctx.feedback == record.feedback
    assert ctx.current_state == record.current_state
    assert ctx.tags == ("algebra", "numbers")
    assert ctx.policy_id == "candidate_7"
    assert ctx.rng_seed == 11
    assert ctx.depth == 2
    assert ctx.best_score == pytest.approx(0.5)
    assert [action.signature() for action in ctx.candidates] == [
        action.signature() for action in record.slate
    ]

    # An empty record suite falls back to the task's suite.
    fallback = context_from_record(dataclasses.replace(record, suite=""))
    assert fallback.suite == "humaneval"
