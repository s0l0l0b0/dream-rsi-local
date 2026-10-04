"""The Dream Engine: offline replay of recorded discovery trees.

This is the component that makes recursive self-improvement affordable.  Instead
of spending model calls to find out whether a candidate policy is better, it
re-asks a *historical* question: given the exact decision points recorded in
past discovery trees, and the exact rewards that were measured there, what would
this policy have done -- and what would it have scored?

How a replay works
------------------
For every recorded decision point (see :meth:`src.core.tree.DiscoveryTree.decisions`):

1. build the policy's input with
   :func:`src.policy.base.context_from_record` -- byte-identical to what the live
   policy saw, and with **no reward fields**, so the answer key cannot leak;
2. ask the policy to pick an action from the recorded slate;
3. look the pick up in the recorded rewards and record it;
4. call ``policy.observe(...)`` with the recorded slate, exactly as the live
   explorer did, so a *learning* policy is replayed faithfully rather than being
   judged as if it had amnesia.

Why the comparison is fair
--------------------------
All policies are replayed over the same decision points and the same slates, so
differences in score are attributable to the *decision rule* alone.  Rewards were
measured by the immutable evaluator when the trees were recorded; replay spawns
no model call at all -- pass a :class:`~src.core.llm_client.NullLLMClient` and any
accidental inference raises instead of costing money.

What replay cannot do
---------------------
* It cannot score an action that was **not in the recorded slate** (no coverage,
  no evidence).  :attr:`ReplayResult.coverage` and ``off_slate_picks`` quantify
  this; a policy that wants an unrecorded action is scored ``0`` for that pick.
* It cannot evaluate a changed **candidate proposal** distribution, because the
  slates are frozen from the recording.  That is why the framework, not the
  policy, owns proposal by default.
* It is only valid across trees recorded with the **same evaluator**: the
  fingerprint stored in each tree is checked before replay.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from benchmarks.evaluator import assert_immutable, evaluator_fingerprint
from src.core.llm_client import LLMClient, NullLLMClient
from src.core.tree import (
    Action,
    CandidateOutcome,
    DecisionRecord,
    DiscoveryTree,
    tree_sort_key,
)
from src.core.utils import mean, safe_div, stdev
from src.policy.base import (
    DEFAULT_ACTION_SPACE,
    ExplorationPolicy,
    PolicyContext,
    context_from_record,
)

#: A selector maps a recorded decision to a chosen action.  Real policies are
#: wrapped as ``lambda record, ctx: policy.select(ctx)``; diagnostics such as the
#: oracle use the record's rewards directly.
Selector = Callable[[DecisionRecord, PolicyContext], Action]


# ======================================================================================
# Results
# ======================================================================================
@dataclass
class ReplayResult:
    """Metrics produced by replaying one policy over a set of trees."""

    policy_id: str = ""
    label: str = ""
    n_trees: int = 0
    n_decisions: int = 0
    #: Mean measured reward of the replayed choices -- the headline "Dream score".
    mean_reward: float = 0.0
    #: Fraction of decisions whose chosen candidate fully passed.
    pass_rate: float = 0.0
    #: Fraction of *trees* where at least one replayed pick fully passed.
    task_solve_rate: float = 0.0
    #: Number of trees with at least one fully-passing replayed pick.
    n_solved_trees: int = 0
    #: Best achievable mean reward given the recorded slates (upper bound).
    oracle_mean: float = 0.0
    #: ``oracle_mean - mean_reward`` (lower is better).
    regret: float = 0.0
    #: ``mean_reward / oracle_mean`` -- normalised skill in ``[0, 1]``.
    efficiency: float = 0.0
    #: Fraction of picks that were the best option available in their slate.
    best_pick_rate: float = 0.0
    #: Fraction of decisions in which the pick existed in the recorded slate.
    coverage: float = 1.0
    off_slate_picks: int = 0
    mean_slate_size: float = 0.0
    distinct_actions: int = 0
    #: Model calls performed during replay (must be 0 for the Dream Engine).
    llm_calls: int = 0
    wall_ms: float = 0.0
    per_suite: dict[str, dict[str, float]] = field(default_factory=dict)
    per_action: dict[str, dict[str, float]] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.task_solve_rate > 0.0

    def to_dict(self) -> dict[str, Any]:
        data = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"rewards", "per_suite", "per_action", "metadata"}
        }
        data["rewards"] = [round(r, 4) for r in self.rewards]
        data["per_suite"] = self.per_suite
        data["per_action"] = self.per_action
        data["metadata"] = self.metadata
        return data

    def summary(self) -> str:
        if self.error:
            return f"{self.label or self.policy_id}: ERROR {self.error}"
        return (
            f"{self.label or self.policy_id}: dream_score={self.mean_reward:.4f} "
            f"pass_rate={self.pass_rate:.3f} solve_rate={self.task_solve_rate:.3f} "
            f"efficiency={self.efficiency:.3f} coverage={self.coverage:.3f} "
            f"n_decisions={self.n_decisions}"
        )

    def row(self) -> dict[str, Any]:
        """Flat row for tables / JSONL reports."""
        return {
            "label": self.label or self.policy_id,
            "mean_reward": round(self.mean_reward, 4),
            "pass_rate": round(self.pass_rate, 4),
            "task_solve_rate": round(self.task_solve_rate, 4),
            "oracle_mean": round(self.oracle_mean, 4),
            "regret": round(self.regret, 4),
            "efficiency": round(self.efficiency, 4),
            "best_pick_rate": round(self.best_pick_rate, 4),
            "coverage": round(self.coverage, 4),
            "distinct_actions": self.distinct_actions,
            "n_decisions": self.n_decisions,
        }


# ======================================================================================
# Diagnostic (non-evolving) policies used as baselines
# ======================================================================================
class FirstCandidatePolicy(ExplorationPolicy):
    """The incumbent's behaviour in this repo: take the mixer's first option.

    Equivalent to the untuned ``policy_v0`` and therefore the honest "did the RSI
    loop actually achieve anything?" floor.
    """

    POLICY_ID = "baseline_first"

    def select(self, ctx: PolicyContext) -> Action:
        return ctx.candidates[0]


class UniformRandomPolicy(ExplorationPolicy):
    """Pick uniformly from the slate (seeded, reproducible)."""

    POLICY_ID = "baseline_uniform"

    def select(self, ctx: PolicyContext) -> Action:
        return ctx.rng("uniform").choice(list(ctx.candidates))


class GlobalBestActionPolicy(ExplorationPolicy):
    """Greedy on the pooled observed reward of each action.

    A learning baseline with no priors and no state features: if an evolved
    policy cannot beat this, evolution has bought nothing but parameters.
    """

    POLICY_ID = "baseline_global_best"

    def __init__(self, params: Mapping[str, float] | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def select(self, ctx: PolicyContext) -> Action:
        return max(
            ctx.candidates,
            key=lambda a: (self.values.get(a.name, 0.0), -ctx.candidates.index(a)),
        )

    def observe(
        self,
        ctx: PolicyContext,
        chosen: Action,
        outcome: CandidateOutcome,
        slate: Sequence[CandidateOutcome],
    ) -> None:
        for candidate in slate:
            name = candidate.action.name
            n = self.counts.get(name, 0)
            self.values[name] = (self.values.get(name, 0.0) * n + float(candidate.score)) / (n + 1)
            self.counts[name] = n + 1


# ======================================================================================
# The Dream Engine
# ======================================================================================
@dataclass
class ReplayConfig:
    """Knobs for a replay run."""

    rng_seed: int = 0
    #: Stop after this many decisions (``None`` = all).
    max_decisions: int | None = None
    suites: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    #: Verify the frozen-evaluator fingerprint before replaying.
    verify_oracle: bool = True
    #: Collapse decision points that are byte-identical evidence (see
    #: :attr:`~src.core.tree.DecisionRecord.dedup_key`).
    deduplicate: bool = True


class Dreamer:
    """Replays discovery trees against candidate policies, at zero LLM cost."""

    def __init__(
        self,
        trees: Iterable[DiscoveryTree],
        *,
        config: ReplayConfig | None = None,
        llm: LLMClient | None = None,
        action_space: Any = None,
    ) -> None:
        self.config = config or ReplayConfig()
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        #: Null by default: replay is structurally forbidden from calling a model.
        self.llm: LLMClient = llm or NullLLMClient()
        self.trees: list[DiscoveryTree] = self._filter(list(trees))
        #: Number of duplicate decision points collapsed by deduplication.
        self.duplicates_skipped: int = 0
        self._validate_oracle()

    # -- setup ------------------------------------------------------------------------
    def _filter(self, trees: list[DiscoveryTree]) -> list[DiscoveryTree]:
        selected = []
        # Deterministic *and* meaningful order (task, world, phase): a policy with
        # online statistics observes trees in this order, so it must not depend on
        # random UUIDs -- see :func:`src.core.tree.tree_sort_key`.
        for tree in sorted(trees, key=tree_sort_key):
            if self.config.task_ids and tree.task_id not in self.config.task_ids:
                continue
            if self.config.suites and str(tree.task_meta.get("suite", "")) not in self.config.suites:
                continue
            selected.append(tree)
        return selected

    def _validate_oracle(self) -> None:
        """Refuse to compare measurements taken by different graders."""
        if not self.config.verify_oracle:
            return
        current = evaluator_fingerprint()
        for tree in self.trees:
            recorded = str(tree.metadata.get("evaluator_sha256", ""))
            if recorded and recorded != current:
                raise RuntimeError(
                    f"tree {tree.tree_id} was recorded with evaluator {recorded[:12]}... "
                    f"but the current evaluator is {current[:12]}...; "
                    "re-record the discovery trees or restore benchmarks/evaluator.py"
                )

    def decision_count(self) -> int:
        return sum(len(t.decisions()) for t in self.trees)

    def _records(self) -> list[tuple[DiscoveryTree, DecisionRecord]]:
        pairs: list[tuple[DiscoveryTree, DecisionRecord]] = []
        seen: set[str] = set()
        for tree in self.trees:
            for record in tree.decisions():
                if self.config.deduplicate:
                    key = record.dedup_key
                    if key in seen:
                        self.duplicates_skipped += 1
                        continue
                    seen.add(key)
                pairs.append((tree, record))
                if self.config.max_decisions and len(pairs) >= self.config.max_decisions:
                    return pairs
        return pairs

    # -- replay -----------------------------------------------------------------------
    def replay(
        self,
        policy: ExplorationPolicy | None = None,
        *,
        selector: Selector | None = None,
        label: str = "",
    ) -> ReplayResult:
        """Replay ``policy`` (or a raw ``selector``) over the recorded decisions.

        The policy instance is deliberately *shared across all trees*, mirroring
        live exploration where one agent works through the whole task list and
        accumulates experience as it goes.
        """
        if policy is None and selector is None:
            raise ValueError("replay needs either a policy or a selector")
        if selector is None:
            assert policy is not None
            selector = lambda record, ctx: policy.select(ctx)  # noqa: E731

        pairs = self._records()
        started = time.perf_counter()
        rewards: list[float] = []
        oracle: list[float] = []
        best_picks: list[bool] = []
        slate_sizes: list[int] = []
        off_slate = 0
        per_suite: dict[str, dict[str, float]] = {}
        per_action: dict[str, dict[str, float]] = {}
        tree_rewards: dict[str, list[float]] = {}
        chosen_actions: set[str] = set()
        error = ""

        for tree, record in pairs:
            ctx = context_from_record(
                record,
                policy_id=getattr(policy, "policy_id", label),
                rng_seed=self.config.rng_seed,
                candidates=record.slate,
            )
            try:
                chosen = selector(record, ctx)
            except Exception as exc:  # noqa: BLE001 - a broken policy must not abort a report
                error = f"{type(exc).__name__}: {exc}"
                break
            if not isinstance(chosen, Action):  # defensive: contract violation
                error = f"selector returned {type(chosen).__name__}, expected Action"
                break

            reward = record.reward_of(chosen)
            if reward is None:
                reward = 0.0
                off_slate += 1
            rewards.append(float(reward))
            oracle.append(record.oracle_score)
            best_picks.append(abs(float(reward) - record.oracle_score) < 1e-9)
            slate_sizes.append(len(record.slate))
            tree_rewards.setdefault(tree.tree_id, []).append(float(reward))
            chosen_actions.add(chosen.name)

            suite = record.suite or str(tree.task_meta.get("suite", ""))
            bucket = per_suite.setdefault(suite, {"n": 0.0, "score_sum": 0.0, "n_solved": 0.0})
            bucket["n"] += 1
            bucket["score_sum"] += float(reward)
            bucket["n_solved"] += 1.0 if float(reward) >= 1.0 else 0.0
            action_bucket = per_action.setdefault(
                chosen.name, {"n": 0.0, "score_sum": 0.0, "n_passed": 0.0}
            )
            action_bucket["n"] += 1
            action_bucket["score_sum"] += float(reward)
            action_bucket["n_passed"] += 1.0 if float(reward) >= 1.0 else 0.0

            # Mirror the learning signal the live agent had at this point.
            if policy is not None:
                slate_outcomes = [
                    CandidateOutcome(action=action, score=float(record.rewards[action.signature()]))
                    for action in record.slate
                ]
                chosen_outcome = next(
                    (c for c in slate_outcomes if c.action.signature() == chosen.signature()),
                    CandidateOutcome(action=chosen, score=float(reward)),
                )
                try:
                    policy.observe(ctx, chosen, chosen_outcome, slate_outcomes)
                except Exception as exc:  # noqa: BLE001
                    error = f"observe() failed: {type(exc).__name__}: {exc}"
                    break

        solved_trees = sum(1 for values in tree_rewards.values() if max(values, default=0.0) >= 1.0)
        oracle_mean = mean(oracle)
        result = ReplayResult(
            policy_id=getattr(policy, "policy_id", label or "selector"),
            label=label or getattr(policy, "policy_id", ""),
            n_trees=len(tree_rewards),
            n_decisions=len(rewards),
            mean_reward=mean(rewards),
            pass_rate=safe_div(sum(1 for r in rewards if r >= 1.0), len(rewards)),
            task_solve_rate=safe_div(solved_trees, len(tree_rewards)),
            n_solved_trees=solved_trees,
            oracle_mean=oracle_mean,
            regret=oracle_mean - mean(rewards),
            efficiency=safe_div(mean(rewards), oracle_mean),
            best_pick_rate=safe_div(sum(1 for b in best_picks if b), len(best_picks)),
            coverage=safe_div(len(rewards) - off_slate, len(rewards), 1.0),
            off_slate_picks=off_slate,
            mean_slate_size=mean([float(s) for s in slate_sizes]),
            distinct_actions=len(chosen_actions),
            llm_calls=self.llm.calls,
            wall_ms=(time.perf_counter() - started) * 1000.0,
            per_suite=self._finalise(per_suite, "score_sum"),
            per_action=self._finalise(per_action, "score_sum"),
            rewards=rewards,
            error=error,
            metadata={
                "config": {
                    "rng_seed": self.config.rng_seed,
                    "suites": list(self.config.suites),
                    "task_ids": list(self.config.task_ids),
                    "max_decisions": self.config.max_decisions,
                },
                "evaluator_sha256": evaluator_fingerprint(),
                "reward_stdev": stdev(rewards),
                "n_suites": len(per_suite),
            },
        )
        return result

    @staticmethod
    def _finalise(buckets: dict[str, dict[str, float]], key: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, bucket in sorted(buckets.items()):
            n = bucket.get("n", 0.0) or 1.0
            entry = dict(bucket)
            entry["mean_reward"] = bucket.get(key, 0.0) / n
            if "n_solved" in bucket:
                entry["pass_rate"] = bucket["n_solved"] / n
            if "n_passed" in bucket:
                entry["pass_rate"] = bucket["n_passed"] / n
            out[name] = entry
        return out

    # -- baselines --------------------------------------------------------------------
    def oracle_selector(self) -> Selector:
        """Upper bound: always take the best recorded option in the slate."""

        def select(record: DecisionRecord, ctx: PolicyContext) -> Action:
            return max(record.slate, key=lambda a: record.rewards.get(a.signature(), 0.0))

        return select

    def baselines(self, uniform_seeds: int = 5) -> dict[str, ReplayResult]:
        """Replay the diagnostic reference points used in every report.

        The random baseline is averaged over several seeds: a single seeded draw
        of a stochastic policy is a very noisy estimate of its expected value, and
        quoting one unlucky draw would flatter the evolved policy for no reason.
        """
        results: dict[str, ReplayResult] = {}
        results["oracle"] = self.replay(selector=self.oracle_selector(), label="oracle")
        results["first_candidate"] = self.replay(FirstCandidatePolicy(), label="first_candidate")
        results["global_best"] = self.replay(GlobalBestActionPolicy(), label="global_best")

        draws: list[ReplayResult] = []
        for offset in range(max(1, uniform_seeds)):
            draws.append(
                self.replay(
                    UniformRandomPolicy(rng_seed=self.config.rng_seed + offset),
                    label="uniform",
                )
            )
        averaged = copy.deepcopy(draws[-1])
        averaged.mean_reward = mean([d.mean_reward for d in draws])
        averaged.pass_rate = mean([d.pass_rate for d in draws])
        averaged.task_solve_rate = mean([d.task_solve_rate for d in draws])
        averaged.efficiency = mean([d.efficiency for d in draws])
        averaged.metadata = {
            **averaged.metadata,
            "n_seeds": len(draws),
            "per_seed_mean_reward": [round(d.mean_reward, 4) for d in draws],
        }
        results["uniform"] = averaged
        return results


__all__ = [
    "Dreamer",
    "FirstCandidatePolicy",
    "GlobalBestActionPolicy",
    "ReplayConfig",
    "ReplayResult",
    "Selector",
    "UniformRandomPolicy",
]
