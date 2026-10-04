"""THE EVOLVING TARGET -- the executable search policy.

Everything in this repository is fixed except this file.  The meta-optimizer
(:mod:`src.meta.optimizer`) profiles past discovery trees, proposes a new version
of *this source*, scores it in the Dream Engine, verifies it on holdout trees and
-- only if it generalises -- promotes it into ``storage/policies/policy_vN/``.

Mutation contract (what the optimizer is allowed to rely on)
------------------------------------------------------------
1. ``POLICY_ID`` is a single string literal.
2. ``PARAMS`` is a flat ``dict[str, float]`` literal with one entry per line.
   The offline mutator rewrites this block textually, leaving the rest of the
   file byte-identical; the LLM route may rewrite the whole file but must keep
   the same public surface.
3. ``SearchPolicy`` subclasses :class:`~src.policy.base.ExplorationPolicy` and
   implements ``select`` (required) plus optional ``observe`` / ``should_stop``.
4. ``param_space()`` returns ``{name: (low, high)}`` for every knob in ``PARAMS``,
   which is how the offline search knows what it may touch.

Version 0 is deliberately *untuned*: every weight that could exploit the
environment is initialised to ``0.0``, so ``policy_v0`` degenerates to
"take the first candidate the coverage mixer offered".  That leaves the entire
improvement budget to the RSI loop instead of hiding it in hand-tuned defaults.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.core.tree import Action, CandidateOutcome
from src.policy.base import ACTION_NAMES, SUITES, ExplorationPolicy, PolicyContext

# --------------------------------------------------------------------------------------
# Identity -- rewritten by the optimizer on promotion.
# --------------------------------------------------------------------------------------
POLICY_ID = "policy_v0"

# --------------------------------------------------------------------------------------
# Tunable weights.  One entry per line; the offline mutator rewrites this block.
# --------------------------------------------------------------------------------------
PARAMS: dict[str, float] = {
    'prior_direct': 0.0,
    'prior_chain_of_thought': 0.0,
    'prior_test_driven': 0.0,
    'prior_repair': 0.0,
    'prior_edge_case_scan': 0.0,
    'prior_decompose': 0.0,
    'prior_restart': 0.0,
    'prior_brute_force': 0.0,
    'prior_optimize': 0.0,
    'sp_humaneval_direct': 0.0,
    'sp_humaneval_chain_of_thought': 0.0,
    'sp_humaneval_test_driven': 0.0,
    'sp_humaneval_repair': 0.0,
    'sp_humaneval_edge_case_scan': 0.0,
    'sp_humaneval_decompose': 0.0,
    'sp_humaneval_restart': 0.0,
    'sp_humaneval_brute_force': 0.0,
    'sp_humaneval_optimize': 0.0,
    'sp_math_direct': 0.0,
    'sp_math_chain_of_thought': 0.0,
    'sp_math_test_driven': 0.0,
    'sp_math_repair': 0.0,
    'sp_math_edge_case_scan': 0.0,
    'sp_math_decompose': 0.0,
    'sp_math_restart': 0.0,
    'sp_math_brute_force': 0.0,
    'sp_math_optimize': 0.0,
    'sp_kernelbench_direct': 0.0,
    'sp_kernelbench_chain_of_thought': 0.0,
    'sp_kernelbench_test_driven': 0.0,
    'sp_kernelbench_repair': 0.0,
    'sp_kernelbench_edge_case_scan': 0.0,
    'sp_kernelbench_decompose': 0.0,
    'sp_kernelbench_restart': 0.0,
    'sp_kernelbench_brute_force': 0.0,
    'sp_kernelbench_optimize': 0.0,
    'w_suite_value': 0.0,
    'w_global_value': 0.0,
    'repair_bonus': 0.0,
    'restart_bonus': 0.0,
    'brute_force_penalty': 0.0,
    'decompose_bonus': 0.0,
    'w_repeat_penalty': 0.0,
    'w_depth_penalty': 0.0,
    'epsilon': 0.0,
    'value_decay': 0.7,
    'stop_at': 1.0,
}

#: Human-readable guidance for the next mutation (read by the LLM route and
#: quoted in the run report).  Kept here so the policy carries its own backlog.
EVOLUTION_NOTES = """
Untuned knobs to consider (see the profiler report for measured evidence):
* prior_<action>: global preference per strategy; the coverage mixer gives every
  strategy a turn, so these priors decide who wins the ties.
* sp_<suite>_<action>: suite-conditional preference; the profiler measures which
  strategy wins inside each suite, which is the signal that transfers to unseen
  tasks of the same suite.
* w_suite_value / w_global_value: how much to trust the reward statistic this
  policy accumulates online per (suite, action) and per action.
* repair_bonus: repair should dominate once a partial artefact exists, because
  evaluator feedback tells us which behaviour is still wrong.
* restart_bonus: re-approaching from scratch helps once the search has stalled.
* brute_force_penalty / decompose_bonus: cheap strategies decay on hard tasks,
  decomposition helps there.
* w_repeat_penalty: avoid re-running a strategy that already failed here.
* epsilon: exploration rate; must stay small once the priors are meaningful.
Known limitation: the coverage mixer caps the slate size, so an action absent
from the slate can never be chosen at that decision point.
""".strip()


def param_space() -> dict[str, tuple[float, float]]:
    """Bounds for every knob in :data:`PARAMS`, used by the offline mutator."""
    space: dict[str, tuple[float, float]] = {
        f"prior_{name}": (-1.0, 1.0) for name in ACTION_NAMES
    }
    for suite in SUITES:
        for name in ACTION_NAMES:
            space[f"sp_{suite}_{name}"] = (-1.0, 1.0)
    space.update(
        {
            "w_suite_value": (-2.0, 3.0),
            "w_global_value": (-2.0, 3.0),
            "repair_bonus": (-1.0, 3.0),
            "restart_bonus": (-1.0, 2.0),
            "brute_force_penalty": (-3.0, 1.0),
            "decompose_bonus": (-1.0, 2.0),
            "w_repeat_penalty": (-1.0, 2.0),
            "w_depth_penalty": (-1.0, 2.0),
            "epsilon": (0.0, 0.5),
            "value_decay": (0.0, 0.99),
            "stop_at": (0.5, 1.2),
        }
    )
    return space


class SearchPolicy(ExplorationPolicy):
    """Value-based strategy selector over the frozen action space.

    At each decision point it scores every candidate in the blind slate and
    picks the best one (with probability ``epsilon`` it explores instead).  The
    score combines three sources, each gated by its own weight:

    * **priors**     -- static preference per strategy (:data:`PARAMS`),
    * **statistics** -- reward statistics accumulated online from *observed*
      outcomes, keyed by ``(suite, action)`` and by ``action``,
    * **state**      -- whether a partial artefact exists, whether the search has
      stalled, and how difficult the task looks.

    No term can see the rewards of the slate it is currently ranking; statistics
    only ever contain outcomes observed at *earlier* decision points.
    """

    POLICY_ID = POLICY_ID
    PARAMS = PARAMS

    def __init__(self, params: Mapping[str, float] | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        # suite -> action -> exponentially-decayed mean reward
        self.suite_values: dict[str, dict[str, float]] = {}
        # action -> exponentially-decayed mean reward (pooled across suites)
        self.global_values: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.tried_here: int = 0

    # -- param plumbing ---------------------------------------------------------------
    @classmethod
    def param_space(cls) -> dict[str, tuple[float, float]]:
        return param_space()

    def prior(self, action_name: str) -> float:
        return float(self.params.get(f"prior_{action_name}", 0.0))

    # -- online statistics ------------------------------------------------------------
    def _decay(self) -> float:
        return max(0.0, min(0.99, self.param("value_decay", 0.7)))

    def _record(self, suite: str, action_name: str, reward: float) -> None:
        """Exponentially-weighted update of the observed reward for an action."""
        decay = self._decay()
        by_action = self.suite_values.setdefault(suite, {})
        by_action[action_name] = by_action.get(action_name, reward) * decay + reward * (1 - decay)
        self.global_values[action_name] = (
            self.global_values.get(action_name, reward) * decay + reward * (1 - decay)
        )
        self.counts[action_name] = self.counts.get(action_name, 0) + 1

    # -- scoring ----------------------------------------------------------------------
    def score_action(self, ctx: PolicyContext, action: Action) -> float:
        """The learned utility of taking ``action`` at ``ctx`` (higher is better)."""
        name = action.name
        suite = ctx.suite

        value = self.prior(name) + self.param(f"sp_{suite}_{name}")
        if self.counts.get(name):
            value += self.param("w_suite_value") * self.suite_values.get(suite, {}).get(name, 0.0)
            value += self.param("w_global_value") * self.global_values.get(name, 0.0)

        best = ctx.best_score
        if name == "repair":
            value += self.param("repair_bonus") * (best if ctx.has_partial_solution() else 0.0)
        if name == "restart":
            value += self.param("restart_bonus") * min(ctx.stagnation, 3.0) / 3.0
        if name == "brute_force":
            value += self.param("brute_force_penalty") * ctx.difficulty
        if name == "decompose":
            value += self.param("decompose_bonus") * ctx.difficulty

        counts = ctx.action_counts()
        value -= self.param("w_repeat_penalty") * min(counts.get(name, 0.0), 3.0) / 3.0
        value -= self.param("w_depth_penalty") * min(ctx.depth, 5.0) / 5.0
        return value

    # -- decision ---------------------------------------------------------------------
    def select(self, ctx: PolicyContext) -> Action:
        """Rank the blind slate and return the winner (epsilon-greedy)."""
        if not ctx.candidates:
            raise ValueError("SearchPolicy.select requires a non-empty slate")

        epsilon = max(0.0, min(0.9, self.param("epsilon", 0.0)))
        if epsilon > 0.0 and ctx.rng(f"explore:{ctx.depth}").random() < epsilon:
            return ctx.rng(f"pick:{ctx.depth}").choice(list(ctx.candidates))

        scores = {a.signature(): self.score_action(ctx, a) for a in ctx.candidates}
        return self.ranked(ctx, scores)

    # -- learning ---------------------------------------------------------------------
    def observe(
        self,
        ctx: PolicyContext,
        chosen: Action,
        outcome: CandidateOutcome,
        slate: Sequence[CandidateOutcome],
    ) -> None:
        """Fold the measured rewards of the whole slate into the statistics."""
        super().observe(ctx, chosen, outcome, slate)
        for candidate in slate:
            self._record(ctx.suite, candidate.action.name, float(candidate.score))

    # -- stopping ---------------------------------------------------------------------
    def should_stop(self, ctx: PolicyContext) -> bool:
        """Stop once the artefact is good enough (``stop_at``)."""
        return ctx.best_score >= self.param("stop_at", 1.0)

    # -- reporting --------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["source"] = __file__
        data["learned"] = {
            "suite_values": {k: dict(v) for k, v in sorted(self.suite_values.items())},
            "global_values": dict(sorted(self.global_values.items())),
            "counts": dict(sorted(self.counts.items())),
        }
        data["evolution_notes"] = EVOLUTION_NOTES
        return data


__all__ = ["EVOLUTION_NOTES", "PARAMS", "POLICY_ID", "SearchPolicy", "param_space"]
