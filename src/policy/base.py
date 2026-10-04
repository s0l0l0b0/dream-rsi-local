"""Policy SDK: the frozen action space plus the abstract exploration policy.

Two things live here, and the split matters:

1. **The frozen strategy library** (:data:`ACTION_SPECS`, :class:`ActionSpace`).
   The set of things a policy may *do* is fixed by the environment, so that an
   evolved policy cannot invent an action the explorer is unable to execute, and
   so that every generation is compared over an identical action space.

2. **The abstract policy interface** (:class:`ExplorationPolicy`) that
   ``src/policy/search_policy.py`` implements.  This is the *only* file the
   meta-optimizer is allowed to rewrite; everything else is the world it lives in.

The framework also owns *candidate proposal* (:meth:`ExplorationPolicy.propose`):
the default implementation is a seeded coverage rotation over applicable actions.
The explorer records every proposed candidate together with its measured reward,
which is what makes offline replay valid.  If each generation proposed wildly
different slates, the Dream Engine would be comparing policies on different
questions -- so proposal is deliberately kept policy-independent by default.

A :class:`PolicyContext` carries **no reward information whatsoever** -- not for
the slate, not for anything.  That is an invariant, not a convention: it is what
stops a replayed policy from reading the answer key.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.core.tree import Action, CandidateOutcome, DecisionRecord
from src.core.utils import seeded_int

# ======================================================================================
# Policy context -- the policy's entire view of the world
# ======================================================================================
@dataclass
class PolicyContext:
    """Everything a policy is allowed to see at a decision point.

    Note the absence of any ``rewards`` / ``scores_of_candidates`` field.  Only
    quantities already observed *before* the decision are present.
    """

    task_id: str
    suite: str
    entry_point: str
    signature: str = ""
    prompt: str = ""
    difficulty: float = 0.5
    #: The artefact produced so far (best attempt on the path).
    current_state: dict[str, Any] = field(default_factory=dict)
    #: Evaluator feedback for the parent attempt.
    feedback: str = ""
    #: Attempts made before this decision point, oldest first.
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Observed feature vector (see ``core.tree.state_features``).
    features: dict[str, float] = field(default_factory=dict)
    #: The slate to choose from (empty when proposing).
    candidates: list[Action] = field(default_factory=list)
    #: Seed for any randomness the policy needs -- keeps runs reproducible.
    rng_seed: int = 0
    policy_id: str = ""
    #: Solver-visible task text.
    tags: tuple[str, ...] = ()

    # -- derived accessors ------------------------------------------------------------
    @property
    def depth(self) -> int:
        return int(self.features.get("depth", 0.0))

    @property
    def best_score(self) -> float:
        return float(self.features.get("best_score", 0.0))

    @property
    def stagnation(self) -> float:
        return float(self.features.get("stagnation", 0.0))

    def has_partial_solution(self) -> bool:
        """True when there is a non-empty artefact that scored above zero."""
        solution = str(self.current_state.get("solution") or "")
        return bool(solution.strip()) and self.best_score > 0.0

    def rng(self, salt: str = "") -> random.Random:
        """Deterministic RNG bound to (decision point, salt)."""
        seed = seeded_int(self.rng_seed, self.task_id, self.depth, salt)
        return random.Random(seed)

    def action_counts(self) -> dict[str, float]:
        return {k[5:]: v for k, v in self.features.items() if k.startswith("act::")}


# ======================================================================================
# The frozen strategy library
# ======================================================================================
@dataclass(frozen=True)
class ActionSpec:
    """Static description of one strategy: how to prompt it, and when it applies.

    ``prompt_hint`` is rendered into the solver prompt by
    :mod:`src.core.llm_client`; ``applicable`` is a pure predicate over the
    context so that both the explorer and the Dream Engine agree on what was
    *possible* at a decision point.
    """

    name: str
    summary: str
    prompt_hint: str
    #: Extra sampling params merged into the action when it is instantiated.
    default_params: Mapping[str, Any] = field(default_factory=dict)
    #: Predicate over the context; default = always applicable.
    applicable: Callable[[PolicyContext], bool] = lambda ctx: True
    #: Which suites this strategy was designed for (documentation / priors only).
    suites: tuple[str, ...] = ()


def _needs_partial(ctx: PolicyContext) -> bool:
    return ctx.has_partial_solution()


def _needs_history(ctx: PolicyContext) -> bool:
    return ctx.depth >= 1


ACTION_SPECS: dict[str, ActionSpec] = {
    spec.name: spec
    for spec in (
        ActionSpec(
            name="direct",
            summary="Answer immediately with the most likely implementation.",
            prompt_hint=(
                "Write the final implementation directly. Do not explore alternatives; "
                "produce the single most likely correct function body."
            ),
            default_params={"temperature": 0.2},
        ),
        ActionSpec(
            name="chain_of_thought",
            summary="Reason step by step about the algorithm before coding.",
            prompt_hint=(
                "First reason carefully about the mathematics/algorithm in a short comment "
                "block, then implement it. Derive the exact formula before writing code."
            ),
            default_params={"temperature": 0.3},
            suites=("math", "kernelbench"),
        ),
        ActionSpec(
            name="test_driven",
            summary="Derive behaviour from the required interface, then implement.",
            prompt_hint=(
                "Design the implementation around the observable input/output contract: "
                "enumerate the concrete cases the function must satisfy, then write code "
                "that handles each of them explicitly."
            ),
            default_params={"temperature": 0.2},
            suites=("humaneval",),
        ),
        ActionSpec(
            name="repair",
            summary="Fix the current partial attempt using evaluator feedback.",
            prompt_hint=(
                "The previous attempt below is partially correct. Repair it: keep the parts "
                "the feedback says are fine, and fix precisely the failing behaviour."
            ),
            default_params={"temperature": 0.1},
            applicable=_needs_partial,
        ),
        ActionSpec(
            name="edge_case_scan",
            summary="Hunt for boundary conditions (empty, single, negative, large).",
            prompt_hint=(
                "Implement the function with explicit handling of boundary conditions: "
                "empty input, single element, negative values, duplicates and maximum size."
            ),
            default_params={"temperature": 0.2},
            suites=("humaneval",),
        ),
        ActionSpec(
            name="decompose",
            summary="Split the problem into sub-problems and solve each.",
            prompt_hint=(
                "Decompose the problem into smaller sub-problems, solve each one inside a "
                "helper, then compose them into the required entry point."
            ),
            default_params={"temperature": 0.3},
        ),
        ActionSpec(
            name="restart",
            summary="Discard the current attempt and re-approach from scratch.",
            prompt_hint=(
                "Ignore the previous attempt entirely (it is a dead end). Approach the "
                "problem from a different angle and write a fresh implementation."
            ),
            default_params={"temperature": 0.6},
            applicable=_needs_history,
        ),
        ActionSpec(
            name="brute_force",
            summary="Write the simplest obviously-correct implementation first.",
            prompt_hint=(
                "Write the simplest, most obviously correct implementation you can, even "
                "if it is inefficient. Correctness first."
            ),
            default_params={"temperature": 0.1},
        ),
        ActionSpec(
            name="optimize",
            summary="Re-implement with numerical stability and efficiency in mind.",
            prompt_hint=(
                "Implement the operation efficiently and with numerical stability in mind "
                "(guard against overflow, division by zero and degenerate shapes)."
            ),
            default_params={"temperature": 0.15},
            suites=("kernelbench",),
        ),
    )
}

#: Stable ordering used by the coverage mixer and by reporting.
ACTION_NAMES: tuple[str, ...] = tuple(ACTION_SPECS)

#: Benchmark suites shipped with the reproduction.
SUITES: tuple[str, ...] = ("humaneval", "math", "kernelbench")


class ActionSpace:
    """Builds concrete :class:`Action` values from the frozen strategy library."""

    def __init__(self, specs: Mapping[str, ActionSpec] | None = None) -> None:
        self.specs: dict[str, ActionSpec] = dict(specs or ACTION_SPECS)

    def names(self) -> list[str]:
        return list(self.specs)

    def spec(self, name: str) -> ActionSpec:
        if name not in self.specs:
            raise KeyError(f"unknown action {name!r}; known: {sorted(self.specs)}")
        return self.specs[name]

    def applicable(self, ctx: PolicyContext) -> list[str]:
        """Names of strategies that are legal at this decision point."""
        return [name for name, spec in self.specs.items() if spec.applicable(ctx)]

    def build(self, name: str, params: Mapping[str, Any] | None = None, rationale: str = "") -> Action:
        """Instantiate an action, merging the spec's default params."""
        spec = self.spec(name)
        merged: dict[str, Any] = dict(spec.default_params)
        merged.update(dict(params or {}))
        return Action(name=name, params=merged, rationale=rationale or spec.summary)

    def slate(self, ctx: PolicyContext, budget: int, *, salt: str = "") -> list[Action]:
        """Seeded coverage rotation over applicable actions.

        The rotation guarantees that, across a tree, every applicable strategy is
        attempted at least once -- without which the Dream Engine could never
        score an action that the incumbent policy happened not to like.  It is
        deterministic in ``(task_id, depth, salt)`` so re-running an exploration
        reproduces the same slates.
        """
        names = self.applicable(ctx)
        if not names:
            names = ["direct"]
        if budget >= len(names):
            return [self.build(n) for n in names]
        rng = random.Random(seeded_int(ctx.task_id, ctx.depth, salt, ctx.rng_seed))
        offset = rng.randrange(len(names))
        rotated = names[offset:] + names[:offset]
        return [self.build(n) for n in rotated[:budget]]


#: Shared default action space instance.
DEFAULT_ACTION_SPACE = ActionSpace()


# ======================================================================================
# Building policy inputs from recorded decisions
# ======================================================================================
def context_from_record(
    record: DecisionRecord,
    *,
    policy_id: str = "",
    rng_seed: int = 0,
    candidates: Sequence[Action] | None = None,
) -> PolicyContext:
    """Turn a :class:`DecisionRecord` into the policy's view of the world.

    Used by **both** sides of the system: the explorer replays the slate it has
    just measured, and the Dream Engine replays the slate stored in the tree.
    Because the feature/history computation lives in ``DiscoveryTree.decision_for``
    and this function only reshapes it, live and replayed inputs are identical by
    construction -- the property that makes offline replay meaningful.

    ``candidates`` defaults to the recorded slate.  It is always a *blind* slate:
    the returned context exposes action labels only, never the recorded rewards.
    """
    task = dict(record.task or {})
    return PolicyContext(
        task_id=record.task_id,
        suite=record.suite or str(task.get("suite", "")),
        entry_point=str(task.get("entry_point", "")),
        signature=str(task.get("signature", "")),
        prompt=str(task.get("prompt", "")),
        difficulty=float(task.get("difficulty", 0.5)),
        current_state=dict(record.current_state),
        feedback=record.feedback,
        history=[dict(h) for h in record.history],
        features=dict(record.features),
        candidates=list(record.slate if candidates is None else candidates),
        rng_seed=int(rng_seed),
        policy_id=policy_id,
        tags=tuple(str(t) for t in (task.get("tags") or ())),
    )


# ======================================================================================
# The abstract policy
# ======================================================================================
class ExplorationPolicy(ABC):
    """Base class every evolved policy implements.

    Subclasses must define:

    * ``POLICY_ID`` -- stable version string (e.g. ``"policy_v3"``),
    * ``PARAMS``     -- a flat ``dict[str, float]`` of tunable knobs,
    * ``param_space``-- ``{name: (low, high)}`` bounds used by the offline mutator,
    * :meth:`select` -- the decision rule (the thing that actually evolves).

    They may override :meth:`propose`, :meth:`should_stop` and :meth:`observe`,
    but the optimizer is instructed to keep :meth:`propose` at its default so
    that replay comparisons stay valid across generations.
    """

    POLICY_ID: str = "policy_abstract"
    PARAMS: dict[str, float] = {}

    def __init__(self, params: Mapping[str, float] | None = None, **kwargs: Any) -> None:
        merged = dict(self.PARAMS)
        merged.update({k: float(v) for k, v in dict(params or {}).items()})
        self.params: dict[str, float] = merged
        self.action_space: ActionSpace = kwargs.get("action_space") or DEFAULT_ACTION_SPACE
        self.rng_seed: int = int(kwargs.get("rng_seed", 0))
        self._observations: int = 0

    # -- identity ---------------------------------------------------------------------
    @property
    def policy_id(self) -> str:
        return str(self.POLICY_ID)

    def param(self, name: str, default: float = 0.0) -> float:
        return float(self.params.get(name, default))

    @classmethod
    def param_space(cls) -> dict[str, tuple[float, float]]:
        """Tunable bounds for PARAMS; override to expose knobs to the mutator."""
        return {}

    def describe(self) -> dict[str, Any]:
        """Machine-readable description, archived with every policy checkpoint."""
        return {
            "policy_id": self.policy_id,
            "params": dict(self.params),
            "param_space": {k: list(v) for k, v in self.param_space().items()},
            "observations": self._observations,
            "class": type(self).__name__,
        }

    # -- proposal ---------------------------------------------------------------------
    def propose(self, ctx: PolicyContext, budget: int = 4) -> list[Action]:
        """Build the candidate slate for this decision point.

        Default: the framework's coverage rotation, salted by a **constant**, so
        the slates a task presents are identical in every generation.  That is
        what keeps the evolving-world comparison clean: the world is held fixed
        and only the decision rule changes.  Overriding this is allowed but
        discouraged; see the module docstring.
        """
        return self.action_space.slate(ctx, budget, salt="coverage")

    # -- decision (the evolving part) --------------------------------------------------
    @abstractmethod
    def select(self, ctx: PolicyContext) -> Action:
        """Choose one action from ``ctx.candidates`` (the blind slate)."""

    # -- control and learning ---------------------------------------------------------
    def should_stop(self, ctx: PolicyContext) -> bool:
        """Return True to end this task's search early (default: never)."""
        return False

    def observe(
        self,
        ctx: PolicyContext,
        chosen: Action,
        outcome: CandidateOutcome,
        slate: Sequence[CandidateOutcome],
    ) -> None:
        """Online learning hook, called after a slate has been evaluated.

        ``slate`` contains the measured reward of *every* candidate, including
        the ones this policy did not pick -- that is what a real agent sees when
        it runs several branches.  The Dream Engine replays the same call with
        recorded rewards, so a learning policy is evaluated faithfully.
        """
        self._observations += 1

    # -- helpers available to subclasses ----------------------------------------------
    def ranked(self, ctx: PolicyContext, scores: Mapping[str, float]) -> Action:
        """Pick the highest-scoring candidate by :class:`Action` signature.

        Ties break deterministically on the slate order, so a policy with no
        preference still behaves reproducibly.
        """
        if not ctx.candidates:
            raise ValueError("cannot select from an empty slate")
        best = max(
            ctx.candidates,
            key=lambda a: (float(scores.get(a.signature(), 0.0)), -ctx.candidates.index(a)),
        )
        return best


__all__ = [
    "ACTION_NAMES",
    "SUITES",
    "ACTION_SPECS",
    "ActionSpace",
    "ActionSpec",
    "DEFAULT_ACTION_SPACE",
    "ExplorationPolicy",
    "PolicyContext",
    "context_from_record",
    "DecisionRecord",
]
