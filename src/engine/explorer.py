"""Real-world explorer: runs tree search on benchmark tasks and records trees.

This is the expensive half of Dream-RSI.  For every task it grows a
:class:`~src.core.tree.DiscoveryTree`, spending model calls, and it records at
each decision point **the whole slate of candidate actions with their measured
rewards**.  That recording discipline is what later lets the Dream Engine
re-rank strategies offline for free, so the explorer deliberately *over-spends*
slightly: a branching factor of ``b`` costs ``b`` model calls per decision point
but yields exact, unbiased offline replay.

Frontier policy
---------------
Best-first over the set of *unexpanded* nodes: highest score first, ties broken
toward shallower (cheaper) nodes.  A node is expanded at most once, so a recorded
slate is never overwritten -- an invariant the Dream Engine relies on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.evaluator import (
    DEFAULT_TIMEOUT_S,
    EvalResult,
    evaluate,
    evaluator_fingerprint,
    task_meta,
)
from src.core.llm_client import GenerationRequest, LLMClient, describe_client
from src.core.tree import Action, CandidateOutcome, DiscoveryTree, Node
from src.core.utils import (
    file_sha256,
    mean,
    new_id,
    read_json,
    stable_hash,
    utc_now,
    write_json,
)
from src.policy.base import DEFAULT_ACTION_SPACE, ExplorationPolicy, context_from_record


# ======================================================================================
# Task features
# ======================================================================================
def task_features(task: Mapping[str, Any]) -> dict[str, float]:
    """Numeric, solver-visible features of a task.

    Persisted inside ``tree.task_meta["features"]`` and merged into every
    :class:`~src.policy.base.PolicyContext` feature vector, so a policy can
    condition on task kind (and so those features also exist during replay).
    """
    meta = task_meta(task)
    suite = meta["suite"]
    features = {
        "difficulty": float(meta["difficulty"]),
        "prompt_len": float(len(str(meta["prompt"]))),
        "n_tags": float(len(meta["tags"])),
    }
    for name in ("humaneval", "math", "kernelbench"):
        features[f"suite_{name}"] = 1.0 if suite == name else 0.0
    return features


# ======================================================================================
# Configuration and reporting
# ======================================================================================
@dataclass
class ExplorerConfig:
    """Budget and I/O settings for an exploration run."""

    #: Decision points per task (each one costs ``candidate_budget`` calls).
    max_steps: int = 4
    #: Slate size -- the branching factor that makes replay possible.
    candidate_budget: int = 4
    timeout_s: float = DEFAULT_TIMEOUT_S
    seed: int = 0
    #: Stop a task early once an artefact fully passes.
    stop_on_solve: bool = True
    #: Reuse evaluations of identical (task, code) pairs (they are deterministic).
    cache_evaluations: bool = True
    cache_path: Path | None = None
    tree_dir: Path | None = None
    persist: bool = True


@dataclass
class ExplorationSummary:
    """Aggregate outcome of one exploration generation."""

    policy_id: str = ""
    n_tasks: int = 0
    n_trees: int = 0
    n_decisions: int = 0
    n_nodes: int = 0
    solve_rate: float = 0.0
    mean_best_score: float = 0.0
    mean_decision_reward: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0
    wall_ms: float = 0.0
    eval_cache_hits: int = 0
    tree_paths: list[str] = field(default_factory=list)
    per_suite: dict[str, dict[str, float]] = field(default_factory=dict)
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ======================================================================================
# Explorer
# ======================================================================================
class Explorer:
    """Runs the live search: policy -> model -> evaluator -> discovery tree."""

    def __init__(
        self,
        policy: ExplorationPolicy,
        client: LLMClient,
        *,
        config: ExplorerConfig | None = None,
        action_space: Any = None,
    ) -> None:
        self.policy = policy
        self.client = client
        self.config = config or ExplorerConfig()
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.fingerprint = evaluator_fingerprint()
        self._eval_cache: dict[str, dict[str, Any]] = {}
        self._cache_hits = 0
        self._saved_entries = 0
        self._load_cache()

    # -- evaluation cache -------------------------------------------------------------
    def _load_cache(self) -> None:
        if not (self.config.cache_evaluations and self.config.cache_path):
            return
        payload = read_json(self.config.cache_path, {}) or {}
        if payload.get("evaluator_sha256") == self.fingerprint:
            self._eval_cache = dict(payload.get("entries") or {})
            self._saved_entries = len(self._eval_cache)

    def save_cache(self) -> None:
        """Flush the evaluation cache when it has grown since the last write."""
        if not (self.config.cache_evaluations and self.config.cache_path):
            return
        if len(self._eval_cache) == self._saved_entries:
            return
        write_json(
            self.config.cache_path,
            {
                "evaluator_sha256": self.fingerprint,
                "entries": self._eval_cache,
                "updated_at": utc_now(),
            },
            indent=None,
        )
        self._saved_entries = len(self._eval_cache)

    def evaluate_code(self, task: Mapping[str, Any], code: str) -> EvalResult:
        """Evaluate ``code`` (memoised: grading is deterministic)."""
        if not self.config.cache_evaluations:
            return evaluate(task, code, timeout_s=self.config.timeout_s)
        key = f"{task['id']}:{stable_hash(code, 24)}"
        cached = self._eval_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return EvalResult(**cached)
        result = evaluate(task, code, timeout_s=self.config.timeout_s)
        self._eval_cache[key] = result.to_dict()
        return result

    # -- context construction ---------------------------------------------------------
    def _context(self, tree: DiscoveryTree, node: Node, candidates: Sequence[Action] = ()):
        """Policy input for ``node``, derived from the same record replay uses."""
        record = tree.decision_for(node.id)
        return context_from_record(
            record,
            policy_id=self.policy.policy_id,
            rng_seed=self.config.seed,
            candidates=list(candidates),
        )

    @staticmethod
    def _frontier(tree: DiscoveryTree, preferred_id: str | None = None) -> Node | None:
        """Next node to expand.

        The policy steers the search: the child it picked is expanded first.  Once
        that branch is exhausted (or if it is unavailable), the search falls back
        to best-first over every unexpanded node, ties broken toward shallower
        (cheaper) nodes.  A node is expanded at most once, so a recorded slate is
        never overwritten.
        """
        if preferred_id is not None:
            preferred = tree.nodes.get(preferred_id)
            if preferred is not None and not preferred.expanded:
                return preferred
        candidates = [n for n in tree.nodes.values() if not n.expanded]
        if not candidates:
            return None
        return max(candidates, key=lambda n: (n.score, -n.depth, n.created_at))

    # -- one task ---------------------------------------------------------------------
    def run_task(self, task: Mapping[str, Any]) -> DiscoveryTree:
        """Explore one task and return its (persisted) discovery tree."""
        meta = task_meta(task)
        meta["features"] = task_features(task)
        tree = DiscoveryTree.create(
            meta["id"],
            policy_id=self.policy.policy_id,
            task_meta=meta,
            root_state={"solution": "", "explanation": f"root for {meta['id']}"},
        )
        tree.metadata.update(
            {
                "evaluator_sha256": self.fingerprint,
                "config": {
                    "max_steps": self.config.max_steps,
                    "candidate_budget": self.config.candidate_budget,
                    "timeout_s": self.config.timeout_s,
                    "seed": self.config.seed,
                },
                "client": self.client.name,
                "started_at": utc_now(),
            }
        )

        preferred: str | None = None
        for step in range(self.config.max_steps):
            node = self._frontier(tree, preferred)
            if node is None:
                break
            ctx_probe = self._context(tree, node)
            if self.config.stop_on_solve and self.policy.should_stop(ctx_probe):
                tree.metadata["stopped_early"] = f"policy_should_stop at depth {node.depth}"
                break

            slate = self.policy.propose(ctx_probe, budget=self.config.candidate_budget)
            if not slate:
                break

            outcomes, artifacts = self._execute_slate(task, ctx_probe, slate)
            # Record the slate *before* choosing, so the decision record the
            # policy sees is the same one the Dream Engine will replay.
            tree.record_slate(node.id, outcomes, None)
            ctx = self._context(tree, node, candidates=slate)
            chosen = self.policy.select(ctx)
            if not isinstance(chosen, Action):
                raise TypeError(
                    f"{self.policy.policy_id}.select() returned {type(chosen).__name__}, "
                    "expected Action"
                )
            matched = self._match_outcome(outcomes, chosen, ctx)
            tree.mark_chosen(node.id, matched.action)

            # Every evaluated candidate becomes a node: the branching factor of the
            # discovery tree is the slate size, and the measurement of each branch
            # was already paid for.  ``selected`` marks the branch the policy
            # chose to steer the search into next.
            chosen_child: Node | None = None
            for outcome in outcomes:
                code, result = artifacts[outcome.action.signature()]
                child = tree.add_node(
                    node.id,
                    outcome.action,
                    {
                        "solution": code,
                        "explanation": f"{outcome.action.name} at depth {node.depth + 1}",
                    },
                    outcome.score,
                    feedback=result.summary() if not result.passed else "",
                    passed=result.passed,
                    policy_id=self.policy.policy_id,
                    metadata={
                        "tests_passed": result.tests_passed,
                        "tests_total": result.tests_total,
                        "duration_ms": round(result.duration_ms, 3),
                        "origin_policy": self.policy.policy_id,
                        "slate_size": len(outcomes),
                        "selected": outcome.action.signature() == matched.action.signature(),
                        "evaluator_error": result.error,
                    },
                )
                if outcome.action.signature() == matched.action.signature():
                    chosen_child = child
            self.policy.observe(ctx, matched.action, matched, outcomes)
            preferred = chosen_child.id if chosen_child is not None else None

            if self.config.stop_on_solve and any(o.passed for o in outcomes):
                tree.metadata["solved_at_depth"] = node.depth + 1
                break

        tree.metadata.update(
            {
                "finished_at": utc_now(),
                "llm_calls": self.client.calls,
                "policy": self.policy.describe(),
                "eval_cache_hits": self._cache_hits,
            }
        )
        tree.metadata["stats"] = tree.stats()
        if self.config.persist and self.config.tree_dir:
            path = Path(self.config.tree_dir) / f"{tree.tree_id}.json"
            tree.save(path)
        # Flush per task: an interrupted run must not throw away paid-for grading
        # or -- far more costly -- model calls that already happened.
        self.save_cache()
        self._flush_client_cache()
        return tree

    def _flush_client_cache(self) -> None:
        """Persist the client's cache if it has one (cheap, saves real money)."""
        for name in ("save_cache", "flush"):
            flush = getattr(self.client, name, None)
            if callable(flush):
                try:
                    flush()
                except OSError:  # pragma: no cover - cache write must never fail a run
                    pass
                return

    def _execute_slate(
        self,
        task: Mapping[str, Any],
        ctx: Any,
        slate: Sequence[Action],
    ) -> tuple[list[CandidateOutcome], dict[str, tuple[str, EvalResult]]]:
        """Run every candidate action: model call, code extraction, grading.

        Returns the recorded outcomes (the slate the Dream Engine will replay)
        together with the artefacts themselves, keyed by action signature, so the
        chosen one can be attached to the tree.
        """
        attempt = len(getattr(ctx, "history", ()) or ())
        outcomes: list[CandidateOutcome] = []
        artifacts: dict[str, tuple[str, EvalResult]] = {}

        for action in slate:
            request = GenerationRequest.from_context(ctx, action, attempt=attempt)
            generation = self.client.generate(request)
            code = generation.code or ""
            if generation.error or not code.strip():
                error = generation.error or "model returned no usable code"
                result = EvalResult(
                    task_id=str(task["id"]),
                    score=0.0,
                    passed=False,
                    error=error,
                    diagnostics=[],
                )
            else:
                result = self.evaluate_code(task, code)
            outcomes.append(
                CandidateOutcome(
                    action=action,
                    score=float(result.score),
                    passed=bool(result.passed),
                    tests_passed=int(result.tests_passed),
                    tests_total=int(result.tests_total),
                    latency_ms=float(generation.latency_ms),
                    error="" if result.passed else (generation.error or result.summary()),
                    origin_policy=self.policy.policy_id,
                    artifact_preview=code[:240],
                )
            )
            artifacts[action.signature()] = (code, result)
        return outcomes, artifacts

    def _match_outcome(
        self,
        outcomes: Sequence[CandidateOutcome],
        chosen: Action,
        ctx: Any,
    ) -> CandidateOutcome:
        """Resolve the policy's pick to the recorded outcome for that action.

        If a policy returns an action that was not in the slate (a contract
        violation), fall back to the closest recorded outcome so the tree stays
        consistent, and flag it in the node metadata.
        """
        for outcome in outcomes:
            if outcome.action.signature() == chosen.signature():
                return outcome
        fallback = outcomes[0]
        fallback.origin_policy = f"{self.policy.policy_id}:off-slate-pick"
        if ctx is not None and not getattr(ctx, "candidates", None):
            raise ValueError("cannot resolve a pick from an empty slate")
        return fallback

    # -- many tasks -------------------------------------------------------------------
    def run(self, tasks: Iterable[Mapping[str, Any]]) -> tuple[list[DiscoveryTree], ExplorationSummary]:
        """Explore every task; persist each tree; return trees plus a summary."""
        started = time.perf_counter()
        trees: list[DiscoveryTree] = []
        per_task: dict[str, dict[str, Any]] = {}
        per_suite: dict[str, dict[str, float]] = {}
        for task in tasks:
            tree = self.run_task(task)
            trees.append(tree)
            best = tree.best_node().score
            suite = str(tree.task_meta.get("suite", ""))
            per_task[tree.task_id] = {
                "suite": suite,
                "best_score": best,
                "solved": tree.solved(),
                "n_nodes": len(tree),
                "n_decisions": len(tree.expanded_nodes()),
            }
            bucket = per_suite.setdefault(suite, {"n": 0.0, "best_sum": 0.0, "n_solved": 0.0})
            bucket["n"] += 1
            bucket["best_sum"] += best
            bucket["n_solved"] += 1.0 if tree.solved() else 0.0
        for bucket in per_suite.values():
            n = bucket["n"] or 1.0
            bucket["mean_best_score"] = bucket["best_sum"] / n
            bucket["solve_rate"] = bucket["n_solved"] / n

        decisions = [len(t.expanded_nodes()) for t in trees]
        summary = ExplorationSummary(
            policy_id=self.policy.policy_id,
            n_tasks=len(trees),
            n_trees=len(trees),
            n_decisions=sum(decisions),
            n_nodes=sum(len(t) for t in trees),
            solve_rate=mean([1.0 if t.solved() else 0.0 for t in trees]),
            mean_best_score=mean([t.best_node().score for t in trees]),
            mean_decision_reward=mean(
                [
                    float(mean([n.score for n in t.expanded_nodes()]))
                    for t in trees
                    if t.expanded_nodes()
                ]
            ),
            llm_calls=self.client.calls,
            total_tokens=self.client.prompt_tokens + self.client.completion_tokens,
            wall_ms=(time.perf_counter() - started) * 1000.0,
            eval_cache_hits=self._cache_hits,
            tree_paths=[
                str(Path(self.config.tree_dir) / f"{t.tree_id}.json")
                for t in trees
                if self.config.tree_dir
            ],
            per_suite=per_suite,
            per_task=per_task,
        )
        self.save_cache()
        return trees, summary


__all__ = ["ExplorationSummary", "Explorer", "ExplorerConfig", "task_features"]
