"""Discovery-tree data structures for Dream-RSI.

A *discovery tree* is the persistent record of one real-world exploration run on
one benchmark task.  It is the shared artefact between the two halves of the
system:

* the :mod:`src.engine.explorer` **writes** trees while spending real LLM calls,
* the :mod:`src.engine.dreamer` **reads** trees to replay them offline at zero
  inference cost.

Design note -- why every decision point stores a *slate*
-------------------------------------------------------
Offline replay can only judge a policy on options that were actually attempted.
We therefore record, at every expanded node, the full slate of candidate actions
the live policy proposed together with the *measured* reward of each one (the
reward comes from the sandboxed evaluator, never from the model).  Because the
slate is exhaustive, replay evaluation is full-information and needs no
importance-sampling correction; the price is that replay can only re-rank the
recorded options -- it cannot score actions nobody ever tried.

Rewards live on :class:`CandidateOutcome`, which is deliberately kept separate
from the action's *representation*: the Dream Engine hands a policy the action
labels only (a "blind slate") and looks up the reward afterwards, so a replayed
policy can never read the answer key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.core.utils import (
    mean,
    new_id,
    read_json,
    stable_hash,
    stdev,
    utc_now,
    write_json,
)

#: Node ids that are valid as "no parent" markers.
ROOT_PARENT = None


# ======================================================================================
# Actions and recorded outcomes
# ======================================================================================
@dataclass
class Action:
    """A single thing the policy can *do* at a decision point.

    An action is a strategy instantiation, not a prompt: the explorer turns it
    into a concrete LLM call.  ``params`` carries the strategy's knobs (e.g.
    ``{"temperature": 0.2, "focus": "edge_cases"}``) and is what makes two
    actions with the same name distinguishable.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Free-form human-readable explanation (never shown to the replayed policy
    #: as a reward, but logged for auditability).
    rationale: str = ""

    def signature(self) -> str:
        """Stable identity of this action, independent of dict ordering."""
        return stable_hash({"name": self.name, "params": self.params}, length=12)

    def describe(self) -> str:
        """Compact human-readable label, e.g. ``repair(focus=edge_cases)``."""
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params), "rationale": self.rationale}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Action":
        return cls(
            name=str(data["name"]),
            params=dict(data.get("params") or {}),
            rationale=str(data.get("rationale") or ""),
        )


@dataclass
class CandidateOutcome:
    """The measured result of executing one :class:`Action` at one decision point.

    ``score`` is produced by the immutable benchmark evaluator, i.e. it is a real
    measurement, not a model self-report.
    """

    action: Action
    score: float = 0.0
    passed: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    latency_ms: float = 0.0
    error: str = ""
    #: Which policy proposed this candidate (for off-policy diagnostics).
    origin_policy: str = ""
    #: Optional short excerpt of the produced artefact, for debugging only.
    artifact_preview: str = ""

    @property
    def key(self) -> str:
        return self.action.signature()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "score": round(float(self.score), 6),
            "passed": bool(self.passed),
            "tests_passed": int(self.tests_passed),
            "tests_total": int(self.tests_total),
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "origin_policy": self.origin_policy,
            "artifact_preview": self.artifact_preview,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateOutcome":
        return cls(
            action=Action.from_dict(data["action"]),
            score=float(data.get("score", 0.0)),
            passed=bool(data.get("passed", False)),
            tests_passed=int(data.get("tests_passed", 0)),
            tests_total=int(data.get("tests_total", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            error=str(data.get("error") or ""),
            origin_policy=str(data.get("origin_policy") or ""),
            artifact_preview=str(data.get("artifact_preview") or ""),
        )


# ======================================================================================
# Nodes and edges
# ======================================================================================
@dataclass
class Node:
    """One state in the search: a candidate artefact plus its measured score.

    ``state`` holds the artefact the policy produced (for code benchmarks:
    ``{"solution": str, "explanation": str}``) together with whatever feedback
    the evaluator returned.  ``candidates`` holds the *slate* that was evaluated
    at this node, turning the node into a replayable decision point.
    """

    id: str
    task_id: str
    parent_id: str | None = None
    #: The action that produced this node (``None`` for the root).
    action: Action | None = None
    state: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    feedback: str = ""
    depth: int = 0
    children: list[str] = field(default_factory=list)
    #: Slate evaluated at this node, with measured rewards.  Recorded in the same
    #: order the live policy proposed it.  ``None`` means "never expanded".
    candidates: list[CandidateOutcome] | None = None
    #: Signature of the candidate that was actually expanded into a child.
    chosen_action_id: str | None = None
    #: Policy that created this node.
    policy_id: str = ""
    created_at: str = field(default_factory=utc_now)
    visit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------------------
    @property
    def expanded(self) -> bool:
        """True when this node is a recorded (replayable) decision point."""
        return bool(self.candidates)

    @property
    def slate_best_score(self) -> float:
        """Best measured reward among the recorded candidates (0.0 if unexpanded)."""
        return max((c.score for c in self.candidates or []), default=0.0)

    def candidate_by_signature(self, signature: str) -> CandidateOutcome | None:
        for candidate in self.candidates or []:
            if candidate.action.signature() == signature:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "action": self.action.to_dict() if self.action else None,
            "state": self.state,
            "score": round(float(self.score), 6),
            "feedback": self.feedback,
            "depth": int(self.depth),
            "children": list(self.children),
            "candidates": [c.to_dict() for c in self.candidates] if self.candidates else None,
            "chosen_action_id": self.chosen_action_id,
            "policy_id": self.policy_id,
            "created_at": self.created_at,
            "visit_count": int(self.visit_count),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        action = data.get("action")
        candidates = data.get("candidates")
        return cls(
            id=str(data["id"]),
            task_id=str(data.get("task_id", "")),
            parent_id=data.get("parent_id"),
            action=Action.from_dict(action) if action else None,
            state=dict(data.get("state") or {}),
            score=float(data.get("score", 0.0)),
            feedback=str(data.get("feedback") or ""),
            depth=int(data.get("depth", 0)),
            children=[str(c) for c in (data.get("children") or [])],
            candidates=[CandidateOutcome.from_dict(c) for c in candidates] if candidates else None,
            chosen_action_id=data.get("chosen_action_id"),
            policy_id=str(data.get("policy_id") or ""),
            created_at=str(data.get("created_at") or utc_now()),
            visit_count=int(data.get("visit_count", 0)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Edge:
    """A directed transition ``parent -> child`` labelled by the action taken.

    Edges are stored explicitly (rather than implied by ``Node.parent_id``) so
    that the persisted tree can be consumed as a plain graph by external tools
    and so that per-transition reward deltas are first-class.
    """

    parent_id: str
    child_id: str
    action: Action
    reward: float = 0.0
    #: Improvement delivered by this transition (child score - parent score).
    delta: float = 0.0
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "action": self.action.to_dict(),
            "reward": round(float(self.reward), 6),
            "delta": round(float(self.delta), 6),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        return cls(
            parent_id=str(data["parent_id"]),
            child_id=str(data["child_id"]),
            action=Action.from_dict(data["action"]),
            reward=float(data.get("reward", 0.0)),
            delta=float(data.get("delta", 0.0)),
            created_at=str(data.get("created_at") or utc_now()),
        )


# ======================================================================================
# Decision records (the Dream Engine's unit of work)
# ======================================================================================
@dataclass
class DecisionRecord:
    """A replayable snapshot of one decision point.

    Everything needed to re-run a *different* policy at this point:

    * ``features`` / ``history`` -- what the live policy observed,
    * ``slate``                  -- the action labels it could choose from,
    * ``rewards``                -- the measured reward behind each label.

    The split between ``slate`` and ``rewards`` is what keeps replay honest.
    """

    decision_id: str
    tree_id: str
    task_id: str
    node_id: str
    depth: int
    suite: str
    features: dict[str, float]
    history: list[dict[str, Any]]
    slate: list[Action]
    rewards: dict[str, float]
    chosen_action_id: str | None
    chosen_score: float
    current_state: dict[str, Any]
    feedback: str
    #: Solver-visible task view (id, suite, entry_point, signature, prompt,
    #: difficulty, tags) -- never the reference solution.
    task: dict[str, Any] = field(default_factory=dict)

    def reward_of(self, action: Action) -> float | None:
        """Measured reward for ``action``, or ``None`` if it was never attempted."""
        return self.rewards.get(action.signature())

    @property
    def oracle_score(self) -> float:
        """Best reward available in the recorded slate."""
        return max(self.rewards.values(), default=0.0)

    @property
    def covered(self) -> bool:
        """True when the live policy chose an action from the recorded slate."""
        return self.chosen_action_id in self.rewards

    @property
    def dedup_key(self) -> str:
        """Identity of this *piece of evidence*.

        Two records with the same task, depth, slate, incoming state, feedback
        and measured rewards are the same observation recorded twice (which is
        what happens when a second generation re-explores a task and hits an
        identical decision point).  Collapsing them keeps significance tests
        honest: duplicated evidence is not extra independent evidence.
        """
        return stable_hash(
            [
                self.task_id,
                self.depth,
                sorted(self.rewards),
                {k: round(v, 6) for k, v in sorted(self.rewards.items())},
                str(self.current_state.get("solution", ""))[:2000],
                self.feedback[:500],
            ],
            length=20,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "depth": self.depth,
            "suite": self.suite,
            "features": self.features,
            "history": self.history,
            "slate": [a.to_dict() for a in self.slate],
            "rewards": self.rewards,
            "chosen_action_id": self.chosen_action_id,
            "chosen_score": self.chosen_score,
            "current_state": self.current_state,
            "feedback": self.feedback,
            "task": self.task,
        }


# ======================================================================================
# State features
# ======================================================================================
def state_features(
    state: Mapping[str, Any] | None,
    *,
    depth: int = 0,
    history: Sequence[Mapping[str, Any]] = (),
    action_counts: Mapping[str, int] | None = None,
    extra: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build the numeric feature vector a policy sees at a decision point.

    Only *observed* quantities may enter this function: scores already measured
    on the path and the actions already taken.  Never feed it the current slate's
    rewards -- that would leak the label the Dream Engine is trying to predict.
    """
    scores = [float(h.get("score", 0.0)) for h in history]
    best = max(scores, default=0.0)
    last = scores[-1] if scores else 0.0

    # Stagnation: how many trailing attempts failed to beat everything before them.
    # ``best_before[i]`` is the best score strictly before index ``i``.
    best_before: list[float] = []
    running = 0.0
    for value in scores:
        best_before.append(running)
        running = max(running, value)
    last_improvement = -1
    for i, value in enumerate(scores):
        if value > best_before[i]:
            last_improvement = i
    stagnation = 0 if len(scores) <= 1 else (len(scores) - 1 - last_improvement)

    features: dict[str, float] = {
        "depth": float(depth),
        "n_attempts": float(len(scores)),
        "best_score": float(best),
        "last_score": float(last),
        "mean_score": mean(scores),
        "std_score": stdev(scores),
        "n_passed": float(sum(1 for s in scores if s >= 1.0)),
        "n_failed": float(sum(1 for s in scores if s < 1.0)),
        "stagnation": float(stagnation),
        "has_feedback": 1.0 if str((state or {}).get("feedback") or "") else 0.0,
        "feedback_len": float(len(str((state or {}).get("feedback") or ""))),
        "artefact_len": float(len(str((state or {}).get("solution") or ""))),
    }
    for name, count in (action_counts or {}).items():
        features[f"act::{name}"] = float(count)
    for name, value in (extra or {}).items():
        features[str(name)] = float(value)
    return features


# ======================================================================================
# The tree itself
# ======================================================================================
@dataclass
class DiscoveryTree:
    """A persisted exploration trace for a single benchmark task.

    The tree is append-mostly: the explorer grows it, then serializes it to
    ``storage/trees/<tree_id>.json``.
    """

    tree_id: str
    task_id: str
    policy_id: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    root_id: str | None = None
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    #: Task context captured at exploration time (prompt, suite, signature...).
    task_meta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Where this tree was loaded from.  A runtime detail, deliberately *not*
    #: serialized: consumers such as the policy-runner manifest must reference the
    #: real file rather than reconstructing ``<tree_dir>/<tree_id>.json``, which is
    #: only correct when the writer chose that name.
    source_path: str = ""

    # -- construction -----------------------------------------------------------------
    @classmethod
    def create(
        cls,
        task_id: str,
        *,
        policy_id: str = "",
        task_meta: Mapping[str, Any] | None = None,
        root_state: Mapping[str, Any] | None = None,
        tree_id: str | None = None,
    ) -> "DiscoveryTree":
        """Create a tree with a single root node representing the empty artefact."""
        tree = cls(
            tree_id=tree_id or new_id("tree"),
            task_id=task_id,
            policy_id=policy_id,
            task_meta=dict(task_meta or {}),
        )
        root = Node(
            id=new_id("node"),
            task_id=task_id,
            parent_id=None,
            action=None,
            state=dict(root_state or {"solution": "", "explanation": "root"}),
            score=0.0,
            feedback="",
            depth=0,
            policy_id=policy_id,
        )
        tree.nodes[root.id] = root
        tree.root_id = root.id
        return tree

    # -- mutation ---------------------------------------------------------------------
    def add_node(
        self,
        parent_id: str,
        action: Action,
        state: Mapping[str, Any],
        score: float,
        *,
        feedback: str = "",
        passed: bool = False,
        policy_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        node_id: str | None = None,
    ) -> Node:
        """Attach a new child produced by ``action`` and record the edge."""
        if parent_id not in self.nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        parent = self.nodes[parent_id]
        node = Node(
            id=node_id or new_id("node"),
            task_id=self.task_id,
            parent_id=parent_id,
            action=action,
            state=dict(state),
            score=float(score),
            feedback=feedback,
            depth=parent.depth + 1,
            policy_id=policy_id or self.policy_id,
            metadata=dict(metadata or {}),
        )
        node.metadata.setdefault("passed", bool(passed))
        self.nodes[node.id] = node
        parent.children.append(node.id)
        self.edges.append(
            Edge(
                parent_id=parent_id,
                child_id=node.id,
                action=action,
                reward=float(score),
                delta=float(score) - float(parent.score),
            )
        )
        self.updated_at = utc_now()
        return node

    def record_slate(
        self,
        node_id: str,
        candidates: Sequence[CandidateOutcome],
        chosen_action_id: str | None,
    ) -> None:
        """Attach the evaluated candidate slate to ``node_id``.

        Records exactly what the live policy considered and what each option
        measured -- the information the Dream Engine later replays.
        """
        node = self.nodes[node_id]
        node.candidates = list(candidates)
        node.chosen_action_id = chosen_action_id
        node.visit_count += 1
        self.updated_at = utc_now()

    def mark_chosen(self, node_id: str, action: Action) -> None:
        """Record which slate member was actually expanded into a child."""
        self.nodes[node_id].chosen_action_id = action.signature()
        self.updated_at = utc_now()

    # -- access -----------------------------------------------------------------------
    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    @property
    def root(self) -> Node:
        if self.root_id is None or self.root_id not in self.nodes:
            raise ValueError(f"tree {self.tree_id} has no root")
        return self.nodes[self.root_id]

    def leaves(self) -> list[Node]:
        return [n for n in self.nodes.values() if not n.children]

    def path_to_root(self, node_id: str) -> list[Node]:
        """Ordered root-to-``node_id`` path (inclusive)."""
        path: list[Node] = []
        current: str | None = node_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:  # defensive: never loop forever on corrupt data
                break
            seen.add(current)
            node = self.nodes.get(current)
            if node is None:
                break
            path.append(node)
            current = node.parent_id
        return list(reversed(path))

    def blind_slate(self, node_id: str) -> list[Action]:
        """The node's candidate actions *without* their rewards (leak-proof)."""
        node = self.nodes[node_id]
        return [c.action for c in (node.candidates or [])]

    def best_node(self) -> Node:
        """Highest-scoring node; ties broken by shallowest depth (cheaper path)."""
        return max(self.nodes.values(), key=lambda n: (n.score, -n.depth))

    def solved(self, threshold: float = 1.0) -> bool:
        return any(n.score >= threshold for n in self.nodes.values())

    # -- replay / analysis ------------------------------------------------------------
    def expanded_nodes(self) -> list[Node]:
        """All nodes that carry a recorded slate, in deterministic creation order."""
        return sorted(
            (n for n in self.nodes.values() if n.expanded),
            key=lambda n: (n.depth, n.created_at, n.id),
        )

    def decision_for(self, node_id: str) -> DecisionRecord:
        """Build the replayable record for a single expanded node.

        This is the **single source of truth** for what a policy observes at a
        decision point: both the live explorer and the Dream Engine derive their
        policy input from this record, so replay features cannot drift away from
        the features the policy saw while exploring.
        """
        node = self.nodes[node_id]
        suite = str(self.task_meta.get("suite", ""))
        path = self.path_to_root(node.id)
        # History *includes* this node: the decision happens on top of the attempt
        # recorded here, and that attempt is what the policy is being asked to
        # improve.  Its score must therefore count towards ``best_score`` (that is
        # what makes ``repair`` applicable whenever a partial artefact exists).
        # The action-less root is skipped, so ``n_attempts == depth`` for every
        # non-root decision point.
        history: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for ancestor in path:
            # The root carries no action: it is the empty starting state, not an
            # attempt, so it must not inflate ``n_attempts``/``stagnation``.
            if ancestor.action is None:
                continue
            counts[ancestor.action.name] = counts.get(ancestor.action.name, 0) + 1
            history.append(
                {
                    "node_id": ancestor.id,
                    "depth": ancestor.depth,
                    "action": ancestor.action.describe() if ancestor.action else "root",
                    "score": float(ancestor.score),
                    "feedback": ancestor.feedback[:400],
                }
            )
        candidates = node.candidates or []
        features = state_features(
            {"solution": str(node.state.get("solution", "")), "feedback": node.feedback},
            depth=node.depth,
            history=history,
            action_counts=counts,
            extra=self.task_meta.get("features", {}),
        )
        return DecisionRecord(
            decision_id=f"{self.tree_id}:{node.id}",
            tree_id=self.tree_id,
            task_id=self.task_id,
            node_id=node.id,
            depth=node.depth,
            suite=suite,
            features=features,
            history=history,
            slate=[c.action for c in candidates],
            rewards={c.action.signature(): float(c.score) for c in candidates},
            chosen_action_id=node.chosen_action_id,
            chosen_score=float(
                (node.candidate_by_signature(node.chosen_action_id).score)
                if node.chosen_action_id and node.candidate_by_signature(node.chosen_action_id)
                else node.score
            ),
            current_state=dict(node.state),
            feedback=node.feedback,
            task=dict(self.task_meta),
        )

    def decisions(self) -> list[DecisionRecord]:
        """Extract replayable decision records (the Dream Engine's input)."""
        return [self.decision_for(node.id) for node in self.expanded_nodes()]

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics used by the meta-optimizer's profiler."""
        nodes = list(self.nodes.values())
        expanded = [n for n in nodes if n.expanded]
        action_stats: dict[str, dict[str, float]] = {}
        for node in expanded:
            for candidate in node.candidates or []:
                entry = action_stats.setdefault(
                    candidate.action.name, {"n": 0.0, "score_sum": 0.0, "n_passed": 0.0}
                )
                entry["n"] += 1
                entry["score_sum"] += float(candidate.score)
                entry["n_passed"] += 1.0 if candidate.passed else 0.0
        for entry in action_stats.values():
            n = entry["n"] or 1.0
            entry["mean_score"] = entry["score_sum"] / n
            entry["pass_rate"] = entry["n_passed"] / n
        return {
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "policy_id": self.policy_id,
            "n_nodes": len(nodes),
            "n_expanded": len(expanded),
            "n_decisions": len(expanded),
            "max_depth": max((n.depth for n in nodes), default=0),
            "best_score": self.best_node().score if nodes else 0.0,
            "solved": self.solved(),
            "llm_calls": int(self.metadata.get("llm_calls", 0)),
            "action_stats": action_stats,
            "evaluator_sha256": self.metadata.get("evaluator_sha256", ""),
        }

    # -- serialization ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dream-rsi/discovery-tree/v1",
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "policy_id": self.policy_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "root_id": self.root_id,
            "task_meta": self.task_meta,
            "metadata": self.metadata,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscoveryTree":
        nodes = {n["id"]: Node.from_dict(n) for n in (data.get("nodes") or [])}
        tree = cls(
            tree_id=str(data["tree_id"]),
            task_id=str(data.get("task_id", "")),
            policy_id=str(data.get("policy_id") or ""),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            root_id=data.get("root_id"),
            nodes=nodes,
            edges=[Edge.from_dict(e) for e in (data.get("edges") or [])],
            task_meta=dict(data.get("task_meta") or {}),
            metadata=dict(data.get("metadata") or {}),
        )
        return tree

    def save(self, path: str | Path) -> Path:
        """Persist the tree atomically; returns the written path."""
        target = Path(path)
        write_json(target, self.to_dict())
        return target

    @classmethod
    def load(cls, path: str | Path) -> "DiscoveryTree":
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(f"no discovery tree at {path}")
        tree = cls.from_dict(data)
        tree.source_path = str(path)
        return tree

    def to_jsonl(self) -> str:
        """One JSON object per line (nodes only) for streaming consumers."""
        return "\n".join(
            json.dumps(n.to_dict(), ensure_ascii=False) for n in self.nodes.values()
        )


# ======================================================================================
# Tree-store helpers
# ======================================================================================
def tree_sort_key(tree: DiscoveryTree) -> tuple[Any, ...]:
    """Deterministic ordering for a corpus of discovery trees.

    Sorting by ``tree_id`` alone is *not* reproducible: ids are random UUIDs, and
    record order is observable -- a policy with online statistics learns in the
    order it sees trees.  Two runs over the same worlds would then pick different
    champions.  Ordering by task, world and phase is both stable and meaningful
    (world 0 before world 1); ``created_at``/``tree_id`` only break ties between
    trees that a normal run never produces for the same (task, world, phase).
    """
    metadata = tree.metadata or {}
    try:
        world = int(metadata.get("world", 0) or 0)
    except (TypeError, ValueError):
        world = 0
    return (
        str(tree.task_id),
        world,
        str(metadata.get("phase", "")),
        str(tree.created_at),
        str(tree.tree_id),
    )


def load_trees(directory: str | Path, *, pattern: str = "*.json") -> list[DiscoveryTree]:
    """Load every discovery tree in ``directory``, skipping unreadable files."""
    trees: list[DiscoveryTree] = []
    for path in sorted(Path(directory).glob(pattern)):
        try:
            trees.append(DiscoveryTree.load(path))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return sorted(trees, key=tree_sort_key)


def decisions_from_trees(trees: Iterable[DiscoveryTree]) -> list[DecisionRecord]:
    """Flatten the decision records of many trees, deterministically ordered."""
    records: list[DecisionRecord] = []
    for tree in sorted(trees, key=tree_sort_key):
        records.extend(tree.decisions())
    return records


__all__ = [
    "Action",
    "CandidateOutcome",
    "DecisionRecord",
    "DiscoveryTree",
    "Edge",
    "Node",
    "ROOT_PARENT",
    "decisions_from_trees",
    "load_trees",
    "tree_sort_key",
    "state_features",
]
