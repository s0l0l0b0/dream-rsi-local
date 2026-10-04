"""The meta-optimizer: profiles past runs, mutates the policy, verifies, promotes.

This closes the recursive self-improvement loop.  Nothing here touches model
weights; the only thing that changes across generations is the *source of the
decision rule* in ``src/policy/search_policy.py``.

One generation, step by step
----------------------------
1. **Profile** the recorded discovery trees (:class:`FailureProfiler`): which
   strategies actually pay off, per suite, per state -- measured, not guessed.
   This is what turns "explore traces" into an improvement signal.
2. **Propose** candidates (:meth:`MetaOptimizer.propose_offline`,
   :meth:`MetaOptimizer.propose_with_llm`).  Two routes, both writing real
   ``.py`` files:
   * *offline route* -- deterministic search over the policy's ``PARAMS`` block,
     informed by the profiler.  Rewrites only that literal, so the diff is exact.
   * *LLM route* -- the meta-optimizer agent is shown the current source, the
     profiler report and the mutation contract, and asked to return an improved
     file.  Requires a real endpoint; skipped when unavailable.
3. **Score** every candidate in the Dream Engine (zero inference): all candidates
   are replayed in one subprocess against the *training* trees, and ranked.
4. **Verify** the champion on *held-out* trees that never influenced the search,
   using a **paired** comparison over identical decision points.  Common random
   numbers make this far more sensitive than comparing two independent means.
5. **Promote or reject**: only a candidate that improves the holdout by
   ``min_holdout_improvement`` with a positive paired z-score, and that does not
   regress solve rate, replaces the incumbent.  Everything -- accepted or not --
   is archived under ``storage/policies/`` so the history is auditable.

Statistical honesty
-------------------
The holdout gate is used to *decide* promotion, so a reported holdout gain is
mildly optimistic (selection on the gate).  ``run_rsi_loop.py`` therefore also
reports a live exploration probe, which is operational evidence rather than a
selection signal.  See the README's limitations section.
"""

from __future__ import annotations

import ast
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from benchmarks.evaluator import evaluator_fingerprint
from src.core.llm_client import LLMClient, LLMError
from src.core.tree import DiscoveryTree, load_trees, tree_sort_key
from src.core.utils import (
    mean,
    new_id,
    read_json,
    safe_div,
    seeded_int,
    stdev,
    utc_now,
    write_json,
)
from src.engine.dreamer import Dreamer, ReplayConfig, ReplayResult
from src.meta.policy_runner import (
    ALLOWED_POLICY_IMPORTS,
    POLICY_CLASS_NAME,
    PolicyContractError,
    load_policy_module,
    validate_module,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ======================================================================================
# Source-level mutation helpers (the mutation contract, enforced mechanically)
# ======================================================================================
def read_params_literal(source: str) -> dict[str, float]:
    """Extract the ``PARAMS`` literal from policy source without executing it."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PARAMS" and node.value is not None:
                return {str(k): float(v) for k, v in ast.literal_eval(node.value).items()}
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets
        ):
            return {str(k): float(v) for k, v in ast.literal_eval(node.value).items()}
    raise PolicyContractError("policy source has no PARAMS literal")


def read_policy_id_literal(source: str) -> str:
    """Extract the ``POLICY_ID`` string literal from policy source."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "POLICY_ID" and node.value is not None:
                return str(ast.literal_eval(node.value))
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "POLICY_ID" for t in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise PolicyContractError("policy source has no POLICY_ID literal")


def _render_params_block(params: Mapping[str, float]) -> str:
    lines = ["PARAMS: dict[str, float] = {"]
    for key, value in params.items():
        lines.append(f"    {key!r}: {float(value)!r},")
    lines.append("}")
    return "\n".join(lines)


def _replace_span(source: str, node: ast.AST, replacement: str) -> str:
    """Replace the source lines spanned by ``node`` (1-based, inclusive)."""
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno) - 1
    return "".join(lines[:start]) + replacement + "\n" + "".join(lines[end + 1 :])


def apply_params(source: str, params: Mapping[str, float], policy_id: str | None = None) -> str:
    """Rewrite the ``PARAMS`` block (and optionally ``POLICY_ID``), byte-exact.

    Everything outside those two literals is preserved verbatim, including
    comments -- so a parameter-only mutation produces a reviewable diff and the
    "only the decision rule evolves" claim stays checkable with ``git diff``.
    """
    tree = ast.parse(source)
    params_node: ast.AST | None = None
    id_node: ast.AST | None = None
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            target = names[0] if names else None
        if target == "PARAMS":
            params_node = node
        elif target == "POLICY_ID":
            id_node = node
    if params_node is None:
        raise PolicyContractError("cannot rewrite PARAMS: literal not found")

    updated = _replace_span(source, params_node, _render_params_block(params))
    if policy_id is not None:
        updated_tree = ast.parse(updated)
        for node in updated_tree.body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                target = names[0] if names else None
            if target == "POLICY_ID":
                updated = _replace_span(updated, node, f'POLICY_ID = "{policy_id}"')
                break
    # The rewritten source must still parse and still satisfy the contract.
    ast.parse(updated)
    return updated


# ======================================================================================
# Profiling
# ======================================================================================
@dataclass
class FailureProfiler:
    """Turns recorded trees into measured evidence about what works where."""

    min_support: int = 3

    def profile(self, trees: Sequence[DiscoveryTree]) -> dict[str, Any]:
        """Aggregate per-action, per-suite and state-conditional reward statistics."""
        action_all: dict[str, dict[str, float]] = {}
        action_suite: dict[str, dict[str, dict[str, float]]] = {}
        #: state-conditional reward samples, keyed by hypothesis
        conditional: dict[str, list[float]] = {
            "repair_when_partial": [],
            "restart_when_stagnant": [],
            "brute_force_high_difficulty": [],
            "brute_force_low_difficulty": [],
        }
        slates: list[list[float]] = []
        oracle_gaps: list[float] = []
        chosen_scores: list[float] = []
        best_of_tree: list[float] = []
        error_tails: dict[str, int] = {}

        for tree in trees:
            best_of_tree.append(tree.best_node().score if len(tree) else 0.0)
            for node in tree.expanded_nodes():
                candidates = node.candidates or []
                if not candidates:
                    continue
                rewards = [float(c.score) for c in candidates]
                slates.append(rewards)
                suite = str(tree.task_meta.get("suite", ""))
                difficulty = float(tree.task_meta.get("features", {}).get("difficulty", 0.5))
                partial = (
                    bool(str(node.state.get("solution") or "").strip()) and float(node.score) > 0.0
                )
                stagnant = self._stagnation(tree, node.id) >= 2
                for candidate in candidates:
                    name = candidate.action.name
                    bucket = action_all.setdefault(
                        name, {"n": 0.0, "score_sum": 0.0, "n_passed": 0.0}
                    )
                    bucket["n"] += 1
                    bucket["score_sum"] += float(candidate.score)
                    bucket["n_passed"] += 1.0 if candidate.passed else 0.0
                    sub = action_suite.setdefault(suite, {}).setdefault(
                        name, {"n": 0.0, "score_sum": 0.0, "n_passed": 0.0}
                    )
                    sub["n"] += 1
                    sub["score_sum"] += float(candidate.score)
                    sub["n_passed"] += 1.0 if candidate.passed else 0.0
                    if name == "repair" and partial:
                        conditional["repair_when_partial"].append(float(candidate.score))
                    if name == "restart" and stagnant:
                        conditional["restart_when_stagnant"].append(float(candidate.score))
                    if name == "brute_force":
                        key = (
                            "brute_force_high_difficulty"
                            if difficulty >= 0.5
                            else "brute_force_low_difficulty"
                        )
                        conditional[key].append(float(candidate.score))
                    if candidate.error:
                        tail = candidate.error.split(":")[0][:60]
                        error_tails[tail] = error_tails.get(tail, 0) + 1

                if node.chosen_action_id:
                    chosen = node.candidate_by_signature(node.chosen_action_id)
                    if chosen is not None:
                        chosen_scores.append(float(chosen.score))
                oracle_gaps.append(max(rewards) - min(rewards))

        actions = {}
        for name, bucket in sorted(action_all.items()):
            n = bucket["n"] or 1.0
            actions[name] = {
                "n": int(bucket["n"]),
                "mean_reward": bucket["score_sum"] / n,
                "pass_rate": bucket["n_passed"] / n,
            }
        per_suite = {}
        for suite, buckets in sorted(action_suite.items()):
            per_suite[suite] = {
                name: {
                    "n": int(b["n"]),
                    "mean_reward": b["score_sum"] / (b["n"] or 1.0),
                    "pass_rate": b["n_passed"] / (b["n"] or 1.0),
                }
                for name, b in sorted(buckets.items())
            }
        return {
            "n_trees": len(trees),
            "n_decisions": len(slates),
            "mean_slate_size": mean([float(len(s)) for s in slates]),
            "mean_oracle_reward": mean([max(s) for s in slates]) if slates else 0.0,
            "mean_worst_reward": mean([min(s) for s in slates]) if slates else 0.0,
            "mean_slate_spread": mean(oracle_gaps),
            "mean_chosen_reward": mean(chosen_scores),
            "mean_best_of_tree": mean(best_of_tree),
            "tree_solve_rate": safe_div(
                sum(1 for s in best_of_tree if s >= 1.0), len(best_of_tree)
            ),
            "actions": actions,
            "per_suite": per_suite,
            "conditional": {
                key: {
                    "values": [round(v, 4) for v in vals],
                    "mean": mean(vals),
                    "n": len(vals),
                }
                for key, vals in conditional.items()
                if vals
            },
            "common_errors": dict(
                sorted(error_tails.items(), key=lambda kv: -kv[1])[:8]
            ),
        }

    @staticmethod
    def _stagnation(tree: DiscoveryTree, node_id: str) -> int:
        """Trailing attempts at ``node_id`` that failed to beat the best before them."""
        scores = [float(n.score) for n in tree.path_to_root(node_id)]
        if len(scores) <= 1:
            return 0
        best_before = 0.0
        last = -1
        for i, value in enumerate(scores):
            if value > best_before:
                last = i
            best_before = max(best_before, value)
        return len(scores) - 1 - last

    # -- textual report + data-driven hypotheses --------------------------------------
    def report(self, profile: Mapping[str, Any]) -> str:
        """Human/LLM-readable summary of the profile."""
        lines = [
            f"Discovery trees: {profile['n_trees']}  decisions: {profile['n_decisions']}",
            f"Mean slate size: {profile['mean_slate_size']:.2f} "
            f"(oracle {profile['mean_oracle_reward']:.3f}, "
            f"worst {profile['mean_worst_reward']:.3f}, "
            f"spread {profile['mean_slate_spread']:.3f})",
            f"Mean reward of the action that was actually taken: "
            f"{profile['mean_chosen_reward']:.3f}",
            "",
            "Strategy reward (pooled over all suites):",
        ]
        for name, stat in sorted(
            profile["actions"].items(), key=lambda kv: -kv[1]["mean_reward"]
        ):
            lines.append(
                f"  {name:<18} n={stat['n']:<4} mean={stat['mean_reward']:.3f} "
                f"pass={stat['pass_rate']:.3f}"
            )
        for suite, stats in sorted(profile["per_suite"].items()):
            ranked = sorted(stats.items(), key=lambda kv: -kv[1]["mean_reward"])
            best = ", ".join(f"{n}={s['mean_reward']:.2f}" for n, s in ranked[:3])
            worst = ", ".join(f"{n}={s['mean_reward']:.2f}" for n, s in ranked[-2:])
            lines.append(f"  suite {suite:<12} best: {best} | worst: {worst}")
        if profile["conditional"]:
            lines.append("")
            lines.append("State-conditional evidence:")
            for key, stat in profile["conditional"].items():
                lines.append(f"  {key}: mean={stat['mean']:.3f} n={stat['n']}")
        if profile["common_errors"]:
            lines.append("")
            lines.append("Most common diagnostics:")
            for err, count in list(profile["common_errors"].items())[:5]:
                lines.append(f"  x{count} {err}")
        return "\n".join(lines)

    def hypotheses(self, profile: Mapping[str, Any]) -> list[dict[str, float]]:
        """Data-driven parameter proposals derived from the measured profile.

        These are *not* answers baked into the optimizer: each one reads a
        statistic out of the profile (which action pays off in which suite, how
        much repair gains when a partial artefact exists) and expresses it as a
        parameter change for the search to evaluate.
        """
        proposals: list[dict[str, float]] = []
        actions = profile.get("actions") or {}
        if not actions:
            return proposals

        ranked_all = sorted(actions.items(), key=lambda kv: -kv[1]["mean_reward"])
        best_all, worst_all = ranked_all[0][0], ranked_all[-1][0]
        proposals.append({f"prior_{best_all}": 0.5})
        proposals.append({f"prior_{worst_all}": -0.5})

        for suite, stats in (profile.get("per_suite") or {}).items():
            ranked = sorted(stats.items(), key=lambda kv: -kv[1]["mean_reward"])
            if len(ranked) >= 2:
                proposals.append(
                    {
                        f"prior_{ranked[0][0]}": 0.4,
                        f"prior_{ranked[-1][0]}": -0.3,
                    }
                )

        conditional = profile.get("conditional") or {}
        repair = conditional.get("repair_when_partial")
        overall = mean([s["mean_reward"] for s in actions.values()])
        if repair and repair["n"] >= self.min_support:
            gain = max(0.0, repair["mean"] - overall)
            proposals.append({"repair_bonus": round(min(3.0, 1.0 + 2.0 * gain), 3)})
        restart = conditional.get("restart_when_stagnant")
        if restart and restart["n"] >= self.min_support:
            gain = max(0.0, restart["mean"] - overall)
            proposals.append({"restart_bonus": round(min(2.0, 0.5 + 2.0 * gain), 3)})
        high = conditional.get("brute_force_high_difficulty")
        low = conditional.get("brute_force_low_difficulty")
        if high and low and high["n"] >= 2 and low["n"] >= 2:
            delta = low["mean"] - high["mean"]
            proposals.append({"brute_force_penalty": round(max(-3.0, -3.0 * delta), 3)})
        proposals.append({"w_suite_value": 1.0})
        proposals.append({"w_global_value": 0.75})
        proposals.append({"w_repeat_penalty": 0.5})
        return [p for p in proposals if p]


# ======================================================================================
# Offline parameter search
# ======================================================================================
class OfflineMutator:
    """Deterministic, seeded search over the policy's ``PARAMS`` block."""

    def __init__(
        self,
        param_space: Mapping[str, Sequence[float]],
        *,
        seed: int = 0,
    ) -> None:
        self.space: dict[str, tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in param_space.items()
        }
        self.rng = random.Random(seed)

    # -- building blocks --------------------------------------------------------------
    def clip(self, params: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in params.items():
            low, high = self.space.get(key, (-1e9, 1e9))
            out[key] = round(max(low, min(high, float(value))), 4)
        return out

    #: Quantiles of each knob's range that coordinate probes sample.
    QUANTILES: tuple[float, ...] = (0.5, 0.85, 0.15, 0.95)

    def coordinate_probes(
        self, base: Mapping[str, float], limit: int | None = None
    ) -> list[dict[str, float]]:
        """One-parameter-at-a-time probes across each knob's range.

        Probes are **round-robin interleaved across knobs and the knob order is
        shuffled**, so a truncated budget still touches every parameter.  A naive
        ``probes[:budget]`` slice would otherwise spend the whole budget on the
        first few keys of the dict and never evaluate the state-conditioned terms.
        """
        keys = list(self.space)
        self.rng.shuffle(keys)
        per_key: list[list[dict[str, float]]] = []
        for key in keys:
            low, high = self.space[key]
            row: list[dict[str, float]] = []
            for q in self.QUANTILES:
                value = low + q * (high - low)
                if abs(value - float(base.get(key, 0.0))) < 1e-9:
                    continue
                row.append(self.clip({**base, key: value}))
            per_key.append(row)
        interleaved: list[dict[str, float]] = []
        depth = max((len(row) for row in per_key), default=0)
        for level in range(depth):
            for row in per_key:
                if level < len(row):
                    interleaved.append(row[level])
        return interleaved[:limit] if limit else interleaved

    def sparse_perturbations(
        self, base: Mapping[str, float], n: int, *, max_changes: int = 5, scale: float = 0.5
    ) -> list[dict[str, float]]:
        """Perturb a random subset of knobs by an amount scaled to each range."""
        out: list[dict[str, float]] = []
        keys = list(self.space)
        for _ in range(n):
            k = self.rng.randint(1, max(max_changes, 1))
            chosen = self.rng.sample(keys, min(k, len(keys)))
            candidate = dict(base)
            for key in chosen:
                low, high = self.space[key]
                span = high - low
                candidate[key] = float(base.get(key, 0.0)) + self.rng.uniform(-scale, scale) * span
            out.append(self.clip(candidate))
        return out

    def uniform_samples(self, n: int) -> list[dict[str, float]]:
        """Fresh samples from the whole space (global exploration)."""
        return [
            self.clip({k: self.rng.uniform(low, high) for k, (low, high) in self.space.items()})
            for _ in range(n)
        ]

    def merge(self, base: Mapping[str, float], patch: Mapping[str, float]) -> dict[str, float]:
        """Apply a sparse patch (used for profiler hypotheses) onto ``base``."""
        return self.clip({**base, **patch})


# ======================================================================================
# Candidates and reports
# ======================================================================================
@dataclass
class Candidate:
    """One proposed successor policy."""

    label: str
    params: dict[str, float]
    source: str
    policy_id: str
    origin: str = "offline"
    path: Path | None = None
    train: ReplayResult | None = None
    holdout: ReplayResult | None = None
    status: str = "proposed"
    error: str = ""
    note: str = ""

    @property
    def train_score(self) -> float:
        return self.train.mean_reward if self.train and not self.train.error else float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "policy_id": self.policy_id,
            "origin": self.origin,
            "status": self.status,
            "error": self.error,
            "note": self.note,
            "n_params_changed": len(
                [k for k, v in self.params.items() if abs(v) > 1e-12]
            ),
            "path": str(self.path) if self.path else "",
            "train": self.train.row() if self.train else None,
            "holdout": self.holdout.row() if self.holdout else None,
        }


@dataclass
class GenerationRecord:
    """Audit trail for one RSI generation."""

    generation: int
    incumbent_id: str
    champion_id: str
    accepted: bool
    reason: str
    n_candidates: int = 0
    n_scored: int = 0
    incumbent_train: dict[str, Any] = field(default_factory=dict)
    champion_train: dict[str, Any] = field(default_factory=dict)
    incumbent_holdout: dict[str, Any] = field(default_factory=dict)
    champion_holdout: dict[str, Any] = field(default_factory=dict)
    paired: dict[str, Any] = field(default_factory=dict)
    baselines: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    llm_route: str = "unavailable"
    archive_dir: str = ""
    wall_ms: float = 0.0
    params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class OptimizationReport:
    """Full history of an optimization run."""

    started_at: str = field(default_factory=utc_now)
    finished_at: str = ""
    initial_policy: str = ""
    final_policy: str = ""
    generations: list[GenerationRecord] = field(default_factory=list)
    evaluator_sha256: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> int:
        return sum(1 for g in self.generations if g.accepted)

    def curve(self) -> list[dict[str, Any]]:
        """Per-generation metrics, including the initial incumbent as gen -1."""
        rows: list[dict[str, Any]] = []
        if self.generations:
            first = self.generations[0]
            rows.append(
                {
                    "generation": -1,
                    "policy": first.incumbent_id,
                    "train": first.incumbent_train.get("mean_reward"),
                    "holdout": first.incumbent_holdout.get("mean_reward"),
                    "accepted": None,
                }
            )
        for gen in self.generations:
            rows.append(
                {
                    "generation": gen.generation,
                    "policy": gen.champion_id,
                    "train": gen.champion_train.get("mean_reward"),
                    "holdout": gen.champion_holdout.get("mean_reward"),
                    "accepted": gen.accepted,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["generations"] = [g.to_dict() for g in self.generations]
        data["curve"] = self.curve()
        data["n_accepted"] = self.accepted
        return data

    def markdown(self) -> str:
        """Compact human-readable report (written next to the JSON)."""
        lines = [
            "# Dream-RSI optimization report",
            "",
            f"* started: {self.started_at}",
            f"* finished: {self.finished_at or 'in progress'}",
            f"* policy: `{self.initial_policy}` -> `{self.final_policy}`",
            f"* generations run: {len(self.generations)}, accepted: {self.accepted}",
            f"* evaluator fingerprint: `{self.evaluator_sha256[:16]}`",
            "",
            "## Dream-score curve",
            "",
            "| generation | policy | train mean reward | holdout mean reward | accepted |",
            "| ---: | --- | ---: | ---: | :---: |",
        ]
        for row in self.curve():
            train = "—" if row["train"] is None else f"{row['train']:.4f}"
            hold = "—" if row["holdout"] is None else f"{row['holdout']:.4f}"
            accepted = "—" if row["accepted"] is None else ("yes" if row["accepted"] else "no")
            lines.append(
                f"| {row['generation']} | `{row['policy']}` | {train} | {hold} | {accepted} |"
            )
        lines += ["", "## Per-generation detail", ""]
        for gen in self.generations:
            lines.append(f"### Generation {gen.generation}: {gen.reason}")
            lines.append("")
            lines.append(
                f"* incumbent `{gen.incumbent_id}` -> champion `{gen.champion_id}` "
                f"({'**promoted**' if gen.accepted else 'rejected'})"
            )
            lines.append(
                f"* candidates proposed: {gen.n_candidates}, scored: {gen.n_scored}, "
                f"LLM route: {gen.llm_route}"
            )
            paired = gen.paired or {}
            if paired:
                lines.append(
                    f"* paired holdout delta: {paired.get('mean_delta', 0.0):+.4f} "
                    f"(se {paired.get('se', 0.0):.4f}, z {paired.get('z', 0.0):.2f}, "
                    f"{paired.get('n_changed', 0)}/{paired.get('n', 0)} decisions changed)"
                )
            if gen.incumbent_holdout and gen.champion_holdout:
                lines.append(
                    f"* holdout: {gen.incumbent_holdout.get('mean_reward', 0):.4f} -> "
                    f"{gen.champion_holdout.get('mean_reward', 0):.4f} "
                    f"(solve rate {gen.incumbent_holdout.get('task_solve_rate', 0):.3f} -> "
                    f"{gen.champion_holdout.get('task_solve_rate', 0):.3f})"
                )
            if gen.baselines:
                base = ", ".join(
                    f"{name}={data.get('mean_reward', 0):.4f}"
                    for name, data in gen.baselines.items()
                )
                lines.append(f"* baselines on holdout: {base}")
            lines.append(f"* archive: `{gen.archive_dir}`")
            lines.append("")
        return "\n".join(lines)


# ======================================================================================
# Statistical gate
# ======================================================================================
def paired_comparison(
    incumbent_rewards: Sequence[float],
    champion_rewards: Sequence[float],
    *,
    seed: int = 0,
    bootstrap: int = 2000,
) -> dict[str, Any]:
    """Paired comparison of two policies replayed on identical decision points.

    Both replays walk the same records in the same order, so element ``i`` of each
    reward list belongs to the same decision.  Differences cancel the enormous
    decision-level variance, which is why this gate can detect a real +0.03 while
    comparing two independent means could not.
    """
    n = min(len(incumbent_rewards), len(champion_rewards))
    if n == 0:
        return {"n": 0, "mean_delta": 0.0, "se": 0.0, "z": 0.0, "n_changed": 0, "p_value": 1.0}
    diffs = [float(champion_rewards[i]) - float(incumbent_rewards[i]) for i in range(n)]
    changed = [d for d in diffs if abs(d) > 1e-9]
    delta = mean(diffs)
    spread = stdev(diffs)
    se = spread / math.sqrt(n) if n > 1 else 0.0
    z = delta / se if se > 0 else (math.inf if delta > 0 else 0.0)
    p_value = 0.5 * math.erfc(z / math.sqrt(2)) if math.isfinite(z) else 0.0

    # Seeded bootstrap CI on the paired mean (cheap and assumption-light).
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(bootstrap):
        boot.append(mean([diffs[rng.randrange(n)] for _ in range(n)]))
    boot.sort()
    ci_low = boot[int(0.05 * len(boot))]
    ci_high = boot[int(0.95 * len(boot)) - 1]
    return {
        "n": n,
        "n_changed": len(changed),
        "mean_delta": delta,
        "mean_delta_changed": mean(changed) if changed else 0.0,
        "sd_delta": spread,
        "se": se,
        "z": z if math.isfinite(z) else 999.0,
        "p_value": p_value,
        "ci90": [ci_low, ci_high],
    }


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass
class OptimizerConfig:
    """Knobs for the RSI loop."""

    #: Fraction of *tasks* used to fit candidates; the rest gate promotion.
    train_ratio: float = 0.6
    #: Minimum paired holdout improvement required for promotion.
    min_holdout_improvement: float = 0.005
    #: Minimum paired z-score for promotion.
    min_z: float = 1.0
    #: A candidate must show at least this paired gain on the *training* trees
    #: before it may become the champion (filters lucky draws).
    min_train_improvement: float = 0.01
    #: ... and this paired z-score on the training trees.
    min_train_z: float = 1.0
    #: Allowed holdout solve-rate regression (relative).
    max_solve_regression: float = 0.05
    #: Allowed net loss of solved holdout trees (absolute, noise tolerance).
    max_solve_loss_tasks: int = 1
    #: Minimum number of holdout decisions before promotion is even considered.
    #: With one or two decisions a "perfect" paired gain is noise, not evidence.
    min_holdout_decisions: int = 6
    #: Candidates proposed per generation.
    n_candidates: int = 200
    #: Greedy refinement rounds inside one generation.
    search_rounds: int = 2
    #: Also compute the diagnostic baselines (oracle/uniform/global-best).
    with_baselines: bool = True
    #: Use the LLM mutation route when a real endpoint is configured.
    use_llm: bool = True
    llm_candidates: int = 1
    llm_temperature: float = 0.4
    #: Replay seed (must match the explorer's seed for identical slates).
    seed: int = 0
    #: Wall-clock budget for one batch subprocess.
    runner_timeout_s: float = 900.0
    per_candidate_timeout_s: float = 20.0
    #: Keep the source of every rejected candidate.
    archive_candidates: bool = True


# ======================================================================================
# The optimizer
# ======================================================================================
class MetaOptimizer:
    """Profiles, mutates, verifies and promotes the exploration policy."""

    def __init__(
        self,
        *,
        policy_path: str | Path = REPO_ROOT / "src" / "policy" / "search_policy.py",
        storage_dir: str | Path = REPO_ROOT / "storage",
        trees_dir: str | Path = REPO_ROOT / "storage" / "trees",
        client: LLMClient | None = None,
        config: OptimizerConfig | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.storage_dir = Path(storage_dir)
        self.trees_dir = Path(trees_dir)
        self.policies_dir = self.storage_dir / "policies"
        self.candidates_dir = self.policies_dir / "candidates"
        self.client = client
        self.config = config or OptimizerConfig()
        self.profiler = FailureProfiler()
        self.fingerprint = evaluator_fingerprint()
        #: Optional extra context for the LLM prompt (task list, budget, ...).
        self.context = dict(context or {})
        self.registry_path = self.policies_dir / "registry.json"

    # -- incumbent --------------------------------------------------------------------
    def incumbent_source(self) -> str:
        return self.policy_path.read_text(encoding="utf-8")

    def incumbent_params(self, source: str | None = None) -> dict[str, float]:
        return read_params_literal(source if source is not None else self.incumbent_source())

    def incumbent_policy_id(self, source: str | None = None) -> str:
        return read_policy_id_literal(source if source is not None else self.incumbent_source())

    def load_incumbent(self) -> Any:
        """Instantiate the current policy from disk (trusted code, same process)."""
        module = load_policy_module(self.policy_path, name="dream_rsi_incumbent")
        problems = validate_module(module)
        if problems:
            raise PolicyContractError("incumbent policy violates the contract: " + "; ".join(problems))
        return module

    def replay_incumbent(self, trees: Sequence[DiscoveryTree], tag: str) -> ReplayResult:
        """Replay the incumbent in-process (it is trusted, so no subprocess needed)."""
        module = self.load_incumbent()
        policy = module.SearchPolicy(rng_seed=self.config.seed)
        return Dreamer(trees, config=ReplayConfig(rng_seed=self.config.seed)).replay(
            policy, label=f"{policy.policy_id}[{tag}]"
        )

    # -- corpus split -----------------------------------------------------------------
    def split_task_ids(
        self,
        task_ids: Iterable[str],
        suites: Mapping[str, str],
    ) -> tuple[set[str], set[str]]:
        """Deterministically split *task ids* into (train, holdout), stratified by suite.

        Shared by the exploration planner (which needs to know the split *before*
        any tree exists) and by the optimizer, so both agree on the partition.
        Splitting by task -- never by tree -- is essential: every tree of a task
        lands on the same side, otherwise a policy could be fitted and verified on
        different attempts at the same task.
        """
        by_suite: dict[str, list[str]] = {}
        for task_id in sorted(set(task_ids)):
            by_suite.setdefault(str(suites.get(task_id, "")), []).append(task_id)

        train_ids: set[str] = set()
        holdout_ids: set[str] = set()
        for suite, ids in sorted(by_suite.items()):
            ordered = sorted(ids, key=lambda t: seeded_int("split", suite, t))
            if len(ordered) == 1:
                # A lone task cannot be split; it trains rather than leaving the
                # holdout empty for its suite.
                train_ids.update(ordered)
                continue
            n_train = max(1, min(len(ordered) - 1, int(round(len(ordered) * self.config.train_ratio))))
            train_ids.update(ordered[:n_train])
            holdout_ids.update(ordered[n_train:])
        return train_ids, holdout_ids

    def split_tasks(
        self, trees: Sequence[DiscoveryTree]
    ) -> tuple[list[DiscoveryTree], list[DiscoveryTree]]:
        """Split a tree corpus into (train, holdout) using :meth:`split_task_ids`."""
        suites = {t.task_id: str(t.task_meta.get("suite", "")) for t in trees}
        train_ids, holdout_ids = self.split_task_ids(suites, suites)
        train = [t for t in trees if t.task_id in train_ids]
        holdout = [t for t in trees if t.task_id in holdout_ids]
        return train, holdout

    # -- candidate construction -------------------------------------------------------
    def _make_candidate(
        self,
        label: str,
        params: Mapping[str, float],
        base_source: str,
        *,
        origin: str,
        policy_id: str,
        note: str = "",
    ) -> Candidate:
        source = apply_params(base_source, params, policy_id)
        return Candidate(
            label=label,
            params=dict(params),
            source=source,
            policy_id=policy_id,
            origin=origin,
            note=note,
        )

    def propose_offline(
        self,
        base_params: Mapping[str, float],
        base_source: str,
        param_space: Mapping[str, Sequence[float]],
        profile: Mapping[str, Any],
        generation: int,
        *,
        n: int | None = None,
    ) -> list[Candidate]:
        """Deterministic parameter search seeded by the profiler's hypotheses."""
        cfg = self.config
        n = n or cfg.n_candidates
        mutator = OfflineMutator(param_space, seed=seeded_int("mutator", cfg.seed, generation))
        budget_per_round = max(1, n // max(1, cfg.search_rounds))

        proposals: list[tuple[str, dict[str, float], str]] = []
        for i, patch in enumerate(self.profiler.hypotheses(profile)):
            proposals.append((f"g{generation}_hyp{i}", mutator.merge(base_params, patch), f"hypothesis {patch}"))
        for params in mutator.coordinate_probes(base_params, limit=max(1, budget_per_round)):
            proposals.append((f"g{generation}_coord{len(proposals)}", params, "coordinate probe"))
        for params in mutator.sparse_perturbations(base_params, max(1, budget_per_round // 2)):
            proposals.append((f"g{generation}_sparse{len(proposals)}", params, "sparse perturbation"))
        for params in mutator.uniform_samples(max(1, budget_per_round // 4)):
            proposals.append((f"g{generation}_uni{len(proposals)}", params, "uniform sample"))
        for params in mutator.sparse_perturbations(base_params, max(1, budget_per_round // 4), scale=0.15):
            proposals.append((f"g{generation}_refine{len(proposals)}", params, "local refinement"))

        candidates: list[Candidate] = []
        seen: set[str] = set()
        for label, params, note in proposals:
            key = json.dumps({k: round(v, 4) for k, v in sorted(params.items())}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                self._make_candidate(
                    f"{label}_{len(candidates):03d}",
                    params,
                    base_source,
                    origin="offline",
                    policy_id=f"policy_v{generation + 1}",
                    note=note,
                )
            )
            if len(candidates) >= n:
                break
        return candidates

    def propose_with_llm(
        self,
        base_source: str,
        profile_report: str,
        profile: Mapping[str, Any],
        generation: int,
    ) -> tuple[Candidate | None, str]:
        """Ask the meta-optimizer's LLM to rewrite the policy source.

        Returns ``(candidate, route_status)``.  The route is skipped -- cleanly,
        with a recorded reason -- whenever no real endpoint is configured, since
        the offline simulator cannot write policy code.
        """
        if not self.config.use_llm:
            return None, "disabled by config"
        if self.client is None:
            return None, "no client configured"
        if "mock" in getattr(self.client, "name", ""):
            return None, "offline simulator cannot author policy code (parameter search used)"

        prompt = self._mutation_prompt(base_source, profile_report, profile)
        try:
            result = self.client.complete(
                [
                    {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=4096,
            )
        except LLMError as exc:
            return None, f"llm error: {exc}"
        if not result.ok:
            return None, f"llm error: {result.error}"
        source = result.code or ""
        if not source.strip():
            return None, "llm returned no code"
        try:
            params = read_params_literal(source)
            problems = self._source_contract_problems(source)
            if problems:
                return None, "llm candidate rejected: " + "; ".join(problems)
            policy_id = f"policy_v{generation + 1}_llm"
            source = apply_params(source, params, policy_id)
        except (PolicyContractError, SyntaxError, ValueError) as exc:
            return None, f"llm candidate rejected: {type(exc).__name__}: {exc}"
        return (
            Candidate(
                label=f"g{generation}_llm",
                params=params,
                source=source,
                policy_id=policy_id,
                origin="llm",
                note="LLM-authored rewrite",
            ),
            f"llm authored candidate ({result.total_tokens} tokens)",
        )

    def _mutation_prompt(
        self, base_source: str, profile_report: str, profile: Mapping[str, Any]
    ) -> str:
        space = {}
        try:
            module = self.load_incumbent()
            space = {k: list(v) for k, v in module.SearchPolicy.param_space().items()}
        except Exception:  # noqa: BLE001 - prompt building must never fail the run
            space = {}
        return "\n".join(
            [
                "You are the meta-optimizer of a recursive self-improvement loop.",
                "Your job: rewrite the exploration policy below so that it scores a higher",
                "mean reward on replayed discovery trees. You may restructure the decision",
                "logic, add derived features and change the weights. Nothing else changes.",
                "",
                "MUTATION CONTRACT (must hold):",
                f"* keep a class named {POLICY_CLASS_NAME} subclassing ExplorationPolicy;",
                "* keep `POLICY_ID` a string literal and `PARAMS` a flat dict[str, float]",
                "  literal with one entry per line (a search process rewrites that block);",
                "* `param_space()` must return (low, high) bounds for every PARAMS key;",
                "* `select(ctx)` must return an Action drawn from ctx.candidates;",
                "* imports limited to: " + ", ".join(sorted(ALLOWED_POLICY_IMPORTS)) + ";",
                "* no file, network, process or clock access; deterministic given ctx;",
                "* `ctx` NEVER contains reward information -- do not invent such fields.",
                "",
                "MEASURED PROFILE OF PAST RUNS:",
                profile_report,
                "",
                f"Current parameter space: {json.dumps(space)}",
                f"Historical reward statistics: {json.dumps(profile.get('actions', {}))}",
                "",
                "CURRENT POLICY SOURCE:",
                "```python",
                base_source,
                "```",
                "",
                "Return the COMPLETE improved file in a single ```python fenced block.",
                "Explain nothing outside the block.",
            ]
        )

    def _source_contract_problems(self, source: str) -> list[str]:
        """Validate candidate source without importing it into this process."""
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]
        problems: list[str] = []
        try:
            read_params_literal(source)
            read_policy_id_literal(source)
        except PolicyContractError as exc:
            problems.append(str(exc))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_POLICY_IMPORTS:
                        problems.append(f"disallowed import {alias.name!r}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in ALLOWED_POLICY_IMPORTS:
                    problems.append(f"disallowed import from {node.module!r}")
        return problems

    # -- scoring ----------------------------------------------------------------------
    def _write_candidates(self, candidates: Sequence[Candidate], generation: int) -> Path:
        gen_dir = self.candidates_dir / f"gen_{generation:03d}" / "proposed"
        gen_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            path = gen_dir / f"{candidate.label}.py"
            path.write_text(candidate.source, encoding="utf-8")
            candidate.path = path
        return gen_dir

    def _manifest(self, trees: Sequence[DiscoveryTree], path: Path) -> Path:
        # Reference where each tree was actually read from; falling back to
        # ``<tree_dir>/<tree_id>.json`` only works when the writer happened to use
        # that name, and a stale manifest makes the runner fail with a confusing
        # FileNotFoundError instead of scoring anything.
        payload = {
            "label": path.stem,
            "evaluator_sha256": self.fingerprint,
            "tree_paths": [
                t.source_path or str(Path(self.trees_dir) / f"{t.tree_id}.json")
                for t in trees
            ],
            "task_ids": sorted({t.task_id for t in trees}),
        }
        write_json(path, payload)
        return path

    def score_candidates(
        self,
        candidates: Sequence[Candidate],
        trees: Sequence[DiscoveryTree],
        generation: int,
        *,
        tag: str,
        into: str = "train",
    ) -> dict[str, ReplayResult]:
        """Replay many candidates in one subprocess and record their metrics.

        ``into`` selects which field of each :class:`Candidate` receives the
        result (``"train"`` for fitting, ``"holdout"`` for verification).
        """
        if not candidates:
            return {}
        work = self.candidates_dir / f"gen_{generation:03d}"
        work.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(trees, work / f"manifest_{tag}.json")
        out_path = work / f"results_{tag}.json"
        paths = [str(c.path) for c in candidates if c.path]
        cmd = [
            sys.executable,
            "-m",
            "src.meta.policy_runner",
            "replay",
            "--policies",
            ",".join(paths),
            "--manifest",
            str(manifest),
            "--out",
            str(out_path),
            "--seed",
            str(self.config.seed),
            "--per-candidate-timeout",
            str(self.config.per_candidate_timeout_s),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONHASHSEED"] = "0"
        payload: dict[str, Any] = {}
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                timeout=self.config.runner_timeout_s,
                check=False,
            )
            if out_path.exists():
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            else:
                # Keep enough stderr to diagnose the failure: a truncated tail
                # hides the first line of the traceback, which is the useful one.
                err = (completed.stderr or b"").decode("utf-8", "replace").strip()
                out = (completed.stdout or b"").decode("utf-8", "replace").strip()
                detail = (err or out)[-4000:]
                for candidate in candidates:
                    candidate.status = "runner_failed"
                    candidate.error = (
                        f"runner produced no output (rc={completed.returncode}, "
                        f"{len(candidates)} policies): {detail}"
                    )
                return {}
        except subprocess.TimeoutExpired:
            for candidate in candidates:
                candidate.status = "runner_timeout"
                candidate.error = f"batch exceeded {self.config.runner_timeout_s:.0f}s"
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            for candidate in candidates:
                candidate.status = "runner_failed"
                candidate.error = f"{type(exc).__name__}: {exc}"
            return {}

        by_name = {Path(r["policy_path"]).name: r for r in payload.get("results", [])}
        scored: dict[str, ReplayResult] = {}
        for candidate in candidates:
            entry = by_name.get(candidate.path.name if candidate.path else "")
            if entry is None:
                candidate.status = "missing"
                candidate.error = "no result in batch output"
                continue
            if not entry.get("ok"):
                candidate.status = "rejected"
                candidate.error = str(entry.get("error", "unknown error"))
                continue
            result = _result_from_dict(entry["replay"])
            setattr(candidate, into, result)
            if into == "train":
                candidate.status = "scored"
            scored[candidate.label] = result
        return scored

    def replay_candidate(
        self,
        candidate: Candidate,
        trees: Sequence[DiscoveryTree],
        generation: int,
        *,
        tag: str,
    ) -> ReplayResult:
        """Replay a single candidate on an arbitrary tree set (used for holdout)."""
        results = self.score_candidates([candidate], trees, generation, tag=tag, into="holdout")
        return results.get(
            candidate.label,
            ReplayResult(policy_id=candidate.policy_id, error=candidate.error or "not scored"),
        )

    def _pair_note(self, incumbent: ReplayResult, candidate: Candidate) -> str:
        """Attach the paired training comparison to a candidate's metadata."""
        if candidate.train is None:
            return candidate.note
        paired = paired_comparison(
            incumbent.rewards, candidate.train.rewards, seed=self.config.seed, bootstrap=200
        )
        candidate.train.metadata["paired_train_delta"] = paired["mean_delta"]
        candidate.train.metadata["paired_train_z"] = paired["z"]
        candidate.train.metadata["paired_train_changed"] = paired["n_changed"]
        return candidate.note or (
            f"paired train delta {paired['mean_delta']:+.4f} "
            f"(z={paired['z']:.2f}, {paired['n_changed']}/{paired['n']} changed)"
        )

    # -- gate -------------------------------------------------------------------------
    def gate(
        self,
        incumbent_holdout: ReplayResult,
        champion_holdout: ReplayResult,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Decide whether the champion may replace the incumbent."""
        cfg = self.config
        if champion_holdout.error:
            return False, f"champion replay error: {champion_holdout.error}", {}
        paired = paired_comparison(
            incumbent_holdout.rewards, champion_holdout.rewards, seed=self.config.seed
        )
        delta = paired["mean_delta"]
        if paired["n"] == 0:
            return False, "no holdout decisions to compare", paired
        if paired["n"] < cfg.min_holdout_decisions:
            return (
                False,
                f"only {paired['n']} holdout decision(s): need at least "
                f"{cfg.min_holdout_decisions} before promoting",
                paired,
            )
        if delta < cfg.min_holdout_improvement:
            return (
                False,
                f"paired holdout gain {delta:+.4f} below threshold "
                f"{cfg.min_holdout_improvement:+.4f}",
                paired,
            )
        if paired["z"] < cfg.min_z:
            return False, f"paired z-score {paired['z']:.2f} below {cfg.min_z:.2f}", paired
        # Solve rate on a handful of holdout trees is a coarse, noisy statistic, so
        # the guard is expressed in *trees*: losing one solve is tolerated, losing
        # more is treated as a real regression in capability.
        lost_solves = incumbent_holdout.n_solved_trees - champion_holdout.n_solved_trees
        allowed_loss = max(
            cfg.max_solve_loss_tasks,
            int(math.ceil(cfg.max_solve_regression * max(1, champion_holdout.n_trees))),
        )
        if lost_solves > allowed_loss:
            return (
                False,
                f"holdout would lose {lost_solves} solved trees "
                f"(limit {allowed_loss}; {incumbent_holdout.n_solved_trees} -> "
                f"{champion_holdout.n_solved_trees})",
                paired,
            )
        if champion_holdout.off_slate_picks > 0:
            # A policy that hallucinates actions outside the recorded slate is
            # scored 0 for them; that is a contract violation, not an improvement.
            return (
                False,
                f"champion made {champion_holdout.off_slate_picks} off-slate picks "
                f"(coverage {champion_holdout.coverage:.3f})",
                paired,
            )
        return (
            True,
            f"paired holdout gain {delta:+.4f} (z={paired['z']:.2f}, "
            f"p={paired['p_value']:.3f}, {paired['n_changed']}/{paired['n']} decisions changed)",
            paired,
        )

    # -- archive ----------------------------------------------------------------------
    def archive_policy(
        self,
        source: str,
        policy_id: str,
        *,
        generation: int,
        train: Mapping[str, Any] | None,
        holdout: Mapping[str, Any] | None,
        accepted: bool,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a policy checkpoint plus its measured metrics."""
        target = self.policies_dir / policy_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "search_policy.py").write_text(source, encoding="utf-8")
        write_json(
            target / "metadata.json",
            {
                "policy_id": policy_id,
                "generation": generation,
                "accepted": accepted,
                "created_at": utc_now(),
                "evaluator_sha256": self.fingerprint,
                "train": dict(train or {}),
                "holdout": dict(holdout or {}),
                **dict(extra or {}),
            },
        )
        self._update_registry(
            policy_id, target, generation, accepted, train=train, holdout=holdout
        )
        return target

    def _update_registry(
        self,
        policy_id: str,
        path: Path,
        generation: int,
        accepted: bool,
        *,
        train: Mapping[str, Any] | None,
        holdout: Mapping[str, Any] | None,
    ) -> None:
        registry = read_json(self.registry_path, {"policies": [], "current": ""}) or {}
        entries = [e for e in registry.get("policies", []) if e.get("policy_id") != policy_id]
        entries.append(
            {
                "policy_id": policy_id,
                "path": str(path),
                "generation": generation,
                "accepted": accepted,
                "created_at": utc_now(),
                "train_mean_reward": (train or {}).get("mean_reward"),
                "holdout_mean_reward": (holdout or {}).get("mean_reward"),
            }
        )
        registry["policies"] = sorted(entries, key=lambda e: (e.get("generation", 0), e["policy_id"]))
        if accepted:
            registry["current"] = policy_id
        write_json(self.registry_path, registry)

    def ensure_incumbent_archived(self, trees: Sequence[DiscoveryTree]) -> Path:
        """Archive ``policy_v0`` (or whichever policy current is) before evolving it."""
        source = self.incumbent_source()
        policy_id = self.incumbent_policy_id(source)
        target = self.policies_dir / policy_id
        if (target / "metadata.json").exists():
            return target
        train, holdout = self.split_tasks(trees)
        incumbent_train = self.replay_incumbent(train, "train") if train else None
        incumbent_holdout = self.replay_incumbent(holdout, "holdout") if holdout else None
        return self.archive_policy(
            source,
            policy_id,
            generation=0,
            train=incumbent_train.row() if incumbent_train else None,
            holdout=incumbent_holdout.row() if incumbent_holdout else None,
            accepted=True,
            extra={"params": self.incumbent_params(source), "origin": "initial"},
        )

    # -- one generation ---------------------------------------------------------------
    def run_generation(
        self,
        trees: Sequence[DiscoveryTree],
        generation: int,
        *,
        on_candidate_round: Callable[[int, int], None] | None = None,
    ) -> GenerationRecord:
        """One full evolve/verify/promote cycle on the recorded corpus."""
        started = time.perf_counter()
        cfg = self.config
        train, holdout = self.split_tasks(trees)
        if not train or not holdout:
            raise ValueError(
                "need both training and holdout trees: "
                f"got {len(train)} train / {len(holdout)} holdout"
            )
        base_source = self.incumbent_source()
        base_params = self.incumbent_params(base_source)
        incumbent_id = self.incumbent_policy_id(base_source)
        module = self.load_incumbent()
        param_space = {k: tuple(v) for k, v in module.SearchPolicy.param_space().items()}

        # Guarantee the incumbent is in the archive before anything can replace it,
        # whichever entry point was used (this is idempotent).
        self.ensure_incumbent_archived(trees)
        incumbent_train = self.replay_incumbent(train, "train")
        incumbent_holdout = self.replay_incumbent(holdout, "holdout")
        profile = self.profiler.profile(train)
        report = self.profiler.report(profile)

        candidates: list[Candidate] = []
        llm_candidate, llm_route = self.propose_with_llm(base_source, report, profile, generation)

        # Greedy search: propose -> score on train -> adopt the round champion as
        # the base for the next round, so improvements compose within a generation.
        best_params = dict(base_params)
        best_train = incumbent_train
        best_train_delta = 0.0
        best_candidate: Candidate | None = None
        all_candidates: list[Candidate] = []
        rounds = max(1, cfg.search_rounds)
        budget_per_round = max(1, cfg.n_candidates // rounds)
        for round_index in range(rounds):
            round_candidates = self.propose_offline(
                best_params,
                base_source,
                param_space,
                profile,
                generation,
                n=budget_per_round,
            )
            if round_index == 0 and llm_candidate is not None:
                round_candidates = [llm_candidate] + round_candidates
            for extra in round_candidates:
                extra.label = f"{extra.label}_r{round_index}" if extra is not llm_candidate else extra.label
            self._write_candidates(round_candidates, generation)
            all_candidates.extend(round_candidates)
            candidates.extend(round_candidates)
            if on_candidate_round is not None:
                on_candidate_round(round_index, len(round_candidates))
            self.score_candidates(round_candidates, train, generation, tag=f"train_r{round_index}")

            for candidate in round_candidates:
                if candidate.status != "scored" or candidate.train is None or candidate.train.error:
                    continue
                candidate.note = self._pair_note(incumbent_train, candidate)
            improved = [
                c
                for c in round_candidates
                if c.status == "scored"
                and c.train is not None
                and not c.train.error
                and float(c.train.metadata.get("paired_train_delta", 0.0))
                >= cfg.min_train_improvement
                and float(c.train.metadata.get("paired_train_z", 0.0)) >= cfg.min_train_z
            ]
            if improved:
                round_best = max(
                    improved, key=lambda c: float(c.train.metadata.get("paired_train_delta", 0.0))
                )
                round_delta = float(round_best.train.metadata.get("paired_train_delta", 0.0))
                if round_delta > best_train_delta:
                    best_candidate = round_best
                    # The offline search works inside the incumbent's declared
                    # parameter space; keys a candidate invented only become
                    # searchable once that candidate is promoted.
                    best_params = {
                        k: v for k, v in round_best.params.items() if k in param_space
                    }
                    best_train = round_best.train
                    best_train_delta = round_delta

        if best_candidate is None:
            if cfg.with_baselines:
                baselines = {
                    name: result.row()
                    for name, result in Dreamer(
                        holdout, config=ReplayConfig(rng_seed=cfg.seed)
                    ).baselines().items()
                }
            else:
                baselines = {}
            record = GenerationRecord(
                generation=generation,
                incumbent_id=incumbent_id,
                champion_id=incumbent_id,
                accepted=False,
                reason="no candidate improved the training score",
                n_candidates=len(all_candidates),
                n_scored=len([c for c in all_candidates if c.status == "scored"]),
                incumbent_train=incumbent_train.row(),
                champion_train=incumbent_train.row(),
                incumbent_holdout=incumbent_holdout.row(),
                champion_holdout=incumbent_holdout.row(),
                baselines=baselines,
                profile=profile,
                llm_route=llm_route,
                wall_ms=(time.perf_counter() - started) * 1000.0,
            )
            self._archive_candidates(all_candidates, generation)
            return record

        # Verify the champion on holdout trees that never influenced the search.
        champion_holdout = self.replay_candidate(
            best_candidate, holdout, generation, tag=f"holdout_{best_candidate.label}"
        )
        accepted, reason, paired = self.gate(incumbent_holdout, champion_holdout)
        best_candidate.status = "accepted" if accepted else "rejected_holdout"
        if accepted:
            self.promote(best_candidate)
        baselines = {}
        if cfg.with_baselines:
            baselines = {
                name: result.row()
                for name, result in Dreamer(
                    holdout, config=ReplayConfig(rng_seed=cfg.seed)
                ).baselines().items()
            }
        archive_dir = self.archive_policy(
            best_candidate.source,
            best_candidate.policy_id,
            generation=generation + 1,
            train=best_candidate.train.row() if best_candidate.train else None,
            holdout=champion_holdout.row(),
            accepted=accepted,
            extra={
                "origin": best_candidate.origin,
                "params": best_candidate.params,
                "note": best_candidate.note,
                "reason": reason,
            },
        )
        self._archive_candidates(all_candidates, generation)

        return GenerationRecord(
            generation=generation,
            incumbent_id=incumbent_id,
            champion_id=best_candidate.policy_id,
            accepted=accepted,
            reason=reason,
            n_candidates=len(all_candidates),
            n_scored=len([c for c in all_candidates if c.status == "scored"]),
            incumbent_train=incumbent_train.row(),
            champion_train=best_candidate.train.row() if best_candidate.train else {},
            incumbent_holdout=incumbent_holdout.row(),
            champion_holdout=champion_holdout.row(),
            paired=paired,
            baselines=baselines,
            profile=profile,
            llm_route=llm_route,
            archive_dir=str(archive_dir),
            wall_ms=(time.perf_counter() - started) * 1000.0,
            params=best_candidate.params,
        )

    def _archive_candidates(self, candidates: Sequence[Candidate], generation: int) -> None:
        if not self.config.archive_candidates:
            return
        target = self.candidates_dir / f"gen_{generation:03d}"
        target.mkdir(parents=True, exist_ok=True)
        write_json(
            target / "candidates.json",
            {
                "generation": generation,
                "created_at": utc_now(),
                "candidates": [c.to_dict() for c in candidates],
            },
        )

    def promote(self, candidate: Candidate) -> Path:
        """Install the champion as the live policy (source + archive)."""
        backup = self.policy_path.with_suffix(".py.bak")
        shutil.copy2(self.policy_path, backup)
        self.policy_path.write_text(candidate.source, encoding="utf-8")
        return self.policy_path

    # -- full run ---------------------------------------------------------------------
    def run(
        self,
        trees: Sequence[DiscoveryTree] | None = None,
        *,
        generations: int = 3,
        on_generation: Callable[[GenerationRecord], None] | None = None,
    ) -> OptimizationReport:
        """Run ``generations`` evolve/verify cycles over the recorded corpus."""
        corpus = (
            sorted(trees, key=tree_sort_key) if trees is not None else load_trees(self.trees_dir)
        )
        if not corpus:
            raise ValueError(f"no discovery trees found in {self.trees_dir}")
        report = OptimizationReport(
            initial_policy=self.incumbent_policy_id(),
            evaluator_sha256=self.fingerprint,
            config={
                "train_ratio": self.config.train_ratio,
                "min_holdout_improvement": self.config.min_holdout_improvement,
                "min_z": self.config.min_z,
                "min_train_improvement": self.config.min_train_improvement,
                "min_train_z": self.config.min_train_z,
                "max_solve_loss_tasks": self.config.max_solve_loss_tasks,
                "min_holdout_decisions": self.config.min_holdout_decisions,
                "n_candidates": self.config.n_candidates,
                "search_rounds": self.config.search_rounds,
                "seed": self.config.seed,
                "use_llm": self.config.use_llm,
            },
        )
        self.ensure_incumbent_archived(corpus)
        for generation in range(generations):
            record = self.run_generation(corpus, generation)
            report.generations.append(record)
            if on_generation is not None:
                on_generation(record)
        report.final_policy = self.incumbent_policy_id()
        report.finished_at = utc_now()
        return report


# ======================================================================================
# Helpers
# ======================================================================================
def _result_from_dict(data: Mapping[str, Any]) -> ReplayResult:
    """Rebuild a :class:`ReplayResult` from the runner's JSON payload."""
    known = {
        "policy_id",
        "label",
        "n_trees",
        "n_decisions",
        "mean_reward",
        "pass_rate",
        "task_solve_rate",
        "n_solved_trees",
        "oracle_mean",
        "regret",
        "efficiency",
        "best_pick_rate",
        "coverage",
        "off_slate_picks",
        "mean_slate_size",
        "distinct_actions",
        "llm_calls",
        "wall_ms",
        "per_suite",
        "per_action",
        "rewards",
        "error",
        "metadata",
    }
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs["rewards"] = [float(r) for r in (data.get("rewards") or [])]
    kwargs["per_suite"] = dict(data.get("per_suite") or {})
    kwargs["per_action"] = dict(data.get("per_action") or {})
    kwargs["metadata"] = dict(data.get("metadata") or {})
    return ReplayResult(**kwargs)  # type: ignore[arg-type]


MUTATION_SYSTEM_PROMPT = (
    "You are the meta-optimizer agent of Dream-RSI. You improve an executable "
    "Python exploration policy by rewriting its source. The model weights are "
    "frozen; the policy source is the only thing that evolves. You must respect "
    "the mutation contract exactly and return one complete Python file."
)


__all__ = [
    "Candidate",
    "FailureProfiler",
    "GenerationRecord",
    "MUTATION_SYSTEM_PROMPT",
    "MetaOptimizer",
    "OfflineMutator",
    "OptimizationReport",
    "OptimizerConfig",
    "apply_params",
    "paired_comparison",
    "read_params_literal",
    "read_policy_id_literal",
]
