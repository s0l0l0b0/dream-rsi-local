#!/usr/bin/env python3
"""Dream-RSI: run the full recursive self-improvement cycle.

    python run_rsi_loop.py --mode offline --generations 3

The cycle, once per generation:

1. **Explore (real inference).** Run the current policy on the training tasks and
   on the held-out tasks, in a *fresh world* (the simulator's sampling seed
   changes every generation, as does the model's own sampling in a real run).
   Each task produces a discovery tree in ``storage/trees/``.
2. **Dream (zero inference).** The Dream Engine replays the accumulated trees
   against candidate policies -- hundreds of them, in one subprocess -- and the
   profiler turns the same trees into measured evidence about which strategies
   pay off where.
3. **Evolve.** The meta-optimizer mutates ``src/policy/search_policy.py`` (an LLM
   rewrite when a real endpoint is configured, otherwise a profiler-guided
   parameter search over the same file).
4. **Verify and promote.** A candidate must beat the incumbent on *training*
   trees by a paired, significant margin, then clear the held-out gate. Only then
   does it replace the live policy, and every attempt is archived.

Because the world is resampled each generation, the held-out evidence
accumulates: a genuine improvement is confirmed by more decisions over time,
while a lucky draw is not.  See ``README.md`` for the honest limitations of the
offline simulator and of the replay-based evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.evaluator import evaluator_fingerprint, load_tasks, task_meta  # noqa: E402
from src.core.llm_client import (  # noqa: E402
    CachingLLMClient,
    LLMClient,
    MockLLMClient,
    OpenAICompatibleClient,
    describe_client,
    make_client,
)
from src.core.tree import DiscoveryTree, load_trees  # noqa: E402
from src.core.utils import new_id, read_json, utc_now, write_json  # noqa: E402
from src.engine.dreamer import Dreamer, ReplayConfig  # noqa: E402
from src.engine.explorer import ExplorationSummary, Explorer, ExplorerConfig  # noqa: E402
from src.meta.optimizer import MetaOptimizer, OptimizationReport, OptimizerConfig  # noqa: E402
from src.policy.base import SUITES  # noqa: E402

#: Distinct prime offsets so each generation simulates a different world.
WORLD_SEED_STRIDE = 7919


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass
class RunConfig:
    """Everything the cycle needs, mirroring the CLI."""

    mode: str = "auto"
    generations: int = 3
    tasks_path: Path | None = None
    suites: tuple[str, ...] = ()
    max_tasks: int = 0
    #: Decision points per task per generation.
    steps: int = 5
    #: Stop a task as soon as an artefact passes every test.  Disabling this costs
    #: extra model calls but makes the replay corpus far richer (every task then
    #: contributes ``steps`` decision points instead of one or two), which is the
    #: exploration-cost/corpus-richness trade-off at the heart of Dream-RSI.
    stop_on_solve: bool = True
    #: Candidate actions evaluated per decision point (the branching factor).
    slate: int = 4
    #: Candidates the meta-optimizer proposes per generation.
    n_candidates: int = 160
    search_rounds: int = 3
    train_ratio: float = 0.6
    seed: int = 0
    storage_dir: Path = field(default_factory=lambda: REPO_ROOT / "storage")
    reports_dir: Path | None = None
    llm_mutation: bool = True
    with_baselines: bool = True
    quiet: bool = False
    #: Also run the diagnostics-only baselines each generation.
    clean: bool = False
    #: Restore this archived policy into src/policy/search_policy.py before running.
    from_policy: str = ""
    #: The policy file that evolves (defaults to the tracked src/policy/search_policy.py).
    policy_path: Path | None = None
    #: Seed ``policy_path`` from this file when it does not exist yet.  Used by
    #: ``--smoke`` so a sanity run never rewrites the tracked policy file.
    seed_policy_from: Path | None = None

    @property
    def resolved_policy_path(self) -> Path:
        return self.policy_path or (REPO_ROOT / "src" / "policy" / "search_policy.py")
    #: Re-explore the holdout tasks with the final policy as a verification probe.
    final_probe: bool = True
    #: Stop starting new generations after this many seconds (0 = no limit).  The
    #: report is still written, so a bounded-cost run always leaves a usable result.
    wall_budget_s: float = 0.0
    #: Tight budget for the final live A/B probe: decision points per task ...
    probe_steps: int = 1
    #: ... and options available at each of them (must be > 1 or the policy has no
    #: choice to make and the probe cannot distinguish two policies).
    probe_slate: int = 3

    @property
    def trees_dir(self) -> Path:
        return self.storage_dir / "trees"

    @property
    def reports_path(self) -> Path:
        return self.reports_dir or (self.storage_dir / "reports")

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        for key in ("tasks_path", "storage_dir", "reports_dir", "policy_path"):
            data[key] = str(data[key]) if data[key] is not None else None
        data["suites"] = list(self.suites)
        return data


# ======================================================================================
# Helpers
# ======================================================================================
def log(config: RunConfig, message: str) -> None:
    """Print unless ``--quiet``."""
    if not config.quiet:
        print(message, flush=True)


def build_client(config: RunConfig, tasks_index: Mapping[str, Mapping[str, Any]], world: int) -> LLMClient:
    """Build a client for one generation's world.

    Offline, the world seed changes per generation so that repeated exploration
    yields *independent* evidence rather than replaying identical outcomes.
    """
    cache_path = config.storage_dir / "llm_cache.json"
    namespace = f"w{world}"
    if config.mode in {"offline", "mock", "simulated"}:
        return CachingLLMClient(
            MockLLMClient(tasks_index, seed=config.seed + world * WORLD_SEED_STRIDE),
            cache_path,
            namespace=namespace,
        )
    client = make_client(
        config.mode,
        tasks=tasks_index,
        seed=config.seed + world * WORLD_SEED_STRIDE,
    )
    return CachingLLMClient(client, cache_path, namespace=namespace)


def clean_artifacts(config: RunConfig) -> list[str]:
    """Remove generated trees, candidate archives and reports (not policies)."""
    removed: list[str] = []
    targets: list[Path] = []
    if config.trees_dir.exists():
        targets.extend(sorted(config.trees_dir.glob("*.json")))
    candidates = config.storage_dir / "policies" / "candidates"
    if candidates.exists():
        for path in sorted(candidates.glob("gen_*")):
            targets.append(path)
    for name in ("dream_report.json", "dream_report.md", "run_report.json", "run_report.md"):
        candidate = config.reports_path / name
        if candidate.exists():
            targets.append(candidate)
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    return removed


def restore_policy(policy_id: str, config: RunConfig) -> Path:
    """Copy an archived checkpoint over the live policy source.

    Promotion deliberately rewrites ``src/policy/search_policy.py`` (that *is* the
    evolving artefact), so starting a run from a known checkpoint is how you make
    a run reproducible instead of inheriting whatever the last run promoted.
    """
    archive = config.storage_dir / "policies" / policy_id / "search_policy.py"
    if not archive.exists():
        # Fresh storage: the shipped source *is* policy_v0, so asking for it is a
        # no-op rather than an error.
        from src.meta.optimizer import read_policy_id_literal

        current = config.resolved_policy_path
        if current.exists() and read_policy_id_literal(current.read_text(encoding="utf-8")) == policy_id:
            return current
        available = sorted(
            path.parent.name
            for path in (config.storage_dir / "policies").glob("*/search_policy.py")
        )
        raise SystemExit(
            f"no archived policy {policy_id!r} at {archive}"
            + (f"; available: {available}" if available else "")
        )
    target = config.resolved_policy_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, target)
    return target


def explore(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    *,
    world: int,
    tag: str,
    client: LLMClient,
    policy: Any,
    persist: bool = True,
    steps: int | None = None,
    slate: int | None = None,
) -> tuple[list[DiscoveryTree], ExplorationSummary]:
    """Run one generation's exploration over ``tasks``.

    ``persist=False`` is used by the final A/B probe: those runs are measurements,
    not exploration corpus, so they must not change the reported corpus.
    """
    explorer = Explorer(
        policy,
        client,
        config=ExplorerConfig(
            max_steps=steps if steps is not None else config.steps,
            candidate_budget=slate if slate is not None else config.slate,
            stop_on_solve=config.stop_on_solve if steps is None else False,
            seed=config.seed,
            tree_dir=config.trees_dir,
            cache_path=config.storage_dir / "eval_cache.json",
            persist=persist,
        ),
    )
    trees, summary = explorer.run(tasks)
    if persist:
        for tree in trees:
            tree.metadata["world"] = world
            tree.metadata["phase"] = tag
            tree.save(config.trees_dir / f"{tree.tree_id}.json")
    return trees, summary


# ======================================================================================
# The cycle
# ======================================================================================
def run(config: RunConfig) -> dict[str, Any]:
    """Execute the full RSI cycle; return the run report as a dict."""
    started = time.perf_counter()
    run_id = new_id("run", 6)
    config.trees_dir.mkdir(parents=True, exist_ok=True)
    config.reports_path.mkdir(parents=True, exist_ok=True)

    if config.clean:
        removed = clean_artifacts(config)
        log(config, f"[clean] removed {len(removed)} generated artefact(s)")

    if config.seed_policy_from and not config.resolved_policy_path.exists():
        config.resolved_policy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.seed_policy_from, config.resolved_policy_path)
        log(config, f"[policy] seeded {config.resolved_policy_path} from {config.seed_policy_from}")

    if config.from_policy:
        restore_policy(config.from_policy, config)
        log(config, f"[policy] restored {config.from_policy} as the starting policy")

    tasks = load_tasks(config.tasks_path)
    if config.suites:
        tasks = [t for t in tasks if str(t.get("suite")) in config.suites]
    if config.max_tasks:
        tasks = tasks[: config.max_tasks]
    if not tasks:
        raise SystemExit("no tasks selected; check --suite / --tasks / --max-tasks")
    index = {str(t["id"]): t for t in tasks}

    fingerprint = evaluator_fingerprint(config.tasks_path)
    optimizer = MetaOptimizer(
        policy_path=config.resolved_policy_path,
        storage_dir=config.storage_dir,
        trees_dir=config.trees_dir,
        client=None,  # a policy-authoring client is attached per generation
        config=OptimizerConfig(
            train_ratio=config.train_ratio,
            n_candidates=config.n_candidates,
            search_rounds=config.search_rounds,
            seed=config.seed,
            with_baselines=config.with_baselines,
            use_llm=config.llm_mutation and config.mode in {"live", "auto"},
        ),
    )

    train_ids, holdout_ids = optimizer.split_task_ids(
        [str(t["id"]) for t in tasks],
        {str(t["id"]): str(t.get("suite", "")) for t in tasks},
    )
    train_tasks = [t for t in tasks if str(t["id"]) in train_ids]
    holdout_tasks = [t for t in tasks if str(t["id"]) in holdout_ids]

    log(config, "=" * 78)
    log(config, f"Dream-RSI run {run_id}   mode={config.mode}   generations={config.generations}")
    log(config, f"tasks: {len(tasks)} total -> {len(train_tasks)} train / {len(holdout_tasks)} holdout")
    log(
        config,
        f"budget: {config.steps} steps x {config.slate} candidates per task "
        f"(stop on solve: {config.stop_on_solve})",
    )
    log(config, f"evaluator fingerprint: {fingerprint[:16]}")
    log(config, "=" * 78)

    initial_policy = optimizer.incumbent_policy_id()
    log(config, f"starting policy: {initial_policy}")

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now(),
        "mode": config.mode,
        "evaluator_sha256": fingerprint,
        "config": config.to_dict(),
        "split": {"train": sorted(train_ids), "holdout": sorted(holdout_ids)},
        "initial_policy": initial_policy,
        "generations": [],
        "artifacts": {},
    }

    optimization = OptimizationReport(
        initial_policy=initial_policy,
        evaluator_sha256=fingerprint,
        config={
            "train_ratio": config.train_ratio,
            "n_candidates": config.n_candidates,
            "search_rounds": config.search_rounds,
            "mode": config.mode,
        },
    )

    def corpus_summary() -> dict[str, Any]:
        """Cheap corpus accounting (no replay), safe to compute at any point."""
        current = load_trees(config.trees_dir)
        decisions = sum(len(tree.expanded_nodes()) for tree in current)
        slates = [
            len(node.candidates or [])
            for tree in current
            for node in tree.expanded_nodes()
        ]
        return {
            "n_trees": len(current),
            "n_decisions": decisions,
            "n_duplicates_skipped": 0,
            "mean_slate_size": round(sum(slates) / len(slates), 3) if slates else 0.0,
        }

    def checkpoint_reports() -> None:
        """Persist what has been achieved so far (survives an interrupted run)."""
        report["finished_at"] = utc_now()
        report["n_accepted"] = optimization.accepted
        report["optimization"] = optimization.to_dict()
        report["corpus"] = report.get("corpus") or corpus_summary()
        report["artifacts"] = write_reports(config, report, optimization)

    for generation in range(config.generations):
        if config.wall_budget_s and (time.perf_counter() - started) > config.wall_budget_s:
            log(
                config,
                f"wall budget of {config.wall_budget_s:.0f}s exhausted after "
                f"{generation} generation(s); stopping and reporting",
            )
            report["stopped_early"] = "wall budget exhausted"
            break
        world = generation
        log(config, "")
        log(config, f"--- generation {generation} (world {world}) " + "-" * 40)
        client = build_client(config, index, world)
        world_started = time.perf_counter()

        # -- 1. explore both splits with the current policy ------------------------
        optimizer.client = client
        train_trees, train_summary = explore(
            config,
            train_tasks,
            world=world,
            tag="train",
            client=client,
            policy=optimizer.load_incumbent().SearchPolicy(rng_seed=config.seed),
        )
        holdout_trees, holdout_summary = explore(
            config,
            holdout_tasks,
            world=world,
            tag="holdout",
            client=client,
            policy=optimizer.load_incumbent().SearchPolicy(rng_seed=config.seed),
        )
        log(
            config,
            f"  live[{initial_policy if generation == 0 else optimizer.incumbent_policy_id()}]: "
            f"train solve {train_summary.solve_rate:.3f} best {train_summary.mean_best_score:.3f} "
            f"| holdout solve {holdout_summary.solve_rate:.3f} best {holdout_summary.mean_best_score:.3f} "
            f"({train_summary.llm_calls + holdout_summary.llm_calls} model calls, "
            f"{time.perf_counter() - world_started:.1f}s)",
        )

        # -- 2..4. dream, evolve, verify -------------------------------------------
        if len(holdout_tasks) < 1 or len(train_tasks) < 1:
            log(config, "  skipped evolution: need at least one task in each split")
            break
        corpus = load_trees(config.trees_dir)
        # Archive the starting policy (with real metrics) before any promotion.
        optimizer.ensure_incumbent_archived(corpus)
        record = optimizer.run_generation(corpus, generation)
        optimization.generations.append(record)
        policy_after = optimizer.incumbent_policy_id()
        log(
            config,
            f"  dream: train {record.incumbent_train.get('mean_reward', 0):.4f} -> "
            f"{record.champion_train.get('mean_reward', 0):.4f} | holdout "
            f"{record.incumbent_holdout.get('mean_reward', 0):.4f} -> "
            f"{record.champion_holdout.get('mean_reward', 0):.4f}",
        )
        log(
            config,
            f"  evolve: {record.n_candidates} candidates ({record.n_scored} scored, "
            f"llm route: {record.llm_route})",
        )
        log(
            config,
            f"  verify: {'PROMOTED' if record.accepted else 'rejected'} -> {policy_after} | {record.reason}",
        )

        report["generations"].append(
            {
                "generation": generation,
                "world": world,
                "policy_before": record.incumbent_id,
                "policy_after": policy_after,
                "live_train": train_summary.to_dict(),
                "live_holdout": holdout_summary.to_dict(),
                "trees_added": len(train_trees) + len(holdout_trees),
                "corpus_trees": len(corpus),
                "optimizer": record.to_dict(),
                "wall_s": round(time.perf_counter() - world_started, 2),
            }
        )

        # Write the report now: a hard kill (CPU limit, OOM, ctrl-C) must not throw
        # away the generations that already completed.
        checkpoint_reports()

        if evaluator_fingerprint(config.tasks_path) != fingerprint:
            raise SystemExit(
                "the benchmark evaluator/suite changed mid-run; refusing to continue "
                "(replay validity depends on a frozen oracle)"
            )

    # -- final verification probe: a live A/B at a fixed tight budget ---------------
    # Not used for any selection.  Both policies see the same world and the same
    # candidate slates, so the only thing that differs is the decision rule.
    if config.final_probe and holdout_tasks:
        from src.meta.policy_runner import load_policy_module
        from src.meta.optimizer import paired_comparison

        probe_world = config.generations
        probe_client = build_client(config, index, probe_world)
        optimizer.client = probe_client
        final_policy = optimizer.incumbent_policy_id()
        probe_kwargs = dict(
            world=probe_world,
            client=probe_client,
            persist=False,
            steps=config.probe_steps,
            slate=config.probe_slate,
        )
        _, final_summary = explore(
            config,
            holdout_tasks,
            tag="probe_final",
            policy=optimizer.load_incumbent().SearchPolicy(rng_seed=config.seed),
            **probe_kwargs,
        )
        log(config, "")
        log(
            config,
            f"final probe with {final_policy}: holdout solve {final_summary.solve_rate:.3f} "
            f"best {final_summary.mean_best_score:.3f}",
        )
        report["final_live_holdout"] = final_summary.to_dict()

        initial_archive = config.storage_dir / "policies" / initial_policy / "search_policy.py"
        if initial_archive.exists() and initial_policy != final_policy:
            initial_module = load_policy_module(initial_archive, name="dream_rsi_initial_probe")
            _, initial_summary = explore(
                config,
                holdout_tasks,
                tag="probe_initial",
                policy=initial_module.SearchPolicy(rng_seed=config.seed),
                **probe_kwargs,
            )
            report["initial_live_holdout"] = initial_summary.to_dict()
            shared = sorted(set(final_summary.per_task) & set(initial_summary.per_task))
            paired = paired_comparison(
                [initial_summary.per_task[t]["best_score"] for t in shared],
                [final_summary.per_task[t]["best_score"] for t in shared],
                seed=config.seed,
                bootstrap=2000,
            )
            report["probe_comparison"] = {
                "budget": {"steps": config.probe_steps, "slate": config.probe_slate},
                "initial_policy": initial_policy,
                "final_policy": final_policy,
                "n_tasks": len(shared),
                "initial_solve_rate": initial_summary.solve_rate,
                "final_solve_rate": final_summary.solve_rate,
                "initial_mean_best_score": initial_summary.mean_best_score,
                "final_mean_best_score": final_summary.mean_best_score,
                "paired": paired,
            }
            log(
                config,
                f"paired probe at {config.probe_steps}x{config.probe_slate}: "
                f"{initial_policy} solve {initial_summary.solve_rate:.3f} "
                f"best {initial_summary.mean_best_score:.3f} -> {final_policy} solve "
                f"{final_summary.solve_rate:.3f} best {final_summary.mean_best_score:.3f} "
                f"({paired['mean_delta']:+.4f} per task, z={paired['z']:.2f})",
            )

    # -- final reporting -----------------------------------------------------------
    optimization.final_policy = optimizer.incumbent_policy_id()
    optimization.finished_at = utc_now()
    corpus = load_trees(config.trees_dir)
    dreamer = Dreamer(corpus, config=ReplayConfig(rng_seed=config.seed))
    final_baselines = {
        name: result.row() for name, result in dreamer.baselines().items()
    }
    final_incumbent = optimizer.replay_incumbent(corpus, "final")
    report.update(
        {
            "finished_at": utc_now(),
            "final_policy": optimization.final_policy,
            "n_accepted": optimization.accepted,
            "corpus": {
                "n_trees": len(corpus),
                "n_decisions": dreamer.decision_count(),
                "n_duplicates_skipped": dreamer.duplicates_skipped,
                "mean_slate_size": round(
                    final_incumbent.mean_slate_size, 3
                ),
            },
            "final_dream": final_incumbent.to_dict(),
            "final_baselines": final_baselines,
            "optimization": optimization.to_dict(),
            "wall_s": round(time.perf_counter() - started, 2),
            "client": describe_client(optimizer.client) if optimizer.client else {},
        }
    )
    report["artifacts"] = write_reports(config, report, optimization)

    if not config.quiet:
        print_final_summary(config, report)
    return report


# ======================================================================================
# Reporting
# ======================================================================================
def write_reports(
    config: RunConfig, report: Mapping[str, Any], optimization: OptimizationReport
) -> dict[str, str]:
    """Write the JSON + Markdown reports and return their paths."""
    json_path = config.reports_path / "run_report.json"
    md_path = config.reports_path / "run_report.md"
    write_json(json_path, report)

    corpus = report.get("corpus") or {}
    md: list[str] = [
        "# Dream-RSI run report",
        "",
        f"* run id: `{report['run_id']}`",
        f"* mode: `{report['mode']}`",
        f"* started: {report['started_at']}",
        f"* finished: {report.get('finished_at', 'in progress')}",
        f"* wall clock: {report.get('wall_s', 0)}s",
        f"* policy: `{report['initial_policy']}` -> "
        f"`{report.get('final_policy', report['initial_policy'])}` "
        f"({report.get('n_accepted', 0)} promotion(s))",
        f"* evaluator fingerprint: `{report['evaluator_sha256']}`",
        f"* corpus: {corpus.get('n_trees', 0)} trees, "
        f"{corpus.get('n_decisions', 0)} replayed decision points "
        f"({corpus.get('n_duplicates_skipped', 0)} duplicate(s) collapsed)",
        "",
        f"* status: {report.get('stopped_early', 'completed all planned generations')}",
        "",
        "## Live exploration (real inference, per generation)",
        "",
        "Live metrics for generation *g* are produced by the policy in force when",
        "that generation started, so a promotion first shows up in the next row.",
        "",
        "| gen | policy | train solve | train best score | holdout solve | holdout best score | model calls |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report.get("generations", []):
        lt, lh = entry["live_train"], entry["live_holdout"]
        md.append(
            f"| {entry['generation']} | `{entry['policy_before']}` | "
            f"{lt['solve_rate']:.3f} | {lt['mean_best_score']:.3f} | "
            f"{lh['solve_rate']:.3f} | {lh['mean_best_score']:.3f} | "
            f"{lt['llm_calls'] + lh['llm_calls']} |"
        )
    if report.get("final_live_holdout"):
        fl = report["final_live_holdout"]
        md += [
            "",
            f"Final probe on the held-out tasks with `{report['final_policy']}` "
            f"(fresh world, verification only): solve rate {fl['solve_rate']:.3f}, "
            f"mean best score {fl['mean_best_score']:.3f}, {fl['llm_calls']} model calls.",
        ]
    probe = report.get("probe_comparison")
    if probe:
        md += [
            "",
            f"### Paired live A/B at a fixed tight budget "
            f"({probe['budget']['steps']} decision point(s) x "
            f"{probe['budget']['slate']} options per task)",
            "",
            "Both policies saw the same world and the same candidate slates, so the",
            "only difference is the decision rule.",
            "",
            "| policy | solve rate | mean best score |",
            "| --- | ---: | ---: |",
            f"| `{probe['initial_policy']}` (initial) | {probe['initial_solve_rate']:.3f} | "
            f"{probe['initial_mean_best_score']:.3f} |",
            f"| `{probe['final_policy']}` (final) | {probe['final_solve_rate']:.3f} | "
            f"{probe['final_mean_best_score']:.3f} |",
            "",
            f"Paired per-task difference: {probe['paired']['mean_delta']:+.4f} "
            f"(z={probe['paired']['z']:.2f}, p={probe['paired']['p_value']:.3f}, "
            f"{probe['paired']['n_changed']}/{probe['paired']['n']} tasks changed, "
            f"n={probe['n_tasks']}).",
        ]
    md += [
        "",
        "## Dream Engine (offline replay, zero inference)",
        "",
        optimization.markdown(),
        "",
        "## Final policy versus reference points (whole corpus)",
        "",
        "| policy | mean reward | pass rate | solve rate | efficiency |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    final = report.get("final_dream")
    if final:
        md.append(
            f"| `{final['policy_id']}` | {final['mean_reward']:.4f} | {final['pass_rate']:.3f} | "
            f"{final['task_solve_rate']:.3f} | {final['efficiency']:.3f} |"
        )
    for name, row in (report.get("final_baselines") or {}).items():
        md.append(
            f"| {name} | {row['mean_reward']:.4f} | {row['pass_rate']:.3f} | "
            f"{row['task_solve_rate']:.3f} | {row['efficiency']:.3f} |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"report_json": str(json_path), "report_markdown": str(md_path)}


def print_final_summary(config: RunConfig, report: Mapping[str, Any]) -> None:
    """Print the end-of-run summary."""
    print("")
    print("=" * 78)
    print(f"Dream-RSI finished: {report['initial_policy']} -> {report['final_policy']} "
          f"({report['n_accepted']} promotion(s)) in {report['wall_s']}s")
    print(f"corpus: {report['corpus']['n_trees']} trees / "
          f"{report['corpus']['n_decisions']} decision points")
    print("-" * 78)
    print(f"{'generation':>10} {'policy':>18} {'train':>9} {'holdout':>9} {'verdict':>10}")
    for entry in report.get("generations", []):
        opt = entry["optimizer"]
        verdict = "PROMOTED" if opt["accepted"] else "rejected"
        train = opt["champion_train"].get("mean_reward")
        hold = opt["champion_holdout"].get("mean_reward")
        print(
            f"{entry['generation']:>10} {entry['policy_after']:>18} "
            f"{(f'{train:.4f}' if train is not None else '—'):>9} "
            f"{(f'{hold:.4f}' if hold is not None else '—'):>9} {verdict:>10}"
        )
    print("-" * 78)
    final = report.get("final_dream")
    if final:
        print(f"final policy dream score: {final['mean_reward']:.4f} "
              f"(efficiency {final['efficiency']:.3f}, coverage {final['coverage']:.3f})")
    for name, row in (report.get("final_baselines") or {}).items():
        print(f"  {name:<16} {row['mean_reward']:.4f}")
    probe = report.get("probe_comparison")
    if probe:
        print("-" * 78)
        print(f"paired live A/B at {probe['budget']['steps']}x{probe['budget']['slate']}: "
              f"{probe['initial_policy']} solve {probe['initial_solve_rate']:.3f} "
              f"-> {probe['final_policy']} solve {probe['final_solve_rate']:.3f} "
              f"(paired {probe['paired']['mean_delta']:+.4f}, z={probe['paired']['z']:.2f})")
    print(f"report: {report['artifacts']['report_markdown']}")
    print("=" * 78)


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_rsi_loop.py",
        description="Run the Dream-RSI recursive self-improvement cycle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "offline", "live"],
        help="auto probes the configured endpoint and falls back to the offline simulator",
    )
    parser.add_argument("--generations", type=int, default=3, help="RSI generations to run")
    parser.add_argument("--tasks", type=Path, default=None, help="task suite JSONL path")
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=list(SUITES),
        help="restrict to a suite (repeatable)",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="use only the first N tasks")
    parser.add_argument("--steps", type=int, default=5, help="decision points per task")
    parser.add_argument(
        "--stop-on-solve",
        dest="stop_on_solve",
        action="store_true",
        default=True,
        help="stop a task as soon as it fully passes (default)",
    )
    parser.add_argument(
        "--no-stop-on-solve",
        dest="stop_on_solve",
        action="store_false",
        help="keep exploring after a task passes: more model calls, much richer corpus",
    )
    parser.add_argument("--slate", type=int, default=4, help="candidate actions per decision point")
    parser.add_argument("--candidates", type=int, default=160, help="policy candidates per generation")
    parser.add_argument("--search-rounds", type=int, default=3, help="greedy search rounds per generation")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="fraction of tasks used to fit")
    parser.add_argument("--seed", type=int, default=0, help="master seed (keeps runs reproducible)")
    parser.add_argument("--storage", type=Path, default=REPO_ROOT / "storage", help="storage directory")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="policy file to evolve (default: src/policy/search_policy.py)",
    )
    parser.add_argument("--reports", type=Path, default=None, help="report output directory")
    parser.add_argument(
        "--no-llm-mutation",
        dest="llm_mutation",
        action="store_false",
        help="never let an LLM rewrite the policy (parameter search only)",
    )
    parser.add_argument(
        "--no-baselines",
        dest="with_baselines",
        action="store_false",
        help="skip the diagnostic baselines (faster)",
    )
    parser.add_argument(
        "--from-policy",
        default="",
        help="start from an archived checkpoint (e.g. policy_v0), restoring it into "
        "src/policy/search_policy.py first; makes runs reproducible",
    )
    parser.add_argument(
        "--probe-steps", type=int, default=1, help="decision points per task in the final A/B probe"
    )
    parser.add_argument(
        "--probe-slate", type=int, default=3, help="options per decision point in the final A/B probe"
    )
    parser.add_argument(
        "--no-final-probe",
        dest="final_probe",
        action="store_false",
        help="skip the final live A/B probe",
    )
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=0.0,
        help="stop starting new generations after N seconds (0 = unlimited)",
    )
    parser.add_argument("--clean", action="store_true", help="delete previously generated trees/reports first")
    parser.add_argument("--quiet", action="store_true", help="only write reports, print nothing")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast configuration for CI (2 generations, 6 tasks, small budget)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build a :class:`RunConfig` from parsed arguments.

    Every field is set once, from one dictionary, with ``--smoke`` applied as an
    overlay.  Duplicating the construction per branch previously dropped
    ``policy_path``/``wall_budget``/probe settings in the normal branch, which
    silently made runs evolve the tracked policy file instead of the one the user
    asked for.
    """
    common: dict[str, Any] = {
        "tasks_path": args.tasks,
        "suites": tuple(args.suite),
        "max_tasks": args.max_tasks,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "storage_dir": args.storage,
        "reports_dir": args.reports,
        "with_baselines": args.with_baselines,
        "quiet": args.quiet,
        "clean": args.clean,
        "from_policy": args.from_policy,
        "policy_path": args.policy_path,
        "wall_budget_s": args.wall_budget,
        "probe_steps": args.probe_steps,
        "probe_slate": args.probe_slate,
        "final_probe": args.final_probe,
    }
    if args.smoke:
        # Never touch the tracked policy file during a sanity run: evolve a copy
        # inside the storage directory instead.
        common["policy_path"] = args.policy_path or (args.storage / "smoke_policy.py")
        common["seed_policy_from"] = args.policy_path or (
            REPO_ROOT / "src" / "policy" / "search_policy.py"
        )
        common.update(
            mode="offline",
            generations=2,
            max_tasks=args.max_tasks or 6,
            steps=2,
            slate=3,
            n_candidates=24,
            search_rounds=1,
            train_ratio=0.5,
            llm_mutation=False,
            with_baselines=True,
            wall_budget_s=0.0,
        )
    else:
        common.update(
            mode=args.mode,
            generations=args.generations,
            steps=args.steps,
            stop_on_solve=args.stop_on_solve,
            slate=args.slate,
            n_candidates=args.candidates,
            search_rounds=args.search_rounds,
            llm_mutation=args.llm_mutation,
        )
    return RunConfig(**common)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    report = run(config)
    # Nonzero exit when nothing could be learned at all, so CI notices.
    return 0 if report["corpus"]["n_decisions"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
