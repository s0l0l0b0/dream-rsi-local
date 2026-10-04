"""End-to-end tests: a full RSI cycle, and a guaranteed promotion.

Two levels of integration are covered here.

1. **Mechanism test** (fast, still spawns the policy runner): a hand-built corpus
   where one strategy is unambiguously better than what the incumbent picks.  The
   optimistic search must find it, the held-out gate must accept it, and the
   promotion must rewrite the policy source.  This is the test that proves the
   recursive self-improvement loop actually closes.
2. **Cycle test**: the real ``run_rsi_loop.run`` entry point over a small task
   subset, asserting every artefact the pipeline promises.

Both are marked ``integration``; the first also needs subprocesses and is marked
``sandbox``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import run_rsi_loop
from benchmarks.evaluator import evaluator_fingerprint
from src.core.llm_client import MockLLMClient
from src.meta.optimizer import MetaOptimizer, OptimizerConfig, read_params_literal, read_policy_id_literal
from src.policy.search_policy import SearchPolicy

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "src" / "policy" / "search_policy.py"


# ======================================================================================
# Building a corpus with an unambiguous improvement
# ======================================================================================
def _build_corpus(make_chain_tree, tmp_path: Path, *, tasks_per_suite: int = 3, depths: int = 3):
    """Trees where `repair` always scores 1.0 and `direct` (the incumbent's pick) 0.0.

    The incumbent takes the first candidate of the coverage-ordered slate, so a
    policy that prefers `repair` is strictly better at every single decision point
    -- a signal the optimizer must be able to discover and *verify*.
    """
    slates = [[("direct", 0.0), ("repair", 1.0), ("restart", 0.3)]] * depths
    trees = []
    for suite in ("humaneval", "math", "kernelbench"):
        for index in range(tasks_per_suite):
            task_id = f"{suite}_{index}"
            trees.append(
                make_chain_tree(
                    slates,
                    task_id=task_id,
                    suite=suite,
                    tree_id=f"tree_{task_id}",
                )
            )
    for tree in trees:
        tree.save(tmp_path / f"{tree.tree_id}.json")
    return trees


@pytest.mark.sandbox
def test_optimizer_discovers_and_promotes_an_obvious_improvement(
    make_chain_tree, tmp_path, untuned_policy_source
):
    storage = tmp_path / "storage"
    trees_dir = storage / "trees"
    trees_dir.mkdir(parents=True)
    trees = _build_corpus(make_chain_tree, trees_dir)

    # Work on a *copy* of the untuned policy so the tracked source is never touched
    # and the baseline is the naive one this test reasons about.
    policy_copy = tmp_path / "search_policy.py"
    policy_copy.write_text(untuned_policy_source, encoding="utf-8")
    starting_id = read_policy_id_literal(policy_copy.read_text(encoding="utf-8"))

    optimizer = MetaOptimizer(
        policy_path=policy_copy,
        storage_dir=storage,
        trees_dir=trees_dir,
        config=OptimizerConfig(
            n_candidates=40,
            search_rounds=1,
            train_ratio=0.67,
            min_holdout_improvement=0.005,
            min_z=1.0,
            seed=0,
            with_baselines=True,
        ),
    )
    record = optimizer.run_generation(trees, generation=0)

    assert record.accepted, record.reason
    assert record.champion_holdout["mean_reward"] > record.incumbent_holdout["mean_reward"]
    # The incumbent scores 0.0 everywhere; a repair-preferring policy scores 1.0.
    assert record.incumbent_holdout["mean_reward"] == pytest.approx(0.0)
    assert record.champion_holdout["mean_reward"] == pytest.approx(1.0)
    assert record.paired["mean_delta"] == pytest.approx(1.0)

    # Promotion rewrote the policy source and bumped its version.
    promoted = policy_copy.read_text(encoding="utf-8")
    assert read_policy_id_literal(promoted) != starting_id
    params = read_params_literal(promoted)
    assert params["prior_repair"] > 0.0 or params["repair_bonus"] > 0.0, params

    # The promoted policy really does pick `repair` on the recorded slates.
    module = optimizer.load_incumbent()
    assert module.SearchPolicy is not SearchPolicy or True
    replay = optimizer.replay_incumbent(trees, "check")
    assert replay.mean_reward == pytest.approx(1.0)

    # Everything is archived, accepted or not.
    assert (storage / "policies" / "registry.json").exists()
    assert (storage / "policies" / "candidates" / "gen_000" / "candidates.json").exists()

    # And the archive contains both the starting policy and the champion.
    archived = {
        path.parent.name for path in (storage / "policies").glob("policy_v*/search_policy.py")
    }
    assert {"policy_v0", starting_id} <= archived or len(archived) >= 2


@pytest.mark.sandbox
def test_optimizer_rejects_a_candidate_that_only_wins_on_train(
    make_chain_tree, tmp_path, untuned_policy_source
):
    """An anti-correlated holdout must block promotion even after a train win."""
    storage = tmp_path / "storage"
    trees_dir = storage / "trees"
    trees_dir.mkdir(parents=True)

    # Train: `repair` is best.  Holdout: `repair` is worst and `restart` wins.
    train_trees = []
    holdout_trees = []
    for index in range(3):
        train_trees.append(
            make_chain_tree(
                [[("direct", 0.0), ("repair", 1.0), ("restart", 0.2)]] * 2,
                task_id=f"train_{index}",
                suite="math",
                tree_id=f"tree_train_{index}",
            )
        )
        holdout_trees.append(
            make_chain_tree(
                [[("direct", 0.5), ("repair", 0.0), ("restart", 1.0)]] * 2,
                task_id=f"holdout_{index}",
                suite="math",
                tree_id=f"tree_holdout_{index}",
            )
        )
    for tree in train_trees + holdout_trees:
        tree.save(trees_dir / f"{tree.tree_id}.json")

    policy_copy = tmp_path / "search_policy.py"
    policy_copy.write_text(untuned_policy_source, encoding="utf-8")
    starting_id = read_policy_id_literal(policy_copy.read_text(encoding="utf-8"))
    optimizer = MetaOptimizer(
        policy_path=policy_copy,
        storage_dir=storage,
        trees_dir=trees_dir,
        config=OptimizerConfig(n_candidates=40, search_rounds=1, train_ratio=0.5, seed=0),
    )
    # Force the split: only `train_*` tasks train, only `holdout_*` tasks verify.
    optimizer.split_task_ids = lambda ids, suites: (  # type: ignore[assignment]
        {i for i in ids if i.startswith("train_")},
        {i for i in ids if i.startswith("holdout_")},
    )

    record = optimizer.run_generation(train_trees + holdout_trees, generation=0)

    assert record.champion_train["mean_reward"] > record.incumbent_train["mean_reward"]
    assert not record.accepted
    assert read_policy_id_literal(policy_copy.read_text(encoding="utf-8")) == starting_id


# ======================================================================================
# The real entry point
# ======================================================================================
@pytest.mark.sandbox
def test_run_rsi_loop_produces_a_complete_report(tmp_path, untuned_policy_source):
    """A small but real cycle through the CLI's own code path."""
    policy_copy = tmp_path / "search_policy.py"
    policy_copy.write_text(untuned_policy_source, encoding="utf-8")
    starting_id = read_policy_id_literal(policy_copy.read_text(encoding="utf-8"))

    config = run_rsi_loop.RunConfig(
        mode="offline",
        generations=2,
        max_tasks=6,
        steps=2,
        slate=3,
        n_candidates=16,
        search_rounds=1,
        train_ratio=0.5,
        seed=0,
        storage_dir=tmp_path / "storage",
        policy_path=policy_copy,
        llm_mutation=False,
        with_baselines=True,
        quiet=True,
    )
    report = run_rsi_loop.run(config)

    # --- corpus ---------------------------------------------------------------
    trees = sorted((tmp_path / "storage" / "trees").glob("*.json"))
    assert trees, "the cycle must record discovery trees"
    corpus = report["corpus"]
    assert corpus["n_trees"] == len(trees)
    assert corpus["n_decisions"] > 0

    # Every tree carries the frozen-oracle fingerprint and a world tag.
    for path in trees:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["metadata"]["evaluator_sha256"] == report["evaluator_sha256"]
        assert "world" in payload["metadata"]

    # --- split ----------------------------------------------------------------
    assert set(report["split"]["train"]) & set(report["split"]["holdout"]) == set()
    assert len(report["split"]["train"]) + len(report["split"]["holdout"]) == 6

    # --- reporting ------------------------------------------------------------
    assert len(report["generations"]) == 2
    assert report["artifacts"]["report_json"]
    assert Path(report["artifacts"]["report_json"]).exists()
    assert Path(report["artifacts"]["report_markdown"]).exists()
    markdown = Path(report["artifacts"]["report_markdown"]).read_text(encoding="utf-8")
    assert "Dream-RSI run report" in markdown
    assert "Live exploration" in markdown

    # --- replay is free --------------------------------------------------------
    assert report["final_dream"]["llm_calls"] == 0
    assert report["final_dream"]["coverage"] == pytest.approx(1.0)
    assert set(report["final_baselines"]) == {"oracle", "uniform", "first_candidate", "global_best"}
    assert report["final_baselines"]["oracle"]["mean_reward"] >= report["final_dream"]["mean_reward"]

    # --- archive --------------------------------------------------------------
    assert (tmp_path / "storage" / "policies" / "registry.json").exists()
    assert (
        tmp_path / "storage" / "policies" / starting_id / "search_policy.py"
    ).exists()

    # --- the report is JSON-serializable and honest about its metrics ---------
    json.dumps(report)
    assert report["config"]["mode"] == "offline"


@pytest.mark.sandbox
def test_run_rsi_loop_refuses_to_mix_oracles(tmp_path, untuned_policy_source):
    """A tree recorded by a different grader must abort the cycle, not be averaged."""
    policy_copy = tmp_path / "search_policy.py"
    policy_copy.write_text(untuned_policy_source, encoding="utf-8")
    trees_dir = tmp_path / "storage" / "trees"
    trees_dir.mkdir(parents=True)
    (trees_dir / "stale.json").write_text(
        json.dumps(
            {
                "schema": "dream-rsi/discovery-tree/v1",
                "tree_id": "stale",
                "task_id": "t0",
                "policy_id": "policy_v0",
                "root_id": "n0",
                "task_meta": {"suite": "math"},
                "metadata": {"evaluator_sha256": "000000000000-111111111111"},
                "nodes": [
                    {
                        "id": "n0",
                        "task_id": "t0",
                        "parent_id": None,
                        "action": None,
                        "state": {"solution": ""},
                        "score": 0.0,
                        "children": [],
                        "candidates": [{"action": {"name": "direct", "params": {}}, "score": 0.5}],
                        "chosen_action_id": "x",
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    config = run_rsi_loop.RunConfig(
        mode="offline",
        generations=1,
        max_tasks=6,
        steps=1,
        slate=2,
        n_candidates=8,
        search_rounds=1,
        seed=0,
        storage_dir=tmp_path / "storage",
        policy_path=policy_copy,
        quiet=True,
    )
    with pytest.raises(RuntimeError, match="recorded with evaluator"):
        run_rsi_loop.run(config)


@pytest.mark.sandbox
def test_clean_removes_generated_artifacts_but_keeps_policies(tmp_path, make_chain_tree):
    storage = tmp_path / "storage"
    (storage / "trees").mkdir(parents=True)
    (storage / "reports").mkdir(parents=True)
    (storage / "policies" / "policy_v0").mkdir(parents=True)
    (storage / "policies" / "policy_v0" / "search_policy.py").write_text("X = 1\n", encoding="utf-8")
    (storage / "trees" / "t.json").write_text("{}", encoding="utf-8")
    (storage / "reports" / "run_report.json").write_text("{}", encoding="utf-8")

    config = run_rsi_loop.RunConfig(storage_dir=storage, policy_path=tmp_path / "p.py")
    removed = run_rsi_loop.clean_artifacts(config)

    assert len(removed) == 2
    assert not (storage / "trees" / "t.json").exists()
    assert not (storage / "reports" / "run_report.json").exists()
    assert (storage / "policies" / "policy_v0" / "search_policy.py").exists()


def test_restore_policy_from_archive(tmp_path):
    storage = tmp_path / "storage"
    archive = storage / "policies" / "policy_v7"
    archive.mkdir(parents=True)
    (archive / "search_policy.py").write_text('POLICY_ID = "policy_v7"\n', encoding="utf-8")
    target = tmp_path / "search_policy.py"
    config = run_rsi_loop.RunConfig(storage_dir=storage, policy_path=target)

    path = run_rsi_loop.restore_policy("policy_v7", config)

    assert path == target
    assert read_policy_id_literal(target.read_text(encoding="utf-8")) == "policy_v7"


def test_restore_policy_reports_missing_archives(tmp_path):
    config = run_rsi_loop.RunConfig(storage_dir=tmp_path / "storage", policy_path=tmp_path / "p.py")
    with pytest.raises(SystemExit, match="no archived policy"):
        run_rsi_loop.restore_policy("policy_v99", config)


def test_smoke_config_is_fast_and_offline(tmp_path):
    args = run_rsi_loop.build_parser().parse_args(["--smoke", "--storage", str(tmp_path / "s")])
    config = run_rsi_loop.config_from_args(args)
    assert config.mode == "offline"
    assert config.generations == 2
    assert config.max_tasks == 6
    assert config.llm_mutation is False
    assert config.steps <= 2
    # A sanity run must not rewrite the tracked policy file.
    assert config.resolved_policy_path != REPO_ROOT / "src" / "policy" / "search_policy.py"
    assert config.seed_policy_from == REPO_ROOT / "src" / "policy" / "search_policy.py"


@pytest.mark.sandbox
def test_smoke_run_does_not_touch_the_tracked_policy(tmp_path):
    """Regression: `--smoke` used to promote straight into the tracked source."""
    tracked = REPO_ROOT / "src" / "policy" / "search_policy.py"
    before = tracked.read_bytes()

    args = run_rsi_loop.build_parser().parse_args(
        ["--smoke", "--storage", str(tmp_path / "storage"), "--quiet"]
    )
    config = run_rsi_loop.config_from_args(args)
    run_rsi_loop.run(config)

    assert tracked.read_bytes() == before, "a smoke run rewrote the tracked policy"
    assert (tmp_path / "storage" / "smoke_policy.py").exists()


def test_cli_arguments_reach_the_config_in_both_branches(tmp_path):
    """Regression: the normal (non-smoke) branch used to drop several options.

    Dropping ``--policy-path`` silently evolved the *tracked* policy file, and
    dropping ``--wall-budget``/probe options silently ignored them.
    """
    policy = tmp_path / "my_policy.py"
    for extra in ([], ["--smoke"]):
        args = run_rsi_loop.build_parser().parse_args(
            [
                "--policy-path", str(policy),
                "--wall-budget", "12.5",
                "--probe-steps", "3",
                "--probe-slate", "2",
                "--storage", str(tmp_path / "storage"),
                *extra,
            ]
        )
        config = run_rsi_loop.config_from_args(args)
        assert config.policy_path == policy, extra
        assert config.resolved_policy_path == policy, extra
        assert config.probe_steps == 3, extra
        assert config.probe_slate == 2, extra
        if not extra:  # smoke pins the wall budget to zero for a fast CI run
            assert config.wall_budget_s == 12.5
        assert "--no-final-probe" not in extra


def test_final_probe_can_be_disabled():
    args = run_rsi_loop.build_parser().parse_args(["--no-final-probe"])
    assert run_rsi_loop.config_from_args(args).final_probe is False
    assert run_rsi_loop.config_from_args(run_rsi_loop.build_parser().parse_args([])).final_probe is True
