"""Tests for the live explorer: tree growth, slate recording and cost accounting.

The explorer is the only component that spends inference, so these tests check
both the scientific invariants (every attempted candidate is measured and
recorded, each decision point is expanded once) and the operational ones (eval
caching, persistence, deterministic re-runs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.evaluator import evaluator_fingerprint
from src.core.llm_client import GenerationRequest, GenerationResult, LLMClient, MockLLMClient
from src.engine.explorer import Explorer, ExplorerConfig, task_features
from src.policy.base import ExplorationPolicy, PolicyContext
from src.policy.search_policy import SearchPolicy


def _config(tmp_path, **overrides) -> ExplorerConfig:
    defaults = dict(
        max_steps=2,
        candidate_budget=3,
        seed=0,
        tree_dir=tmp_path,
        cache_path=tmp_path / "eval_cache.json",
        persist=True,
    )
    defaults.update(overrides)
    return ExplorerConfig(**defaults)


# ======================================================================================
# Task features
# ======================================================================================
def test_task_features_are_numeric_and_one_hot_the_suite(tasks):
    for task in tasks:
        features = task_features(task)
        assert all(isinstance(v, float) for v in features.values())
        assert sum(features[f"suite_{name}"] for name in ("humaneval", "math", "kernelbench")) == 1.0
        assert 0.0 <= features["difficulty"] <= 1.0


# ======================================================================================
# Tree growth
# ======================================================================================
def test_run_task_builds_a_tree_with_recorded_slates(mock_client, small_tasks, tmp_path):
    task = small_tasks[0]
    explorer = Explorer(SearchPolicy(), mock_client, config=_config(tmp_path))
    tree = explorer.run_task(task)

    assert tree.task_id == task["id"]
    assert tree.metadata["evaluator_sha256"] == evaluator_fingerprint()
    assert tree.metadata["stats"]["n_expanded"] >= 1
    assert (tmp_path / f"{tree.tree_id}.json").exists()

    # Every evaluated candidate becomes a node; the branching factor is the slate size.
    for node in tree.expanded_nodes():
        slate = node.candidates or []
        assert 1 <= len(slate) <= 3
        assert len(node.children) == len(slate)

    # Each decision point is expanded at most once, so slates are never overwritten.
    expanded_ids = [n.id for n in tree.expanded_nodes()]
    assert len(expanded_ids) == len(set(expanded_ids))

    # Every child records the artefact it produced and the measurement of it.
    for node in tree.nodes.values():
        if node.action is None:
            continue
        assert isinstance(node.state.get("solution"), str)
        assert node.metadata["tests_total"] >= 0
        assert node.metadata["selected"] in (True, False)
        assert 0.0 <= node.score <= 1.0


def test_selected_flag_marks_exactly_the_policy_pick(mock_client, small_tasks, tmp_path):
    task = small_tasks[1]
    explorer = Explorer(SearchPolicy(), mock_client, config=_config(tmp_path))
    tree = explorer.run_task(task)
    for node in tree.expanded_nodes():
        selected = [child for child in node.children if tree.nodes[child].metadata["selected"]]
        assert len(selected) == 1
        assert tree.nodes[selected[0]].action.signature() == node.chosen_action_id


def test_run_reports_operational_summary(mock_client, small_tasks, tmp_path):
    explorer = Explorer(SearchPolicy(), mock_client, config=_config(tmp_path))
    trees, summary = explorer.run(small_tasks)

    assert summary.n_trees == len(small_tasks)
    assert summary.n_tasks == len(small_tasks)
    assert summary.llm_calls == mock_client.calls > 0
    assert summary.total_tokens > 0
    assert 0.0 <= summary.solve_rate <= 1.0
    assert set(summary.per_suite) <= {"humaneval", "math", "kernelbench"}
    assert len(summary.tree_paths) == len(small_tasks)
    assert all(Path(p).exists() for p in summary.tree_paths)
    assert summary.n_decisions >= len(small_tasks)
    assert sum(entry["n"] for entry in summary.per_suite.values()) == len(small_tasks)


def test_more_steps_produce_more_decisions(mock_client, small_tasks, tmp_path):
    task = small_tasks[0]
    small = Explorer(
        SearchPolicy(),
        MockLLMClient({t["id"]: t for t in small_tasks}, seed=1),
        config=_config(tmp_path / "a", max_steps=1),
    ).run_task(task)
    large = Explorer(
        SearchPolicy(),
        MockLLMClient({t["id"]: t for t in small_tasks}, seed=1),
        config=_config(tmp_path / "b", max_steps=3, stop_on_solve=False),
    ).run_task(task)

    assert len(large.expanded_nodes()) >= len(small.expanded_nodes())
    assert len(large) > len(small)


def test_stop_on_solve_disables_the_policy_stop(mock_client, small_tasks, tmp_path):
    task = next(t for t in small_tasks if t.get("suite") == "humaneval")
    explorer = Explorer(
        SearchPolicy(),
        mock_client,
        config=_config(tmp_path, max_steps=3, stop_on_solve=False),
    )
    tree = explorer.run_task(task)
    # Even if an artefact passes, the budget is spent in full.
    assert len(tree.expanded_nodes()) == 3 or tree.root.expanded is False


# ======================================================================================
# Evaluation caching
# ======================================================================================
def test_identical_artefacts_are_graded_once(mock_client, small_tasks, tmp_path):
    """Grading is deterministic, so repeated code must not re-spawn the sandbox."""
    task = small_tasks[0]
    explorer = Explorer(SearchPolicy(), mock_client, config=_config(tmp_path, max_steps=2))
    explorer.run_task(task)
    first_pass_calls = explorer._cache_hits

    # Re-run in the same process: every evaluation should now be a cache hit.
    explorer2 = Explorer(
        SearchPolicy(),
        MockLLMClient({t["id"]: t for t in small_tasks}, seed=0),
        config=_config(tmp_path, max_steps=2),
    )
    assert explorer2._eval_cache, "the cache must be reloaded from disk"
    assert explorer2._saved_entries > 0
    explorer2.run_task(task)
    assert explorer2._cache_hits > 0
    assert first_pass_calls >= 0


def test_cache_is_invalidated_when_the_fingerprint_changes(small_tasks, tmp_path):
    cache = tmp_path / "eval_cache.json"
    cache.write_text('{"evaluator_sha256": "stale", "entries": {"x": {}}}', encoding="utf-8")
    explorer = Explorer(SearchPolicy(), MockLLMClient({}, seed=0), config=_config(tmp_path, cache_path=cache))
    assert explorer._eval_cache == {}


# ======================================================================================
# Determinism and pluggability
# ======================================================================================
def test_same_seed_and_client_produce_the_same_tree(tmp_path, small_tasks):
    task = small_tasks[0]
    client_a = MockLLMClient({t["id"]: t for t in small_tasks}, seed=5)
    client_b = MockLLMClient({t["id"]: t for t in small_tasks}, seed=5)
    tree_a = Explorer(SearchPolicy(), client_a, config=_config(tmp_path / "a")).run_task(task)
    tree_b = Explorer(SearchPolicy(), client_b, config=_config(tmp_path / "b")).run_task(task)

    scores_a = [(n.depth, n.score, n.action.name if n.action else None) for n in tree_a.nodes.values()]
    scores_b = [(n.depth, n.score, n.action.name if n.action else None) for n in tree_b.nodes.values()]
    assert scores_a == scores_b


def test_a_broken_model_response_is_recorded_as_a_failure(small_tasks, tmp_path):
    class BrokenClient(LLMClient):
        name = "broken"

        def generate(self, request: GenerationRequest) -> GenerationResult:
            self.calls += 1
            return GenerationResult(error="endpoint unreachable")

    explorer = Explorer(SearchPolicy(), BrokenClient(), config=_config(tmp_path))
    tree = explorer.run_task(small_tasks[0])

    assert all(node.score == 0.0 for node in tree.nodes.values() if node.action is not None)
    assert tree.best_node().score == 0.0
    for node in tree.expanded_nodes():
        assert all("endpoint unreachable" in c.error for c in node.candidates or [])


def test_off_slate_pick_falls_back_to_a_recorded_outcome(small_tasks, tmp_path):
    class ConfusedPolicy(ExplorationPolicy):
        POLICY_ID = "confused"

        def select(self, ctx: PolicyContext) -> Action:  # noqa: F821
            return Action(name="not_in_the_slate")  # noqa: F821

    from src.core.tree import Action

    explorer = Explorer(ConfusedPolicy(), MockLLMClient({t["id"]: t for t in small_tasks}, seed=2),
                        config=_config(tmp_path, max_steps=1))
    tree = explorer.run_task(small_tasks[0])
    # The tree must stay consistent: the recorded chosen action is one of the slate.
    for node in tree.expanded_nodes():
        assert node.candidate_by_signature(node.chosen_action_id) is not None


def test_persist_false_writes_nothing(mock_client, small_tasks, tmp_path):
    explorer = Explorer(SearchPolicy(), mock_client, config=_config(tmp_path, persist=False))
    tree = explorer.run_task(small_tasks[0])
    assert not (tmp_path / f"{tree.tree_id}.json").exists()


def test_client_cache_is_flushed_after_every_task(small_tasks, tmp_path):
    """Model calls are the expensive resource; an interrupted run must keep them."""
    from src.core.llm_client import GenerationResult, LLMClient

    class FlushCountingClient(LLMClient):
        name = "flush-probe"

        def __init__(self):
            super().__init__()
            self.flushes = 0

        def generate(self, request):
            self.calls += 1
            return GenerationResult(text="```python\ndef f():\n    return 1\n```",
                                    code="def f():\n    return 1\n")

        def save_cache(self):
            self.flushes += 1

    client = FlushCountingClient()
    explorer = Explorer(SearchPolicy(), client, config=_config(tmp_path, max_steps=1))
    explorer.run_task(small_tasks[0])
    explorer.run_task(small_tasks[1])

    assert client.calls > 0
    assert client.flushes == 2, "the client cache must be persisted once per task"

