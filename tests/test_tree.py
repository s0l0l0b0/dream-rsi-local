"""Tests for the discovery-tree data structures and their replay semantics.

These are the invariants the whole system rests on: the Dream Engine can only be
trusted if decision records are faithful, leak-free and stable across
serialization.
"""

from __future__ import annotations

import json

import pytest

from src.core.tree import (
    Action,
    CandidateOutcome,
    DiscoveryTree,
    Edge,
    Node,
    decisions_from_trees,
    load_trees,
    state_features,
)


# ======================================================================================
# Actions
# ======================================================================================
def test_action_signature_is_order_independent_and_param_sensitive():
    a = Action(name="repair", params={"temperature": 0.1, "focus": "edge"})
    b = Action(name="repair", params={"focus": "edge", "temperature": 0.1})
    c = Action(name="repair", params={"temperature": 0.2, "focus": "edge"})
    d = Action(name="restart", params={"temperature": 0.1, "focus": "edge"})

    assert a.signature() == b.signature()
    assert a.signature() != c.signature()
    assert a.signature() != d.signature()
    assert a.describe() == "repair(focus=edge,temperature=0.1)"
    assert Action(name="direct").describe() == "direct"


def test_action_roundtrip():
    action = Action(name="decompose", params={"k": 3}, rationale="split it")
    restored = Action.from_dict(action.to_dict())
    assert restored == action
    assert restored.signature() == action.signature()


# ======================================================================================
# Nodes and edges
# ======================================================================================
def test_node_roundtrip():
    node = Node(
        id="n1",
        task_id="t1",
        parent_id="n0",
        action=Action(name="direct", params={"temperature": 0.2}),
        state={"solution": "def f(): return 1"},
        score=0.75,
        feedback="1/4 failed",
        depth=2,
        children=["n2"],
        candidates=[CandidateOutcome(action=Action(name="direct"), score=0.75)],
        chosen_action_id="abc",
        policy_id="policy_v0",
        metadata={"passed": False},
    )
    restored = Node.from_dict(node.to_dict())
    assert restored.id == node.id
    assert restored.action == node.action
    assert restored.score == node.score
    assert restored.candidates[0].score == 0.75
    assert restored.metadata == {"passed": False}
    assert restored.expanded is True


def test_edge_roundtrip():
    edge = Edge(
        parent_id="a", child_id="b", action=Action(name="restart"), reward=0.5, delta=0.25
    )
    assert Edge.from_dict(edge.to_dict()).to_dict() == edge.to_dict()


# ======================================================================================
# Tree construction
# ======================================================================================
def test_create_and_add_node_links_children_and_edges():
    tree = DiscoveryTree.create("task_x", policy_id="p", task_meta={"suite": "math"})
    assert tree.root.parent_id is None
    assert tree.root.depth == 0
    assert len(tree) == 1

    child = tree.add_node(
        tree.root.id, Action(name="direct"), {"solution": "x"}, 0.5, feedback="half"
    )
    assert child.parent_id == tree.root.id
    assert child.depth == 1
    assert tree.root.children == [child.id]
    assert len(tree.edges) == 1
    assert tree.edges[0].delta == pytest.approx(0.5)
    assert tree.edges[0].action.name == "direct"


def test_add_node_rejects_unknown_parent():
    tree = DiscoveryTree.create("task_x")
    with pytest.raises(KeyError):
        tree.add_node("missing", Action(name="direct"), {}, 0.0)


def test_path_to_root_is_ordered_and_best_node_prefers_shallow_ties(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.5)], [("repair", 0.5)], [("restart", 0.5)]])
    leaf = max(tree.nodes.values(), key=lambda n: n.depth)
    path = tree.path_to_root(leaf.id)
    assert [n.depth for n in path] == [0, 1, 2, 3]
    assert path[0].id == tree.root.id

    # The three children all score 0.5; the tie is broken toward the shallowest
    # of the tied nodes (depth 1), not toward the deeper ones.
    assert tree.best_node().score == pytest.approx(0.5)
    assert tree.best_node().depth == 1


def test_path_to_root_survives_a_cycle():
    tree = DiscoveryTree.create("task_x")
    child = tree.add_node(tree.root.id, Action(name="direct"), {}, 0.1)
    # Corrupt the parent pointer into a cycle; the traversal must still terminate.
    tree.nodes[tree.root.id].parent_id = child.id
    assert {n.id for n in tree.path_to_root(child.id)} == {tree.root.id, child.id}


# ======================================================================================
# Slates, decision records and the anti-leak invariant
# ======================================================================================
def test_record_slate_and_mark_chosen(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.2), ("repair", 0.8)]], choose="repair")
    root = tree.root
    assert root.expanded
    assert root.visit_count == 1
    assert root.chosen_action_id == Action(name="repair").signature()
    assert [c.action.name for c in root.candidates] == ["direct", "repair"]
    assert root.slate_best_score == pytest.approx(0.8)


def test_blind_slate_exposes_no_rewards(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.2), ("repair", 0.8)]])
    slate = tree.blind_slate(tree.root.id)
    assert [a.name for a in slate] == ["direct", "repair"]
    assert all(isinstance(a, Action) for a in slate)
    # An Action carries no reward channel at all.
    assert not hasattr(slate[0], "score")


def test_decisions_expose_history_including_the_current_attempt(make_chain_tree):
    # Scores along the chain: root(empty) -> 0.0 -> 0.5 -> 0.5
    tree = make_chain_tree(
        [[("direct", 0.0)], [("repair", 0.5)], [("restart", 0.5)]]
    )
    records = tree.decisions()
    assert len(records) == 3

    root_decision, second, third = records
    # The action-less root is not an attempt: nothing has been tried yet.
    assert root_decision.depth == 0
    assert root_decision.features["n_attempts"] == 0.0
    assert root_decision.features["best_score"] == 0.0
    assert root_decision.features["stagnation"] == 0.0

    # At depth 1 exactly one attempt exists and it scored 0.0.
    assert second.depth == 1
    assert second.features["n_attempts"] == 1.0
    assert second.features["best_score"] == 0.0
    assert second.current_state["solution"].strip()

    # At depth 2 the attempt on top of this decision point scored 0.5 *and* it is
    # counted, which is what makes `repair` applicable from here on.
    assert third.depth == 2
    assert third.features["n_attempts"] == 2.0
    assert third.features["best_score"] == pytest.approx(0.5)
    assert third.features["stagnation"] == 0.0
    assert third.features["n_passed"] == 0.0
    assert third.features["n_failed"] == 2.0


def test_decision_rewards_and_metadata(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.25), ("repair", 1.0)]], choose="direct")
    record = tree.decisions()[0]
    direct = Action(name="direct")
    repair = Action(name="repair")
    assert record.rewards[direct.signature()] == pytest.approx(0.25)
    assert record.rewards[repair.signature()] == pytest.approx(1.0)
    assert record.oracle_score == pytest.approx(1.0)
    assert record.chosen_action_id == direct.signature()
    assert record.chosen_score == pytest.approx(0.25)  # the *chosen* candidate
    assert record.covered is True
    assert record.task["suite"] == "humaneval"
    assert record.reward_of(Action(name="nonexistent")) is None


def test_decision_uses_task_meta_features(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.5)]], suite="math", difficulty=0.8)
    record = tree.decisions()[0]
    # Task features are copied into the record via task_meta["features"], so they
    # are available to replayed policies exactly as they were live.
    assert record.features["difficulty"] == pytest.approx(0.8)
    assert record.features["suite_math"] == 1.0
    assert record.features["suite_humaneval"] == 0.0
    assert record.task["difficulty"] == pytest.approx(0.8)


def test_dedup_key_identifies_identical_evidence(make_chain_tree):
    a = make_chain_tree([[("direct", 0.2), ("repair", 0.8)]], tree_id="tree_a")
    b = make_chain_tree([[("direct", 0.2), ("repair", 0.8)]], tree_id="tree_b")
    c = make_chain_tree([[("direct", 0.2), ("repair", 0.4)]], tree_id="tree_c")

    assert a.decisions()[0].dedup_key == b.decisions()[0].dedup_key
    assert a.decisions()[0].dedup_key != c.decisions()[0].dedup_key


# ======================================================================================
# Statistics
# ======================================================================================
def test_stats_reports_action_statistics(make_chain_tree):
    tree = make_chain_tree(
        [[("direct", 0.5), ("brute_force", 0.0)], [("direct", 1.0), ("brute_force", 0.5)]]
    )
    stats = tree.stats()
    assert stats["n_nodes"] == 3
    assert stats["n_expanded"] == 2
    assert stats["best_score"] == pytest.approx(1.0)
    assert stats["solved"] is True
    assert stats["action_stats"]["brute_force"]["n"] == 2
    assert stats["action_stats"]["brute_force"]["mean_score"] == pytest.approx(0.25)
    assert stats["action_stats"]["direct"]["pass_rate"] == pytest.approx(0.5)


def test_state_features_stagnation_counts_trailing_non_improvements():
    history = [{"score": 0.0}, {"score": 0.4}, {"score": 0.4}, {"score": 0.3}]
    features = state_features({}, depth=4, history=history)
    assert features["stagnation"] == 2.0  # the last two attempts failed to improve

    improving = [{"score": 0.0}, {"score": 0.2}, {"score": 0.6}]
    assert state_features({}, depth=3, history=improving)["stagnation"] == 0.0
    assert state_features({}, depth=0, history=[{"score": 0.0}])["stagnation"] == 0.0


def test_state_features_action_counts_and_extra():
    features = state_features(
        {"solution": "abc", "feedback": "nope"},
        depth=2,
        history=[{"score": 1.0}],
        action_counts={"direct": 2},
        extra={"difficulty": 0.9},
    )
    assert features["act::direct"] == 2.0
    assert features["difficulty"] == 0.9
    assert features["has_feedback"] == 1.0
    assert features["feedback_len"] == 4.0
    assert features["artefact_len"] == 3.0
    assert features["n_passed"] == 1.0


# ======================================================================================
# Serialization
# ======================================================================================
def test_save_and_load_roundtrip(make_chain_tree, tmp_path):
    tree = make_chain_tree([[("direct", 0.25), ("repair", 1.0)], [("restart", 0.5)]])
    path = tree.save(tmp_path / "tree.json")
    restored = DiscoveryTree.load(path)
    assert restored.to_dict() == tree.to_dict()
    assert restored.decisions()[0].rewards == tree.decisions()[0].rewards
    assert restored.task_meta["suite"] == "humaneval"


def test_load_missing_tree_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DiscoveryTree.load(tmp_path / "nope.json")


def test_load_trees_skips_corrupt_files(make_chain_tree, tmp_path):
    good = make_chain_tree([[("direct", 0.5)]], tree_id="good")
    good.save(tmp_path / "good.json")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "wrong_schema.json").write_text(json.dumps({"nodes": [{}]}), encoding="utf-8")

    trees = load_trees(tmp_path)
    assert [t.tree_id for t in trees] == ["good"]


def test_decisions_from_trees_and_to_jsonl(make_chain_tree):
    a = make_chain_tree([[("direct", 0.5)]], tree_id="tree_a", task_id="t_a")
    b = make_chain_tree([[("direct", 0.5)], [("repair", 1.0)]], tree_id="tree_b", task_id="t_b")
    records = decisions_from_trees([b, a])
    assert [r.tree_id for r in records] == ["tree_a", "tree_b", "tree_b"]

    lines = [json.loads(line) for line in a.to_jsonl().splitlines()]
    assert len(lines) == 2  # root + one child
    assert lines[0]["id"] == a.root.id


def test_node_expanded_flag(make_chain_tree):
    tree = make_chain_tree([[("direct", 0.5)]])
    assert tree.root.expanded is True
    leaf = max(tree.nodes.values(), key=lambda n: n.depth)
    assert leaf.expanded is False
    assert tree.leaves()[0].id == leaf.id
