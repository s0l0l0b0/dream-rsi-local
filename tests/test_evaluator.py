"""Tests for the immutable sandboxed evaluator (``benchmarks/evaluator.py``).

The evaluator is the ground-truth oracle: candidate artefacts are executed in a
*fresh, isolated interpreter*, so tests that reach :func:`evaluate` with code that
is not rejected by the static gate are marked ``sandbox``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks import evaluator as ev

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO_ROOT / "benchmarks" / "tasks.jsonl"


@pytest.fixture(scope="module")
def suite() -> list[dict[str, Any]]:
    return ev.load_tasks(TASKS_PATH)


@pytest.fixture(scope="module")
def index(suite: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return ev.task_index(suite)


# ======================================================================================
# Task suite loading
# ======================================================================================
def test_load_tasks_returns_wellformed_unique_suite(suite: list[dict[str, Any]]) -> None:
    assert len(suite) >= 18, f"expected 18+ tasks, found {len(suite)}"

    ids = [str(task["id"]) for task in suite]
    assert len(set(ids)) == len(ids), f"duplicate task ids: {sorted(ids)}"

    for task in suite:
        label = task.get("id")
        for key in ("id", "entry_point", "prompt", "tests", "reference_solution"):
            assert task.get(key), f"task {label!r} is missing required key {key!r}"
        tests = ev.normalise_tests(task)
        assert tests, f"task {label!r} has no normalised tests"
        assert all(entry["name"] and entry["source"] for entry in tests)


# ======================================================================================
# End-to-end scoring
# ======================================================================================
@pytest.mark.sandbox
def test_every_reference_solution_scores_exactly_one(suite: list[dict[str, Any]]) -> None:
    for task in suite:
        expected = len(ev.normalise_tests(task))
        result = ev.evaluate(task, task["reference_solution"])
        assert result.passed is True, (
            f"{task['id']}: reference solution did not fully pass "
            f"(score={result.score}, error={result.error!r})"
        )
        assert result.score == 1.0, f"{task['id']}: score {result.score}"
        assert result.tests_passed == result.tests_total == expected, (
            f"{task['id']}: {result.tests_passed}/{result.tests_total} tests, expected {expected}"
        )


@pytest.mark.sandbox
def test_partial_credit_for_a_partly_wrong_solution(index: dict[str, dict[str, Any]]) -> None:
    task = index["he_001"]
    reference = task["reference_solution"]
    # ``sum_even`` must only count non-negative even numbers after the alteration,
    # so the test that covers negative inputs fails while the others still pass.
    altered = reference.replace("n % 2 == 0)", "n % 2 == 0 and n > 0)")
    assert altered != reference, "task he_001 changed shape; update this test"

    result = ev.evaluate(task, altered)
    assert 0.0 < result.score < 1.0, f"expected partial credit, got score={result.score}"
    assert result.passed is False
    assert result.tests_passed == round(result.score * result.tests_total)
    assert 0 < result.tests_passed < result.tests_total
    assert any(not entry["passed"] for entry in result.diagnostics)
    assert any(entry["passed"] for entry in result.diagnostics)
    # The score is exactly the observed pass fraction.
    assert result.score == pytest.approx(result.tests_passed / result.tests_total)


@pytest.mark.sandbox
def test_diagnostics_have_one_entry_per_test(index: dict[str, dict[str, Any]]) -> None:
    task = index["he_001"]
    names = [entry["name"] for entry in ev.normalise_tests(task)]
    result = ev.evaluate(task, task["reference_solution"])

    assert result.tests_total == len(names)
    assert len(result.diagnostics) == len(names)
    assert [entry["name"] for entry in result.diagnostics] == names
    for entry in result.diagnostics:
        assert set(entry) == {"name", "passed", "error"}, entry
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["passed"], bool)
        assert isinstance(entry["error"], str)
    assert all(entry["passed"] for entry in result.diagnostics)


@pytest.mark.sandbox
def test_infinite_loop_triggers_the_timeout_path(index: dict[str, dict[str, Any]]) -> None:
    task = index["he_001"]
    code = f"def {task['entry_point']}(*args, **kwargs):\n    while True:\n        pass\n"

    result = ev.evaluate(task, code, timeout_s=1.0)

    assert result.score == 0.0
    assert result.passed is False
    assert result.violation == "", "a runaway artefact is a runtime failure, not a gate rejection"
    assert "timeout" in result.error.lower(), f"unexpected error: {result.error!r}"
    assert result.diagnostics and all(not entry["passed"] for entry in result.diagnostics)


# ======================================================================================
# Cheap rejections: these never reach a subprocess
# ======================================================================================
def test_empty_and_whitespace_code_score_zero_with_an_error(
    index: dict[str, dict[str, Any]],
) -> None:
    task = index["he_001"]
    for code in ("", "   ", "\n\t  \n"):
        result = ev.evaluate(task, code)
        assert result.score == 0.0
        assert result.passed is False
        assert result.error, f"code {code!r} produced no error message"
        assert result.violation == result.error
        assert len(result.diagnostics) == result.tests_total


def test_syntax_error_scores_zero_and_mentions_syntaxerror(
    index: dict[str, dict[str, Any]],
) -> None:
    task = index["he_001"]
    result = ev.evaluate(task, "def broken(:\n    pass\n")

    assert result.score == 0.0
    assert result.passed is False
    assert "SyntaxError" in result.error, f"error should mention SyntaxError: {result.error!r}"
    assert result.violation == result.error


@pytest.mark.parametrize(
    ("code", "hint"),
    [
        ("import os\n", "os"),
        ("import subprocess\n", "subprocess"),
        ("import socket\n", "socket"),
        ('__import__("os")\n', "__import__"),
        ('open("x")\n', "open"),
        ('eval("1")\n', "eval"),
        ('getattr(x, "__class__")\n', "getattr"),
    ],
)
def test_static_gate_rejects_dangerous_artefacts(
    code: str, hint: str, index: dict[str, dict[str, Any]]
) -> None:
    violation = ev.static_check(code)
    assert violation, f"static gate accepted {code!r}"
    assert hint in violation, f"hint {hint!r} missing from {violation!r}"

    # The gate rejects before the harness is ever spawned, so the score is zero
    # and the artefact is reported as a violation rather than a test failure.
    result = ev.evaluate(index["he_001"], code)
    assert result.score == 0.0
    assert result.passed is False
    assert result.violation == violation
    assert result.error == violation


def test_static_gate_accepts_a_normal_solution(index: dict[str, dict[str, Any]]) -> None:
    assert ev.static_check(index["he_001"]["reference_solution"]) == ""
    assert ev.static_check("def sum_even(nums):\n    return sum(n for n in nums if n % 2 == 0)\n") == ""


# ======================================================================================
# normalise_tests
# ======================================================================================
def test_normalise_tests_supports_native_list_and_humaneval_check() -> None:
    native: dict[str, Any] = {
        "tests": [
            "def t0(candidate):\n    pass\n",
            {"name": "custom", "source": "def custom(candidate):\n    pass\n"},
            {"source": "def t2(candidate):\n    pass\n"},
        ]
    }
    tests = ev.normalise_tests(native)
    assert [entry["name"] for entry in tests] == ["t0", "custom", "t2"]
    assert tests[0]["source"] == "def t0(candidate):\n    pass\n"
    assert tests[1]["source"] == "def custom(candidate):\n    pass\n"

    check = "def check(candidate):\n    assert candidate(1) == 1\n"
    tests = ev.normalise_tests({"check": check})
    assert [entry["name"] for entry in tests] == ["humaneval_check"]
    assert "check(candidate)" in tests[0]["source"]
    assert check in tests[0]["source"]

    both = ev.normalise_tests({"tests": ["def t0(candidate):\n    pass\n"], "check": check})
    assert [entry["name"] for entry in both] == ["t0", "humaneval_check"]


@pytest.mark.sandbox
def test_humaneval_style_check_task_runs_end_to_end() -> None:
    task = {
        "id": "synthetic_check_task",
        "suite": "humaneval",
        "entry_point": "double",
        "signature": "def double(x):",
        "prompt": "Return twice x.",
        "difficulty": 0.1,
        "tags": [],
        "check": (
            "def check(candidate):\n"
            "    assert candidate(2) == 4\n"
            "    assert candidate(0) == 0\n"
        ),
    }
    good = ev.evaluate(task, "def double(x):\n    return 2 * x\n")
    assert good.passed is True
    assert good.tests_passed == good.tests_total == 1
    assert good.score == 1.0

    bad = ev.evaluate(task, "def double(x):\n    return x\n")
    assert bad.passed is False
    assert bad.score == 0.0


# ======================================================================================
# Immutability
# ======================================================================================
def test_evaluator_fingerprint_is_stable_and_task_suite_sensitive(tmp_path: Path) -> None:
    current = ev.evaluator_fingerprint()
    assert isinstance(current, str) and current
    assert current == ev.evaluator_fingerprint()

    tiny = tmp_path / "tiny_tasks.jsonl"
    tiny.write_text(
        json.dumps(
            {"id": "tiny", "entry_point": "f", "prompt": "p", "tests": ["def t0(c):\n    pass\n"]}
        )
        + "\n",
        encoding="utf-8",
    )
    other = ev.evaluator_fingerprint(tiny)
    assert other == ev.evaluator_fingerprint(tiny)
    assert other != current, "a different task suite must change the fingerprint"


def test_assert_immutable_guards_the_recorded_hash(tmp_path: Path) -> None:
    current = ev.evaluator_fingerprint()
    ev.assert_immutable(current)  # no raise
    ev.assert_immutable(current, TASKS_PATH)
    ev.assert_immutable("")  # empty record = "unrecorded", tolerated

    with pytest.raises(RuntimeError, match="oracle changed"):
        ev.assert_immutable("deadbeef" * 4)

    tiny = tmp_path / "tiny_tasks.jsonl"
    tiny.write_text(
        json.dumps(
            {"id": "tiny", "entry_point": "f", "prompt": "p", "tests": ["def t0(c):\n    pass\n"]}
        )
        + "\n",
        encoding="utf-8",
    )
    ev.assert_immutable(ev.evaluator_fingerprint(tiny), tiny)  # no raise
    with pytest.raises(RuntimeError):
        ev.assert_immutable(current, tiny)


def test_task_meta_never_contains_the_reference_solution(suite: list[dict[str, Any]]) -> None:
    for task in suite:
        meta = ev.task_meta(task)
        assert "reference_solution" not in meta
        assert meta["id"] == str(task["id"])
        assert meta["entry_point"] == str(task["entry_point"])
        blob = json.dumps(meta)
        assert task["reference_solution"] not in blob, f"{task['id']}: reference leaked into meta"
