"""Benchmark package: deterministic task suites and the frozen evaluator."""

from benchmarks.evaluator import (
    EvalResult,
    assert_immutable,
    evaluate,
    evaluate_batch,
    evaluator_fingerprint,
    load_tasks,
    normalise_tests,
    static_check,
    task_index,
    task_meta,
)

__all__ = [
    "EvalResult",
    "assert_immutable",
    "evaluate",
    "evaluate_batch",
    "evaluator_fingerprint",
    "load_tasks",
    "normalise_tests",
    "static_check",
    "task_index",
    "task_meta",
]
