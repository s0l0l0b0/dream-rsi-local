"""Deterministic, sandboxed benchmark evaluator -- **IMMUTABLE**.

This module is the ground-truth oracle of the whole system: it is the only
component allowed to turn a candidate artefact into a *measured reward*.  It is
declared immutable because every discovery tree stores
``metadata["evaluator_sha256"]``; if the fingerprint of this file (or of the task
suite) changes, the Dream Engine refuses to compare trees recorded under
different oracles.  See :func:`assert_immutable`.

Threat model
------------
Candidate code is produced by a language model and may be wrong, hostile, or
merely pathological.  It is therefore never executed in-process.  Each
evaluation runs in a fresh interpreter with:

* ``-I -B`` (isolated: no ``PYTHONPATH``, no user site-packages, no bytecode),
* a scratch working directory and a scrubbed environment (no API keys leak in),
* a static AST gate that rejects dangerous imports, dunder-attribute escapes and
  the classic sandbox-escape builtins,
* POSIX resource limits (CPU seconds, address space, file size, core dumps),
* a wall-clock timeout plus process-group kill, so a runaway loop or a forked
  child cannot outlive the evaluation.

This is a *research* sandbox: it stops accidents and casual escapes, not a
determined attacker with a kernel exploit.  Untrusted code from the open
internet should still be run inside a container or VM.

Public API
----------
``load_tasks`` / ``task_index``  -- read ``benchmarks/tasks.jsonl``
``evaluate``                     -- run one candidate against one task
``evaluate_batch``               -- convenience loop
``evaluator_fingerprint``        -- hash of this file + the task suite
``assert_immutable``             -- guard used by the explorer / dreamer
"""

from __future__ import annotations

import ast
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Marker documenting (for readers and linters) that this file must not change
#: semantics between generations of an RSI run.
IMMUTABLE = True

#: Hard cap on candidate source size (bytes).
MAX_CODE_BYTES = 20_000

#: Default wall-clock budget for one full task evaluation.
DEFAULT_TIMEOUT_S = 6.0

#: Address-space cap for the child (macOS/Linux).  Generous: CPython itself
#: needs a few hundred MB of virtual memory to start.
CHILD_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

#: Imports a candidate artefact may use.  Anything else is a policy violation --
#: a whitelist, because the sandbox's job is correctness grading, not I/O.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "bisect",
        "collections",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "math",
        "numbers",
        "operator",
        "re",
        "statistics",
        "string",
        "textwrap",
        "types",
        "typing",
    }
)

#: Builtins that either escape the sandbox or introduce nondeterminism.
FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "exit",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "vars",
    }
)

#: Attribute names used in CPython sandbox escapes.
FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "__bases__",
        "__builtins__",
        "__class__",
        "__closure__",
        "__code__",
        "__dict__",
        "__getattribute__",
        "__globals__",
        "__import__",
        "__mro__",
        "__reduce__",
        "__reduce_ex__",
        "__subclasses__",
        "f_back",
        "f_builtins",
        "f_globals",
        "f_locals",
        "gi_frame",
    }
)

_FORBIDDEN_CALL_NAMES = FORBIDDEN_BUILTINS | {"getattr", "setattr", "delattr"}


# ======================================================================================
# Results
# ======================================================================================
@dataclass
class EvalResult:
    """Outcome of one evaluation: a score in ``[0, 1]`` plus per-test diagnostics."""

    task_id: str
    score: float = 0.0
    passed: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    duration_ms: float = 0.0
    error: str = ""
    #: Per-test detail: ``[{"name": str, "passed": bool, "error": str}, ...]``
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    #: Populated when the static gate rejected the artefact.
    violation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": round(float(self.score), 6),
            "passed": bool(self.passed),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "duration_ms": round(float(self.duration_ms), 3),
            "error": self.error,
            "violation": self.violation,
            "diagnostics": self.diagnostics,
        }

    def summary(self, limit: int = 3) -> str:
        """Compact human-readable feedback string (fed back to the solver)."""
        if self.violation:
            return f"rejected by static gate: {self.violation}"
        if self.error and not self.diagnostics:
            return f"error: {self.error}"
        failures = [d for d in self.diagnostics if not d["passed"]]
        head = f"{self.tests_passed}/{self.tests_total} tests passed"
        if not failures:
            return head
        details = "; ".join(f"{d['name']}: {d['error'][:160]}" for d in failures[:limit])
        more = "" if len(failures) <= limit else f" (+{len(failures) - limit} more)"
        return f"{head} | {details}{more}"


# ======================================================================================
# Task loading
# ======================================================================================
def default_tasks_path() -> Path:
    """Path of the shipped task suite (``<repo>/benchmarks/tasks.jsonl``)."""
    return Path(__file__).resolve().parent / "tasks.jsonl"


def load_tasks(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the JSONL task suite, validating each record's required keys."""
    target = Path(path) if path else default_tasks_path()
    if not target.exists():
        raise FileNotFoundError(f"task suite not found: {target}")
    tasks: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - malformed suite
                raise ValueError(f"{target}:{lineno}: invalid JSON ({exc})") from exc
            for key in ("id", "entry_point", "prompt", "tests"):
                if key not in task:
                    raise ValueError(f"{target}:{lineno}: task missing required key {key!r}")
            if not task["tests"]:
                raise ValueError(f"{target}:{lineno}: task {task['id']} has no tests")
            tasks.append(task)
    if not tasks:
        raise ValueError(f"task suite {target} is empty")
    return tasks


def task_index(tasks: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index tasks by id."""
    return {str(t["id"]): dict(t) for t in tasks}


def task_meta(task: Mapping[str, Any]) -> dict[str, Any]:
    """The public, solver-visible view of a task (never includes the solution)."""
    return {
        "id": str(task["id"]),
        "suite": str(task.get("suite", "")),
        "entry_point": str(task["entry_point"]),
        "signature": str(task.get("signature", "")),
        "prompt": str(task.get("prompt", "")),
        "difficulty": float(task.get("difficulty", 0.5)),
        "tags": list(task.get("tags") or []),
    }


def normalise_tests(task: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return ``[{"name", "source"}, ...]`` for a task.

    Supports both the native ``tests`` list and a HumanEval-style single
    ``check(candidate)`` function so the suite can ingest upstream benchmarks
    without rewriting them.
    """
    tests: list[dict[str, str]] = []
    for i, entry in enumerate(task.get("tests") or []):
        if isinstance(entry, str):
            tests.append({"name": f"t{i}", "source": entry})
        else:
            tests.append(
                {
                    "name": str(entry.get("name") or f"t{i}"),
                    "source": str(entry["source"]),
                }
            )
    check = task.get("check")
    if check:
        tests.append(
            {
                "name": "humaneval_check",
                "source": f"def t_check(candidate):\n    check(candidate)\n\n{check}",
            }
        )
    return tests


# ======================================================================================
# Static gate
# ======================================================================================
def _root_module(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".")[0] if node.names else ""
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0]
    return ""


def static_check(code: str) -> str:
    """Return a violation message, or ``""`` when the artefact passes the gate.

    Deliberately conservative: rejects anything that could touch the filesystem,
    network, process table, interpreter internals, or that would make grading
    nondeterministic.
    """
    if not isinstance(code, str):
        return "artefact is not a string"
    if not code.strip():
        return "artefact is empty"
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return f"artefact exceeds {MAX_CODE_BYTES} bytes"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = _root_module(node)
            if module not in ALLOWED_IMPORTS:
                return f"import of disallowed module {module!r}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_BUILTINS:
            return f"use of forbidden builtin {node.id!r}"
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            return f"access to forbidden attribute {node.attr!r}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALL_NAMES:
                return f"call to forbidden builtin {node.func.id!r}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in {"__init__", "__name__", "__doc__", "__len__", "__add__", "__eq__"}:
                return f"access to dunder attribute {node.attr!r}"
    return ""


# ======================================================================================
# Harness
# ======================================================================================
_HARNESS = r'''
import json, sys

CANDIDATE_SRC = __CANDIDATE__
TESTS = __TESTS__
ENTRY_POINT = __ENTRY__
OUT_PATH = sys.argv[1]


def _run():
    results = []
    ns = {"__name__": "candidate_under_test"}
    cand_error = ""
    try:
        exec(compile(CANDIDATE_SRC, "<candidate>", "exec"), ns)
    except BaseException as exc:
        cand_error = "%s: %s" % (type(exc).__name__, exc)
    candidate = ns.get(ENTRY_POINT)
    if candidate is None and not cand_error:
        cand_error = "NameError: entry point %r not defined" % (ENTRY_POINT,)
    for name, src in TESTS:
        if cand_error:
            results.append({"name": name, "passed": False, "error": cand_error})
            continue
        tns = {"__name__": "tests"}
        try:
            exec(compile(src, "<test:%s>" % name, "exec"), tns)
            fn = tns.get(name)
            if fn is None:
                candidates = [
                    v for k, v in tns.items()
                    if k.startswith("t_") and callable(v) and getattr(v, "__module__", "") == "tests"
                ]
                fn = candidates[0] if candidates else None
            if fn is None:
                results.append({"name": name, "passed": False, "error": "test function not found"})
                continue
            fn(candidate)
            results.append({"name": name, "passed": True, "error": ""})
        except BaseException as exc:
            results.append({"name": name, "passed": False, "error": "%s: %s" % (type(exc).__name__, exc)})
    return results


try:
    _results = _run()
except BaseException as exc:
    _results = [
        {"name": n, "passed": False, "error": "harness error: %s: %s" % (type(exc).__name__, exc)}
        for n, _ in TESTS
    ]

with open(OUT_PATH, "w", encoding="utf-8") as fh:
    json.dump({"tests": _results}, fh)
'''


def build_harness(task: Mapping[str, Any], code: str) -> str:
    """Render the child-process harness with the artefact and tests embedded.

    Sources are embedded via ``repr``-safe JSON so that quotes, backslashes and
    unicode in the artefact cannot break out of the scaffolding.
    """
    tests = normalise_tests(task)
    return (
        _HARNESS.replace("__CANDIDATE__", json.dumps(code))
        .replace("__TESTS__", json.dumps([[t["name"], t["source"]] for t in tests]))
        .replace("__ENTRY__", json.dumps(str(task["entry_point"])))
    )


def _apply_child_limits(timeout_s: float) -> None:  # pragma: no cover - child only
    """``preexec_fn``: cap CPU, memory, file size and core dumps in the child."""
    try:
        cpu = max(1, int(timeout_s) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(
                resource.RLIMIT_AS, (CHILD_MEMORY_LIMIT_BYTES, CHILD_MEMORY_LIMIT_BYTES)
            )
    except (ValueError, OSError):
        pass


def _child_env(scratch: str) -> dict[str, str]:
    """Minimal environment: no credentials, deterministic hashing."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": scratch,
        "TMPDIR": scratch,
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def _kill_group(process: subprocess.Popen[bytes]) -> None:  # pragma: no cover - signal path
    """Terminate the child and every process it spawned."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _run_harness(harness_src: str, timeout_s: float) -> tuple[dict[str, Any] | None, str]:
    """Execute the harness in a sandboxed child; return ``(payload, error)``."""
    with tempfile.TemporaryDirectory(prefix="dreamrsi-eval-") as scratch:
        script = Path(scratch) / "harness.py"
        out = Path(scratch) / "results.json"
        script.write_text(harness_src, encoding="utf-8")
        cmd = [sys.executable, "-I", "-B", str(script), str(out)]
        process = subprocess.Popen(
            cmd,
            cwd=scratch,
            env=_child_env(scratch),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_apply_child_limits(timeout_s),
        )
        try:
            _, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            process.communicate()
            return None, f"timeout after {timeout_s:.1f}s (artefact did not terminate)"
        if not out.exists():
            tail = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
            detail = tail[-1] if tail else f"exit code {process.returncode}"
            return None, f"harness produced no results: {detail}"
        try:
            with out.open("r", encoding="utf-8") as handle:
                return json.load(handle), ""
        except (json.JSONDecodeError, OSError) as exc:
            return None, f"unreadable harness results: {exc}"


# ======================================================================================
# Public evaluation entry point
# ======================================================================================
def evaluate(
    task: Mapping[str, Any],
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> EvalResult:
    """Grade ``code`` against ``task`` in a sandbox; return score and diagnostics.

    ``score`` is the fraction of the task's tests that pass (partial credit), so
    the reward signal is graded rather than binary.  A rejected or crashed
    artefact always scores ``0.0`` -- never a silent pass.
    """
    started = time.perf_counter()
    tests = normalise_tests(task)
    task_id = str(task.get("id", "?"))

    def finish(result: EvalResult) -> EvalResult:
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    violation = static_check(code)
    if violation:
        return finish(
            EvalResult(
                task_id=task_id,
                score=0.0,
                passed=False,
                tests_total=len(tests),
                error=violation,
                violation=violation,
                diagnostics=[
                    {"name": t["name"], "passed": False, "error": violation} for t in tests
                ],
            )
        )

    payload, error = _run_harness(build_harness(task, code), timeout_s)
    if payload is None:
        return finish(
            EvalResult(
                task_id=task_id,
                score=0.0,
                passed=False,
                tests_total=len(tests),
                error=error,
                diagnostics=[{"name": t["name"], "passed": False, "error": error} for t in tests],
            )
        )

    diagnostics = list(payload.get("tests") or [])
    passed_count = sum(1 for d in diagnostics if d.get("passed"))
    total = len(diagnostics) or len(tests)
    score = passed_count / total if total else 0.0
    return finish(
        EvalResult(
            task_id=task_id,
            score=score,
            passed=(passed_count == total and total > 0),
            tests_passed=passed_count,
            tests_total=total,
            error="" if passed_count == total else "some tests failed",
            diagnostics=diagnostics,
        )
    )


def evaluate_batch(
    tasks: Sequence[Mapping[str, Any]],
    codes: Sequence[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[EvalResult]:
    """Evaluate parallel ``tasks``/``codes`` sequences element-wise."""
    return [evaluate(t, c, timeout_s=timeout_s) for t, c in zip(tasks, codes)]


# ======================================================================================
# Immutability
# ======================================================================================
def evaluator_fingerprint(tasks_path: str | Path | None = None) -> str:
    """Hash of this evaluator *and* the task suite it grades against.

    Stored in every discovery tree so that the Dream Engine can detect a changed
    oracle and refuse to mix incompatible records.
    """
    from src.core.utils import file_sha256

    grader = file_sha256(__file__)[:12]
    suite = Path(tasks_path) if tasks_path else default_tasks_path()
    suite_hash = file_sha256(suite)[:12] if suite.exists() else "no-suite"
    # Both halves are shown so a changed *task suite* is visible in reports, not
    # only a changed grader.
    return f"{grader}-{suite_hash}"


def assert_immutable(recorded: str, tasks_path: str | Path | None = None) -> None:
    """Raise if the oracle no longer matches ``recorded``.

    Called by the dream engine and the meta-optimizer before they compare two
    generations of trees: comparing measurements taken with different graders
    would be meaningless.
    """
    current = evaluator_fingerprint(tasks_path)
    if recorded and recorded != current:
        raise RuntimeError(
            "benchmark oracle changed since the trees were recorded "
            f"(recorded={recorded[:12]}..., current={current[:12]}...); "
            "re-record discovery trees or restore benchmarks/evaluator.py"
        )


__all__ = [
    "ALLOWED_IMPORTS",
    "DEFAULT_TIMEOUT_S",
    "EvalResult",
    "IMMUTABLE",
    "MAX_CODE_BYTES",
    "assert_immutable",
    "build_harness",
    "default_tasks_path",
    "evaluate",
    "evaluate_batch",
    "evaluator_fingerprint",
    "load_tasks",
    "normalise_tests",
    "static_check",
    "task_index",
    "task_meta",
]
