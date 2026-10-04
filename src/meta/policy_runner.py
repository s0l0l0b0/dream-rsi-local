"""Loads, validates and replays *candidate* policies in a separate process.

Candidate policies are produced by a language model or by a parameter search, so
they are treated as untrusted code:

* a static gate rejects filesystem/process/network imports before execution;
* the runner is launched as its own process by the optimizer, and a per-candidate
  ``SIGALRM`` timeout means one hanging policy cannot take the batch down;
* every candidate is validated against the mutation contract (must expose
  ``SearchPolicy``, ``POLICY_ID``, ``PARAMS`` and ``param_space``) and smoke
  tested on a synthetic decision point before it is scored.

Batch mode is what makes the RSI loop cheap: the optimizer hands over ~100
candidate files and one tree manifest, and gets back a metrics table for all of
them from a single interpreter start.  Every number in that table comes from
:class:`~src.engine.dreamer.Dreamer`, i.e. from replaying recorded rewards -- no
model call is made anywhere in this file.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import signal
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Make ``src``/``benchmarks`` importable when run as a script from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.evaluator import (  # noqa: E402
    evaluator_fingerprint,
)
from src.core.tree import Action, CandidateOutcome, DiscoveryTree, state_features  # noqa: E402
from src.core.utils import stable_hash, write_json  # noqa: E402
from src.engine.dreamer import Dreamer, ReplayConfig  # noqa: E402
from src.policy.base import (  # noqa: E402
    DEFAULT_ACTION_SPACE,
    ExplorationPolicy,
    PolicyContext,
)

POLICY_CLASS_NAME = "SearchPolicy"

#: Import roots a candidate policy may use.  A whitelist, because policy code has
#: no legitimate reason to touch the filesystem, the network or the process table
#: -- and it runs inside the optimizer's own interpreter.
ALLOWED_POLICY_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "bisect",
        "collections",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "types",
        "typing",
        "src",
        "benchmarks",
    }
)

FORBIDDEN_POLICY_CALLS: frozenset[str] = frozenset(
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
        "open",
        "quit",
        "vars",
    }
)

FORBIDDEN_POLICY_ATTRS: frozenset[str] = frozenset(
    {
        "__bases__",
        "__builtins__",
        "__closure__",
        "__code__",
        "__globals__",
        "__mro__",
        "__reduce__",
        "__reduce_ex__",
        "__subclasses__",
        "f_back",
        "f_builtins",
        "f_globals",
        "f_locals",
    }
)

MAX_POLICY_BYTES = 200_000


class PolicyContractError(RuntimeError):
    """Raised when a candidate policy does not satisfy the mutation contract."""


# ======================================================================================
# Static gate
# ======================================================================================
def policy_static_gate(source: str) -> str:
    """Return a violation message, or ``""`` when the candidate passes the gate."""
    if not source.strip():
        return "policy source is empty"
    if len(source.encode("utf-8")) > MAX_POLICY_BYTES:
        return f"policy source exceeds {MAX_POLICY_BYTES} bytes"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_POLICY_IMPORTS:
                    return f"import of disallowed module {root!r}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_POLICY_IMPORTS:
                return f"import from disallowed module {root!r}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_POLICY_CALLS:
                return f"call to forbidden builtin {node.func.id!r}"
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_POLICY_ATTRS:
            return f"access to forbidden attribute {node.attr!r}"
    return ""


# ======================================================================================
# Loading and contract validation
# ======================================================================================
def load_policy_module(path: str | Path, name: str = "dream_rsi_candidate_policy"):
    """Import a policy module from an arbitrary file path, ALWAYS from source.

    Two details matter here, and both are deliberate:

    * the module name is derived from the file *content*, so loading a rewritten
      file can never return the previously imported module object;
    * the source is compiled in memory instead of going through
      ``SourceFileLoader.exec_module``.  CPython validates its ``__pycache__``
      entry using mtimes **truncated to whole seconds**, so rewriting a policy
      within the same second it was last loaded -- exactly what a promotion does --
      and keeping the file size identical (``0.0`` -> ``0.5`` is the same number of
      characters) makes CPython silently reuse the stale bytecode.  That caused a
      promoted policy to be ignored by the live loop while the Dream Engine, which
      loads candidates in a fresh subprocess, saw it correctly.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise PolicyContractError(f"policy file not found: {file_path}")
    source = file_path.read_text(encoding="utf-8")
    violation = policy_static_gate(source)
    if violation:
        raise PolicyContractError(f"static gate rejected {file_path.name}: {violation}")

    module_name = f"{name}_{stable_hash([str(file_path), source], 8)}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None:  # pragma: no cover - defensive
        raise PolicyContractError(f"cannot import policy from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__spec__ = spec
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(file_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def validate_module(module: Any) -> list[str]:
    """Return the list of contract violations for an imported policy module."""
    problems: list[str] = []
    policy_cls = getattr(module, POLICY_CLASS_NAME, None)
    if policy_cls is None:
        return [f"module does not define {POLICY_CLASS_NAME}"]
    if not (isinstance(policy_cls, type) and issubclass(policy_cls, ExplorationPolicy)):
        problems.append(f"{POLICY_CLASS_NAME} must subclass ExplorationPolicy")
    policy_id = getattr(module, "POLICY_ID", None)
    if not isinstance(policy_id, str) or not policy_id.strip():
        problems.append("POLICY_ID must be a non-empty string literal")
    params = getattr(module, "PARAMS", None)
    if not isinstance(params, dict) or not params:
        problems.append("PARAMS must be a non-empty dict literal")
        return problems
    for key, value in params.items():
        if not isinstance(key, str):
            problems.append(f"PARAMS key {key!r} is not a string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"PARAMS[{key!r}] must be a number, got {type(value).__name__}")
    space_fn = getattr(policy_cls, "param_space", None)
    if not callable(space_fn):
        problems.append(f"{POLICY_CLASS_NAME}.param_space() is missing")
        return problems
    space = space_fn()
    if not isinstance(space, dict):
        problems.append("param_space() must return a dict")
        return problems
    for key, bounds in space.items():
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            problems.append(f"param_space()[{key!r}] must be a (low, high) pair")
            continue
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            problems.append(f"param_space()[{key!r}] has empty range ({low}, {high})")
    missing = sorted(set(params) - set(space))
    if missing:
        problems.append(f"PARAMS entries missing from param_space(): {missing}")
    if not callable(getattr(policy_cls, "select", None)):
        problems.append(f"{POLICY_CLASS_NAME}.select() is missing")
    return problems


def instantiate(module: Any, **kwargs: Any) -> ExplorationPolicy:
    """Build a policy instance from an imported module."""
    policy_cls = getattr(module, POLICY_CLASS_NAME, None)
    if policy_cls is None:
        raise PolicyContractError(f"module does not define {POLICY_CLASS_NAME}")
    return policy_cls(**kwargs)


# ======================================================================================
# Smoke test
# ======================================================================================
def synthetic_context(*, n_candidates: int = 4) -> PolicyContext:
    """A representative decision point used to smoke test candidate policies."""
    history = [
        {"node_id": "n0", "depth": 0, "action": "root", "score": 0.0, "feedback": ""},
        {"node_id": "n1", "depth": 1, "action": "direct", "score": 0.5, "feedback": "1/2 failed"},
    ]
    features = state_features(
        {"solution": "def f():\n    return 1", "feedback": "1/2 tests passed"},
        depth=2,
        history=history,
        action_counts={"direct": 1},
        extra={"difficulty": 0.5, "suite_humaneval": 1.0},
    )
    return PolicyContext(
        task_id="synthetic_1",
        suite="humaneval",
        entry_point="f",
        signature="def f(x):",
        prompt="Return x.",
        difficulty=0.5,
        current_state={"solution": "def f(x):\n    return 1"},
        feedback="1/2 tests passed",
        history=history,
        features=features,
        candidates=[],
        rng_seed=7,
        policy_id="candidate",
    )


def smoke_test(policy: ExplorationPolicy, action_space: Any | None = None) -> dict[str, Any]:
    """Exercise the policy's public surface on a synthetic decision point."""
    space = action_space or DEFAULT_ACTION_SPACE
    probe = synthetic_context()
    slate = space.slate(probe, 4, salt="smoke")
    ctx = PolicyContext(**{**probe.__dict__, "candidates": slate})

    chosen = policy.select(ctx)
    if not isinstance(chosen, Action):
        raise PolicyContractError(f"select() returned {type(chosen).__name__}, expected Action")
    if chosen.signature() not in {a.signature() for a in slate}:
        raise PolicyContractError("select() returned an action outside the slate")

    stop = policy.should_stop(ctx)
    if not isinstance(stop, bool):
        raise PolicyContractError(f"should_stop() returned {type(stop).__name__}, expected bool")

    outcomes = [CandidateOutcome(action=a, score=0.5, passed=False) for a in slate]
    policy.observe(ctx, chosen, outcomes[0], outcomes)

    second = type(policy)(params=dict(policy.params), rng_seed=policy.rng_seed)
    repeat = second.select(ctx)
    if repeat.signature() != chosen.signature():
        raise PolicyContractError("select() is not deterministic for identical input")

    proposal = policy.propose(synthetic_context(), budget=3)
    if not proposal or not all(isinstance(a, Action) for a in proposal):
        raise PolicyContractError("propose() must return a non-empty list of Actions")

    return {
        "chosen": chosen.name,
        "slate_size": len(slate),
        "should_stop": stop,
        "proposed": [a.name for a in proposal],
        "deterministic": True,
    }


# ======================================================================================
# Replay
# ======================================================================================
def load_manifest(path: str | Path) -> list[DiscoveryTree]:
    """Load the trees listed in a manifest file (or a single tree path)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    paths = data.get("tree_paths") or data.get("trees") or []
    if not paths:
        raise ValueError(f"manifest {path} lists no trees")
    trees: list[DiscoveryTree] = []
    for item in paths:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        trees.append(DiscoveryTree.load(candidate))
    return trees


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Raise ``TimeoutError`` if the block runs longer than ``seconds``.

    Unix-only, which is why the optimizer's own subprocess timeout is the outer
    safety net.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):  # pragma: no cover
        yield
        return

    def handler(signum, frame):  # noqa: ANN001, ARG001
        raise TimeoutError(f"candidate exceeded {seconds:.1f}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def evaluate_policy_file(
    policy_path: str | Path,
    trees: Sequence[DiscoveryTree],
    *,
    config: ReplayConfig | None = None,
    per_candidate_timeout: float = 20.0,
    label: str = "",
) -> dict[str, Any]:
    """Validate + smoke test + replay one candidate file; never raises."""
    label = label or Path(policy_path).stem
    payload: dict[str, Any] = {"label": label, "policy_path": str(policy_path), "ok": False}
    try:
        with time_limit(per_candidate_timeout):
            module = load_policy_module(policy_path, name=f"candidate_{label}")
            problems = validate_module(module)
            if problems:
                raise PolicyContractError("; ".join(problems))
            policy = instantiate(module, rng_seed=int((config or ReplayConfig()).rng_seed))
            payload["smoke"] = smoke_test(policy)
            payload["policy_id"] = policy.policy_id
            payload["params"] = dict(policy.params)
            policy = instantiate(module, rng_seed=int((config or ReplayConfig()).rng_seed))
            dreamer = Dreamer(trees, config=config)
            result = dreamer.replay(policy)
            payload["replay"] = result.to_dict()
            payload["ok"] = not result.error
            payload["error"] = result.error
    except PolicyContractError as exc:
        payload["error"] = f"contract: {exc}"
    except TimeoutError as exc:
        payload["error"] = f"timeout: {exc}"
    except Exception as exc:  # noqa: BLE001 - a bad candidate must not kill the batch
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-2000:]
    return payload


def replay_batch(
    policy_paths: Sequence[str | Path],
    trees: Sequence[DiscoveryTree],
    *,
    config: ReplayConfig | None = None,
    with_baselines: bool = False,
    per_candidate_timeout: float = 20.0,
) -> dict[str, Any]:
    """Evaluate many candidate files in one interpreter start."""
    results = [
        evaluate_policy_file(
            path,
            trees,
            config=config,
            per_candidate_timeout=per_candidate_timeout,
            label=Path(path).parent.name or Path(path).stem,
        )
        for path in policy_paths
    ]
    payload: dict[str, Any] = {
        "ok": True,
        "n_policies": len(results),
        "n_trees": len(trees),
        "n_decisions": sum(len(t.decisions()) for t in trees),
        "evaluator_sha256": evaluator_fingerprint(),
        "results": results,
    }
    if with_baselines:
        payload["baselines"] = {
            name: result.to_dict()
            for name, result in Dreamer(trees, config=config).baselines().items()
        }
    return payload


# ======================================================================================
# CLI
# ======================================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and replay candidate policies offline.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate a candidate policy file")
    check.add_argument("--policy", required=True)

    replay = sub.add_parser("replay", help="replay one or more candidate policies on a tree manifest")
    replay.add_argument("--policy", action="append", default=[], help="repeatable")
    replay.add_argument("--policies", default="", help="comma-separated list of policy files")
    replay.add_argument("--manifest", required=True, help="JSON manifest listing tree paths")
    replay.add_argument("--out", required=True, help="where to write the JSON report")
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--baselines", action="store_true")
    replay.add_argument("--per-candidate-timeout", type=float, default=20.0)

    args = parser.parse_args(argv)

    if args.command == "check":
        payload: dict[str, Any] = {"ok": False, "policy_path": args.policy}
        try:
            module = load_policy_module(args.policy)
            problems = validate_module(module)
            payload["problems"] = problems
            if problems:
                payload["error"] = "contract: " + "; ".join(problems)
            else:
                policy = instantiate(module, rng_seed=0)
                payload["policy_id"] = policy.policy_id
                payload["params"] = dict(policy.params)
                payload["param_space"] = {k: list(v) for k, v in policy.param_space().items()}
                payload["smoke"] = smoke_test(policy)
                payload["ok"] = True
        except Exception as exc:  # noqa: BLE001
            payload["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["ok"] else 1

    paths = [p for p in (args.policy or []) if p]
    paths += [p.strip() for p in args.policies.split(",") if p.strip()]
    if not paths:
        parser.error("replay needs at least one --policy or --policies")
    trees = load_manifest(args.manifest)
    payload = replay_batch(
        paths,
        trees,
        config=ReplayConfig(rng_seed=args.seed),
        with_baselines=args.baselines,
        per_candidate_timeout=args.per_candidate_timeout,
    )
    write_json(args.out, payload, indent=None)
    print(json.dumps({"ok": True, "out": args.out, "n_results": len(payload["results"])}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
