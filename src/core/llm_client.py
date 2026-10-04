"""OpenAI-compatible LLM client (dsh / Ollama / vLLM) with an offline simulator.

The project must be runnable with **no** endpoint, **no** API key and **no**
network, because the interesting half of Dream-RSI (the Dream Engine) is
supposed to cost zero inference.  So this module ships three clients behind one
interface:

``OpenAICompatibleClient``
    Plain ``urllib`` ``POST {base_url}/chat/completions``.  Works with dsh's
    router, Ollama (``http://localhost:11434/v1``), vLLM, LM Studio or the real
    OpenAI API.  Only the standard library is used.

``MockLLMClient``
    A **deterministic solver simulator**, not a model.  It fabricates plausible
    Python artefacts from the benchmark's reference solution, with a
    success probability that depends on the strategy, the task suite and the
    current search state -- so an evolved policy faces a structured but
    nontrivial world.  Crucially it only ever produces *code*; every reward is
    still measured by the real sandboxed evaluator in :mod:`benchmarks.evaluator`.

    This is a test double for pipeline validation.  It reads reference solutions
    that a real model would not have, so **offline numbers are not scientific
    evidence about real models** -- they show that the RSI machinery works.
    See the README's limitations section.

``CachingLLMClient``
    Content-addressed cache in front of any client.  Re-running an exploration
    with the same prompts costs nothing, and cache hits are reported separately
    from real calls so "zero inference" claims stay auditable.

``NullLLMClient``
    Raises on use.  The Dream Engine takes one of these to make "replay performs
    no inference" a structural guarantee rather than a promise.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.tree import Action
from src.core.utils import monotonic_ms, new_id, read_json, seeded_int, stable_hash, write_json
from src.policy.base import ACTION_SPECS, PolicyContext

# ======================================================================================
# Errors and data types
# ======================================================================================
class LLMError(RuntimeError):
    """Raised when an LLM call cannot be completed."""


@dataclass
class GenerationRequest:
    """A single solver call: task + strategy + current state."""

    task_id: str
    suite: str
    entry_point: str
    prompt: str
    signature: str
    action: Action
    current_state: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    history: Sequence[Mapping[str, Any]] = ()
    difficulty: float = 0.5
    #: Monotonic index of this attempt within its task (for cache keys).
    attempt: int = 0
    best_score: float = 0.0
    depth: int = 0
    #: Free-form extras forwarded to the prompt template.
    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived ----------------------------------------------------------------------
    @property
    def temperature(self) -> float:
        return float(self.action.params.get("temperature", 0.2))

    @property
    def strategy_hint(self) -> str:
        spec = ACTION_SPECS.get(self.action.name)
        return spec.prompt_hint if spec else "Write the implementation."

    @property
    def cache_key(self) -> str:
        """Identity of the *physical* request: the rendered prompt plus sampling.

        Derived from :meth:`messages` rather than from the individual fields, so
        that anything which reaches the model is necessarily part of the identity.
        An earlier field-by-field version omitted ``history`` while
        :func:`render_messages` rendered it, so two genuinely different prompts
        could share a cache key -- and, worse, the offline simulator seeds its RNG
        from this key, which made it fabricate the same artefact for both.
        """
        return stable_hash(
            {
                "messages": self.messages(),
                "temperature": self.temperature,
                "action": self.action.name,
                "attempt": self.attempt,
                "extra": self.extra,
            },
            length=24,
        )

    def messages(self) -> list[dict[str, str]]:
        """Render the chat messages for an OpenAI-compatible endpoint."""
        return render_messages(self)

    @classmethod
    def from_context(
        cls,
        ctx: PolicyContext,
        action: Action,
        *,
        attempt: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> "GenerationRequest":
        """Build a request from a :class:`PolicyContext` and an action."""
        return cls(
            task_id=ctx.task_id,
            suite=ctx.suite,
            entry_point=ctx.entry_point,
            prompt=ctx.prompt,
            signature=ctx.signature,
            action=action,
            current_state=dict(ctx.current_state),
            feedback=ctx.feedback,
            history=list(ctx.history),
            difficulty=ctx.difficulty,
            attempt=attempt,
            best_score=ctx.best_score,
            depth=ctx.depth,
            extra=dict(extra or {}),
        )


@dataclass
class GenerationResult:
    """A solver response plus the accounting needed for cost reporting."""

    text: str = ""
    code: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "code_chars": len(self.code),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "cached": self.cached,
            "error": self.error,
        }


# ======================================================================================
# Prompt rendering (mechanical; deliberately not part of the evolved policy)
# ======================================================================================
SYSTEM_PROMPT = (
    "You are an expert Python programmer solving benchmark tasks in a sandbox.\n"
    "Return ONLY the complete Python source for the requested function.\n"
    "Rules: standard library only; no file, network or process access; no I/O; "
    "no printing; the entry point must keep the exact name and signature given.\n"
    "Wrap the code in a single ```python fenced block."
)


def render_messages(request: GenerationRequest) -> list[dict[str, str]]:
    """Render the chat messages for a request.

    Kept deliberately simple and *fixed*: the RSI loop is supposed to evolve
    decision-making, so prompt construction must not drift between generations
    or the comparison would confound two changes at once.
    """
    lines: list[str] = [
        f"TASK ID: {request.task_id}   SUITE: {request.suite}",
        "",
        "PROBLEM:",
        request.prompt.strip(),
        "",
        "REQUIRED SIGNATURE:",
        request.signature.strip() or f"def {request.entry_point}(...):",
    ]

    previous = str(request.current_state.get("solution") or "").strip()
    if previous:
        lines += ["", "CURRENT BEST ATTEMPT (may be wrong):", "```python", previous[:4000], "```"]
    if request.feedback:
        lines += ["", "EVALUATOR FEEDBACK ON THE CURRENT ATTEMPT:", request.feedback[:1500]]
    if request.history:
        lines.append("")
        lines.append("ATTEMPTS SO FAR (most recent last):")
        for item in list(request.history)[-6:]:
            lines.append(
                f"- depth {item.get('depth')}: {item.get('action')} -> score "
                f"{float(item.get('score', 0.0)):.2f}"
                + (f" ({str(item.get('feedback'))[:120]})" if item.get("feedback") else "")
            )

    lines += ["", f"STRATEGY: {request.action.name} -- {request.strategy_hint}"]
    if request.action.params:
        knob = ", ".join(f"{k}={v}" for k, v in sorted(request.action.params.items()))
        lines.append(f"STRATEGY PARAMETERS: {knob}")
    lines += ["", "Write the complete Python source now."]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str, entry_point: str | None = None) -> str:
    """Extract runnable Python from a model response.

    Prefers a fenced block that defines ``entry_point``; falls back to the first
    fence, then to the raw text.  Returns ``""`` when nothing usable is present.
    """
    if not text or not text.strip():
        return ""
    blocks = [b.strip() for b in _FENCE_RE.findall(text) if b.strip()]
    if entry_point:
        for block in blocks:
            if f"def {entry_point}" in block:
                return block
    if blocks:
        return blocks[0]
    stripped = text.strip()
    looks_like_code = f"def {entry_point}" in stripped if entry_point else "def " in stripped
    return stripped if looks_like_code else ""


# ======================================================================================
# Client interface
# ======================================================================================
class LLMClient(ABC):
    """Minimal solver interface: one request in, one :class:`GenerationResult` out."""

    name: str = "llm"

    def __init__(self) -> None:
        self.calls: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce an artefact for ``request`` (never raises for solver errors)."""

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Raw chat completion, used by the meta-optimizer to rewrite policy code.

        The default implementation reports "unsupported" rather than raising, so
        the optimizer can fall back to its deterministic mutation route on any
        client that cannot do free-form code generation.
        """
        return GenerationResult(
            error=f"{self.name} does not support raw completions",
            model=self.name,
        )

    def counters(self) -> dict[str, Any]:
        return {
            "client": self.name,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def _account(self, result: GenerationResult) -> GenerationResult:
        if not result.cached:
            self.calls += 1
            self.prompt_tokens += result.prompt_tokens
            self.completion_tokens += result.completion_tokens
        return result

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap token estimate (~4 chars/token); avoids a tokenizer dependency."""
        return max(1, len(text) // 4) if text else 0


class NullLLMClient(LLMClient):
    """A client that refuses to run -- used by the Dream Engine.

    Passing this to the replay engine makes "offline replay performs no
    inference" a structural property: any accidental model call raises loudly
    instead of silently costing money.
    """

    name = "null"

    def generate(self, request: GenerationRequest) -> GenerationResult:  # pragma: no cover
        raise LLMError(
            "NullLLMClient was invoked: offline replay must never call a model "
            f"(task={request.task_id}, action={request.action.name})"
        )


# ======================================================================================
# OpenAI-compatible HTTP client
# ======================================================================================
class OpenAICompatibleClient(LLMClient):
    """``POST {base_url}/chat/completions`` via :mod:`urllib` (no dependencies)."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        model: str = "qwen2.5-coder:7b",
        timeout_s: float = 60.0,
        max_tokens: int = 1024,
        max_retries: int = 2,
        extra_headers: Mapping[str, str] | None = None,
        completion_path: str = "/chat/completions",
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.extra_headers = dict(extra_headers or {})
        self.completion_path = completion_path
        #: Extra top-level fields merged into every request body.  Endpoint-specific
        #: switches live here so the client stays generic: with Ollama this is how
        #: you disable a thinking model's reasoning pass (``{"think": false}``) or
        #: hold the model in memory (``{"keep_alive": "30m"}``).  Thinking models
        #: otherwise spend hundreds of tokens per answer -- measured here at 33-62s
        #: per call, versus ~15s with reasoning disabled.
        self.extra_body = dict(extra_body or {})

    # -- configuration from the environment -------------------------------------------
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> "OpenAICompatibleClient":
        """Build a client from ``DREAM_RSI_*`` / ``OPENAI_*`` environment variables."""
        e = dict(os.environ if env is None else env)
        kwargs: dict[str, Any] = {
            "base_url": (
                e.get("DREAM_RSI_BASE_URL")
                or e.get("OPENAI_BASE_URL")
                or e.get("OPENAI_API_BASE")
                or "http://localhost:11434/v1"
            ),
            "api_key": e.get("DREAM_RSI_API_KEY") or e.get("OPENAI_API_KEY") or "",
            "model": e.get("DREAM_RSI_MODEL") or e.get("OPENAI_MODEL") or "qwen2.5-coder:7b",
        }
        if e.get("DREAM_RSI_MAX_TOKENS"):
            kwargs["max_tokens"] = int(e["DREAM_RSI_MAX_TOKENS"])
        if e.get("DREAM_RSI_TIMEOUT_S"):
            kwargs["timeout_s"] = float(e["DREAM_RSI_TIMEOUT_S"])
        if e.get("DREAM_RSI_EXTRA_BODY"):
            try:
                extra = json.loads(e["DREAM_RSI_EXTRA_BODY"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"DREAM_RSI_EXTRA_BODY is not valid JSON: {exc}. "
                    'Example: DREAM_RSI_EXTRA_BODY=\'{"think": false}\''
                ) from exc
            if not isinstance(extra, dict):
                raise ValueError("DREAM_RSI_EXTRA_BODY must be a JSON object")
            kwargs["extra_body"] = extra
        kwargs.update(overrides)
        return cls(**kwargs)

    def _payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens if max_tokens is not None else self.max_tokens),
            "stream": False,
        }
        # Applied last so an operator can override anything the client sets.
        payload.update(self.extra_body)
        return payload

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Raw chat completion (used for LLM-driven policy mutation)."""
        result = self._post(self._payload(messages, temperature=temperature, max_tokens=max_tokens))
        if result.ok:
            result.code = extract_code(result.text)
        return self._account(result)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Call the endpoint, retrying transient failures with backoff."""
        payload = self._payload(
            request.messages(),
            temperature=request.temperature,
            max_tokens=int(request.action.params.get("max_tokens", self.max_tokens)),
        )
        result = self._post(payload)
        if result.ok:
            result.code = extract_code(result.text, request.entry_point)
        return self._account(result)

    def _post(self, payload: dict[str, Any]) -> GenerationResult:
        """POST a payload, retrying transient failures with exponential backoff."""
        started = monotonic_ms()
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                body = json.dumps(payload).encode("utf-8")
                http_request = urllib.request.Request(
                    f"{self.base_url}{self.completion_path}",
                    data=body,
                    headers=self._headers(),
                    method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    or data.get("choices", [{}])[0].get("text", "")
                )
                usage = data.get("usage") or {}
                return GenerationResult(
                    text=text,
                    code=text,
                    model=str(data.get("model") or self.model),
                    prompt_tokens=int(usage.get("prompt_tokens") or self.estimate_tokens(
                        json.dumps(payload["messages"]))),
                    completion_tokens=int(usage.get("completion_tokens") or self.estimate_tokens(text)),
                    latency_ms=monotonic_ms() - started,
                    raw={"id": data.get("id", ""), "finish_reason": (
                        data.get("choices", [{}])[0].get("finish_reason", ""))},
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                last_error = f"malformed response: {type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(min(2.0**attempt * 0.5, 4.0))
        return GenerationResult(
            error=last_error or "unknown LLM error",
            model=self.model,
            latency_ms=monotonic_ms() - started,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe (used by ``--mode auto``)."""
        try:
            probe = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
            with urllib.request.urlopen(probe, timeout=min(self.timeout_s, 5.0)) as response:
                return response.status == 200, f"HTTP {response.status}"
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            return False, f"{type(exc).__name__}: {exc}"


# ======================================================================================
# Offline deterministic simulator
# ======================================================================================
#: Base probability that a strategy yields a fully-correct artefact, per suite.
#: These encode *structure the policy can learn*: some strategies suit some kinds
#: of problem.  Suite-level (not task-level) structure is what lets a policy
#: fitted on training tasks transfer to the holdout.
_BASE_SUCCESS: dict[str, dict[str, float]] = {
    "direct": {"humaneval": 0.13, "math": 0.09, "kernelbench": 0.10},
    "chain_of_thought": {"humaneval": 0.15, "math": 0.23, "kernelbench": 0.18},
    "test_driven": {"humaneval": 0.23, "math": 0.12, "kernelbench": 0.15},
    "repair": {"humaneval": 0.10, "math": 0.09, "kernelbench": 0.09},
    "edge_case_scan": {"humaneval": 0.18, "math": 0.11, "kernelbench": 0.13},
    "decompose": {"humaneval": 0.14, "math": 0.17, "kernelbench": 0.15},
    "restart": {"humaneval": 0.11, "math": 0.11, "kernelbench": 0.10},
    "brute_force": {"humaneval": 0.16, "math": 0.08, "kernelbench": 0.08},
    "optimize": {"humaneval": 0.08, "math": 0.06, "kernelbench": 0.21},
}
_DEFAULT_SUCCESS = 0.2

#: Behaviour-altering edits used to fabricate plausible *wrong* solutions.  They
#: are applied to the syntax tree of the entry point rather than to its text, so
#: a mutation is guaranteed to change the program (the old text-substitution
#: version silently no-opped on solutions whose formatting did not match, which
#: made "failed" generations pass all tests).
_MUTATION_KINDS: tuple[str, ...] = (
    "integer",
    "comparison",
    "binary_op",
    "return_shift",
    "boolean",
)

#: Operator flips that keep an expression type-correct but change its behaviour.
_COMPARISON_FLIPS: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

_BINARY_FLIPS: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Add,
    ast.Pow: ast.Mult,
}


class _Gremlin(ast.NodeTransformer):
    """Applies exactly one behaviour-altering edit, in place, in the target body."""

    def __init__(self, kind: str, rng: random.Random) -> None:
        self.kind = kind
        self.rng = rng
        self.done = False

    # -- integer literals ------------------------------------------------------------
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if (
            not self.done
            and self.kind == "integer"
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            self.done = True
            delta = 1 if node.value >= 0 else -1
            return ast.copy_location(ast.Constant(value=node.value + delta), node)
        return node

    # -- comparison operators --------------------------------------------------------
    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if not self.done and self.kind == "comparison":
            for index, op in enumerate(node.ops):
                flip = _COMPARISON_FLIPS.get(type(op))
                if flip is not None:
                    node.ops[index] = flip()
                    self.done = True
                    break
        return self.generic_visit(node) if not self.done else node

    # -- arithmetic ------------------------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if not self.done and self.kind == "binary_op":
            flip = _BINARY_FLIPS.get(type(node.op))
            if flip is not None:
                node.op = flip()
                self.done = True
                return node
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        if not self.done and self.kind == "binary_op":
            flip = _BINARY_FLIPS.get(type(node.op))
            if flip is not None:
                node.op = flip()
                self.done = True
                return node
        return self.generic_visit(node)

    # -- boolean connectives ---------------------------------------------------------
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if not self.done and self.kind == "boolean":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.done = True
            return node
        return self.generic_visit(node)

    # -- returned values -------------------------------------------------------------
    def visit_Return(self, node: ast.Return) -> ast.AST:
        if not self.done and self.kind == "return_shift" and node.value is not None:
            if isinstance(node.value, (ast.Name, ast.Constant, ast.Call, ast.UnaryOp)):
                node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=1))
                self.done = True
                return node
        return self.generic_visit(node)


def _find_entry_node(tree: ast.Module, entry_point: str) -> ast.AST | None:
    """Locate the function the tests call, so mutations stay inside it."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


class MockLLMClient(LLMClient):
    """Deterministic solver simulator: fabricates code with structured competence.

    Given the same request, it always returns the same artefact (like a
    temperature-0 model), and the artefact is a *function of the prompt*: it is
    derived from the task's reference solution plus a strategy- and
    state-dependent chance of being correct.

    Rewards are never simulated -- the caller grades the returned code with the
    real evaluator.
    """

    name = "mock-offline"

    def __init__(
        self,
        tasks: Mapping[str, Mapping[str, Any]],
        *,
        seed: int = 1234,
        strict_baseline: bool = False,
    ) -> None:
        super().__init__()
        self.tasks = {str(k): dict(v) for k, v in tasks.items()}
        self.seed = int(seed)
        #: When True, ignore the repair/restart/state bonuses (a blunter world,
        #: useful for sanity tests that expect no learnable structure).
        self.strict_baseline = bool(strict_baseline)

    # -- probability model ------------------------------------------------------------
    def success_probability(self, request: GenerationRequest) -> float:
        """Probability the simulated solver returns a fully correct artefact."""
        table = _BASE_SUCCESS.get(request.action.name, {})
        p = float(table.get(request.suite, _DEFAULT_SUCCESS))

        if not self.strict_baseline:
            # Repair is the right move precisely when there is something to repair.
            if request.action.name == "repair" and request.best_score > 0.0:
                p += 0.55 * float(request.best_score)
            # Restart pays off once the search has clearly stalled.
            if request.action.name == "restart" and request.depth >= 2:
                p += 0.15
            # Cheap strategies degrade on hard tasks; decomposition helps there.
            if request.action.name == "brute_force":
                p -= 0.40 * float(request.difficulty)
            if request.action.name == "decompose":
                p += 0.25 * float(request.difficulty)

        # Small task-specific jitter: keeps the world from being a lookup table
        # while leaving the suite-level structure (the learnable signal) intact.
        jitter = ((stable_hash([request.task_id, request.action.name], 8))[0:4])
        p += (int(jitter, 16) / 65535.0 - 0.5) * 0.10
        return max(0.02, min(0.95, p))

    # -- generation -------------------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = monotonic_ms()
        task = self.tasks.get(request.task_id)
        if task is None:
            return self._account(
                GenerationResult(error=f"unknown task {request.task_id!r}", model=self.name)
            )
        reference = str(task.get("reference_solution") or "")
        if not reference.strip():
            return self._account(
                GenerationResult(error=f"task {request.task_id} has no reference solution", model=self.name)
            )

        rng = random.Random(
            seeded_int(self.seed, request.cache_key, request.action.name, round(request.depth))
        )
        p_success = self.success_probability(request)
        roll = rng.random()

        if roll < p_success:
            code = self._variant(reference, rng)
            kind = "correct"
        elif roll < p_success + (1.0 - p_success) * 0.55:
            code = self._mutate(reference, rng, n=1, entry_point=request.entry_point)
            kind = "near_miss"
        else:
            code = self._mutate(reference, rng, n=rng.choice([2, 2, 3]), entry_point=request.entry_point)
            kind = "wrong"

        messages = request.messages()
        prompt_text = json.dumps(messages)
        return self._account(
            GenerationResult(
                text=f"```python\n{code}\n```",
                code=code,
                model=self.name,
                prompt_tokens=self.estimate_tokens(prompt_text),
                completion_tokens=self.estimate_tokens(code),
                latency_ms=monotonic_ms() - started,
                raw={"simulated": True, "kind": kind, "p_success": round(p_success, 4)},
            )
        )

    # -- artefact fabrication ---------------------------------------------------------
    @staticmethod
    def _variant(reference: str, rng: random.Random) -> str:
        """A correct-but-differently-formatted copy of the reference solution."""
        header = (
            "# solution generated by the offline simulator\n"
            if rng.random() < 0.3
            else ""
        )
        return header + reference

    @staticmethod
    def _parses(source: str) -> bool:
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    def _mutate(
        self, reference: str, rng: random.Random, *, n: int, entry_point: str
    ) -> str:
        """Apply ``n`` guaranteed AST-level mutations to the entry point.

        Returns source that is syntactically valid and *provably different* from
        the reference (compared on the syntax tree, not on formatting).
        """
        source = reference
        applied = 0
        for _ in range(max(n, 1) * 3):
            if applied >= max(n, 1):
                break
            try:
                tree = ast.parse(source)
            except SyntaxError:  # pragma: no cover - defensive
                break
            target = _find_entry_node(tree, entry_point) or tree
            before = ast.dump(tree)
            mutator = _Gremlin(rng.choice(_MUTATION_KINDS), rng)
            mutator.visit(target)
            if not mutator.done:
                continue
            ast.fix_missing_locations(tree)
            try:
                candidate = ast.unparse(tree)
            except Exception:  # pragma: no cover - unparse is very robust
                continue
            if ast.dump(ast.parse(candidate)) != before and self._parses(candidate):
                source = candidate
                applied += 1
        if applied == 0:
            # Last resort: break the interface so the artefact cannot silently pass.
            source = f"def {entry_point}(*args, **kwargs):\n    raise NotImplementedError\n"
        return source

    def health(self) -> tuple[bool, str]:
        return True, "offline simulator ready"


# ======================================================================================
# Caching wrapper
# ======================================================================================
class CachingLLMClient(LLMClient):
    """Content-addressed cache: identical prompts are answered without inference."""

    name = "caching"

    def __init__(
        self,
        inner: LLMClient,
        cache_path: str | Path | None = None,
        *,
        namespace: str = "",
    ) -> None:
        super().__init__()
        self.inner = inner
        self.cache_path = Path(cache_path) if cache_path else None
        #: Sampling namespace mixed into every cache key.  The RSI loop sets it to
        #: the current world, so re-exploring a task in a new generation draws a
        #: *fresh* sample (independent evidence) while repeats inside one
        #: generation are still free.
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self._memory: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            self._memory = dict(read_json(self.cache_path, {}) or {})

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"caching({self.inner.name})"

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Cached raw completion (policy-mutation prompts are large and repeated)."""
        key = (
            f"complete:{self.inner.name}:{self.namespace}:"
            f"{stable_hash([list(messages), temperature], 24)}"
        )
        entry = self._memory.get(key)
        if entry is not None:
            self.hits += 1
            return GenerationResult(
                text=entry.get("text", ""),
                code=entry.get("code", ""),
                model=entry.get("model", ""),
                cached=True,
                error=entry.get("error", ""),
            )
        self.misses += 1
        result = self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
        self._account(result)
        if result.ok:
            self._memory[key] = {
                "text": result.text,
                "code": result.code,
                "model": result.model,
                "error": result.error,
            }
        return result

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = f"{self.inner.name}:{self.namespace}:{request.cache_key}"
        entry = self._memory.get(key)
        if entry is not None:
            self.hits += 1
            return GenerationResult(
                text=entry.get("text", ""),
                code=entry.get("code", ""),
                model=entry.get("model", ""),
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                cached=True,
                error=entry.get("error", ""),
                raw={"cache_hit": True},
            )
        self.misses += 1
        result = self.inner.generate(request)
        self._account(result)
        if result.ok:
            self._memory[key] = {
                "text": result.text,
                "code": result.code,
                "model": result.model,
                "error": result.error,
            }
            if self.cache_path and self.misses % 8 == 0:
                self.flush()
        return result

    def flush(self) -> None:
        if self.cache_path:
            write_json(self.cache_path, self._memory, indent=None)

    def counters(self) -> dict[str, Any]:
        data = super().counters()
        data.update(
            {
                "client": self.name,
                "hits": self.hits,
                "misses": self.misses,
                "inner": self.inner.counters(),
                "cache_entries": len(self._memory),
            }
        )
        return data

    def save_cache(self) -> None:
        self.flush()


# ======================================================================================
# Factory
# ======================================================================================
def make_client(
    mode: str = "auto",
    *,
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
    cache_path: str | Path | None = None,
    seed: int = 1234,
    **overrides: Any,
) -> LLMClient:
    """Build a client for ``mode`` in ``{"auto", "offline", "live", "null"}``.

    ``auto`` probes the configured endpoint once and silently falls back to the
    offline simulator, so the pipeline always runs.
    """
    mode = (mode or "auto").lower()
    if mode == "null":
        return NullLLMClient()

    if mode in {"offline", "mock", "simulated"}:
        if tasks is None:
            raise ValueError("offline mode needs the task suite for the simulator")
        client: LLMClient = MockLLMClient(tasks, seed=seed)
    elif mode in {"live", "remote", "openai"}:
        client = OpenAICompatibleClient.from_env(**overrides)
    elif mode == "auto":
        remote = OpenAICompatibleClient.from_env(**overrides)
        healthy, detail = remote.health()
        if healthy:
            client = remote
        else:
            if tasks is None:
                raise ValueError("auto mode fell back to offline but no task suite was given")
            client = MockLLMClient(tasks, seed=seed)
            client.name = f"mock-offline (no endpoint: {detail})"
    else:
        raise ValueError(f"unknown LLM mode {mode!r}; expected auto|offline|live|null")

    if cache_path:
        return CachingLLMClient(client, cache_path)
    return client


def describe_client(client: LLMClient) -> dict[str, Any]:
    """Audit-friendly description of a client (used in run reports)."""
    data = client.counters()
    data["offline_simulator"] = "mock" in client.name
    return data


__all__ = [
    "CachingLLMClient",
    "GenerationRequest",
    "GenerationResult",
    "LLMClient",
    "LLMError",
    "MockLLMClient",
    "NullLLMClient",
    "OpenAICompatibleClient",
    "SYSTEM_PROMPT",
    "describe_client",
    "extract_code",
    "make_client",
    "render_messages",
]
