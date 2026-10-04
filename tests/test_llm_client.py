"""Tests for the solver clients in ``src/core/llm_client.py``.

Everything here is offline: the module ships a deterministic simulator
(:class:`MockLLMClient`) precisely so the pipeline can be exercised with no
endpoint and no network.
"""

from __future__ import annotations

import ast
import random
from typing import Any, Mapping

import pytest

from benchmarks.evaluator import load_tasks
from src.core.llm_client import (
    CachingLLMClient,
    OpenAICompatibleClient,
    GenerationRequest,
    GenerationResult,
    LLMClient,
    LLMError,
    MockLLMClient,
    NullLLMClient,
    SYSTEM_PROMPT,
    extract_code,
    make_client,
    render_messages,
)
from src.core.tree import Action
from src.policy.base import ACTION_SPECS

SHIPPED_TASKS = load_tasks()


@pytest.fixture(scope="module")
def index() -> dict[str, dict[str, Any]]:
    return {str(task["id"]): task for task in SHIPPED_TASKS}


@pytest.fixture(scope="module")
def mock(index: Mapping[str, Mapping[str, Any]]) -> MockLLMClient:
    return MockLLMClient(index)


def make_action(name: str = "direct", temperature: float = 0.2) -> Action:
    return Action(name=name, params={"temperature": temperature})


def make_request(task: Mapping[str, Any], *, action: Action | None = None, suite: str | None = None, **kwargs: Any) -> GenerationRequest:
    data: dict[str, Any] = {
        "task_id": str(task["id"]),
        "suite": suite if suite is not None else str(task.get("suite", "")),
        "entry_point": str(task["entry_point"]),
        "prompt": str(task["prompt"]),
        "signature": str(task.get("signature", "")),
        "action": action or make_action(),
    }
    data.update(kwargs)
    return GenerationRequest(**data)


# ======================================================================================
# extract_code
# ======================================================================================
def test_extract_code_reads_a_fenced_block() -> None:
    assert extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
    assert extract_code("```\ndef f():\n    return 1\n```") == "def f():\n    return 1"


def test_extract_code_reads_a_fenced_block_among_prose() -> None:
    text = "Sure, here it is:\n\n```python\ndef g(x):\n    return x + 1\n```\n\nHope that helps."
    assert extract_code(text, "g") == "def g(x):\n    return x + 1"
    assert extract_code(text) == "def g(x):\n    return x + 1"


def test_extract_code_without_a_fence_but_with_a_def() -> None:
    code = "def h(a):\n    return a * 2\n"
    assert extract_code(code, "h") == code.strip()
    assert extract_code(code) == code.strip()
    assert extract_code("def h(a):\n    return a\n", "other") == ""
    assert extract_code("Just an explanation, no code.") == ""
    assert extract_code("Just an explanation, no code.", "f") == ""


def test_extract_code_empty_string() -> None:
    for text in ("", "   ", "\n\t  \n"):
        assert extract_code(text) == ""
        assert extract_code(text, "f") == ""


def test_extract_code_prefers_the_fence_with_the_entry_point() -> None:
    text = (
        "```python\ndef alpha(x):\n    return x\n```\n"
        "middle prose\n"
        "```python\ndef beta(y):\n    return y\n```\n"
    )
    assert extract_code(text, "beta") == "def beta(y):\n    return y"
    # No block defines the requested entry point -> fall back to the first fence.
    assert extract_code(text, "gamma") == "def alpha(x):\n    return x"
    assert extract_code(text) == "def alpha(x):\n    return x"


# ======================================================================================
# render_messages
# ======================================================================================
def test_render_messages_includes_prompt_signature_strategy_and_attempt(
    index: Mapping[str, Mapping[str, Any]],
) -> None:
    task = index["he_001"]
    action = make_action("repair", 0.1)
    solution = "def sum_even(nums):\n    return 0\n"
    request = make_request(
        task, action=action, current_state={"solution": solution}, attempt=2
    )

    messages = render_messages(request)
    assert request.messages() == messages
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(messages) == 2 and messages[1]["role"] == "user"
    content = messages[1]["content"]

    assert str(task["prompt"]).strip() in content
    assert str(task["signature"]) in content
    assert action.name in content
    assert ACTION_SPECS[action.name].prompt_hint in content
    assert solution.strip() in content
    assert "CURRENT BEST ATTEMPT" in content
    assert "temperature=0.1" in content

    history_request = make_request(
        task,
        history=[{"depth": 1, "action": "direct", "score": 0.5, "feedback": "1/2 failed"}],
    )
    history_content = render_messages(history_request)[1]["content"]
    assert "ATTEMPTS SO FAR" in history_content
    assert "direct" in history_content and "0.50" in history_content

    # With no attempt recorded the section is omitted entirely.
    assert "CURRENT BEST ATTEMPT" not in render_messages(make_request(task))[1]["content"]


# ======================================================================================
# GenerationRequest.cache_key
# ======================================================================================
def test_cache_key_is_stable_and_sensitive_to_the_physical_request(
    index: Mapping[str, Mapping[str, Any]],
) -> None:
    task = index["he_001"]
    base = make_request(task)
    assert base.cache_key == make_request(task).cache_key
    assert len(base.cache_key) == 24

    parent = make_request(task, current_state={"solution": "def sum_even(nums):\n    return 0\n"})
    other_action = make_request(task, action=make_action("chain_of_thought", 0.3))
    other_temperature = make_request(task, action=make_action("direct", 0.9))

    assert parent.cache_key != base.cache_key, "parent solution must change the key"
    assert other_action.cache_key != base.cache_key, "action must change the key"
    assert other_temperature.cache_key != base.cache_key, "temperature must change the key"
    assert other_action.cache_key != other_temperature.cache_key


def test_cache_key_identifies_the_rendered_history(
    index: Mapping[str, Mapping[str, Any]],
) -> None:
    task = index["he_001"]
    base = make_request(
        task, attempt=1, history=[{"depth": 0, "action": "root", "score": 0.0, "feedback": ""}]
    )
    other = make_request(
        task, attempt=1, history=[{"depth": 0, "action": "direct", "score": 0.5, "feedback": "1/2"}]
    )

    assert render_messages(base) != render_messages(other), "history must reach the prompt"
    assert base.cache_key != other.cache_key, (
        "cache_key claims to identify the physical request (prompt + sampling params) "
        "but ignores rendered history"
    )


# ======================================================================================
# MockLLMClient
# ======================================================================================
def test_mock_generate_is_deterministic_and_parses(
    mock: MockLLMClient, index: Mapping[str, Mapping[str, Any]]
) -> None:
    task = index["he_001"]
    request = make_request(task, attempt=0)

    first = mock.generate(request)
    second = mock.generate(request)

    assert first.error == "" and second.error == ""
    assert first.code == second.code
    assert first.code.strip(), "the simulator returned no code"
    ast.parse(first.code)  # raises SyntaxError on failure
    assert extract_code(first.text, task["entry_point"]) == first.code
    assert first.raw.get("simulated") is True


def test_mock_success_probability_is_state_and_suite_dependent(
    mock: MockLLMClient, index: Mapping[str, Mapping[str, Any]]
) -> None:
    task = index["he_001"]

    cold = make_request(task, action=make_action("repair", 0.1), best_score=0.0)
    warm = make_request(task, action=make_action("repair", 0.1), best_score=0.6)
    assert mock.success_probability(warm) > mock.success_probability(cold), (
        "repair must be likelier when there is a partial artefact to repair"
    )

    kernelbench = make_request(task, action=make_action("optimize", 0.15), suite="kernelbench")
    math = make_request(task, action=make_action("optimize", 0.15), suite="math")
    assert mock.success_probability(kernelbench) > mock.success_probability(math), (
        "optimize must be suite-dependent (kernelbench > math)"
    )

    for request in (cold, warm, kernelbench, math):
        probability = mock.success_probability(request)
        assert 0.0 <= probability <= 1.0


def test_mock_rejects_an_unknown_task_id(mock: MockLLMClient) -> None:
    request = make_request(
        {"id": "does_not_exist", "suite": "humaneval", "entry_point": "f", "prompt": "p"}
    )
    result = mock.generate(request)
    assert result.error, "an unknown task must produce an error result, not code"
    assert "does_not_exist" in result.error
    assert result.code == ""


def test_mutator_never_returns_the_reference_unchanged() -> None:
    mutator = MockLLMClient({})
    seeds = (0, 1, 2, 7, 13)

    for task in SHIPPED_TASKS:
        reference = str(task["reference_solution"])
        reference_tree = ast.parse(reference)
        for seed in seeds:
            mutated = mutator._mutate(
                reference, random.Random(seed), n=1, entry_point=str(task["entry_point"])
            )
            assert mutated != reference, f"{task['id']} seed={seed}: mutation was a no-op"
            try:
                mutated_tree = ast.parse(mutated)
            except SyntaxError as exc:  # pragma: no cover - failure path
                pytest.fail(f"{task['id']} seed={seed}: mutation is not valid Python: {exc}")
            assert ast.dump(mutated_tree) != ast.dump(reference_tree), (
                f"{task['id']} seed={seed}: syntax tree is unchanged"
            )


# ======================================================================================
# CachingLLMClient
# ======================================================================================
class FakeClient(LLMClient):
    """Counts real ``generate`` calls and returns a canned artefact."""

    name = "fake"

    def __init__(self, code: str = "def f():\n    return 1\n") -> None:
        super().__init__()
        self.code = code

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            text=f"```python\n{self.code}```",
            code=self.code,
            model=self.name,
            prompt_tokens=11,
            completion_tokens=7,
        )


def test_caching_client_reports_hits_misses_and_real_calls(
    index: Mapping[str, Mapping[str, Any]],
) -> None:
    inner = FakeClient()
    cache = CachingLLMClient(inner, namespace="world_0")
    request = make_request(index["he_001"])

    first = cache.generate(request)
    assert first.error == "" and first.cached is False
    assert inner.calls == 1
    assert cache.calls == 1, "a miss is a real call"
    assert (cache.hits, cache.misses) == (0, 1)

    second = cache.generate(request)
    assert second.cached is True
    assert second.code == first.code
    assert inner.calls == 1, "a cache hit must not reach the inner client"
    assert cache.calls == 1, "a cache hit must not be counted as a real call"
    assert (cache.hits, cache.misses) == (1, 1)

    counters = cache.counters()
    assert counters["hits"] == 1 and counters["misses"] == 1
    assert counters["inner"]["calls"] == 1
    assert counters["cache_entries"] == 1

    other = cache.generate(make_request(index["he_002"]))
    assert other.cached is False
    assert inner.calls == 2 and cache.calls == 2 and cache.misses == 2


def test_caching_namespaces_do_not_share_entries(
    index: Mapping[str, Mapping[str, Any]],
) -> None:
    inner = FakeClient()
    request = make_request(index["he_001"])
    first = CachingLLMClient(inner, namespace="generation_a")
    second = CachingLLMClient(inner, namespace="generation_b")

    assert first.generate(request).cached is False
    assert second.generate(request).cached is False
    assert inner.calls == 2, "different namespaces must not share cache entries"

    # ... while each namespace still caches its own repeats.
    assert first.generate(request).cached is True
    assert second.generate(request).cached is True
    assert inner.calls == 2


# ======================================================================================
# Null client and factory
# ======================================================================================
def test_null_client_refuses_to_generate(index: Mapping[str, Mapping[str, Any]]) -> None:
    client = NullLLMClient()
    with pytest.raises(LLMError):
        client.generate(make_request(index["he_001"]))
    assert client.calls == 0


def test_make_client_null_and_offline(index: Mapping[str, Mapping[str, Any]]) -> None:
    assert isinstance(make_client("null"), NullLLMClient)

    offline = make_client("offline", tasks=index)
    assert isinstance(offline, MockLLMClient)
    assert "mock" in offline.name.lower(), offline.name

    with pytest.raises(ValueError):
        make_client("offline")  # the simulator needs the task suite


# ======================================================================================
# extra_body passthrough (endpoint-specific switches)
# ======================================================================================
def test_extra_body_is_merged_into_the_request_payload():
    client = OpenAICompatibleClient(model="m", extra_body={"think": False, "keep_alive": "30m"})
    payload = client._payload([{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=64)
    assert payload["think"] is False
    assert payload["keep_alive"] == "30m"
    assert payload["max_tokens"] == 64 and payload["stream"] is False


def test_extra_body_can_override_client_defaults():
    """Applied last on purpose, so an operator can win against any client default."""
    client = OpenAICompatibleClient(model="m", max_tokens=1024, extra_body={"max_tokens": 7})
    payload = client._payload([{"role": "user", "content": "hi"}], temperature=0.2)
    assert payload["max_tokens"] == 7


def test_from_env_parses_extra_body_json_and_rejects_junk():
    env = {
        "DREAM_RSI_BASE_URL": "http://127.0.0.1:11434/v1",
        "DREAM_RSI_MODEL": "gemma4:latest",
        "DREAM_RSI_EXTRA_BODY": '{"think": false}',
    }
    client = OpenAICompatibleClient.from_env(env)
    assert client.extra_body == {"think": False}
    assert client.base_url == "http://127.0.0.1:11434/v1"

    with pytest.raises(ValueError, match="not valid JSON"):
        OpenAICompatibleClient.from_env({**env, "DREAM_RSI_EXTRA_BODY": "{oops"})
    with pytest.raises(ValueError, match="must be a JSON object"):
        OpenAICompatibleClient.from_env({**env, "DREAM_RSI_EXTRA_BODY": "[1, 2]"})

