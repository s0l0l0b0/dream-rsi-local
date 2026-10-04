# AGENTS.md

## 1. Project Mission & Identity
This project is an open-source, local reproduction of **Dream-RSI: Recursive Self-Improvement through Evolving Worlds** (Zheng et al., 2026).
The goal is to build an autonomous coding exploration system where:
1. Model weights are **frozen / unmodified**.
2. Real-world exploration traces are logged into persistent **Discovery Trees**.
3. A **Replay Simulator ("Dream Engine")** replays past exploration trees offline with zero additional LLM inference costs.
4. A **Meta-Optimizer Agent** evolves the executable Python search policy (`src/policy/search_policy.py`) against the Dream Engine and verifies improvements on holdout benchmarks.

---

## 2. Target Repository Architecture
Construct the codebase following this exact module structure:

```text
├── benchmarks/              # Deterministic task suites (HumanEval / Kernel-bench / math)
│   ├── tasks.jsonl          # Task definitions (prompt, signature, validation test)
│   └── evaluator.py         # Sandboxed test runner (IMMUTABLE, returns score & diagnostics)
├── storage/
│   ├── trees/               # Serialized discovery trees (.json or SQLite)
│   └── policies/            # Archive of evolved policy checkpoints (policy_v0, policy_v1...)
├── src/
│   ├── core/
│   │   ├── tree.py          # Node, Edge, and DiscoveryTree data structures
│   │   └── llm_client.py    # OpenAI-compatible client (routed via dsh / Ollama / vLLM)
│   ├── policy/
│   │   ├── base.py          # Base abstract class for ExplorationPolicy
│   │   └── search_policy.py # THE EVOLVING TARGET: active search heuristics and logic
│   ├── engine/
│   │   ├── explorer.py      # Real-world tree search runner using search_policy.py
│   │   └── dreamer.py       # Offline replay engine (evaluates candidate policies on saved trees)
│   └── meta/
│       └── optimizer.py     # RSI loop: profiles past runs, asks LLM to mutate search_policy.py
├── tests/                   # Unit & integration tests for all engine components
├── pyproject.toml           # Poetry / uv configuration
└── run_rsi_loop.py          # Main entry point running the full RSI cycle
