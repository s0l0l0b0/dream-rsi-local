# Dream-RSI (local reproduction)

A dependency-free, offline-runnable reproduction of **Dream-RSI: Recursive
Self-Improvement through Evolving Worlds**.

The idea in one paragraph: the model's weights never change. What changes is the
*executable search policy* — the Python code that decides which strategy to try
next. An agent explores benchmark tasks for real, every attempt is recorded in a
persistent **Discovery Tree**, and an offline **Dream Engine** then replays those
trees to score hundreds of candidate policies at **zero inference cost**. A
meta-optimizer proposes mutations of `src/policy/search_policy.py`, keeps only
those that clear a paired statistical gate on held-out tasks, and promotes the
winners into the live policy.

```
        explore (real inference)              dream (zero inference)
   ┌──────────────────────────────┐      ┌──────────────────────────────┐
   │  src/engine/explorer.py      │      │  src/engine/dreamer.py       │
   │  policy → model → evaluator  │─────▶│  replay recorded rewards     │
   │  records full candidate      │      │  score ~150 candidates in    │
   │  slates with measured scores │      │  one subprocess, no model    │
   └──────────────┬───────────────┘      └──────────────┬───────────────┘
                  │ storage/trees/*.json                │
                  ▼                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │  src/meta/optimizer.py                                │
        │  profile → mutate → score on train → verify on holdout │
        │  → promote into src/policy/search_policy.py (archived) │
        └───────────────────────────────────────────────────────┘
```

---

## Quickstart

Runtime is **standard library only** (Python ≥ 3.10). No install needed:

```bash
uv sync --extra dev          # creates .venv + uv.lock, installs pytest

# Full RSI cycle, offline, from the shipped untuned policy_v0
uv run python run_rsi_loop.py --generations 3 --steps 4 --slate 3 \
    --search-rounds 3 --no-stop-on-solve --probe-steps 2 --probe-slate 3

# ~20-second sanity run used by CI; it evolves a copy inside the storage
# directory, so it never rewrites the tracked policy file
uv run dream-rsi --smoke

# Tests (pytest is the only dev dependency; uv manages the environment)
uv sync --extra dev
uv run pytest tests/
```

`--mode auto` (the default) probes for an OpenAI-compatible endpoint and falls
back to the offline simulator, so the pipeline always runs. To use a real model:

```bash
export DREAM_RSI_BASE_URL=http://localhost:11434/v1   # Ollama / vLLM / dsh / OpenAI
export DREAM_RSI_MODEL=qwen2.5-coder:7b
export DREAM_RSI_API_KEY=...                          # if the endpoint needs one
python run_rsi_loop.py --mode live --generations 3
```

With a real endpoint the meta-optimizer also uses the **LLM mutation route**: it
is shown the profiler report, the current policy source and the mutation
contract, and asked to return an improved file. Offline, that route reports
"offline simulator cannot author policy code" and the deterministic
profiler-guided parameter search is used instead.

---

## Running against a local model (measured)

Verified against **Ollama 0.35.1 + `gemma4:latest`** (7.5B, Q4_K_M). Measured facts,
because they change how you budget a run:

* **~30-45 s per call**, and gemma4 spends **~840 completion tokens on a single
  small function** (it runs a reasoning pass). Do **not** cap `max_tokens` low:
  at 512 the answer was truncated mid-function and the evaluator scored it `0.0`,
  while 1024 finished it and scored `1.0`.
* Budget is `tasks x steps x slate` calls per split per generation. At 40 s/call,
  6 calls is ~2 minutes; the offline reference run's ~1500 calls would be ~17 hours.
* **Plan for `min_holdout_decisions` (default 6).** The gate refuses to promote on
  fewer holdout decisions, so `tasks x steps` for the holdout split should exceed
  ~6 before a live run can promote anything.
* At the end of the run, the exploration is thrown away and replayed for free --
  that asymmetry is the whole point, and it is very visible in live mode.

```bash
export DREAM_RSI_BASE_URL=http://127.0.0.1:11434/v1
export DREAM_RSI_MODEL=gemma4:latest
export DREAM_RSI_TIMEOUT_S=180          # slow local models need headroom
export DREAM_RSI_MAX_TOKENS=1024        # gemma4 needs >=1024 to finish a function
export DREAM_RSI_EXTRA_BODY='{"keep_alive":"30m"}'   # endpoint-specific extras

uv run python run_rsi_loop.py --mode live --suite humaneval --max-tasks 2 \
    --steps 1 --slate 2 --generations 1 --candidates 16 --search-rounds 1
```

`DREAM_RSI_EXTRA_BODY` is JSON merged into every request body, so endpoint-specific
switches stay out of the code: `{"keep_alive":"30m"}` keeps a local model resident
(avoiding reloads between sparse calls), `{"think":false}` asks a reasoning model to
skip its reasoning pass where the server supports it.

A measured 2-task run: 6 model calls, 128.6 s of inference, then 16 candidate
policies replayed for **0 s of inference** -- and **no promotion**, because gemma4
scored `1.000` on every task and the policy had nothing left to improve. That is an
expected outcome, not a failure: to *see* the loop improve a policy you need tasks
the model actually fails (try `--suite math` or `--suite kernelbench`) plus enough
holdout decisions for the gate to have something to measure.

---

## What one generation does

| Step | Where | Cost |
| --- | --- | --- |
| 1. Explore both splits with the current policy, in a fresh world | `src/engine/explorer.py` | real model calls |
| 2. Profile the accumulated trees: which strategy pays off, per suite, per state | `src/meta/optimizer.py` (`FailureProfiler`) | free |
| 3. Propose ~160 candidate policies (LLM rewrite and/or parameter search) | `src/meta/optimizer.py` | free / 1 call |
| 4. Replay every candidate in the Dream Engine, rank on **train** trees | `src/engine/dreamer.py` | **zero** |
| 5. Verify the champion on **held-out** trees with a paired test; promote or reject | `src/meta/optimizer.py` (`gate`) | **zero** |

Then the loop repeats. Because each generation re-explores in a *fresh world*
(the simulator's sampling seed changes, as a real model's sampling would), the
held-out evidence **accumulates**: a genuine improvement is confirmed by more
decisions over time, a lucky draw is not.

### Why the offline comparison is valid

At every decision point the explorer evaluates the **whole candidate slate** and
records the measured reward of *each* option. Replay therefore re-asks a
fully-observed question — "given these options and these measured rewards, what
would this policy have picked?" — with no importance-sampling correction needed.
All policies are replayed over identical decision points, in identical order,
and the replay also feeds each policy the same `observe()` calls it had live, so a
*learning* policy is judged fairly rather than as if it had amnesia.

Replay is structurally incapable of calling a model: the Dream Engine holds a
`NullLLMClient` that raises, and every report records `llm_calls == 0`.

### What replay cannot do (please read before quoting numbers)

* It cannot score an action **absent from the recorded slate**. Slates are frozen
  from the recording, so a policy that wants an untried action is scored 0 for
  that pick; `coverage` and `off_slate_picks` quantify it.
* It cannot evaluate a changed **candidate-proposal** distribution. That is why
  the framework — not the evolving policy — owns proposal (`ActionSpace.slate`,
  a constant-salted coverage rotation), keeping the world fixed across
  generations so only the decision rule changes.
* It is only meaningful against a **frozen oracle**. Every tree stores
  `metadata["evaluator_sha256"]`; replay and the optimizer refuse to mix trees
  graded by a different evaluator or task suite.

---

## Reference run (reproducible)

```bash
uv run python run_rsi_loop.py --generations 3 --steps 4 --slate 3 --candidates 200 \
    --search-rounds 3 --no-stop-on-solve --probe-steps 2 --probe-slate 3
```

32 tasks · 144 tests · 3 suites · 96 discovery trees · 384 recorded decision points
(64 exact duplicate decision points collapsed, so significance tests never count the
same observation twice).

| gen | policy | champion train | champion holdout | paired holdout Δ | verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | `policy_v0` | 0.5881 | 0.3844 | −0.0052 (z −0.08) | rejected |
| 1 | `policy_v2` | 0.5585 | 0.4500 | **+0.0621 (z 1.17, p 0.120, 40/95 decisions changed)** | **PROMOTED** |
| 2 | `policy_v2` | 0.5330 | 0.4619 | 0.0000 | rejected (no candidate beat train) |

Final Dream-Engine scores over the whole corpus (higher is better):

| policy | mean reward | efficiency |
| --- | ---: | ---: |
| `policy_v2` (evolved) | 0.4957 | 0.660 |
| `policy_v0` (untuned baseline) | 0.4528 | 0.603 |
| uniform-random (5-seed mean) | 0.4641 | 0.618 |
| global-best-action baseline | 0.5148 | 0.686 |
| oracle (best recorded option per decision) | 0.7508 | 1.000 |

And the live paired A/B at a fixed tight budget (2 decision points × 3 options per
task, 12 held-out tasks, same world and same slates for both policies):

| policy | solve rate | mean best score |
| --- | ---: | ---: |
| `policy_v0` (initial) | 0.750 | 0.817 |
| `policy_v2` (evolved) | 0.917 | 0.933 |
| paired per-task difference | **+0.1167, z = 1.40** | |

Two independent runs with the same seed produce **identical** results — same
promotions, same paired deltas and z-scores, same final policy — because policy
decisions never depend on Python's randomised `hash()` and the corpus order is
derived from task/world/phase rather than from random tree ids.

### Honest reading of those numbers

* The evolved policy clearly beats the policy it started from, both offline
  (`0.453 → 0.496`) and live at a tight budget (`0.750 → 0.917` solve rate).
* It does **not** beat the purpose-built `global_best` baseline offline
  (`0.496 < 0.515`). The meta-optimizer is a greedy search over a 47-parameter
  space, not an oracle; ~34 % of the oracle gap remains.
* The gate is used to *decide* promotion, so a reported holdout gain is
  "validated by a gate", not "measured on an untouched test set". The live A/B
  probe is the closest thing here to an unbiased operational measurement.
* With a generous budget the live **solve rate saturates at 1.0**, so decision
  quality (mean measured reward) is the discriminating metric, not solve rate.

---

## The offline simulator is a test double, not a model

With no endpoint configured, `MockLLMClient` fabricates artefacts from the
benchmark's reference solution, with a success probability that depends on the
strategy, the suite and the search state (`_BASE_SUCCESS`, plus bonuses for
`repair` when a partial artefact exists and `restart` when the search has
stalled). Two consequences you should keep in mind:

1. **Every reward is still real.** The simulator only produces *code*; the score
   comes from the frozen sandboxed evaluator running the real tests. Mutations
   are applied at the AST level and verified to change the program, and a
   deliberate probe shows ~4.6 % of mutated artefacts still pass every test
   (behaviourally equivalent mutants) — those are genuine passes.
2. **It reads reference solutions a real model would not have.** Offline numbers
   therefore demonstrate that the *RSI machinery* works end to end; they are not
   evidence about any real model's behaviour, and the embedded structure
   (suite-level strategy competence, repair-when-partial) is learnable by
   construction. Reproduce the pipeline claim with `--mode live` on your own
   endpoint before drawing scientific conclusions.

---

## Repository map

```text
├── benchmarks/
│   ├── tasks.jsonl           32 deterministic tasks (humaneval 13, math 11, kernelbench 8), 144 tests
│   └── evaluator.py          IMMUTABLE sandboxed grader → score + diagnostics
├── storage/
│   ├── trees/                discovery trees (runtime data, gitignored)
│   └── policies/             policy_vN/ checkpoints, candidates/, registry.json
├── src/
│   ├── core/
│   │   ├── tree.py           Action, Edge, Node, DiscoveryTree, DecisionRecord
│   │   ├── llm_client.py     OpenAI-compatible client, offline simulator, cache, null client
│   │   └── utils.py          shared helpers (atomic JSON, stable hashing, stats)
│   ├── policy/
│   │   ├── base.py           frozen action space + ExplorationPolicy SDK + PolicyContext
│   │   └── search_policy.py  THE EVOLVING TARGET (policy_v0 ships untuned)
│   ├── engine/
│   │   ├── explorer.py       live tree search: policy → model → evaluator → tree
│   │   └── dreamer.py        offline replay + diagnostic baselines
│   └── meta/
│       ├── optimizer.py      profiler, mutation, paired gate, promotion, archive
│       └── policy_runner.py  untrusted candidate loading + batch replay (subprocess)
├── tests/                    169 tests: unit, sandbox-isolation, sandboxed timeouts, integration
├── pyproject.toml            stdlib runtime; pytest as the only dev extra
└── run_rsi_loop.py           full RSI cycle CLI
```

### Immutability of the grader

`benchmarks/evaluator.py` is the only component allowed to turn an artefact into a
reward, so it is frozen and fingerprinted. Candidate code never runs in the host
process: each evaluation gets a fresh `python -I -B` child with a scratch cwd, a
scrubbed environment (no credentials inherited), an AST gate (import allowlist,
no dunder escapes, no `eval`/`exec`/`open`), POSIX resource limits, and a
wall-clock timeout with process-group kill. This stops accidents and casual
escapes; it is not a VM — run genuinely untrusted code in a container.

### The policy mutation contract

`src/meta/optimizer.py` may rewrite **only** `src/policy/search_policy.py`, and
only within these rules (enforced on every candidate):

* `POLICY_ID` stays a string literal, `PARAMS` a flat `dict[str, float]` literal
  with one entry per line — the offline mutator rewrites *just that block*, so a
  parameter-only change leaves the rest of the file byte-identical and reviewable
  with `git diff`;
* `param_space()` returns `(low, high)` bounds for every `PARAMS` key;
* `SearchPolicy.select(ctx)` returns an `Action` drawn from `ctx.candidates`;
* imports are limited to a safe allowlist; no file, network, process or clock
  access; `select` must be deterministic for identical input.

`PolicyContext` carries **no reward information at all** — that is an invariant,
not a convention, and it is what stops a replayed policy from reading the answer
key. Candidates that violate the contract, hang, or fail to import are rejected
with a recorded reason, never silently.

Every proposal is archived (`storage/policies/candidates/gen_NNN/`) and every
checkpoint accepts *or rejects* with its measured metrics
(`storage/policies/policy_vN/metadata.json`, `registry.json`), so the whole
history is auditable — including the failures.

---

## CLI reference

```
--mode {auto,offline,live}   auto probes the endpoint, else offline simulator
--generations N              RSI generations (default 3)
--steps N                    decision points per task per generation (default 5)
--slate N                    candidate actions per decision point (default 4)
--no-stop-on-solve           keep exploring after a task passes: more model calls,
                             a much richer replay corpus (this is the central
                             exploration-cost / offline-quality trade-off)
--candidates N               policy candidates proposed per generation (default 200)
--search-rounds N            greedy refinement rounds per generation
--train-ratio F              fraction of tasks used to fit (rest gates promotion)
--probe-steps/--probe-slate  budget of the final paired live A/B probe
--no-final-probe             skip that probe
--wall-budget SECONDS        stop starting generations after N seconds (a report is
                             still written; reports are also written after every
                             generation, so an interrupted run keeps its results)
--from-policy policy_vN      restore an archived checkpoint first (reproducible runs)
--policy-path PATH           evolve a different policy file than the tracked one
--storage / --reports        output locations
--seed N                     master seed
--clean                      delete previously generated trees/reports (not policies)
--smoke                      tiny fast configuration for CI (non-destructive: it
                             evolves a copy under --storage, not the tracked file)
```

Promotion deliberately rewrites `src/policy/search_policy.py` — that *is* the
evolving artefact, and its history lives in `storage/policies/`. The repository
ships that file at the untuned `policy_v0`; use `--from-policy policy_v0` (or
`--policy-path`) to start a run from a known checkpoint.

---

## Testing

169 tests, ~8 s total, no network:

```bash
uv run pytest tests/test_tree.py tests/test_dreamer.py tests/test_policy.py \
    tests/test_llm_client.py tests/test_evaluator.py        # 97
uv run pytest tests/test_policy_runner.py tests/test_explorer.py \
    tests/test_optimizer.py                                 # 61
uv run pytest tests/test_rsi_integration.py                 # 11 (marked "integration")
```

What is covered: tree/decision-record invariants (including that the policy input
contains no rewards), replay arithmetic against hand-computed expectations,
evaluator sandbox behaviour (partial credit, static gate, timeout kill),
determinism of the mock and the split, every branch of the promotion gate,
paired statistics, the parameter-only source mutation, the policy contract, and
an end-to-end cycle on a hand-built corpus where a promotion is guaranteed.

Not covered, deliberately: `OpenAICompatibleClient` (needs a live endpoint), the
LLM policy-mutation route end to end, and POSIX-specific resource limits beyond
the timeout path. A few bugs found while building the suite are pinned by
regression tests, notably:

* `GenerationRequest.cache_key` used to omit the rendered history (two different
  prompts shared one cache key);
* `load_policy_module` used to return **stale bytecode**: CPython validates
  `__pycache__` with second-truncated mtimes, so promoting a policy within the
  same second it was last loaded — with an unchanged file size, e.g. `0.0 → 0.5`
  — made the live loop keep running the old policy while the Dream Engine saw the
  new one. Loading now compiles from source in memory.

---

## Extending this

* **Multi-step (path) replay.** Trees already store *every* evaluated branch with
  its score, so a replay engine could score whole decision *sequences*, not just
  one-step choices. `Dreamer` currently replays single decision points.
* **Richer offline evaluation.** Off-policy estimators (IPS/doubly robust) would
  let replay score actions that were never tried, removing the coverage limit —
  at the cost of variance.
* **Program-space search.** The offline mutator tunes parameters; the LLM route
  can restructure the policy. A middle ground (a library of policy *templates*
  scored by the Dream Engine) is a natural next step.

## License

Apache-2.0 (see `pyproject.toml`).
