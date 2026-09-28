# HalluWorld-Chess

Engineering reference for the chess track: the probe battery, the determinism contract, the
configuration surface, and the bugs that are fixed and must stay fixed.

Sourced from the working notes that were the only record of any of this. Paths have been updated
for the current layout; `examples/chess_full_probes.py` is now
`halluworld/tracks/chess/battery.py`, and the environment variables it documents are now expressed
as YAML under `halluworld/data/configs/chess/` (they still work as environment variables -- the
config layer exports them).

### Layout

```
halluworld/
  benchmark.py                 # run_benchmark, SYSTEM_PROMPT, BenchmarkRun, EpisodeResult
  lm/                          # OpenAI, Anthropic, Baseten, xAI clients
  tracks/chess/
    battery.py                 # the probe battery (python -m halluworld.tracks.chess.battery)
    replay.py                  # scores a model against the frozen bank (halluworld eval chess)
    question_set.py            # exports a question set under a stub model (bank building)
    probes/                    # standard, old_bishop, atomic
    evaluators.py              # ChessYesNo / ChessInteger / ChessLegalUciSet / ChessPieceName
    serializers.py             # ChessSerializer, FenDisplayConfig, render_full, render_history
    envs/, rules/              # chess environments and variant rules
    description.md
  data/configs/chess/          # fen_off / fen_on / fen_transpose run configs
scripts/
  run_chess_benchmark.sh       # orchestration: models x FEN modes x replicates
results/chess/<RESULTS_SUBDIR>/...
```

### Probe battery and taxonomy

Seven active probes, 50 episodes each (`chess_hidden_side_capture_stats` yields 48 after skips; `ALL` n = 348).

| Probe | Class | Cognitive demand |
|---|---|---|
| `chess_can_capture` | P — Perceptual | One-move capture legality on the visible board |
| `chess_defended` | P | Defender/attack graph on the visible board |
| `chess_hanging` | P | Attacked-and-undefended on the visible board |
| `chess_hypothetical_in_check` | C — Causal | Apply one legal move, read a rule-defined property |
| `chess_after_move_undefended_count` | C | 1–2 ply rollout, then count a derived predicate |
| `chess_hidden_side_capture_stats` | M — Memory | Count capture events over a long explicit move list |
| `chess_san_legal_move` | X — Compound | SAN-only history (40+ plies, no board): maintain state (M) + apply rules (C) + emit a legal UCI move (V) |

Notes:
- The M/X split is newer than the original P/C/M grouping. Older docs put both long-sequence probes under M; the current convention is M = `hidden_side_capture_stats`, X = `san_legal_move`.
- No primary U (uncertainty) probe exists in this battery.
- `chess_two_step_hanging` and `chess_mate_stalemate_confusion` are present but commented out in `build_probes_and_evaluators()`.

### Determinism contract

Every stochastic component is explicitly seeded. Do not remove these.

- `benchmark_seed = 42` (passed to `run_benchmark`); env constructed with `seed=42`; Lichess FEN load `seed=42`.
- Explicit probe RNGs in `build_probes_and_evaluators()`:

| Probe | Seed |
|---|---|
| `ChessCanCaptureProbe` | 101 |
| `ChessDefendedProbe` | 102 |
| `ChessHangingProbe` | 103 |
| `ChessHypotheticalProbe` | 104 |
| `ChessSanLegalContinuationProbe` | 105 |
| `ChessHiddenSideCaptureStatsProbe` | 501 |
| `ChessAfterMoveUndefendedCountProbe` | 502 |

Historical bug: these probes originally defaulted to bare `random.Random()`, which seeds from system entropy, so two runs with identical env seeds produced *different questions*. If exported question sets stop matching across runs, suspect a newly added unseeded probe.

### Running the battery

Canonical single run (this is the parameter set behind the reported results):

```bash
CHESS_EXTREME=1 \
HARD_HIDDEN_MIN_PLIES=70 HARD_HIDDEN_MAX_PLIES=120 HARD_HIDDEN_CAPTURE_BIAS=0.62 \
USE_LICHESS=1 N_EPISODES=50 BENCHMARK_VERBOSE=1 \
INCLUDE_FEN=0 \
OPENAI_MODEL=o3 OPENAI_REASONING_EFFORT=medium OPENAI_MAX_COMPLETION_TOKENS=16384 \
python3 halluworld/tracks/chess/battery.py
```

The three observation conditions:

```bash
INCLUDE_FEN=0                                                   # fen-off  ("No FEN")
INCLUDE_FEN=1                                                   # fen-on
INCLUDE_FEN=1 OBSERVATION_FEN_MODE=transpose OBSERVATION_FEN_SEED=7   # fen-transpose ("Incorrect FEN")
```

Orchestrator (`scripts/run_chess_benchmark.sh`) runs models in parallel background jobs, FEN modes sequentially within each model, and supports temperature replicates:

```bash
TEMPERATURE=0.7 RUN_REPLICATES=5 RESULTS_SUBDIR=chess_temp_07 bash scripts/run_chess_benchmark.sh
```

It writes each replicate to `rep-NN/` and then aggregates mean and sample std (ddof=1) of accuracy and hallucination rate per `(model, fen, probe)` into `replicate_stats.tsv` and `replicate_stats.md`.

Only temperature-capable models are enabled in the sweep: `gpt-4o`, `gpt-4o-mini`, Claude Opus/Sonnet 4.6 (± thinking), and Baseten GLM-5 / DeepSeek-V3.1 / Kimi-K2.6.

### Key environment variables

Difficulty and battery: `CHESS_EXTREME`, `STRESS_MODE`, `N_EPISODES`, `STEPS_BEFORE_PROBE`, `USE_LICHESS`, `HARD_HIDDEN_{MIN_PLIES,MAX_PLIES,CAPTURE_BIAS}`, `SAN_*`, `UNDEFENDED_*`, `SQUARE_SCOPE`, `ENABLE_BENCHMARKS`, `DISABLE_BENCHMARKS`.

Observation: `INCLUDE_FEN`, `OBSERVATION_FEN_MODE` (`truth` | `flip_turn` | `transpose` | `startpos` | `override`), `OBSERVATION_FEN_OVERRIDE`, `OBSERVATION_FEN_SEED`, `CHESS_CHAT_CONTEXT` and `CHESS_CHAT_*`.

Providers: `LM_PROVIDER` (`openai` | `anthropic` | `baseten`); `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT`, `OPENAI_MAX_COMPLETION_TOKENS`, `OPENAI_TEMPERATURE`; `ANTHROPIC_MODEL`, `ANTHROPIC_THINKING_EFFORT`, `ANTHROPIC_TEMPERATURE`, `ANTHROPIC_MAX_TOKENS`; `BASETEN_MODEL`, `BASETEN_BASE_URL`, `BASETEN_TEMPERATURE`, `BASETEN_MAX_TOKENS`.

Output: `RESULTS_SUBDIR`, `BENCHMARK_VERBOSE`, `USE_STUB_LM`, `QUESTION_SET_NAME`, `TEMPERATURE`, `RUN_REPLICATES`.

`SQUARE_SCOPE=midboard` only restricts which squares get *asked about* — the full board is still rendered. It is not a visibility mask.

### Results layout

```
results/chess/<RESULTS_SUBDIR>/<model_slug>/<fen_tag>/summary.log
results/chess/<RESULTS_SUBDIR>/<model_slug>/<fen_tag>/examples.txt
results/chess/<RESULTS_SUBDIR>/logs/<run_name>.log
# with replicates:
results/chess/<RESULTS_SUBDIR>/rep-NN/<model_slug>/<fen_tag>/...
results/chess/<RESULTS_SUBDIR>/replicate_stats.{tsv,md}
```

`_model_slug()`: StubLM → `stub`; Anthropic → `<model>[--thinking-<effort>]`; Baseten → `<model with / and : replaced by ->--baseten`; OpenAI → `<model>[--r<effort>]`.

`_fen_tag()`: `INCLUDE_FEN=0` → `fen-off`; else by mode → `fen-transpose` / `fen-flip` / `fen-startpos` / `fen-on`.

**Skip behavior:** if `<out_dir>/summary.log` already exists, the run prints `SKIP` and exits. Delete the file to force a re-run. Per-model `logs/<name>.log` files are truncated on rerun (`>`), so they are not a durable record — `summary.log` is.

**Naming gotcha:** `gpt-4o` lands in `gpt-4o--rlow/` because `OPENAI_REASONING_EFFORT` defaults to `low` in `chess_full_probes.py`. That flag is ignored for non-reasoning models; the directory name does not mean reasoning was used.

### Provider and temperature semantics

`OpenAILM` treats models starting with `gpt-5`, `o1`, `o3`, `o4` as reasoning models: temperature is forced to `None` (with a warning), `max_completion_tokens` is used instead of `max_tokens` and raised to at least 16000, and `reasoning_effort` is passed through. Non-reasoning models honor temperature and default to `max_tokens=256`.

`AnthropicLM` always sends temperature, including alongside adaptive thinking (`thinking={"type":"adaptive"}` plus `output_config={"effort": ...}` for Opus/Sonnet 4.6). In `chess_full_probes.py` the default temperature is 1.0 when `ANTHROPIC_THINKING_EFFORT` is set, else 0.0; override with `ANTHROPIC_TEMPERATURE`.

`BasetenLM` always sends temperature. Deployed endpoints (URL contains `model-`) must be called with `model=""`; serverless endpoints take the real model ID. Qwen models get `max_tokens` bumped to at least 4096, and the wrapper falls back to `message.reasoning` when `content` is `None`.

Determinism caveat: OpenAI reasoning models ignoring temperature does **not** make them deterministic — backend nondeterminism and reasoning-path variation still cause run-to-run drift.

### Baseten endpoint status (checked 2026-07-24)

Reachable (serverless `https://inference.baseten.co/v1`): `zai-org/GLM-5`, `deepseek-ai/DeepSeek-V3.1`, `moonshotai/Kimi-K2.6`.

Not usable: `qwen-3-30b-instruct` (model version deactivated), `qwen-3-30b-thinking` (request timeout), `openai/gpt-oss-120b` (HTTP 200 but empty content on a smoke test).

Re-check with a tiny `chat/completions` request (`max_tokens=4`) before enabling anything; 404s here are usually account-level access, not wrong model IDs.

### Environment setup

`ModuleNotFoundError: No module named 'minigrid'` (or `tqdm`, or `chess`) means the wrong
interpreter, not a code bug.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[chess]'
python -c "import minigrid, chess, openai, anthropic, tqdm; print('deps OK')"
```

On a cluster, be careful that a shared virtualenv actually targets the node you are on: an
interpreter built for one partition can symlink to paths that do not exist on another, which
surfaces as a missing-module error rather than a clear failure.

### Exporting the shared question set

`halluworld/tracks/chess/question_set.py` reuses the exact probe/env config path with `StubLM` (no API calls) and writes the canonical items so every model can be queried on an identical set:

```bash
CHESS_EXTREME=1 HARD_HIDDEN_MIN_PLIES=70 HARD_HIDDEN_MAX_PLIES=120 \
HARD_HIDDEN_CAPTURE_BIAS=0.62 USE_LICHESS=1 N_EPISODES=50 \
INCLUDE_FEN=0 QUESTION_SET_NAME=20260505_fen-off \
python3 halluworld/tracks/chess/question_set.py
```

Outputs `questions.jsonl`, `questions.txt`, and `by_probe/<probe>.txt` under `results/chess/question_sets/<QUESTION_SET_NAME>/`. Prompts are trimmed to start at `Current game:` to drop synthetic chat/commentary. `chess_san_legal_move` prompts have no `Current game:` marker (they start at `## Current observation`), so they fall through untrimmed — expected.

### Aggregation rules

For the May 2026 results, merge two run directories:
- five non-memory probes from `results/chess/20260505/`
- `chess_san_legal_move` and `chess_hidden_side_capture_stats` from `results/chess/20260505_rerun_memory_fix/` (the earlier memory numbers predate the evaluator and observation fixes and are wrong)

Category values are n-weighted averages of `hallucination_rate` across the probes in each class; `Overall` is the n-weighted average across all seven. Leave a cell blank only when a required probe is missing for that `(model, fen)` pair.

This rule describes the two original May run directories specifically. The published numbers now
come from a more complete pipeline that also folds in a corrected P/C rerun, a DeepSeek M/X
backfill, and two July replacement replicates, aggregated with a position-cluster hierarchical
bootstrap (not a flat n-weighted average) so each cell carries a confidence interval. That
per-response corpus and its bootstrap scripts are not part of this release.

### LaTeX table conventions

Colors are precomputed in Python and written as literal integers. Attempts to compute blends inside LaTeX with `\pgfmathsetmacro` were repeatedly broken by expansion and float-vs-integer issues — don't reintroduce them.

Preamble (load `xcolor` with `[table]` **exactly once**; a second bare `\usepackage{xcolor}` prevents `colortbl` from initializing and silently disables `\cellcolor`):

```latex
\usepackage[table]{xcolor}
\definecolor{pastelgreen}{RGB}{186,238,186}
\definecolor{pastelred}{RGB}{244,187,187}
```

Cell generation, with `HRMAX = 65` clipping the scale to the observed data range:

```python
def mix(v, hrmax=65):           # v is a percentage, e.g. 24.7
    t = max(0.0, min(1.0, v / hrmax))
    return int(round(100 * (1 - t)))   # 100 = green (best), 0 = red (worst)

cell = f"\\cellcolor{{pastelgreen!{mix(v)}!pastelred}}{v:.1f}\\%"
```

Bold the column minimum. Model naming in tables: `(T)` for thinking/reasoning enabled, `(NT)` for reasoning disabled.

### Bugs already fixed — do not regress

1. **`run_benchmark` ignored `observation_override`.** `halluworld/benchmark.py` must honor both `probe_result.observation_override` and `probe_result.prepend_env_observation`; otherwise SAN-only probes get a full board diagram and the question references a "sequence above" that isn't shown.
2. **`ChessIntegerEvaluator` grabbed the first integer.** Parse order is: exact single-integer response, then a "final answer: N" pattern, then the *last* integer as a chain-of-thought fallback. Records `integer_parse_mode` in metadata. Without this, verbose models scored near zero on `chess_hidden_side_capture_stats` despite correct answers.
3. **Redundant history preamble.** `render_history(companion_board=True)` suppresses the "evaluating by move history alone / Starting FEN" block when a board is prepended.
4. **Unseeded probe RNGs** — see the determinism contract.
5. **`from examples import chess_full_probes` collides** with a site-packages `examples` module. `export_chess_question_set.py` loads it by absolute file path via `importlib.util.spec_from_file_location`, and inserts the repo root on `sys.path`.