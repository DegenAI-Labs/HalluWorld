# Probe Benchmark: Methods

## Overview

We evaluate 11 frontier language models on 529 **runtime probes** extracted from 110 real software-engineering trajectories executed in [Terminal-Bench](https://github.com/harbor-framework/terminal-bench). Each probe is a factual question whose ground-truth answer is fully determined by a portion of the terminal context visible to the agent at a specific step in its trajectory. Models receive only that context and must return an answer in a constrained schema — no generation of prose, no chain-of-thought in the output.

---

## 1. Probe Construction

### 1.1 Trajectory Source

Probes are derived from 132 Terminal-Bench trajectories across 110 distinct software-engineering tasks (e.g., `build-linux-kernel-qemu`, `inspect-dtype`, `torch-tensor-parallelism`). Each trajectory was executed by a `gpt-4o-mini` agent operating in a live Linux environment, producing a sequence of shell commands, terminal outputs, and file-system snapshots.

### 1.2 Probe Injection and Generation

All 529 probes were **LLM-generated** (`injector=llm_generated`) using `gpt-5.4` at `reasoning_effort=high`. For each candidate trajectory step (a shell command), the generator was given:

- The full terminal pane history up to and including that step (`context_scope=through_trigger`)
- The command itself (`trigger_command`)
- File-system snapshot diffs for nominated paths (e.g., SHA-256 hashes, mtimes, directory listings)

The generator was asked to produce:
1. A factual question answerable solely from the provided evidence
2. A constrained answer schema (e.g., `idx=<0250,0256|0250|0256|none>`, `outcome=<ok|fail>`)
3. A ground-truth answer with an explicit reasoning trace (`golden_command_or_heuristic`)
4. A `difficulty_score` and `answerability_score` (both 1–5)
5. A `failure_mode_target` identifying which hallucination failure mode the probe stresses

All generated probes passed a post-hoc answerability filter: probes with `answerability_score < 5` were discarded. The resulting set has mean `difficulty_score = 4.36` (min 4, max 5), indicating all probes are at the harder end of the scale.

### 1.3 Probe Types

Probes are classified into five types by the cognitive demand they place on a model reading the terminal context:

| Probe Type | N | Description |
|---|---|---|
| **causal** | 106 | Requires reasoning about cause-effect relationships in the command trace (e.g., which command caused a file hash to change) |
| **cross-tier compound** | 106 | Requires integrating evidence across multiple context sections or artifact types simultaneously |
| **memory** | 108 | Requires tracking a value or state across multiple prior steps (e.g., what was the value of X at step 42?) |
| **perceptual** | 105 | Requires accurately reading a value directly present in the context (e.g., a specific column value in a snapshot table) |
| **uncertainty** | 104 | The correct answer is "cannot determine" or requires acknowledging an epistemic limit; confident wrong answers are scored incorrect |

The five types are approximately balanced (~105–108 each) to ensure no single probe type dominates the overall accuracy signal.

### 1.4 Failure Mode Targets

Each probe explicitly targets one of the following hallucination failure modes, based on the generator's classification:

| Failure Mode | N | Description |
|---|---|---|
| **stale memory** | 187 | Model relies on an outdated value rather than the most recent terminal output |
| **cross-tier reasoning** | 110 | Model fails to correctly bridge between different context sections (e.g., file listing vs. process output) |
| **uncertainty overclaim** | 107 | Model asserts a definite answer when the context does not support one |
| **causal shortcut** | 103 | Model infers the wrong causal attribution (e.g., confuses which command caused an effect) |
| **version/API hallucination** | 19 | Model substitutes prior training knowledge about an API for the actual observed behavior |

### 1.5 Context Presentation

Each probe's context is the raw terminal pane text captured immediately before the trigger command executes (`selected_context_kind=context_path`, `context_scope=through_trigger`). This is unedited terminal output — ANSI-stripped but otherwise not curated. Context length ranges from **1,409 to 60,055 characters** (mean ~50,000 chars), placing most probes at or near the upper end of typical context windows.

All probes come from the **agent session** (`step_src=command`), meaning the context reflects what the agent itself observed, not a separate oracle channel.

---

## 2. Evaluation Protocol

### 2.1 Prompt Format

Each model receives a two-turn prompt:

- **System:** `"Answer probe questions with only the exact requested answer string. Do not include prose, markdown, or explanation."`
- **User:** A structured message containing:
  - A framing preamble explaining the terminal context format and the constraint to answer from visible context only
  - The full terminal context (`CONTEXT` block)
  - The question with its answer schema

Models must return only the answer string in the requested schema. No explanation, no markdown, no chain-of-thought in the output.

### 2.2 Answer Scoring

New runs are scored by `halluworld/tracks/terminal/evaluation.py`. The production scorer first
accepts normalized exact equality. For structured schemas it then locates every declared key and
compares the complete value between adjacent keys. All declared keys must be present and correct;
there is no permissive bare-value fallback. Normalization is limited to casing, surrounding markup,
and insignificant whitespace around separators. Meaningful commas, semicolons, pipes, arrows,
equals signs, paths, spaces, and angle brackets remain part of the value. This prevents a truncated
prefix from receiving credit for a longer structured answer.

The historical leaderboard used the older `scripts/terminal/tools/evaluate_probes.py` artifact
evaluator. New results should identify the scorer/version in their manifest and use the installed
`halluworld terminal eval` interface.

### 2.3 Model Configuration

| Model | Provider | Thinking/Reasoning | Max Output Tokens |
|---|---|---|---|
| GPT-5.5 | OpenAI Responses API | `reasoning_effort=medium` | 16,384 |
| o3 | OpenAI Responses API | `reasoning_effort=medium` | 16,384 |
| o4-mini | OpenAI Responses API | `reasoning_effort=medium` | 16,384 |
| o3-mini | OpenAI Responses API | `reasoning_effort=medium` | 16,384 |
| Claude Opus 4.6 | Anthropic Messages API | `thinking=adaptive, effort=medium` | 16,384 |
| Claude Sonnet 4.6 | Anthropic Messages API | `thinking=adaptive, effort=medium` | 16,384 |
| GLM-5 | Baseten serverless | — | 256 |
| GPT-4o | OpenAI Responses API | — | 256 |
| GPT-5.4-mini | OpenAI Responses API | — | 256 |
| GPT-4o-mini | OpenAI Responses API | — | 256 |
| Kimi-K2.6 | Baseten serverless | — | 256 |

Reasoning/thinking models use `max_output_tokens=16,384` to ensure reasoning tokens do not exhaust the budget before the answer is emitted. Standard models use `max_output_tokens=256`, sufficient for the constrained schema answers (typically 5–20 characters).

Temperature is unset (default) for all models; all other parameters are provider defaults.

### 2.4 Concurrency and Reliability

Each model run uses 8 concurrent workers (2 for Baseten models to respect rate limits), with per-request timeouts of 90–120 seconds and up to 6 retries with exponential backoff (capped at 30s). Results are written atomically per probe, and `--resume` allows partial runs to be continued without re-evaluating completed probes. All 529 probes were successfully evaluated for all 11 models (0 errors in the final dataset).

---

## 3. Design Decisions and Benchmark Properties

### 3.1 Grounded Ground Truth

Every probe's ground truth is derived from concrete, observable terminal evidence — file hashes, timestamps, directory listings, command outputs — not from model judgments or human annotation of subjective quality. The `golden_command_or_heuristic` field contains the explicit derivation trace (e.g., "column 3 of the snapshot at step 0250 shows hash 48dd..., which differs from the prior step's 661d..."). This makes the benchmark resistant to annotation disputes and re-scorable without re-running any model.

### 3.2 Constrained Output Schemas

By requiring answers in a structured schema (`key=value` pairs with a fixed option set), we avoid conflating answer correctness with answer verbosity or formatting. LLM judges are not used for scoring; all grading is rule-based. This eliminates judge model bias and makes the benchmark fully reproducible.

### 3.3 Context Fidelity

Contexts are the raw terminal pane text as-seen by the agent — not cleaned, not summarized, not reformatted. This means the probe evaluates the same perceptual challenge the agent itself faced. If the relevant information was buried in 50,000 characters of noisy terminal output, the probe model faces the same challenge.

### 3.4 Answerability Filter

All probes have `answerability_score=5` (fully answerable from context). The generator was explicitly instructed to avoid probes that require external knowledge, inference beyond the provided evidence, or subjective judgment. This ensures that `correct=False` unambiguously indicates a model failure, not an ambiguous probe.

### 3.5 Balanced Difficulty

All probes have `difficulty_score ∈ {4, 5}` (mean 4.36). Easier probes (score 1–3) were either not generated or filtered. This ceiling-lifts the evaluation: even the strongest models (GPT-5.5) achieve only 94.1%, and the hardest probe type (uncertainty) caps out at 76.9% for top models. The benchmark is not yet saturated.

### 3.6 Task and Domain Diversity

Probes span 110 distinct software-engineering tasks covering compiled languages (C/C++, Rust), interpreted languages (Python, JS), build systems (CMake, Make, Bazel), databases, containerization, networking, and ML frameworks. This breadth reduces the risk of a model over-fitting to a narrow domain.

---

## 4. Limitations

- **Single trajectory per task**: Each task contributes probes from one agent run. A different agent trajectory through the same task might yield different probes at different difficulty levels.
- **LLM-generated questions**: While all questions passed an answerability filter and have explicit derivation traces, subtle errors in the generator's reasoning could produce incorrect ground truth for a small fraction of probes. Manual auditing of a random sample found no such cases in the final 529.
- **Non-thinking baseline gap**: The six thinking/reasoning models all receive substantially more compute budget than the five standard models. The performance gap between tiers partly reflects this, not solely capability. Specifically, whether extended reasoning hurts on uncertainty probes (a "thinking paradox" documented in some gridworld evaluations, where overconfident inference increases under epistemic ambiguity) **cannot be assessed** from these results — no non-thinking Claude runs were performed on Terminal-Bench. The comparison between thinking and non-thinking Claude models would require a controlled paired run.
- **No multi-turn or action evaluation**: Probes test comprehension of terminal context, not the model's ability to take the correct next action. High probe accuracy does not guarantee high task-completion rate.
