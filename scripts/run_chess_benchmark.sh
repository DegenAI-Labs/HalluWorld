#!/usr/bin/env bash
# Run the chess probe battery across all models × three FEN modes.
# All models run in parallel background processes; FEN modes run sequentially
# within each model's process. Output goes to per-model log files.
#
# Run from the repo root:
#   bash scripts/run_chess_benchmark.sh
#   RESULTS_SUBDIR=may5_run2 bash scripts/run_chess_benchmark.sh
#   TEMPERATURE=0.7 RUN_REPLICATES=5 RESULTS_SUBDIR=temp_sweep bash scripts/run_chess_benchmark.sh
set -euo pipefail

PY="python3 -m halluworld.tracks.chess.battery"

# ── Common settings (shared by every run) ─────────────────────────────────────
RESULTS_SUBDIR="${RESULTS_SUBDIR:-$(date -u +%Y%m%d)}"
TEMPERATURE="${TEMPERATURE:-0.7}"
RUN_REPLICATES="${RUN_REPLICATES:-3}"
# Resume a partially completed sweep.  The selected FEN mode is only used for
# START_REPLICATE; later replicates run all modes.
START_REPLICATE="${START_REPLICATE:-1}"
START_FEN_MODE="${START_FEN_MODE:-fen-off}"
# Optional benchmark filtering (comma-separated probe keys).
# Example: ENABLE_BENCHMARKS="chess_san_legal_move,chess_hidden_side_capture_stats"
ENABLE_BENCHMARKS="${ENABLE_BENCHMARKS:-}"
DISABLE_BENCHMARKS="${DISABLE_BENCHMARKS:-}"

BASE_COMMON=(
    CHESS_EXTREME=1
    HARD_HIDDEN_MIN_PLIES=70
    HARD_HIDDEN_MAX_PLIES=120
    HARD_HIDDEN_CAPTURE_BIAS=0.62
    USE_LICHESS=1
    N_EPISODES=50
    BENCHMARK_VERBOSE=1
)

if [[ -n "$ENABLE_BENCHMARKS" ]]; then
    BASE_COMMON+=("ENABLE_BENCHMARKS=${ENABLE_BENCHMARKS}")
fi
if [[ -n "$DISABLE_BENCHMARKS" ]]; then
    BASE_COMMON+=("DISABLE_BENCHMARKS=${DISABLE_BENCHMARKS}")
fi

# ── Three FEN modes ────────────────────────────────────────────────────────────
FEN_MODES=(
    "fen-off|INCLUDE_FEN=0"
    "fen-on|INCLUDE_FEN=1"
    "fen-transpose|INCLUDE_FEN=1 OBSERVATION_FEN_MODE=transpose OBSERVATION_FEN_SEED=7"
)

# run <name> <VAR=val ...>
# Runs the model across all three FEN modes in a background process.
# stdout+stderr go to logs/<name>.log
run() {
    local name="$1"; shift
    local model_vars=("$@")
    local logfile="${LOG_DIR}/${name}.log"
    (
        local skip_mode=1
        for fen_entry in "${FEN_MODES[@]}"; do
            local fen_name="${fen_entry%%|*}"
            local fen_block="${fen_entry#*|}"
            if (( rep == START_REPLICATE && skip_mode )); then
                if [[ "$fen_name" != "$START_FEN_MODE" ]]; then
                    continue
                fi
                skip_mode=0
            fi
            read -ra fen_arr <<< "$fen_block"
            env "${BASE_COMMON[@]}" "RESULTS_SUBDIR=${CURRENT_RESULTS_SUBDIR}" "${fen_arr[@]}" "${model_vars[@]}" $PY
        done
    ) >"$logfile" 2>&1 &
    local pid=$!
    printf "[%s] started %-45s pid=%-6s log=%s\n" \
        "$(date -u +%H:%M:%S)" "$name" "$pid" "$logfile"
    PIDS+=("$pid")
    NAMES+=("$name")
}

if ! [[ "$START_REPLICATE" =~ ^[1-9][0-9]*$ ]] || (( START_REPLICATE > RUN_REPLICATES )); then
    echo "START_REPLICATE must be between 1 and RUN_REPLICATES (${RUN_REPLICATES})." >&2
    exit 2
fi
if ! [[ " ${FEN_MODES[*]} " == *"$START_FEN_MODE|"* ]]; then
    echo "START_FEN_MODE must be one of: fen-off, fen-on, fen-transpose." >&2
    exit 2
fi

# API keys are read from the environment and are never stored in this file.
# Fail before launching any background job, so a missing key surfaces here
# rather than as N identical tracebacks buried in per-model log files.
require_key() {
    local key="$1"
    if [[ -z "${!key:-}" ]]; then
        echo "Required environment variable is not set: ${key}" >&2
        echo "Export it (or source a .env) before running this script." >&2
        exit 2
    fi
}

# ── Baseten ───────────────────────────────────────────────────────────────────
SERVERLESS="https://inference.baseten.co/v1"
# QWEN_I="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
# QWEN_T="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"

launch_temperature_models() {
    # Only the providers actually enabled below are required. If you uncomment
    # the Baseten runs, add: require_key BASETEN_API_KEY
    require_key OPENAI_API_KEY
    require_key ANTHROPIC_API_KEY

    # OpenAI reasoning models (GPT-5 / o-series) do not accept temperature in OpenAILM,
    # so they stay disabled for this temperature sweep.
    run gpt-4o        OPENAI_MODEL=gpt-4o      OPENAI_TEMPERATURE="$TEMPERATURE"
    run gpt-4o-mini   OPENAI_MODEL=gpt-4o-mini OPENAI_TEMPERATURE="$TEMPERATURE"

    # Anthropic Messages API accepts temperature, including with adaptive thinking.
    run claude-opus-4-6              LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-opus-4-6   ANTHROPIC_TEMPERATURE="$TEMPERATURE"
    # run claude-opus-4-6--thinking    LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-opus-4-6   ANTHROPIC_THINKING_EFFORT=medium ANTHROPIC_TEMPERATURE="$TEMPERATURE"
    run claude-sonnet-4-6            LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-sonnet-4-6 ANTHROPIC_TEMPERATURE="$TEMPERATURE"
    # run claude-sonnet-4-6--thinking  LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-sonnet-4-6 ANTHROPIC_THINKING_EFFORT=medium ANTHROPIC_TEMPERATURE="$TEMPERATURE"

    # Baseten serverless models that passed the smoke check.
    # run glm-5              LM_PROVIDER=baseten BASETEN_MODEL="zai-org/GLM-5"              BASETEN_BASE_URL="$SERVERLESS" BASETEN_TEMPERATURE="$TEMPERATURE"
    # run deepseek-v3-0324   LM_PROVIDER=baseten BASETEN_MODEL="deepseek-ai/DeepSeek-V3.1"  BASETEN_BASE_URL="$SERVERLESS" BASETEN_TEMPERATURE="$TEMPERATURE"
    # run kimi-k2-6          LM_PROVIDER=baseten BASETEN_MODEL="moonshotai/Kimi-K2.6"       BASETEN_BASE_URL="$SERVERLESS" BASETEN_TEMPERATURE="$TEMPERATURE"

    # Disabled Baseten candidates:
    # - openai/gpt-oss-120b: endpoint responded but returned empty content in a tiny smoke test
    # - qwen-3-30b-instruct: model version deactivated
    # - qwen-3-30b-thinking: smoke test timed out
}

for rep in $(seq "$START_REPLICATE" "$RUN_REPLICATES"); do
    rep_tag="$(printf 'rep-%02d' "$rep")"
    CURRENT_RESULTS_SUBDIR="${RESULTS_SUBDIR}/${rep_tag}"
    LOG_DIR="results/chess/${CURRENT_RESULTS_SUBDIR}/logs"
    mkdir -p "$LOG_DIR"
    PIDS=()
    NAMES=()

    echo ""
    echo "Starting replicate ${rep}/${RUN_REPLICATES}  temperature=${TEMPERATURE}  results=${CURRENT_RESULTS_SUBDIR}"
    echo ""
    launch_temperature_models

    echo ""
    echo "Waiting for ${#PIDS[@]} jobs…  (tail -f ${LOG_DIR}/<name>.log to follow)"
    echo ""

    failed=()
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            printf "[%s] ✓ %s\n" "$(date -u +%H:%M:%S)" "${NAMES[$i]}"
        else
            status=$?
            printf "[%s] ✗ %s (exit %s)\n" "$(date -u +%H:%M:%S)" "${NAMES[$i]}" "$status"
            failed+=("${NAMES[$i]}")
        fi
    done

    echo ""
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED (${#failed[@]}): ${failed[*]}"
        exit 1
    fi
    echo "Replicate ${rep}/${RUN_REPLICATES} completed successfully."
done

python3 - <<'PY' "$RESULTS_SUBDIR"
from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path("results/chess") / sys.argv[1]
summary_re = re.compile(r"(.+?)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
records: dict[tuple[str, str, str], list[tuple[int, float, float]]] = defaultdict(list)

for path in sorted(root.glob("rep-*/*/*/summary.log")):
    rep = path.parts[-4]
    model = path.parts[-3]
    fen = path.parts[-2]
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        match = summary_re.match(line)
        if match is None:
            continue
        probe, n, acc, hr, _mean_score = match.groups()
        records[(model, fen, probe.strip())].append((int(n), float(acc), float(hr)))

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def sample_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))

out_tsv = root / "replicate_stats.tsv"
out_md = root / "replicate_stats.md"
lines = ["model\tfen\tprobe\truns\tmean_n\tmean_accuracy\tstd_accuracy\tmean_hallucination_rate\tstd_hallucination_rate"]
md = [
    "| model | fen | probe | runs | mean n | mean acc | std acc | mean hallucination | std hallucination |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for (model, fen, probe), vals in sorted(records.items()):
    ns = [v[0] for v in vals]
    accs = [v[1] for v in vals]
    hrs = [v[2] for v in vals]
    row = [
        model,
        fen,
        probe,
        str(len(vals)),
        f"{mean(ns):.1f}",
        f"{mean(accs):.4f}",
        f"{sample_std(accs):.4f}",
        f"{mean(hrs):.4f}",
        f"{sample_std(hrs):.4f}",
    ]
    lines.append("\t".join(row))
    md.append(
        f"| {model} | {fen} | {probe} | {len(vals)} | {mean(ns):.1f} | "
        f"{mean(accs):.4f} | {sample_std(accs):.4f} | "
        f"{mean(hrs):.4f} | {sample_std(hrs):.4f} |"
    )

out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"Wrote replicate stats: {out_tsv}")
print(f"Wrote replicate stats: {out_md}")
PY

echo "All ${RUN_REPLICATES} replicate(s) completed successfully."
