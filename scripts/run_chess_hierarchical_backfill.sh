#!/usr/bin/env bash
# Build a complete position-level dataset for hierarchical chess bootstrap CIs.
#
# Stages:
#   1. Corrected May P/C sweep for all currently served historical models.
#   2. Missing DeepSeek M/X sweep.
#   3. Two complete replacement replicates for the July four-model sweep.
#
# API keys must already be exported in the environment. Outputs go to a fresh
# tree so the historical summaries are never overwritten.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260726_hierarchical_backfill}"
RUN_MAY_PC="${RUN_MAY_PC:-1}"
RUN_DEEPSEEK_MX="${RUN_DEEPSEEK_MX:-1}"
RUN_JULY_REPLACEMENTS="${RUN_JULY_REPLACEMENTS:-1}"
JULY_REPLICATES="${JULY_REPLICATES:-2}"
JULY_START_REPLICATE="${JULY_START_REPLICATE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"

SERVERLESS="https://inference.baseten.co/v1"

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "PYTHON must point to Python 3.10+ with the HalluWorld dependencies installed." >&2
    echo "Activate .venv_dtai or set PYTHON=/path/to/.venv_dtai/bin/python." >&2
    exit 2
fi

PC_PROBES="chess_can_capture,chess_defended,chess_hanging,chess_hypothetical_in_check,chess_after_move_undefended_count"
MX_PROBES="chess_hidden_side_capture_stats,chess_san_legal_move"

COMMON=(
    OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    CHESS_EXTREME=1
    HARD_HIDDEN_MIN_PLIES=70
    HARD_HIDDEN_MAX_PLIES=120
    HARD_HIDDEN_CAPTURE_BIAS=0.62
    USE_LICHESS=1
    N_EPISODES=50
    BENCHMARK_VERBOSE=1
)

FEN_MODES=(
    "fen-off|INCLUDE_FEN=0"
    "fen-on|INCLUDE_FEN=1"
    "fen-transpose|INCLUDE_FEN=1 OBSERVATION_FEN_MODE=transpose OBSERVATION_FEN_SEED=7"
)

PIDS=()
NAMES=()
OVERALL_FAILED=()

require_key() {
    local key="$1"
    if [[ -z "${!key:-}" ]]; then
        echo "Required environment variable is not set: ${key}" >&2
        exit 2
    fi
}

result_slug() {
    case "$1" in
        gpt-4o) echo "gpt-4o--rlow" ;;
        gpt-4o-mini) echo "gpt-4o-mini--rlow" ;;
        gpt-5.4) echo "gpt-5.4--rmedium" ;;
        gpt-5.4-mini) echo "gpt-5.4-mini--rlow" ;;
        gpt-5.4-mini--rnone) echo "gpt-5.4-mini" ;;
        gpt-5.5) echo "gpt-5.5--rmedium" ;;
        gpt-5.5--rnone) echo "gpt-5.5" ;;
        o3) echo "o3--rmedium" ;;
        o3-mini) echo "o3-mini--rmedium" ;;
        o4-mini) echo "o4-mini--rmedium" ;;
        claude-opus-4-6--thinking) echo "claude-opus-4-6--thinking-medium" ;;
        claude-sonnet-4-6--thinking) echo "claude-sonnet-4-6--thinking-medium" ;;
        deepseek-v3-1) echo "deepseek-ai-DeepSeek-V3.1--baseten" ;;
        kimi-k2-6) echo "moonshotai-Kimi-K2.6--baseten" ;;
        *) echo "$1" ;;
    esac
}

start_model() {
    local stage="$1"; shift
    local name="$1"; shift
    local result_subdir="$1"; shift
    local probes="$1"; shift
    local log_dir="results/chess/${result_subdir}/logs"
    local logfile="${log_dir}/${name}.log"
    local model_vars=("$@")
    local model_slug
    local complete=1
    local fen_entry fen_name

    model_slug="$(result_slug "$name")"
    for fen_entry in "${FEN_MODES[@]}"; do
        fen_name="${fen_entry%%|*}"
        if [[ ! -f "results/chess/${result_subdir}/${model_slug}/${fen_name}/summary.log" ||
              ! -f "results/chess/${result_subdir}/${model_slug}/${fen_name}/results.jsonl" ]]; then
            complete=0
            break
        fi
    done
    if ((complete)); then
        printf "[%s] skip complete %-49s\n" "$(date -u +%H:%M:%S)" "${stage}/${name}"
        return
    fi

    mkdir -p "$log_dir"
    (
        printf "\n===== %s stage=%s model=%s =====\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$name"
        for fen_entry in "${FEN_MODES[@]}"; do
            local fen_name="${fen_entry%%|*}"
            local fen_block="${fen_entry#*|}"
            local fen_vars=()
            read -ra fen_vars <<< "$fen_block"
            printf "\n----- fen=%s -----\n" "$fen_name"
            env \
                "${COMMON[@]}" \
                "RESULTS_SUBDIR=${result_subdir}" \
                "ENABLE_BENCHMARKS=${probes}" \
                "${fen_vars[@]}" \
                "${model_vars[@]}" \
                "$PYTHON" -m halluworld.tracks.chess.battery
        done
    ) >>"$logfile" 2>&1 &

    PIDS+=("$!")
    NAMES+=("${stage}/${name}")
    printf "[%s] started %-55s log=%s\n" "$(date -u +%H:%M:%S)" "${stage}/${name}" "$logfile"

    if ((${#PIDS[@]} >= MAX_PARALLEL)); then
        wait_stage
    fi
}

wait_stage() {
    local i status
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            printf "[%s] ✓ %s\n" "$(date -u +%H:%M:%S)" "${NAMES[$i]}"
        else
            status=$?
            printf "[%s] ✗ %s (exit %s)\n" "$(date -u +%H:%M:%S)" "${NAMES[$i]}" "$status"
            OVERALL_FAILED+=("${NAMES[$i]}")
        fi
    done
    PIDS=()
    NAMES=()
}

launch_may_pc() {
    local out="${RESULTS_SUBDIR}/may-pc"
    require_key OPENAI_API_KEY
    require_key ANTHROPIC_API_KEY
    require_key BASETEN_API_KEY

    start_model may-pc gpt-4o "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-4o OPENAI_REASONING_EFFORT=low OPENAI_TEMPERATURE=0
    start_model may-pc gpt-4o-mini "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-4o-mini OPENAI_REASONING_EFFORT=low OPENAI_TEMPERATURE=0
    start_model may-pc gpt-5.4 "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-5.4 OPENAI_REASONING_EFFORT=medium
    start_model may-pc gpt-5.4-mini "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-5.4-mini OPENAI_REASONING_EFFORT=low
    start_model may-pc gpt-5.4-mini--rnone "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-5.4-mini OPENAI_REASONING_EFFORT=
    start_model may-pc gpt-5.5 "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-5.5 OPENAI_REASONING_EFFORT=medium
    start_model may-pc gpt-5.5--rnone "$out" "$PC_PROBES" \
        OPENAI_MODEL=gpt-5.5 OPENAI_REASONING_EFFORT=
    start_model may-pc o3 "$out" "$PC_PROBES" \
        OPENAI_MODEL=o3 OPENAI_REASONING_EFFORT=medium
    start_model may-pc o3-mini "$out" "$PC_PROBES" \
        OPENAI_MODEL=o3-mini OPENAI_REASONING_EFFORT=medium
    start_model may-pc o4-mini "$out" "$PC_PROBES" \
        OPENAI_MODEL=o4-mini OPENAI_REASONING_EFFORT=medium

    start_model may-pc claude-opus-4-6 "$out" "$PC_PROBES" \
        LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-opus-4-6 ANTHROPIC_TEMPERATURE=0
    start_model may-pc claude-opus-4-6--thinking "$out" "$PC_PROBES" \
        LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-opus-4-6 ANTHROPIC_THINKING_EFFORT=medium ANTHROPIC_TEMPERATURE=1
    start_model may-pc claude-sonnet-4-6 "$out" "$PC_PROBES" \
        LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-sonnet-4-6 ANTHROPIC_TEMPERATURE=0
    start_model may-pc claude-sonnet-4-6--thinking "$out" "$PC_PROBES" \
        LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-sonnet-4-6 ANTHROPIC_THINKING_EFFORT=medium ANTHROPIC_TEMPERATURE=1

    start_model may-pc deepseek-v3-1 "$out" "$PC_PROBES" \
        LM_PROVIDER=baseten BASETEN_MODEL=deepseek-ai/DeepSeek-V3.1 BASETEN_BASE_URL="$SERVERLESS" BASETEN_TEMPERATURE=0
    start_model may-pc kimi-k2-6 "$out" "$PC_PROBES" \
        LM_PROVIDER=baseten BASETEN_MODEL=moonshotai/Kimi-K2.6 BASETEN_BASE_URL="$SERVERLESS" BASETEN_TEMPERATURE=0
}

launch_deepseek_mx() {
    local out="${RESULTS_SUBDIR}/may-mx-deepseek"
    require_key BASETEN_API_KEY
    start_model may-mx-deepseek deepseek-v3-1 "$out" "$MX_PROBES" \
        LM_PROVIDER=baseten BASETEN_MODEL=deepseek-ai/DeepSeek-V3.1 BASETEN_BASE_URL="$SERVERLESS" BASETEN_TEMPERATURE=0
}

launch_july_replacement() {
    local rep="$1"
    local rep_tag out
    rep_tag="$(printf 'rep-%02d' "$rep")"
    out="${RESULTS_SUBDIR}/july/${rep_tag}"
    require_key OPENAI_API_KEY
    require_key ANTHROPIC_API_KEY

    start_model "july/${rep_tag}" gpt-4o "$out" "" \
        OPENAI_MODEL=gpt-4o OPENAI_TEMPERATURE=0.7
    start_model "july/${rep_tag}" gpt-4o-mini "$out" "" \
        OPENAI_MODEL=gpt-4o-mini OPENAI_TEMPERATURE=0.7
    start_model "july/${rep_tag}" claude-opus-4-6 "$out" "" \
        LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-opus-4-6 ANTHROPIC_TEMPERATURE=0.7
    start_model "july/${rep_tag}" claude-sonnet-4-6 "$out" "" \
        LM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-sonnet-4-6 ANTHROPIC_TEMPERATURE=0.7
}

echo "Writing backfill results under results/chess/${RESULTS_SUBDIR}"

if [[ "$RUN_MAY_PC" == "1" ]]; then
    launch_may_pc
    wait_stage
fi

if [[ "$RUN_DEEPSEEK_MX" == "1" ]]; then
    launch_deepseek_mx
    wait_stage
fi

if [[ "$RUN_JULY_REPLACEMENTS" == "1" ]]; then
    for rep in $(seq "$JULY_START_REPLICATE" "$JULY_REPLICATES"); do
        launch_july_replacement "$rep"
        wait_stage
    done
fi

if ((${#OVERALL_FAILED[@]})); then
    echo "FAILED (${#OVERALL_FAILED[@]}): ${OVERALL_FAILED[*]}" >&2
    exit 1
fi

echo "Backfill completed successfully."
