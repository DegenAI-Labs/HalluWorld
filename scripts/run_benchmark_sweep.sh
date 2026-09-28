#!/usr/bin/env bash
#
# Run the HalluWorld benchmark across many models, several at a time.
#
#     scripts/run_benchmark_sweep.sh                 # run everything enabled below
#     scripts/run_benchmark_sweep.sh --dry-run       # print the plan, spend nothing
#     scripts/run_benchmark_sweep.sh --models opus-5,sonnet-5
#     scripts/run_benchmark_sweep.sh --tracks chess --version v0.2
#
# Resumable: a job whose output directory already holds records.jsonl is
# skipped, so re-running after a failure only repeats what failed. Pass
# --force to redo everything.
#
# ---------------------------------------------------------------------------
# BEFORE THE FIRST RUN -- three things need your input
# ---------------------------------------------------------------------------
# 1. API KEYS. Export before running; none are read from a file:
#      OPENAI_API_KEY, ANTHROPIC_API_KEY, BASETEN_API_KEY, XAI_API_KEY
#    Each provider reads its own key, so all four can be set at once. A model
#    whose provider key is unset is skipped with a warning.
#
# 2. MODEL IDS. The table below holds each provider's exact API model string.
#    An entry reading FILL_ME is skipped rather than guessed at, because a
#    wrong id fails per-request and looks like a model problem.
#
# 3. CONCURRENCY. Defaults are deliberately low. See CONCURRENCY below.
#
# The venv is found automatically; no need to activate it first.
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
# Every job shells out to `halluworld`, which lives in the project venv rather
# than on a login shell's PATH. Resolve it once, here: otherwise each of the N
# jobs fails separately with "command not found" buried in its own log, and a
# real run builds a tree of empty result directories before anyone notices.
if ! command -v halluworld >/dev/null 2>&1; then
  for candidate in \
      "${HALLUWORLD_VENV:-}" \
      "$REPO_ROOT/.venv" \
      "$REPO_ROOT/../../.venv" \
      "$REPO_ROOT/../.venv"; do
    [[ -n "$candidate" && -f "$candidate/bin/activate" ]] || continue
    # shellcheck disable=SC1091
    source "$candidate/bin/activate"
    command -v halluworld >/dev/null 2>&1 && break
  done
fi
if ! command -v halluworld >/dev/null 2>&1; then
  echo "error: the 'halluworld' command is not available." >&2
  echo "       Activate the project venv first, e.g." >&2
  echo "         source .venv/bin/activate" >&2
  echo "       or set HALLUWORLD_VENV to its path and re-run." >&2
  exit 2
fi

VERSION="${VERSION:-v0.2}"
TRACKS="${TRACKS:-chess grid terminal}"
OUT_ROOT="${OUT_ROOT:-results/sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
EPISODES="${EPISODES:-50}"
# 16000, not 1024. OpenAILM silently floors reasoning models to 16000, so at
# 1024 the OpenAI models answered with a 16x larger budget than the Anthropic
# and Baseten ones -- and 15% of all scored answers were cut off mid-answer,
# including ~85% of kimi-k3's and gpt-oss-120b's chess answers. Those
# accuracies measured the token cap, not the model. Matching the floor makes
# the budget uniform. A cap costs nothing unless it is reached: a model that
# answers in 6 tokens is billed for 6.
MAX_TOKENS="${MAX_TOKENS:-16000}"
RETRIES="${RETRIES:-5}"
# Re-ask a refused prompt up to this many more times (anthropic only). Off by
# default: it biases the sample toward content the model will engage with, so
# a run that uses it should say so. Set REFUSAL_RETRIES=2 to test whether the
# ~100 Anthropic refusals are consistent or sampling artifacts.
REFUSAL_RETRIES="${REFUSAL_RETRIES:-0}"
DRY_RUN=0
FORCE=0
ONLY_MODELS=""

# ---------------------------------------------------------------------------
# CONCURRENCY
# ---------------------------------------------------------------------------
# Two independent dials:
#
#   JOBS          how many (model x track) jobs run at once, across providers.
#   WORKERS_*     threads inside one job, per provider. chess, grid and
#                 terminal replay paths thread; innav runs sequentially.
#
# Peak in-flight requests to one provider is roughly
# (its concurrent jobs) x (its WORKERS_*), so these multiply. The binding
# limit is your ACCOUNT's rate limit, not this machine -- request-per-minute
# and token-per-minute caps are per-account and tier-dependent, and the same
# model on a lower tier will throttle far sooner. Nothing here can discover
# your tier, so these are conservative starting points, not recommendations
# tuned to your account.
#
# The clients retry rate-limit errors with exponential backoff (--retries),
# so moderate over-subscription costs latency rather than lost work. Sustained
# over-subscription wastes real money on retried requests, and a request that
# exhausts its retries is recorded with error=rate_limit and NOT scored --
# excluded, not counted wrong. Check the excluded count before trusting a
# number, and lower concurrency if it is not ~0.
#
# Tune by watching: run one model, raise until rate_limit errors appear in
# records.jsonl, then back off ~25%.
JOBS="${JOBS:-4}"
WORKERS_OPENAI="${WORKERS_OPENAI:-4}"
WORKERS_ANTHROPIC="${WORKERS_ANTHROPIC:-4}"
WORKERS_BASETEN="${WORKERS_BASETEN:-2}"
WORKERS_XAI="${WORKERS_XAI:-4}"

# ---------------------------------------------------------------------------
# MODELS -- label|provider|model_id|api_base
# ---------------------------------------------------------------------------
# api_base is blank except where a provider needs an explicit endpoint.
# A label is what you pass to --models; it also names the output directory.
#
# xai is a first-class provider now: its API is OpenAI-compatible, so it reuses
# OpenAILM against https://api.x.ai/v1 and reads XAI_API_KEY. No key collision
# with Baseten, and it keeps OpenAILM's bad_request handling (a too-small token
# budget is recorded as an exclusion rather than scored as a wrong answer).
MODELS=(
  # --- OpenAI ---
  "gpt-6-astra|openai|gpt-6-astra|"
  "gpt-5.6-sol|openai|gpt-5.6-sol|"
  "gpt-5.6-terra|openai|gpt-5.6-terra|"
  "o3|openai|o3|"

  # --- Anthropic (ids known) ---
  "fable-5.1|anthropic|claude-fable-5-1|"
  "opus-5|anthropic|claude-opus-5|"
  "sonnet-5|anthropic|claude-sonnet-5|"

  # --- Baseten (needs each model's exact served slug) ---
  "glm-5.3|baseten|zai-org/GLM-5.3|"
  "kimi-k3|baseten|moonshotai/Kimi-K3|"
  # "qwen-3.8-27b|baseten|Qwen/Qwen3.8-27B|"
  "deepseek-v4.1-flash|baseten|deepseek-ai/DeepSeek-V4.1-Flash|"
  "gpt-oss-120b|baseten|openai/gpt-oss-120b|"

  # --- xAI (id from docs.x.ai; base_url is implied by the provider) ---
  "grok-4.6|xai|grok-4.6|"

  # Google
  # "gemini-3.8-flash|baseten|FILL_ME|"
)

# ---------------------------------------------------------------------------
# LIMITATIONS -- read before trusting a sweep
# ---------------------------------------------------------------------------
# * v0.2 covers chess (347), grid (231) and terminal (200). No innav.
# * grid v0.2 is r4 only, by decision -- r3 is not being used.
# * terminal v0.2 reuses v0.1's evidence contexts: 198 of its 200 probes ask a
#   NEW question over a pane that also appears in v0.1. Questions and ground
#   truth are new; the evidence is not. Held out at the question level, not the
#   observation level -- worth stating next to any number.
# * chess, grid and terminal all replay their bank and honour --workers. innav
#   still regenerates, so for it only JOBS gives parallelism.
# * grid v0.1 REGENERATES rather than replays, because that bank stores probe
#   recipes rather than rendered questions. innav likewise -- so --version on
#   those stamps the manifest without changing the questions asked.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --models) ONLY_MODELS="$2"; shift 2 ;;
    --tracks) TRACKS="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
for track in $TRACKS; do
  case "$track" in
    chess|grid|innav|terminal) ;;
    *) echo "error: unknown track '$track'" >&2; exit 2 ;;
  esac
  bank="halluworld/data/questions/$VERSION/$track.jsonl.gz"
  if [[ ! -f "$bank" ]]; then
    echo "error: no $track bank for $VERSION ($bank)." >&2
    echo "       $VERSION contains: $(ls halluworld/data/questions/$VERSION/*.jsonl.gz 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/.jsonl.gz//' | tr '\n' ' ')" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT"

# ---------------------------------------------------------------------------
# Build the job list
# ---------------------------------------------------------------------------
JOBLIST="$OUT_ROOT/jobs.tsv"
: > "$JOBLIST"
skipped=0

for row in "${MODELS[@]}"; do
  IFS='|' read -r label provider model_id api_base <<< "$row"
  [[ -z "${label:-}" ]] && continue
  if [[ -n "$ONLY_MODELS" && ",$ONLY_MODELS," != *",$label,"* ]]; then
    continue
  fi
  if [[ "$model_id" == "FILL_ME" || -z "$model_id" ]]; then
    echo "skip: $label has no model id yet (edit MODELS in this script)" >&2
    skipped=$((skipped + 1))
    continue
  fi
  case "$provider" in
    openai)    key="${OPENAI_API_KEY:-}";    workers="$WORKERS_OPENAI" ;;
    anthropic) key="${ANTHROPIC_API_KEY:-}"; workers="$WORKERS_ANTHROPIC" ;;
    baseten)   key="${BASETEN_API_KEY:-}";   workers="$WORKERS_BASETEN" ;;
    xai)       key="${XAI_API_KEY:-}";       workers="$WORKERS_XAI" ;;
    *) echo "skip: $label has unknown provider '$provider'" >&2; skipped=$((skipped+1)); continue ;;
  esac
  if [[ -z "$key" && $DRY_RUN -eq 0 ]]; then
    echo "skip: $label needs the $provider API key exported" >&2
    skipped=$((skipped + 1))
    continue
  fi
  for track in $TRACKS; do
    # '|' rather than tab: tab is an IFS *whitespace* character, so bash
    # collapses runs of it and an empty api_base field silently vanishes,
    # shifting every field after it.
    printf '%s|%s|%s|%s|%s|%s\n' \
      "$label" "$provider" "$model_id" "$api_base" "$track" "$workers" >> "$JOBLIST"
  done
done

total=$(wc -l < "$JOBLIST" | tr -d ' ')
echo "=============================================================="
echo "  HalluWorld sweep"
echo "=============================================================="
echo "bank version : $VERSION"
echo "tracks       : $TRACKS"
echo "jobs         : $total ($skipped skipped)"
echo "parallel     : $JOBS jobs at once"
echo "threads/job  : openai=$WORKERS_OPENAI anthropic=$WORKERS_ANTHROPIC baseten=$WORKERS_BASETEN xai=$WORKERS_XAI"
echo "output       : $OUT_ROOT"
echo "halluworld   : $(command -v halluworld)"
echo

if [[ "$total" -eq 0 ]]; then
  echo "nothing to run." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Run one job
# ---------------------------------------------------------------------------
run_job() {
  # One whole TSV line per call. Splitting on whitespace instead would drop the
  # empty api_base field and shift every argument after it.
  local label provider model_id api_base track workers
  IFS='|' read -r label provider model_id api_base track workers <<< "$1"
  # Dry runs write real files (terminal emits a dry_run record per question),
  # and the resume check below only looks for a non-empty results file -- so
  # sharing a directory would make a dry run mask the real work as "done".
  local real_out="$OUT_ROOT/$track/$label"
  local out="$real_out"
  [[ $DRY_RUN -eq 1 ]] && out="$OUT_ROOT/_dryrun/$track/$label"
  local log="$out/run.log"

  # chess/grid/innav write records.jsonl; terminal writes results.jsonl.
  # "Done" means at least one question was actually SCORED. A run where every
  # request failed still writes a full-length file -- 200 api_error records,
  # say -- and treating that as done would silently skip the retry it needs.
  # Skip only when nothing is left unanswered. "Has some scored records" is
  # not done: a run that died partway -- rate limited, out of credits -- leaves
  # a mix, and skipping it would silently drop the unanswered questions. Every
  # job is invoked with --resume, so a re-run pays only for what is missing.
  # Always read the REAL output directory, even on a dry run: the point of a
  # dry run is to preview the plan, and checking the throwaway _dryrun path
  # would report every job as pending no matter what has already been answered.
  if [[ $FORCE -eq 0 ]]; then
    for f in "$real_out/records.jsonl" "$real_out/results.jsonl"; do
      [[ -s "$f" ]] || continue
      remaining=$(python3 -c "
import json,sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('score') is None))
" "$f" 2>/dev/null || echo 0)
      if [[ "$remaining" -eq 0 ]]; then
        echo "  = $track/$label already complete, skipping"
        return 0
      fi
      echo "  ~ $track/$label resuming: $remaining unanswered"
    done
  fi
  mkdir -p "$out"

  local cmd=(halluworld eval "$track"
             --provider "$provider" --model "$model_id"
             --out "$out" --version "$VERSION"
             --max-tokens "$MAX_TOKENS")
  # terminal selects with --limit; the others count episodes per level/probe.
  if [[ "$track" == "terminal" ]]; then
    [[ -n "${TERMINAL_LIMIT:-}" ]] && cmd+=(--limit "$TERMINAL_LIMIT")
  else
    cmd+=(--episodes "$EPISODES")
  fi
  # Only the chess replay path threads or takes these flags.
  if [[ "$track" == "chess" || "$track" == "grid" || "$track" == "terminal" ]]; then
    cmd+=(--workers "$workers" --retries "$RETRIES" --refusal-retries "$REFUSAL_RETRIES")
    # --force means re-ask everything. Passing --resume alongside it kept every
    # already-scored record and re-asked only the unscored ones -- so --force
    # on a completed job was a no-op that reported success.
    [[ $FORCE -eq 0 ]] && cmd+=(--resume)
    [[ -n "$api_base" ]] && cmd+=(--api-base "$api_base")
  fi
  [[ $DRY_RUN -eq 1 ]] && cmd+=(--dry-run)

  local started=$SECONDS
  if "${cmd[@]}" > "$log" 2>&1; then
    local took=$((SECONDS - started))
    local n=0 scored=0
    for f in "$out/records.jsonl" "$out/results.jsonl"; do
      [[ -f "$f" ]] || continue
      n=$(wc -l < "$f" | tr -d ' ')
      scored=$(python3 -c "
import json,sys
print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('score') is not None))
" "$f" 2>/dev/null || echo 0)
      break
    done
    # A job that wrote a full-length file of provider errors exits 0 on some
    # paths. Reporting that as a tick is how an entirely failed sweep looked
    # successful: every question excluded, nothing measured.
    if [[ $DRY_RUN -eq 0 && "$n" -gt 0 && "$scored" -eq 0 ]]; then
      echo "  ✗ $track/$label NOTHING SCORED -- $n records, all excluded. See $log"
      python3 -c "
import json,collections,sys
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
notes=collections.Counter(r.get('parse_note','')[:120] for r in recs if r.get('parse_note'))
for note,c in notes.most_common(1): print('      [%dx] %s' % (c, note))
" "$out/records.jsonl" 2>/dev/null || python3 -c "
import json,collections,sys
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
notes=collections.Counter(r.get('parse_note','')[:120] for r in recs if r.get('parse_note'))
for note,c in notes.most_common(1): print('      [%dx] %s' % (c, note))
" "$out/results.jsonl" 2>/dev/null
      return 1
    fi
    echo "  ✓ $track/$label (${took}s, $n records, $scored scored)"
  else
    echo "  ✗ $track/$label FAILED -- see $log"
    tail -3 "$log" | sed 's/^/      /'
    return 1
  fi
}
export -f run_job
export OUT_ROOT VERSION EPISODES MAX_TOKENS RETRIES REFUSAL_RETRIES DRY_RUN FORCE

# xargs -P gives job-level parallelism without requiring GNU parallel.
# Records go to per-job files, so concurrent jobs never write the same path.
set +e
xargs -P "$JOBS" -d '\n' -n 1 bash -c 'run_job "$@"' _ < "$JOBLIST"
sweep_status=$?
set -e

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo "  Results"
echo "=============================================================="
if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry run: no requests were made)"
else
  SWEEP_MAX_TOKENS="$MAX_TOKENS" python3 - "$OUT_ROOT" <<'PY'
import json, os, pathlib, sys

root = pathlib.Path(sys.argv[1])
cap = int(os.environ.get("SWEEP_MAX_TOKENS") or 0)
rows = []
paths = sorted(root.glob("*/*/records.jsonl")) + sorted(root.glob("*/*/results.jsonl"))
paths = [p for p in paths if "_dryrun" not in p.parts]
for path in paths:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        continue
    scored = [r for r in recs if r.get("score") is not None]
    excluded = len(recs) - len(scored)
    acc = sum(r["score"] for r in scored) / len(scored) if scored else None
    # An answer that stopped exactly at the cap was cut off mid-answer, and it
    # is still scored on whatever fragment arrived. Counting it is what turns a
    # token-budget problem into a visible one instead of a model ranking.
    truncated = sum(
        1 for r in scored if cap and (r.get("completion_tokens") or 0) >= cap
    )
    # A refusal is the model declining, not hallucinating -- but it is scored 0,
    # which reads as a wrong answer. Whether that is right is a metric decision,
    # so the count is surfaced rather than folded silently into the accuracy.
    # A refusal is excluded (score=None) and carries error="refusal". The rule
    # -- strictly zero output tokens plus the provider's refusal flag -- lives
    # in halluworld.results.exclusion_reason, so this is a count, not a test.
    refused = sum(1 for r in recs if r.get("error") == "refusal")
    # Only these are worth a "re-run" nudge: they are the provider or the
    # network, not the model. Refusals and budget-exhausted empties also sit
    # inside `excluded`, and telling someone to check rate limits over those
    # sends them looking in the wrong place.
    transient = sum(1 for r in recs if r.get("error") in ("rate_limit", "api_error"))
    rows.append((path.parent.parent.name, path.parent.name, len(recs), excluded,
                 transient, truncated, refused, "n/a" if acc is None else "%.4f" % acc))

if not rows:
    print("no records written.")
else:
    print("%-9s %-22s %7s %9s %10s %8s %9s"
          % ("track", "model", "n", "excluded", "truncated", "refused", "accuracy"))
    for track, model, n, exc, trans, trunc, ref, acc in rows:
        flag = ""
        if trunc:
            flag = "  <- TRUNCATED, accuracy unreliable"
        elif trans:
            flag = "  <- %d transient (rate_limit/api_error): re-run recovers these" % trans
        elif exc:
            # Say what the exclusions ARE, measured against the exclusions,
            # not against n: 13 refusals out of 13 exclusions is "all
            # refusals" even though 13 is under 5% of 347.
            empty = exc - ref
            if ref and not empty:
                flag = "  <- exclusions are refusals (0 tokens, provider-flagged)"
            elif ref:
                flag = "  <- %d refusals + %d empty (budget exhausted)" % (ref, empty)
            elif exc > n * 0.02:
                flag = "  <- exclusions are empty answers: model exhausted its budget"
        print("%-9s %-22s %7d %9d %10d %8d %9s%s"
              % (track, model, n, exc, trunc, ref, acc, flag))
    print()
    print("excluded = provider failures, empty answers and refusals; NOT scored as wrong.")
    print("A nonzero excluded count means the number above rests on fewer")
    print("questions than the bank holds -- lower concurrency and re-run those.")
    if any(ref for *_, ref, _ in rows):
        print()
        print("refused = provider-flagged refusal with STRICTLY zero output tokens.")
        print("Counted inside `excluded`: the model gave no answer, so there is")
        print("nothing to grade. A refusal that emitted text is graded as that text.")
    if any(trunc for *_, trunc, _, _ in rows):
        print()
        print("truncated = answers that ran into --max-tokens and stopped mid-answer.")
        print("They are scored on the fragment that arrived, so a high count means")
        print("the accuracy beside it measures the cap, not the model. Raise")
        print("MAX_TOKENS and re-run those models with --force.")
PY
fi
echo
echo "output: $OUT_ROOT"
echo "re-run the same command to retry only what failed (--force to redo all)."
exit $sweep_status
