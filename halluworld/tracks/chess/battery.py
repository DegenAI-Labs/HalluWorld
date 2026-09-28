"""chess_full_probes.py — chess probe battery (HalluWorld), sane defaults + optional stress.

What "capture_bias" means (not a model setting):
  When a probe *synthesizes* a random move sequence (SAN continuation or capture
  count), each ply rolls a die: with probability ``capture_bias`` it picks a
  uniform random *capture* among legal captures (if any); otherwise a uniform
  random legal move. That only makes the generated game tactically messier;
  ground truth still comes from python-chess. ``0`` = purely random legal moves.

  ``setup_capture_bias`` on hypotheticals is the same idea but only for the
  single "setup" move in the what-if question.

Default run is **moderate** (FEN on, short warm-up, full board squares, mild
SAN depth). Set ``STRESS_MODE=1`` for the previous extreme profile (no FEN,
long capture-heavy SAN, midboard-only questions, etc.).

Run from repo root:
    export OPENAI_API_KEY=sk-...
    python -m halluworld.tracks.chess.battery

Environment:
    STRESS_MODE=1              enable extreme settings (see below)
    OPENAI_MODEL               (default: gpt-5)
    OPENAI_REASONING_EFFORT    (default: high; empty string = omit → API default)
    OPENAI_MAX_COMPLETION_TOKENS (optional; reasoning models default to 8192 in OpenAILM)
    OPENAI_TEMPERATURE         sampling temperature for non-reasoning OpenAI models
                               (reasoning models ignore this in OpenAILM; default: 0.0)
    N_EPISODES                 (default: 20)
    STEPS_BEFORE_PROBE         warm-up random plies (default: 2; stress: 10)
    INCLUDE_FEN                1/0 — include FEN (and ASCII board) in the **serializer** path for
                               probes that use the default board observation. ``chess_san_legal_move``
                               stays text-only (SAN ± optional text FEN lines); it never prepends the grid.
                               Stress defaults to 0 unless overridden; extreme defaults to 0 unless
                               overridden or ``OBSERVATION_FEN_MODE`` forces a misleading FEN line.
    OBSERVATION_FEN_MODE       ``truth`` (default), ``flip_turn``, ``transpose`` (swap two non-king
                               pieces), ``startpos`` (always show start FEN text), ``override`` (use
                               ``OBSERVATION_FEN_OVERRIDE``). Applies to the FEN line in the board
                               observation and to ``Starting FEN:`` in history-only probes when set
                               via :func:`configure_observation_fen_display` (set automatically in
                               ``chess_full_probes``). When this is not ``truth``/``none``, orthodox
                               runs force ``INCLUDE_FEN=1`` unless you explicitly set ``INCLUDE_FEN=0``.
                               ``SAN_OBSERVATION_MODE=san_only`` still hides the diagram; a separate
                               ``Reference FEN`` line is then appended to the SAN block so transpose
                               can appear.
    OBSERVATION_FEN_OVERRIDE   full FEN string when ``OBSERVATION_FEN_MODE=override``
    OBSERVATION_FEN_SEED       integer seed for ``transpose`` pair sampling (default ``0``)
    CONFLICTING_PROMPT_LIE     ``piece`` (default), ``fen_transpose``, ``fen_startpos``, or
                               ``fen_override`` (requires ``CONFLICTING_PROMPT_FALSE_FEN``) when
                               ``INCLUDE_CONFLICTING_PROMPT=1`` adds :class:`ChessConflictingPromptProbe`.
    CONFLICTING_PROMPT_FALSE_FEN  explicit bogus FEN for ``fen_override`` lie mode
    SQUARE_SCOPE               ``all`` or ``midboard`` (default: all; stress: midboard)
    SAN_N_PLIES, SAN_MIN_PLIES, SAN_CAPTURE_BIAS, SAN_MAX_REPLAY_ATTEMPTS
    SAN_OBSERVATION_MODE       ``fen_and_san`` (default) or ``san_only`` (moves only;
                               requires SAN_REPLAY_FROM=startpos)
    SAN_REPLAY_FROM            ``env`` (default) or ``startpos`` (replay from initial setup)
    SAN_REPLY_CONSTRAINT       ``any`` (default); stress defaults to ``quiet``;
                               extreme defaults to ``quiet_non_promotion``. Also
                               ``non_capture``, ``non_promotion``, ``quiet_np`` (alias).
    SAN_MAX_CONSTRAINED_REPLIES  cap on how many moves may satisfy the reply
                               constraint in the terminal position (else retry); ``0``
                               disables. Defaults: stress ``12``, extreme ``8``.
    SAN_OMIT_SIDE_HINT         1/0 — with ``san_only`` + ``startpos``, omit the
                               “It is now …'s turn” line (default: 1 under stress/extreme
                               when those modes apply).
    DEFENDED_UNDER_ATTACK_ONLY 1/0 — defended probe only asks about attacked pieces
                               (default: 0; stress: 1 unless overridden)
    HYPOTHETICAL_SETUP_POOL    ``all`` (default) or ``non_capture`` (stress: non_capture)
    CHESS_CHAT_CONTEXT         1/0 — prepend synthetic past games + kibitz (+ optional lobby
                               lines) before the real observation (default: 0; stress or
                               ``CHESS_VARIANT=old_bishop`` / ``atomic``: 1 unless overridden).
    CHESS_CHAT_MIN_GAMES, CHESS_CHAT_MAX_GAMES — past-game block count (old-bishop defaults 10–16).
    CHESS_CHAT_SAN_MIN_PLIES, CHESS_CHAT_SAN_MAX_PLIES — length of each fake SAN line (defaults 18–44).
    CHESS_CHAT_RUMOR_RATE      probability of a bogus ``{uci}`` rumor per fake game (default 0.82
                               for old-bishop heavy prefix).
    CHESS_CHAT_LOBBY_LINES     extra synthetic chat lines after kibitz (default 12 for old-bishop
                               heavy prefix; 6 extreme / 4 stress when chat is on; else 0).
    CHESS_CHAT_PREFIX_STACK    repeat the full noise block N times before ``Current game:`` (default 3
                               for old-bishop path, 1 otherwise). Each repeat is freshly randomized.
    CHESS_CHAT_POST_OBS_LINES  extra noise lines *after* the real diagram (default 20 for old-bishop).
    STEPS_BEFORE_PROBE         random warm-up plies before each probe (default 48 for old-bishop / atomic when
                               not stress/extreme; else 2 / 10 / 14). Massively buries the task board.
    CHESS_EXTREME=1            frontier-tier profile: longer SAN-only + stricter reply,
                               extra probes (attacker, pin, capture count, conflicting
                               prompt, legal-move foil, history readout, is_capture hypo),
                               heavier chat prefix, no FEN unless overridden.
    CHESS_HARD_CAPTURES=1      bias ``can_capture`` negatives toward pseudo-legal
                               captures that are illegal (pins); default on under stress
                               or extreme unless overridden.
    CAPTURE_COUNT_N_PLIES, CAPTURE_COUNT_CAPTURE_BIAS
    HARD_HIDDEN_MIN_PLIES, HARD_HIDDEN_MAX_PLIES, HARD_HIDDEN_CAPTURE_BIAS
                               — long hidden-history side capture stats (orthodox battery); defaults
                               scale up under STRESS_MODE / CHESS_EXTREME unless you set these explicitly.
    TWO_STEP_HANG_MAX_TRIES    resampling cap for ``ChessTwoStepHangingProbe`` (default ``60``).
    MATE_STALE_MAX_ATTEMPTS, MATE_STALE_MAX_PLIES, MATE_STALE_CAPTURE_BIAS
                               — mate/stalemate confusion probe (random self-play from startpos).
    UNDEFENDED_COUNT_MODE      ``post`` (count pieces with no defender after the line) or ``delta``
                               (count pieces that **lost** all friendly defenders vs the diagram);
                               default ``post``; ``CHESS_EXTREME=1`` defaults to ``delta`` unless set.
    UNDEFENDED_HYPOTHESIS_PLIES  ``1`` or ``2`` hypothetical plies (default ``1``; extreme defaults ``2``).
    UNDEFENDED_MAX_TRIES       resampling cap for the two-ply case (default scales with stress/extreme).
    HYPOTHETICAL_SETUP_CAPTURE_BIAS
    BENCHMARK_VERBOSE=1        per-trial lines from run_benchmark (default: off)
    ENABLE_BENCHMARKS         comma-separated benchmark keys to run (allowlist),
                              e.g. ``chess_san_legal_move,chess_hidden_side_capture_stats``.
    DISABLE_BENCHMARKS        comma-separated benchmark keys to skip (blocklist).
    RESULTS_SUBDIR             subdirectory under results/chess/ for this batch of runs
                               (default: today's UTC date, e.g. ``20260504``). Results are
                               laid out as ``results/chess/<RESULTS_SUBDIR>/<model>/<fen-mode>/``.
    USE_STUB_LM=1              offline (will score ~0 on non-yes/no probes)
    LM_PROVIDER                ``openai`` (default), ``anthropic``, or ``baseten``.
    ANTHROPIC_MODEL            Anthropic model slug (default: claude-sonnet-4-6).
    ANTHROPIC_THINKING_EFFORT  low / medium / high / max — adaptive thinking (Claude 4.6 only).
    ANTHROPIC_TEMPERATURE      sampling temperature; defaults to 1.0 when thinking_effort is set,
                               0.0 otherwise.
    ANTHROPIC_MAX_TOKENS       max tokens to generate (default: 16384).
    BASETEN_MODEL              required: Baseten model ID (e.g. ``zai-org/GLM-5``,
                               ``qwen-3-30b-instruct``).
    BASETEN_BASE_URL           Baseten endpoint (default: serverless
                               ``https://inference.baseten.co/v1``; deployed models use a
                               model-specific URL from the BASETEN_PORT_GUIDE).
    BASETEN_TEMPERATURE        sampling temperature (default: 0.0).
    BASETEN_MAX_TOKENS         max tokens to generate (default: 16384).
    USE_LICHESS=1
    CHESS_VARIANT              ``old_bishop`` (aliases ``old-bishop``, ``short_bishop``): short-range-bishop
                               board + **UCI** probes (any legal, bishop-only, quiet-only, and
                               FIDE-legal-but-variant-illegal bishop slides) scored with
                               ``ChessLegalUciSetEvaluator``. Legacy yes/no probes remain in
                               ``chess_old_bishop.py`` but are not in the default battery. ``USE_LICHESS`` ignored.
                               Chat stuffing before the observation is **on by default** (see
                               ``CHESS_CHAT_CONTEXT``); tune volume with ``CHESS_CHAT_*`` vars below.
    OLD_BISHOP_WARMUP_STEPS    default ``STEPS_BEFORE_PROBE`` when variant is on and not stress/extreme
                               (default ``48`` random plies before each probe).
    OLD_BISHOP_HIDE_FEN        1/0 — hide FEN from the serializer in variant mode (default: 1).
    OLD_BISHOP_UCI_FAKE_FIDE_RATE, OLD_BISHOP_UCI_MINIMAL_RULES_RATE — rule-bloat / red-herring rates
                               for the UCI probes (defaults ``0.5`` / ``0.4``).
    CHESS_VARIANT=atomic       (alias ``atomic_chess``): Lichess atomic + **UCI-named hypotheticals** only
                               (one or more legal moves in order, then piece-on-square), scored with
                               ``ChessPieceNameEvaluator``. ``USE_LICHESS`` (puzzles) ignored.
                               Heavy chat prefix on by default like old_bishop.
    ATOMIC_OBS_MODE            ``grid`` (default) or ``san_history`` — SAN list from episode root FEN
                               through warm-up (no board diagram).
    ATOMIC_HYPOTHESIS_PLIES_MIN, ATOMIC_HYPOTHESIS_PLIES_MAX — length of the hypothetical move chain
                               (defaults ``1`` / ``3``).
    ATOMIC_USE_HF_FENS=1      stream rated atomic games from Hugging Face ``Lichess/atomic-chess-games``
                               into midgame FENs (needs ``pip install datasets``). If loading fails or yields
                               nothing, falls back to the builtin **10** FEN pool (6 default + 4 extras).
    ATOMIC_HF_TARGET_FENS, ATOMIC_HF_MAX_SCAN_ROWS, ATOMIC_HF_MIN_PLIES, ATOMIC_HF_MAX_PLIES, ATOMIC_HF_SEED
    ATOMIC_HF_FENS_CACHE       optional JSON path to cache / read the HF-derived FEN list
    ATOMIC_WARMUP_STEPS        default ``STEPS_BEFORE_PROBE`` for atomic when not stress/extreme (``48``).
    ATOMIC_HIDE_FEN            1/0 — hide FEN in serializer (default: ``1``).
    ATOMIC_UCI_FAKE_FIDE_RATE, ATOMIC_UCI_MINIMAL_RULES_RATE — preamble noise (defaults ``0.5`` / ``0.4``).
"""
from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except ImportError:
    def load_dotenv() -> None:
        return None

from halluworld.benchmark import SYSTEM_PROMPT, run_benchmark
from halluworld.tracks.chess.distractor_context import prefix_chess_observation_with_chat_context
from halluworld.tracks.chess import load_lichess_puzzle_fens, make_chess_env
from halluworld.tracks.chess import ChessIntegerEvaluator, ChessLegalUciSetEvaluator, ChessPieceNameEvaluator, ChessYesNoEvaluator
from halluworld.tracks.chess.probes.atomic import AtomicHypotheticalPieceAfterUciProbe
from halluworld.tracks.chess.probes.old_bishop import OldBishopFideIllegalBishopUciProbe, OldBishopLegalAnyUciProbe, OldBishopLegalBishopUciProbe, OldBishopLegalQuietUciProbe
from halluworld.lm import StubLM

try:
    from halluworld.lm import OpenAILM  # type: ignore[attr-defined]
except ImportError:
    OpenAILM = None  # type: ignore[assignment]

try:
    from halluworld.lm import AnthropicLM  # type: ignore[attr-defined]
except ImportError:
    AnthropicLM = None  # type: ignore[assignment]

try:
    from halluworld.lm import BasetenLM  # type: ignore[attr-defined]
except ImportError:
    BasetenLM = None  # type: ignore[assignment]
from halluworld.tracks.chess import ChessAfterMoveUndefendedCountProbe, ChessAttackerProbe, ChessCanCaptureProbe, ChessCaptureCountProbe, ChessConflictingPromptProbe, ChessDefendedProbe, ChessHangingProbe, ChessHiddenSideCaptureStatsProbe, ChessHistoryReadoutProbe, ChessHypotheticalProbe, ChessLegalMoveProbe, ChessMateStalemateConfusionProbe, ChessPinProbe, ChessSanLegalContinuationProbe, ChessTwoStepHangingProbe
from halluworld.tracks.chess import ChessSerializer
from halluworld.tracks.chess.serializers import FenDisplayConfig, configure_observation_fen_display


class _TransformedObservationSerializer:
    """Serializer wrapper that post-processes observation text.

    Needed for compatibility with benchmark APIs that removed
    ``user_observation_transform`` and only accept ``include_obs``.
    """

    def __init__(self, base_serializer, transform_fn, rng_seed: int):
        self._base = base_serializer
        self._transform_fn = transform_fn
        self._rng = random.Random(rng_seed)

    def serialize(self, env):
        obs = self._base.serialize(env)
        return self._transform_fn(obs, self._rng)


def _env_bool(key: str, default: bool) -> bool:
    if key not in os.environ:
        return default
    return os.environ[key].strip().lower() not in ("0", "false", "no", "")


def _parse_observation_fen_display() -> FenDisplayConfig:
    mode = os.environ.get("OBSERVATION_FEN_MODE", "truth").strip().lower()
    allowed = ("truth", "none", "flip_turn", "transpose", "swap", "startpos", "override", "replace")
    if mode not in allowed:
        raise ValueError(f"OBSERVATION_FEN_MODE must be one of {allowed}, got {mode!r}")
    override = os.environ.get("OBSERVATION_FEN_OVERRIDE", "").strip() or None
    seed = int(os.environ.get("OBSERVATION_FEN_SEED", "0"))
    return FenDisplayConfig(mode=mode, override_fen=override, seed=seed)


def _make_chat_observation_transform(
    *,
    heavy_chat: bool,
    extreme: bool,
    stress: bool,
):
    """Synthetic past games + kibitz + optional lobby noise before ``Current game:``."""

    def _transform(obs: str, rng: random.Random) -> str:
        if heavy_chat:
            return prefix_chess_observation_with_chat_context(
                obs,
                rng,
                min_games=int(os.environ.get("CHESS_CHAT_MIN_GAMES", "10")),
                max_games=int(os.environ.get("CHESS_CHAT_MAX_GAMES", "16")),
                past_san_min_plies=int(os.environ.get("CHESS_CHAT_SAN_MIN_PLIES", "18")),
                past_san_max_plies=int(os.environ.get("CHESS_CHAT_SAN_MAX_PLIES", "44")),
                rumor_rate=float(os.environ.get("CHESS_CHAT_RUMOR_RATE", "0.82")),
                lobby_lines=int(os.environ.get("CHESS_CHAT_LOBBY_LINES", "12")),
                prefix_stack=int(os.environ.get("CHESS_CHAT_PREFIX_STACK", "3")),
                post_observation_lines=int(os.environ.get("CHESS_CHAT_POST_OBS_LINES", "20")),
            )
        if extreme:
            return prefix_chess_observation_with_chat_context(
                obs,
                rng,
                min_games=6,
                max_games=9,
                past_san_min_plies=10,
                past_san_max_plies=32,
                rumor_rate=0.68,
                lobby_lines=int(os.environ.get("CHESS_CHAT_LOBBY_LINES", "6")),
                prefix_stack=int(os.environ.get("CHESS_CHAT_PREFIX_STACK", "1")),
                post_observation_lines=int(os.environ.get("CHESS_CHAT_POST_OBS_LINES", "0")),
            )
        if stress:
            return prefix_chess_observation_with_chat_context(
                obs,
                rng,
                min_games=4,
                max_games=6,
                past_san_min_plies=6,
                past_san_max_plies=26,
                rumor_rate=0.52,
                lobby_lines=int(os.environ.get("CHESS_CHAT_LOBBY_LINES", "4")),
                prefix_stack=int(os.environ.get("CHESS_CHAT_PREFIX_STACK", "1")),
                post_observation_lines=int(os.environ.get("CHESS_CHAT_POST_OBS_LINES", "0")),
            )
        return prefix_chess_observation_with_chat_context(
            obs,
            rng,
            lobby_lines=int(os.environ.get("CHESS_CHAT_LOBBY_LINES", "0")),
            prefix_stack=int(os.environ.get("CHESS_CHAT_PREFIX_STACK", "1")),
            post_observation_lines=int(os.environ.get("CHESS_CHAT_POST_OBS_LINES", "0")),
        )

    return _transform


def _timestamp() -> str:
    # ISO-ish but filesystem friendly.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _model_slug(lm) -> str:
    """Return a filesystem-safe name describing the LM configuration."""
    if isinstance(lm, StubLM):
        return "stub"
    # AnthropicLM
    if AnthropicLM is not None and isinstance(lm, AnthropicLM):
        slug = lm.model
        if getattr(lm, "thinking_effort", None):
            slug += f"--thinking-{lm.thinking_effort}"
        return slug
    # BasetenLM
    if BasetenLM is not None and isinstance(lm, BasetenLM):
        safe = lm.model.replace("/", "-").replace(":", "-")
        return f"{safe}--baseten"
    # OpenAILM (and fallback)
    model = getattr(lm, "model", "unknown")
    effort = getattr(lm, "reasoning_effort", None)
    slug = model
    if effort:
        slug += f"--r{effort}"
    return slug


def _fen_tag() -> str:
    """Derive a short tag from the current FEN environment variables."""
    include_fen = os.environ.get("INCLUDE_FEN", "").strip()
    if include_fen == "0":
        return "fen-off"
    mode = os.environ.get("OBSERVATION_FEN_MODE", "truth").strip().lower()
    if mode == "transpose":
        return "fen-transpose"
    if mode == "flip_turn":
        return "fen-flip"
    if mode == "startpos":
        return "fen-startpos"
    return "fen-on"


def _results_dir(lm=None) -> Path:
    """Return (and create) the output directory for this run.

    Layout::

        results/chess/<RESULTS_SUBDIR>/<model_slug>/<fen_tag>/

    ``RESULTS_SUBDIR`` defaults to today's date (e.g. ``20260504``).

    The root is the current working directory unless HALLUWORLD_RESULTS_ROOT
    says otherwise. It deliberately does NOT count parent directories from
    __file__: this module used to live at examples/chess_full_probes.py, where
    parents[1] was the repository root, and moving it to
    halluworld/tracks/chess/ silently redirected every run's output into
    halluworld/tracks/results/ -- inside the installed package.
    """
    direct = os.environ.get("HALLUWORLD_RESULTS_DIR", "").strip()
    if direct:
        # Unified `halluworld eval chess --out ...` evaluates one model and
        # one condition per invocation, so the requested directory is exact.
        base = Path(direct)
    else:
        root = Path(os.environ.get("HALLUWORLD_RESULTS_ROOT") or Path.cwd())
        subdir = os.environ.get(
            "RESULTS_SUBDIR",
            datetime.now(timezone.utc).strftime("%Y%m%d"),
        ).strip()
        base = root / "results" / "chess" / subdir
        if lm is not None:
            base = base / _model_slug(lm) / _fen_tag()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_run_artifacts(run, out_dir: Path) -> tuple[Path, Path]:
    summary_path = out_dir / "summary.log"
    examples_path = out_dir / "examples.txt"

    # If files already exist (re-run), rotate them with a timestamp suffix.
    if summary_path.exists() or examples_path.exists():
        ts = _timestamp()
        summary_path = out_dir / f"summary_{ts}.log"
        examples_path = out_dir / f"examples_{ts}.txt"

    summary_text = run.summary().to_string(index=False)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    # Save one concrete prompt per probe type (first occurrence).
    seen: set[str] = set()
    lines: list[str] = []
    for r in run.results:
        if r.probe_name in seen:
            continue
        seen.add(r.probe_name)

        # Prefer the exact prompt as sent to the model.
        user_prompt = ""
        if getattr(r, "messages", None):
            for msg in r.messages:
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break
        if not user_prompt:
            user_prompt = f"## Current observation\n(unavailable)\n\n## Question\n{r.question}"
        lines.append("=" * 88)
        lines.append(f"probe: {r.probe_name}")
        lines.append(f"episode_id: {r.episode_id}")
        lines.append("")
        lines.append("SYSTEM:")
        lines.append(SYSTEM_PROMPT)
        lines.append("")
        lines.append("USER:")
        lines.append(user_prompt)
        lines.append("")
        lines.append(f"ground_truth: {r.ground_truth!r}")
        lines.append(f"lm_response: {r.lm_response!r}")
        lines.append(f"is_correct: {r.is_correct} score: {r.score}")
        lines.append("")
        lines.append("metadata:")
        try:
            lines.append(str({k: r.metadata[k] for k in sorted(r.metadata.keys())}))
        except Exception:
            lines.append(str(r.metadata))
        lines.append("")

    examples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, examples_path


def _write_episode_results(run, out_dir: Path) -> Path:
    """Atomically persist every scored episode for downstream resampling."""
    results_path = out_dir / "results.jsonl"
    temporary_path = out_dir / "results.jsonl.tmp"
    run.to_jsonl(str(temporary_path))
    temporary_path.replace(results_path)
    return results_path


def lichess_pool_kwargs() -> dict:
    """Environment-driven arguments for load_lichess_puzzle_fens().

    These were hardcoded (seed=42, rating 1200-2200), which meant USE_LICHESS=1
    drew the identical 200 puzzles on every run no matter what BENCHMARK_SEED
    said -- so a held-out set could not get new positions out of the puzzle
    corpus. The defaults reproduce the previous behaviour exactly.

    question_set.py (generation) and this module (scoring) both call the loader,
    so both must read the same knobs or a v0.2 bank and the run that scores it
    would silently disagree about which positions they are using.

    LICHESS_MIN_RATING/LICHESS_MAX_RATING also give non-overlap by construction:
    a band disjoint from v0.1's 1200-2200 cannot return any of its puzzles.
    """
    themes = [t.strip() for t in os.environ.get("LICHESS_THEMES", "").split(",") if t.strip()]
    return {
        "num_samples": int(os.environ.get("LICHESS_SAMPLES", "200")),
        "seed": int(os.environ.get("LICHESS_SEED", "42")),
        "min_rating": int(os.environ.get("LICHESS_MIN_RATING", "1200")),
        "max_rating": int(os.environ.get("LICHESS_MAX_RATING", "2200")),
        "themes": themes or None,
    }


def _csv_env(name: str) -> set[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _probe_key(probe) -> str:
    """Canonical key used by ENABLE_BENCHMARKS / DISABLE_BENCHMARKS."""
    name = probe.__class__.__name__
    mapping = {
        "ChessCanCaptureProbe": "chess_can_capture",
        "ChessDefendedProbe": "chess_defended",
        "ChessHangingProbe": "chess_hanging",
        "ChessHypotheticalProbe": "chess_hypothetical_in_check",
        "ChessSanLegalContinuationProbe": "chess_san_legal_move",
        "ChessHiddenSideCaptureStatsProbe": "chess_hidden_side_capture_stats",
        "ChessAfterMoveUndefendedCountProbe": "chess_after_move_undefended_count",
        "ChessConflictingPromptProbe": "chess_conflicting_prompt",
        "OldBishopLegalAnyUciProbe": "chess_old_bishop_legal_any_uci",
        "OldBishopLegalBishopUciProbe": "chess_old_bishop_legal_bishop_uci",
        "OldBishopLegalQuietUciProbe": "chess_old_bishop_legal_quiet_uci",
        "OldBishopFideIllegalBishopUciProbe": "chess_old_bishop_fide_illegal_bishop_uci",
        "AtomicHypotheticalPieceAfterUciProbe": "chess_atomic_hypothetical_piece_after_uci",
    }
    return mapping.get(name, name.lower())


def _apply_probe_filters(pairs: list[tuple]) -> list[tuple]:
    """Filter probes using ENABLE_BENCHMARKS / DISABLE_BENCHMARKS env vars."""
    enable = _csv_env("ENABLE_BENCHMARKS")
    disable = _csv_env("DISABLE_BENCHMARKS")
    if not enable and not disable:
        return pairs

    pair_with_keys = [(_probe_key(p), p, e) for p, e in pairs]
    available = {k for k, _, _ in pair_with_keys}

    unknown_enable = sorted(enable - available)
    unknown_disable = sorted(disable - available)
    if unknown_enable or unknown_disable:
        raise ValueError(
            "Unknown benchmark key(s): "
            + ", ".join(unknown_enable + unknown_disable)
            + f". Available: {', '.join(sorted(available))}"
        )

    if enable and disable and (enable & disable):
        overlap = ", ".join(sorted(enable & disable))
        raise ValueError(f"Benchmark key(s) present in both ENABLE and DISABLE: {overlap}")

    filtered: list[tuple] = []
    selected_keys: list[str] = []
    for key, probe, evaluator in pair_with_keys:
        if enable and key not in enable:
            continue
        if key in disable:
            continue
        filtered.append((probe, evaluator))
        selected_keys.append(key)

    if not filtered:
        raise ValueError("Probe filtering removed all benchmarks; nothing to run.")

    print(
        f"[chess_full_probes] probe filter: ENABLE_BENCHMARKS={sorted(enable) if enable else None}  "
        f"DISABLE_BENCHMARKS={sorted(disable) if disable else None}"
    )
    print(f"[chess_full_probes] selected benchmarks ({len(selected_keys)}): {', '.join(selected_keys)}")
    return filtered


def _validate_probe_generation(env, serializer: ChessSerializer, probes: list, *, n_env_resets: int) -> None:
    """Catch misconfigurations (excess skips, empty legal sets, etc.)."""
    import random as _random

    rng = _random.Random(0)
    per_probe = {p.__class__.__name__: {"ok": 0, "skipped": 0, "bad": 0} for p in probes}

    for _ in range(n_env_resets):
        env.reset(seed=rng.randint(0, 2**31 - 1))
        _ = serializer.serialize(env)  # ensure serializer runs
        for p in probes:
            name = p.__class__.__name__
            pr = p.generate(env)
            if pr is None or pr.metadata.get("skipped"):
                per_probe[name]["skipped"] += 1
                continue
            # Minimal invariants.
            bad = False
            if pr.question.strip() == "":
                bad = True
            if pr.ground_truth is None:
                bad = True
            if pr.probe_type in (
                "chess_san_legal_move",
                "chess_old_bishop_legal_any_uci",
                "chess_old_bishop_legal_bishop_uci",
                "chess_old_bishop_legal_quiet_uci",
                "chess_old_bishop_fide_illegal_bishop_uci",
            ):
                legal_ucis = pr.metadata.get("legal_ucis") or pr.ground_truth
                if not isinstance(legal_ucis, list) or len(legal_ucis) == 0:
                    bad = True
            if pr.probe_type == "chess_atomic_hypothetical_piece_after_uci":
                gt = pr.ground_truth
                if not isinstance(gt, str) or not gt.strip():
                    bad = True
            if pr.probe_type in (
                "chess_capture_count",
                "chess_hidden_side_capture_stats",
                "chess_after_move_undefended_count",
                "chess_piece_count",
            ):
                if not isinstance(pr.ground_truth, int) or pr.ground_truth < 0:
                    bad = True
            if pr.probe_type in ("chess_two_step_hanging", "chess_mate_stalemate_confusion"):
                if not isinstance(pr.ground_truth, bool):
                    bad = True
            if bad:
                per_probe[name]["bad"] += 1
            else:
                per_probe[name]["ok"] += 1

    # Fail fast on obviously broken probes.
    failures: list[str] = []
    for name, s in per_probe.items():
        total = s["ok"] + s["skipped"] + s["bad"]
        if total == 0:
            failures.append(f"{name}: never generated")
            continue
        # If we can't generate a non-skipped instance at least sometimes, it's probably too strict.
        if s["ok"] == 0:
            failures.append(f"{name}: 0 ok, skipped={s['skipped']} bad={s['bad']}")
        # Bad should basically never happen.
        if s["bad"] > 0:
            failures.append(f"{name}: bad={s['bad']} (ok={s['ok']} skipped={s['skipped']})")

    if failures:
        raise RuntimeError(
            "Probe validation failed:\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\nTry lowering SAN_MIN_PLIES, lowering capture biases, raising "
            "SAN_MAX_CONSTRAINED_REPLIES (or set to 0 to disable), disabling CHESS_EXTREME, "
            "or disabling STRESS_MODE."
        )


def build_probes_and_evaluators() -> tuple[list, list, bool]:
    stress = os.environ.get("STRESS_MODE", "0").strip() == "1"
    extreme = os.environ.get("CHESS_EXTREME", "0").strip() == "1"
    stress_or_extreme = stress or extreme

    variant = os.environ.get("CHESS_VARIANT", "").strip().lower()
    if variant in ("atomic", "atomic_chess"):
        if "INCLUDE_FEN" in os.environ:
            include_fen_at = _env_bool("INCLUDE_FEN", True)
        else:
            include_fen_at = not _env_bool("ATOMIC_HIDE_FEN", True)
        uci_fake = float(os.environ.get("ATOMIC_UCI_FAKE_FIDE_RATE", "0.5"))
        uci_min = float(os.environ.get("ATOMIC_UCI_MINIMAL_RULES_RATE", "0.4"))
        hp_lo = int(os.environ.get("ATOMIC_HYPOTHESIS_PLIES_MIN", "1"))
        hp_hi = int(os.environ.get("ATOMIC_HYPOTHESIS_PLIES_MAX", "3"))
        piece_ev = ChessPieceNameEvaluator
        pairs_at = [
            (
                AtomicHypotheticalPieceAfterUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(921),
                    hypothesis_plies_min=hp_lo,
                    hypothesis_plies_max=hp_hi,
                ),
                piece_ev(),
            ),
            (
                AtomicHypotheticalPieceAfterUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(922),
                    hypothesis_plies_min=hp_lo,
                    hypothesis_plies_max=hp_hi,
                ),
                piece_ev(),
            ),
            (
                AtomicHypotheticalPieceAfterUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(923),
                    hypothesis_plies_min=hp_lo,
                    hypothesis_plies_max=hp_hi,
                ),
                piece_ev(),
            ),
            (
                AtomicHypotheticalPieceAfterUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(924),
                    hypothesis_plies_min=hp_lo,
                    hypothesis_plies_max=hp_hi,
                ),
                piece_ev(),
            ),
        ]
        print(
            f"[chess_full_probes] CHESS_VARIANT={variant!r}  "
            f"INCLUDE_FEN={include_fen_at!r}  battery=atomic_hypothetical_piece×4  "
            f"hypothesis_plies=[{hp_lo},{hp_hi}]  "
            f"ATOMIC_UCI_FAKE_FIDE_RATE={uci_fake}  ATOMIC_UCI_MINIMAL_RULES_RATE={uci_min}"
        )
        pairs_at_filtered = _apply_probe_filters(pairs_at)
        return [p for p, _ in pairs_at_filtered], [e for _, e in pairs_at_filtered], include_fen_at

    if variant in ("old_bishop", "old-bishop", "short_bishop"):
        if "INCLUDE_FEN" in os.environ:
            include_fen_ob = _env_bool("INCLUDE_FEN", True)
        else:
            include_fen_ob = not _env_bool("OLD_BISHOP_HIDE_FEN", True)
        uci_fake = float(os.environ.get("OLD_BISHOP_UCI_FAKE_FIDE_RATE", "0.5"))
        uci_min = float(os.environ.get("OLD_BISHOP_UCI_MINIMAL_RULES_RATE", "0.4"))
        legal_uci_ev = ChessLegalUciSetEvaluator
        pairs_ob = [
            (
                OldBishopLegalAnyUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(911),
                ),
                legal_uci_ev(),
            ),
            (
                OldBishopLegalBishopUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(912),
                ),
                legal_uci_ev(),
            ),
            (
                OldBishopLegalQuietUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(913),
                ),
                legal_uci_ev(),
            ),
            (
                OldBishopFideIllegalBishopUciProbe(
                    fake_fide_rate=uci_fake,
                    minimal_rules_rate=uci_min,
                    rng=random.Random(914),
                ),
                legal_uci_ev(),
            ),
        ]
        print(
            f"[chess_full_probes] CHESS_VARIANT={variant!r}  "
            f"INCLUDE_FEN={include_fen_ob!r}  battery=uci×4  "
            f"OLD_BISHOP_UCI_FAKE_FIDE_RATE={uci_fake}  OLD_BISHOP_UCI_MINIMAL_RULES_RATE={uci_min}"
        )
        pairs_ob_filtered = _apply_probe_filters(pairs_ob)
        return [p for p, _ in pairs_ob_filtered], [e for _, e in pairs_ob_filtered], include_fen_ob

    if "INCLUDE_FEN" in os.environ:
        include_fen = _env_bool("INCLUDE_FEN", True)
    else:
        if extreme:
            include_fen = False
        else:
            include_fen = not stress

    # Misleading FEN modes need an actual ``FEN:`` line in ``render_full``; extreme/stress default to
    # include_fen=False, so turn it back on unless the user explicitly set INCLUDE_FEN=0.
    obs_fen_mode_for_include = os.environ.get("OBSERVATION_FEN_MODE", "truth").strip().lower()
    if obs_fen_mode_for_include not in ("truth", "none", ""):
        if "INCLUDE_FEN" in os.environ:
            if _env_bool("INCLUDE_FEN", True):
                include_fen = True
        else:
            include_fen = True

    square_scope = os.environ.get("SQUARE_SCOPE", "midboard" if stress_or_extreme else "all")

    if extreme:
        san_n_d, san_min_d, san_cap_d, san_retries_d = 108, 60, 0.80, 72
        san_obs_d, san_replay_d, san_reply_d = "san_only", "startpos", "quiet_non_promotion"
    elif stress:
        san_n_d, san_min_d, san_cap_d, san_retries_d = 88, 54, 0.72, 52
        san_obs_d, san_replay_d, san_reply_d = "fen_and_san", "env", "quiet"
    else:
        san_n_d, san_min_d, san_cap_d, san_retries_d = 40, 22, 0.28, 20
        san_obs_d, san_replay_d, san_reply_d = "fen_and_san", "env", "any"

    san_n = int(os.environ.get("SAN_N_PLIES", str(san_n_d)))
    san_min = int(os.environ.get("SAN_MIN_PLIES", str(san_min_d)))
    san_cap = float(os.environ.get("SAN_CAPTURE_BIAS", str(san_cap_d)))
    san_retries = int(os.environ.get("SAN_MAX_REPLAY_ATTEMPTS", str(san_retries_d)))
    san_obs = os.environ.get("SAN_OBSERVATION_MODE", san_obs_d).strip().lower()
    if san_obs not in ("fen_and_san", "san_only"):
        raise ValueError(f"SAN_OBSERVATION_MODE must be fen_and_san or san_only, got {san_obs!r}")
    san_replay = os.environ.get(
        "SAN_REPLAY_FROM", "startpos" if san_obs == "san_only" else san_replay_d
    ).strip().lower()
    if san_replay not in ("env", "startpos"):
        raise ValueError(f"SAN_REPLAY_FROM must be env or startpos, got {san_replay!r}")
    if san_obs == "san_only" and san_replay != "startpos":
        raise ValueError("SAN_OBSERVATION_MODE=san_only requires SAN_REPLAY_FROM=startpos")

    cc_n = int(os.environ.get("CAPTURE_COUNT_N_PLIES", str(44 if extreme else 36 if stress else 18)))
    cc_cap = float(
        os.environ.get("CAPTURE_COUNT_CAPTURE_BIAS", str(0.62 if extreme else 0.58 if stress else 0.18))
    )

    hyp_cap = float(
        os.environ.get(
            "HYPOTHETICAL_SETUP_CAPTURE_BIAS",
            str(0.85 if extreme else 0.75 if stress else 0.35),
        )
    )

    if "SAN_REPLY_CONSTRAINT" in os.environ:
        san_reply = os.environ["SAN_REPLY_CONSTRAINT"].strip().lower()
    else:
        san_reply = san_reply_d
    if san_reply not in (
        "any",
        "none",
        "non_capture",
        "non_promotion",
        "quiet",
        "quiet_np",
        "quiet_non_promotion",
        "",
    ):
        raise ValueError(
            "SAN_REPLY_CONSTRAINT must be any, non_capture, non_promotion, quiet, "
            f"quiet_non_promotion (or quiet_np), got {san_reply!r}"
        )

    if "SAN_MAX_CONSTRAINED_REPLIES" in os.environ:
        _raw_mc = os.environ["SAN_MAX_CONSTRAINED_REPLIES"].strip().lower()
        san_max_constrained = None if _raw_mc in ("", "0", "off", "none") else int(_raw_mc)
    else:
        san_max_constrained = 8 if extreme else (12 if stress else None)

    san_omit_side = _env_bool(
        "SAN_OMIT_SIDE_HINT",
        bool(stress_or_extreme and san_obs == "san_only" and san_replay == "startpos"),
    )

    defended_under_attack = _env_bool("DEFENDED_UNDER_ATTACK_ONLY", stress_or_extreme)

    hyp_pool = os.environ.get(
        "HYPOTHETICAL_SETUP_POOL", "non_capture" if stress_or_extreme else "all"
    ).strip().lower()
    if hyp_pool not in ("all", "non_capture"):
        raise ValueError(f"HYPOTHETICAL_SETUP_POOL must be all or non_capture, got {hyp_pool!r}")

    hist_n = int(os.environ.get("HISTORY_READOUT_N_PLIES", str(28 if extreme else 18 if stress else 12)))

    # Long hidden-history capture stats: scale with stress/extreme (override any time via HARD_HIDDEN_*).
    if extreme:
        _hhl, _hhh, _hhc = 52, 88, 0.58
    elif stress:
        _hhl, _hhh, _hhc = 34, 56, 0.52
    else:
        _hhl, _hhh, _hhc = 25, 40, 0.48
    hard_hidden_lo = int(os.environ.get("HARD_HIDDEN_MIN_PLIES", str(_hhl)))
    hard_hidden_hi = int(os.environ.get("HARD_HIDDEN_MAX_PLIES", str(_hhh)))
    hard_hidden_cap = float(os.environ.get("HARD_HIDDEN_CAPTURE_BIAS", str(_hhc)))
    two_step_hang_tries = int(
        os.environ.get("TWO_STEP_HANG_MAX_TRIES", str(120 if extreme else 90 if stress else 60))
    )
    mate_stale_attempts = int(
        os.environ.get("MATE_STALE_MAX_ATTEMPTS", str(180 if extreme else 150 if stress else 120))
    )
    mate_stale_plies = int(
        os.environ.get("MATE_STALE_MAX_PLIES", str(200 if extreme else 160 if stress else 120))
    )
    mate_stale_cap = float(
        os.environ.get("MATE_STALE_CAPTURE_BIAS", str(0.48 if extreme else 0.45 if stress else 0.42))
    )

    undef_mode = os.environ.get("UNDEFENDED_COUNT_MODE", "delta" if extreme else "post").strip().lower()
    if undef_mode not in ("post", "delta"):
        raise ValueError(f"UNDEFENDED_COUNT_MODE must be post or delta, got {undef_mode!r}")
    undef_hyp_plies = int(os.environ.get("UNDEFENDED_HYPOTHESIS_PLIES", str(2 if extreme else 1)))
    if undef_hyp_plies not in (1, 2):
        raise ValueError(f"UNDEFENDED_HYPOTHESIS_PLIES must be 1 or 2, got {undef_hyp_plies}")
    undef_max_tries = int(
        os.environ.get("UNDEFENDED_MAX_TRIES", str(120 if extreme else 90 if stress else 60))
    )

    hard_capture_default = stress_or_extreme
    if "CHESS_HARD_CAPTURES" in os.environ:
        hard_capture = _env_bool("CHESS_HARD_CAPTURES", False)
    else:
        hard_capture = hard_capture_default

    print(
        f"[chess_full_probes] STRESS_MODE={stress!r}  CHESS_EXTREME={extreme!r}  "
        f"INCLUDE_FEN={include_fen!r}  SQUARE_SCOPE={square_scope!r}  CHESS_HARD_CAPTURES={hard_capture!r}"
    )
    print(
        f"[chess_full_probes] SAN: n_plies={san_n}  min_plies={san_min}  capture_bias={san_cap}  "
        f"max_replay_attempts={san_retries}  observation_mode={san_obs!r}  replay_from={san_replay!r}  "
        f"reply_constraint={san_reply!r}  max_constrained_replies={san_max_constrained!r}  "
        f"omit_side_hint={san_omit_side!r}"
    )
    print(
        f"[chess_full_probes] capture_count: n_plies={cc_n}  capture_bias={cc_cap}  "
        f"hypothetical setup_capture_bias={hyp_cap}  hypothetical_setup_pool={hyp_pool!r}  "
        f"defended_under_attack_only={defended_under_attack!r}  history_readout_n_plies={hist_n}"
    )
    print(
        f"[chess_full_probes] hard_hidden_side_capture: min_plies={hard_hidden_lo}  max_plies={hard_hidden_hi}  "
        f"capture_bias={hard_hidden_cap!r}  two_step_hang_max_tries={two_step_hang_tries}  "
        f"mate_stale: max_attempts={mate_stale_attempts}  max_plies={mate_stale_plies}  "
        f"capture_bias={mate_stale_cap!r}"
    )
    print(
        f"[chess_full_probes] undefended_count: mode={undef_mode!r}  hypothesis_plies={undef_hyp_plies}  "
        f"max_tries={undef_max_tries}"
    )
    if _env_bool("INCLUDE_CONFLICTING_PROMPT", False):
        print(
            f"[chess_full_probes] conflicting_prompt: lie={os.environ.get('CONFLICTING_PROMPT_LIE', 'piece')!r}  "
            f"false_fen_set={bool(os.environ.get('CONFLICTING_PROMPT_FALSE_FEN', '').strip())}"
        )

    yes_no = ChessYesNoEvaluator
    integer = ChessIntegerEvaluator
    legal_uci_ev = ChessLegalUciSetEvaluator

    pairs: list = [
        # Apr 26 orthodox dynamics battery (see examples/chess_apr26_probes.py).
        (ChessCanCaptureProbe(square_scope=square_scope, prefer_ghost_captures=hard_capture, rng=random.Random(101)), yes_no()),
        (ChessDefendedProbe(square_scope=square_scope, under_attack_only=defended_under_attack, rng=random.Random(102)), yes_no()),
        (ChessHangingProbe(square_scope=square_scope, rng=random.Random(103)), yes_no()),
        (
            ChessHypotheticalProbe(
                predicate="in_check",
                setup_capture_bias=hyp_cap,
                setup_pool=hyp_pool,
                rng=random.Random(104),
            ),
            yes_no(),
        ),
        (
            ChessSanLegalContinuationProbe(
                n_plies=san_n,
                min_plies=san_min,
                capture_bias=san_cap,
                max_replay_attempts=san_retries,
                observation_mode=san_obs,
                replay_from=san_replay,
                reply_constraint=san_reply or "any",
                max_constrained_replies=san_max_constrained,
                omit_side_to_move=san_omit_side,
                prepend_terminal_board_grid=False,
                rng=random.Random(105),
            ),
            legal_uci_ev(),
        ),
        (
            ChessHiddenSideCaptureStatsProbe(
                min_plies=hard_hidden_lo,
                max_plies=hard_hidden_hi,
                rng=random.Random(501),
                capture_bias=hard_hidden_cap,
            ),
            integer(),
        ),
        (
            ChessAfterMoveUndefendedCountProbe(
                rng=random.Random(502),
                mode=undef_mode,
                hypothesis_plies=undef_hyp_plies,
                max_tries=undef_max_tries,
            ),
            integer(),
        ),
        # (
        #     ChessTwoStepHangingProbe(rng=random.Random(503), max_tries=two_step_hang_tries),
        #     yes_no(),
        # ),
        # (
        #     ChessMateStalemateConfusionProbe(
        #         rng=random.Random(504),
        #         max_attempts=mate_stale_attempts,
        #         max_plies=mate_stale_plies,
        #         capture_bias=mate_stale_cap,
        #     ),
        #     yes_no(),
        # ),
    ]

    if _env_bool("INCLUDE_CONFLICTING_PROMPT", False):
        lie = os.environ.get("CONFLICTING_PROMPT_LIE", "piece").strip().lower()
        if lie not in ("piece", "fen_transpose", "fen_startpos", "fen_override"):
            raise ValueError(
                f"CONFLICTING_PROMPT_LIE must be piece, fen_transpose, fen_startpos, or fen_override, got {lie!r}"
            )
        false_fen = os.environ.get("CONFLICTING_PROMPT_FALSE_FEN", "").strip() or None
        pairs.append(
            (
                ChessConflictingPromptProbe(
                    rng=random.Random(505),
                    lie_kind=lie,
                    false_fen_override=false_fen,
                ),
                yes_no(),
            )
        )

    # if extreme:
    #     pairs.extend(
    #         [
    #             (ChessAttackerProbe(square_scope=square_scope), yes_no()),
    #             (ChessPinProbe(square_scope=square_scope), yes_no()),
    #             (ChessCaptureCountProbe(n_plies=cc_n, capture_bias=cc_cap), integer()),
    #             (ChessConflictingPromptProbe(rng=random.Random(401)), yes_no()),
    #             (ChessLegalMoveProbe(legal_rate=0.22, rng=random.Random(402)), yes_no()),
    #             (ChessHistoryReadoutProbe(n_plies=hist_n, rng=random.Random(403)), ChessPieceNameEvaluator()),
    #             (
    #                 ChessHypotheticalProbe(
    #                     predicate="is_capture",
    #                     setup_capture_bias=hyp_cap,
    #                     setup_pool="all",
    #                     rng=random.Random(404),
    #                 ),
    #                 yes_no(),
    #             ),
    #         ]
    #     )

    pairs = _apply_probe_filters(pairs)
    return [p for p, _ in pairs], [e for _, e in pairs], include_fen


def main():
    load_dotenv()
    configure_observation_fen_display(None)
    benchmark_seed = 42

    variant = os.environ.get("CHESS_VARIANT", "").strip().lower()
    use_old_bishop = variant in ("old_bishop", "old-bishop", "short_bishop")
    use_atomic = variant in ("atomic", "atomic_chess")

    if (use_old_bishop or use_atomic) and os.environ.get("USE_LICHESS", "0").strip() == "1":
        print(f"[chess_full_probes] CHESS_VARIANT={variant!r}: ignoring USE_LICHESS.")

    if os.environ.get("USE_LICHESS", "0") == "1" and not use_old_bishop and not use_atomic:
        pool = lichess_pool_kwargs()
        print("[chess_full_probes] Loading Lichess puzzle FENs … %s" % pool)
        fens = load_lichess_puzzle_fens(**pool)
        env = make_chess_env(fens=fens, seed=pool["seed"])
        print("[chess_full_probes] Environment: ChessEnv (Lichess puzzle FENs)")
    elif use_old_bishop:
        from halluworld.tracks.chess.envs.old_bishop_chess_env import make_old_bishop_chess_env

        env = make_old_bishop_chess_env(seed=42)
        print("[chess_full_probes] Environment: OldBishopChessEnv")
    elif use_atomic:
        from halluworld.tracks.chess.envs.atomic_chess_env import make_atomic_chess_env

        fens_arg = None
        if _env_bool("ATOMIC_USE_HF_FENS", False):
            from halluworld.tracks.chess.rules.atomic_hf_fens import load_atomic_hf_fen_pool

            cache_raw = os.environ.get("ATOMIC_HF_FENS_CACHE", "").strip()
            cache_path = Path(cache_raw).expanduser() if cache_raw else None
            try:
                fens_arg = load_atomic_hf_fen_pool(
                    target=int(os.environ.get("ATOMIC_HF_TARGET_FENS", "400")),
                    max_scan=int(os.environ.get("ATOMIC_HF_MAX_SCAN_ROWS", "20000")),
                    seed=int(os.environ.get("ATOMIC_HF_SEED", "42")),
                    min_plies=int(os.environ.get("ATOMIC_HF_MIN_PLIES", "6")),
                    max_plies=int(os.environ.get("ATOMIC_HF_MAX_PLIES", "40")),
                    cache_path=cache_path,
                )
            except Exception as e:
                print(f"[chess_full_probes] ATOMIC_USE_HF_FENS failed ({e!r}); using builtin FEN pool.")
                fens_arg = None
            if fens_arg:
                print(f"[chess_full_probes] Atomic FEN pool: {len(fens_arg)} positions from Hugging Face / cache.")
            else:
                print("[chess_full_probes] Atomic FEN pool: HF empty or unavailable; using builtin FENs.")
        env = make_atomic_chess_env(fens=fens_arg, seed=42)
        print("[chess_full_probes] Environment: AtomicChessEnv")
    else:
        env = make_chess_env(seed=42)
        print("[chess_full_probes] Environment: ChessEnv (default FEN pool)")

    use_stub = os.environ.get("USE_STUB_LM", "0") == "1"
    if use_stub:
        lm = StubLM(mode="yes")
        print("[chess_full_probes] LM: StubLM(mode=yes)")
    else:
        provider = os.environ.get("LM_PROVIDER", "openai").strip().lower()

        if provider == "anthropic":
            if AnthropicLM is None:
                raise RuntimeError("anthropic not installed; pip install anthropic")
            ant_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            ant_effort = os.environ.get("ANTHROPIC_THINKING_EFFORT", "").strip() or None
            ant_temp_default = 1.0 if ant_effort else 0.0
            ant_temp = float(os.environ.get("ANTHROPIC_TEMPERATURE", str(ant_temp_default)))
            ant_max = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16384"))
            lm = AnthropicLM(
                model=ant_model,
                api_key=os.environ["ANTHROPIC_API_KEY"],
                temperature=ant_temp,
                max_tokens=ant_max,
                thinking_effort=ant_effort,
            )
            print(
                f"[chess_full_probes] LM: AnthropicLM(model={ant_model!r}"
                f" thinking_effort={ant_effort!r} temperature={ant_temp}"
                f" max_tokens={ant_max})"
            )

        elif provider == "baseten":
            if BasetenLM is None:
                raise RuntimeError("openai package not installed (required for Baseten); pip install openai")
            bt_model = os.environ.get("BASETEN_MODEL", "").strip()
            if not bt_model:
                raise ValueError("BASETEN_MODEL must be set when LM_PROVIDER=baseten")
            bt_url = os.environ.get(
                "BASETEN_BASE_URL", "https://inference.baseten.co/v1"
            ).strip()
            bt_temp = float(os.environ.get("BASETEN_TEMPERATURE", "0.0"))
            bt_max = int(os.environ.get("BASETEN_MAX_TOKENS", "16384"))
            lm = BasetenLM(
                model=bt_model,
                base_url=bt_url,
                api_key=os.environ.get("BASETEN_API_KEY"),
                temperature=bt_temp,
                max_tokens=bt_max,
            )
            print(
                f"[chess_full_probes] LM: BasetenLM(model={bt_model!r}"
                f" base_url={bt_url!r} temperature={bt_temp} max_tokens={bt_max})"
            )

        else:  # openai (default)
            if OpenAILM is None:
                raise RuntimeError(
                    "openai is not installed; set USE_STUB_LM=1 or `pip install openai`."
                )
            model = os.environ.get("OPENAI_MODEL", "gpt-5.5")
            effort_raw = os.environ.get("OPENAI_REASONING_EFFORT", "low").strip()
            reasoning_effort = effort_raw if effort_raw else None
            raw_max = os.environ.get("OPENAI_MAX_COMPLETION_TOKENS", "").strip()
            max_tokens = int(raw_max) if raw_max else None
            openai_temp = float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))
            lm = OpenAILM(
                model=model,
                api_key=os.environ["OPENAI_API_KEY"],
                temperature=openai_temp,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            )
            effort_sent = reasoning_effort if reasoning_effort is not None else "(omitted → API default)"
            max_sent = getattr(lm, "max_tokens", max_tokens)
            print(f"[chess_full_probes] LM: OpenAILM(model={model!r})")
            print(f"[chess_full_probes]     OPENAI_REASONING_EFFORT env={effort_raw!r}  sent={effort_sent!r}")
            print(f"[chess_full_probes]     OPENAI_TEMPERATURE env={openai_temp!r}  sent={getattr(lm, 'temperature', None)!r}")
            print(f"[chess_full_probes]     max_tokens (completion budget)={max_sent!r}")

    # Skip early if results already exist for this (model, FEN-mode) combination.
    out_dir = _results_dir(lm=lm)
    if (out_dir / "summary.log").exists() and (out_dir / "results.jsonl").exists():
        print(
            f"[chess_full_probes] SKIP — summary.log and results.jsonl already exist at {out_dir}\n"
            f"[chess_full_probes]   Delete both to force a re-run."
        )
        configure_observation_fen_display(None)
        return
    if (out_dir / "summary.log").exists():
        print(
            f"[chess_full_probes] INCOMPLETE — summary.log exists without results.jsonl at {out_dir}; "
            "rerunning to recover episode-level data."
        )

    probes, evaluators, include_fen = build_probes_and_evaluators()

    stress = os.environ.get("STRESS_MODE", "0").strip() == "1"
    extreme = os.environ.get("CHESS_EXTREME", "0").strip() == "1"
    if "CHESS_CHAT_CONTEXT" in os.environ:
        chat_context = _env_bool("CHESS_CHAT_CONTEXT", False)
    else:
        chat_context = use_old_bishop or use_atomic or stress or extreme
    if extreme:
        steps_default = 14
    elif stress:
        steps_default = 10
    elif use_old_bishop:
        steps_default = int(os.environ.get("OLD_BISHOP_WARMUP_STEPS", "48"))
    elif use_atomic:
        steps_default = int(os.environ.get("ATOMIC_WARMUP_STEPS", "48"))
    else:
        steps_default = 2
    steps_before = int(os.environ.get("STEPS_BEFORE_PROBE", str(steps_default)))

    obs_mode = os.environ.get("ATOMIC_OBS_MODE", "grid").strip().lower()
    if obs_mode not in ("grid", "san_history"):
        obs_mode = "grid"
    fen_cfg = _parse_observation_fen_display()
    configure_observation_fen_display(fen_cfg)
    serializer = ChessSerializer(
        include_legal_moves=False,
        include_fen=include_fen,
        observation_mode=obs_mode if use_atomic else "grid",
        fen_display=fen_cfg,
    )
    if chat_context:
        serializer = _TransformedObservationSerializer(
            serializer,
            _make_chat_observation_transform(
                heavy_chat=use_old_bishop or use_atomic,
                extreme=extreme,
                stress=stress,
            ),
            rng_seed=benchmark_seed,
        )
    print(
        f"[chess_full_probes] Serializer: include_legal_moves=False  include_fen={include_fen!r}  "
        f"observation_mode={obs_mode if use_atomic else 'grid'!r}  "
        f"observation_fen_mode={fen_cfg.mode!r}  observation_fen_seed={fen_cfg.seed}  "
        f"steps_before_probe={steps_before}  CHESS_CHAT_CONTEXT={chat_context!r}"
    )

    # Pre-run validations to catch “we broke the generator” vs “model is failing”.
    validate_resets = int(
        os.environ.get("VALIDATE_RESETS", str(55 if extreme else 35 if stress else 20))
    )
    _validate_probe_generation(env, serializer, probes, n_env_resets=validate_resets)
    n_eps = int(os.environ.get("N_EPISODES", "20"))
    print(f"[chess_full_probes] validate_resets={validate_resets}  N_EPISODES={n_eps}")

    print("[chess_full_probes] Running benchmark …")
    run = run_benchmark(
        env=env,
        serializer=serializer,
        probes=probes,
        lm=lm,
        evaluators=evaluators,
        n_episodes=n_eps,
        steps_before_probe=steps_before,
        seed=benchmark_seed,
        verbose=os.environ.get("BENCHMARK_VERBOSE", "0").strip() == "1",
        include_obs=True,
    )

    print(run.summary().to_string(index=False))
    run_context = {
        "results_subdir": os.environ.get("RESULTS_SUBDIR", "").strip(),
        "model_slug": _model_slug(lm),
        "fen_mode": _fen_tag(),
        "include_fen": include_fen,
        "observation_fen_mode": fen_cfg.mode,
        "observation_fen_seed": fen_cfg.seed,
    }
    for result in run.results:
        result.metadata.update(run_context)
    results_path = _write_episode_results(run, out_dir)
    summary_path, examples_path = _write_run_artifacts(run, out_dir)
    print(f"Wrote episode results to: {results_path}")
    print(f"Wrote results to: {summary_path}")
    print(f"Wrote examples to: {examples_path}")
    configure_observation_fen_display(None)


if __name__ == "__main__":
    main()
