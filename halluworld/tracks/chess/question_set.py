"""Export deterministic chess benchmark question sets (prompt + ground truth).

This script uses the same probe/environment configuration path as
`halluworld/tracks/chess/battery.py`, but runs with `StubLM` and exports every
generated benchmark item to files for reuse/auditing.

Outputs:
  - questions.jsonl            machine-readable full dataset
  - questions.txt              human-readable full dataset
  - by_probe/<probe>.txt       human-readable grouped by probe

Usage (repo root):
  python3 -m halluworld.tracks.chess.question_set

Environment mirrors `battery.py` for probe/env configuration.
Useful knobs:
  N_EPISODES=50
  STEPS_BEFORE_PROBE=...
  STRESS_MODE=1 / CHESS_EXTREME=1
  INCLUDE_FEN=...
  OBSERVATION_FEN_MODE=...
  ENABLE_BENCHMARKS=...
  DISABLE_BENCHMARKS=...
  QUESTION_SET_NAME=my_tag
"""

from __future__ import annotations

import json
import os
import sys
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halluworld.benchmark import run_benchmark
from halluworld.tracks.chess import load_lichess_puzzle_fens, make_chess_env
from halluworld.lm import StubLM
from halluworld.tracks.chess import ChessSerializer
from halluworld.tracks.chess.serializers import configure_observation_fen_display

_THIS_DIR = Path(__file__).resolve().parent
# battery.py was examples/chess_full_probes.py before the track reorganization
# (2026-08-26); this loads it by path rather than a normal import because
# question_set.py needs battery.py's private helpers (_env_bool,
# build_probes_and_evaluators, _TransformedObservationSerializer, ...), not
# just its public re-exports.
_CFP_PATH = _THIS_DIR / "battery.py"
_CFP_SPEC = importlib.util.spec_from_file_location("chess_battery_local", _CFP_PATH)
if _CFP_SPEC is None or _CFP_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Failed to load local module from {_CFP_PATH}")
chess_full_probes = importlib.util.module_from_spec(_CFP_SPEC)
_CFP_SPEC.loader.exec_module(chess_full_probes)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _trim_to_current_game(user_prompt: str) -> str:
    """Drop everything before the real position, for the human-readable dumps.

    Only for the .txt files. The JSONL must keep the untrimmed prompt: the
    distractor chat block that STRESS_MODE / CHESS_CHAT_CONTEXT prepends sits
    *before* this marker, so trimming it out of the machine-readable export
    stores an easier prompt than the model was actually given -- and anything
    replaying the bank would ask the question without the distractors that are
    the entire point of those knobs.
    """
    marker = "Current game:"
    idx = user_prompt.find(marker)
    if idx == -1:
        return user_prompt
    return user_prompt[idx:]


def _default_out_dir() -> Path:
    """Return (and create) the output directory for this export.

    The root is the current working directory unless HALLUWORLD_RESULTS_ROOT
    says otherwise -- deliberately never derived from __file__. battery.py had
    the same bug (Path(__file__).resolve().parents[1]) from when it lived at
    examples/chess_full_probes.py, where parents[1] was the repository root;
    moving both files to halluworld/tracks/chess/ during the 2026-08-26 track
    reorganization silently redirected output into
    halluworld/tracks/results/ -- inside the installed package. battery.py was
    fixed then; this sibling script was missed and reproduced the same path
    computation independently.
    """
    root = Path(os.environ.get("HALLUWORLD_RESULTS_ROOT") or Path.cwd())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = os.environ.get("QUESTION_SET_NAME", "").strip() or stamp
    out = root / "results" / "chess" / "question_sets" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_env():
    variant = os.environ.get("CHESS_VARIANT", "").strip().lower()
    use_old_bishop = variant in ("old_bishop", "old-bishop", "short_bishop")
    use_atomic = variant in ("atomic", "atomic_chess")

    if os.environ.get("USE_LICHESS", "0") == "1" and not use_old_bishop and not use_atomic:
        pool = chess_full_probes.lichess_pool_kwargs()
        print("[export_chess_question_set] Loading Lichess puzzle FENs … %s" % pool)
        fens = load_lichess_puzzle_fens(**pool)
        print("[export_chess_question_set] Environment: ChessEnv (Lichess puzzle FENs)")
        return make_chess_env(fens=fens, seed=pool["seed"]), use_atomic, use_old_bishop

    if use_old_bishop:
        from halluworld.tracks.chess.envs.old_bishop_chess_env import make_old_bishop_chess_env

        print("[export_chess_question_set] Environment: OldBishopChessEnv")
        return make_old_bishop_chess_env(seed=42), use_atomic, use_old_bishop

    if use_atomic:
        from halluworld.tracks.chess.envs.atomic_chess_env import make_atomic_chess_env

        fens_arg = None
        if chess_full_probes._env_bool("ATOMIC_USE_HF_FENS", False):
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
            except Exception as exc:  # pragma: no cover
                print(f"[export_chess_question_set] ATOMIC_USE_HF_FENS failed ({exc!r}); using builtin pool.")
                fens_arg = None
            if fens_arg:
                print(f"[export_chess_question_set] Atomic FEN pool from HF/cache: {len(fens_arg)}")
        print("[export_chess_question_set] Environment: AtomicChessEnv")
        return make_atomic_chess_env(fens=fens_arg, seed=42), use_atomic, use_old_bishop

    print("[export_chess_question_set] Environment: ChessEnv (default FEN pool)")
    return make_chess_env(seed=42), use_atomic, use_old_bishop


def _write_outputs(run, out_dir: Path) -> None:
    jsonl_path = out_dir / "questions.jsonl"
    text_path = out_dir / "questions.txt"
    by_probe_dir = out_dir / "by_probe"
    by_probe_dir.mkdir(parents=True, exist_ok=True)

    serializable_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}

    for idx, r in enumerate(run.results, start=1):
        user_prompt = ""
        for msg in r.messages:
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
        row = {
            "index": idx,
            "probe": r.probe_name,
            "episode_id": r.episode_id,
            "env_reset_id": r.metadata.get("env_reset_id"),
            "env_seed": r.metadata.get("env_seed"),
            "question": r.question,
            "ground_truth": _jsonable(r.ground_truth),
            # The exact model-visible prompt, distractor block included, so the
            # frozen bank can be replayed rather than regenerated.
            "user_prompt": user_prompt,
            "display_prompt": _trim_to_current_game(user_prompt),
            "metadata": _jsonable(r.metadata),
        }
        serializable_rows.append(row)
        grouped.setdefault(r.probe_name, []).append(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in serializable_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    lines: list[str] = []
    for row in serializable_rows:
        lines.append("=" * 88)
        lines.append(f"index: {row['index']}")
        lines.append(f"probe: {row['probe']}")
        lines.append(f"episode_id: {row['episode_id']} env_reset_id: {row['env_reset_id']}")
        lines.append("")
        lines.append(row["display_prompt"])
        lines.append("")
        lines.append(f"ground_truth: {row['ground_truth']!r}")
        lines.append("")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for probe_name, rows in sorted(grouped.items()):
        probe_path = by_probe_dir / f"{probe_name}.txt"
        probe_lines: list[str] = []
        for row in rows:
            probe_lines.append("=" * 88)
            probe_lines.append(f"index: {row['index']}")
            probe_lines.append(f"probe: {probe_name}")
            probe_lines.append(f"episode_id: {row['episode_id']} env_reset_id: {row['env_reset_id']}")
            probe_lines.append("")
            probe_lines.append(row["display_prompt"])
            probe_lines.append("")
            probe_lines.append(f"ground_truth: {row['ground_truth']!r}")
            probe_lines.append("")
        probe_path.write_text("\n".join(probe_lines) + "\n", encoding="utf-8")

    print(f"[export_chess_question_set] Wrote {len(serializable_rows)} items")
    print(f"[export_chess_question_set] JSONL: {jsonl_path}")
    print(f"[export_chess_question_set] text:  {text_path}")
    print(f"[export_chess_question_set] by_probe directory: {by_probe_dir}")


def main() -> None:
    configure_observation_fen_display(None)
    benchmark_seed = int(os.environ.get("BENCHMARK_SEED", "42"))

    env, use_atomic, use_old_bishop = _make_env()
    probes, evaluators, include_fen = chess_full_probes.build_probes_and_evaluators()

    stress = os.environ.get("STRESS_MODE", "0").strip() == "1"
    extreme = os.environ.get("CHESS_EXTREME", "0").strip() == "1"
    if "CHESS_CHAT_CONTEXT" in os.environ:
        chat_context = chess_full_probes._env_bool("CHESS_CHAT_CONTEXT", False)
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
    fen_cfg = chess_full_probes._parse_observation_fen_display()
    configure_observation_fen_display(fen_cfg)
    serializer = ChessSerializer(
        include_legal_moves=False,
        include_fen=include_fen,
        observation_mode=obs_mode if use_atomic else "grid",
        fen_display=fen_cfg,
    )
    if chat_context:
        serializer = chess_full_probes._TransformedObservationSerializer(
            serializer,
            chess_full_probes._make_chat_observation_transform(
                heavy_chat=use_old_bishop or use_atomic,
                extreme=extreme,
                stress=stress,
            ),
            rng_seed=benchmark_seed,
        )

    n_eps = int(os.environ.get("N_EPISODES", "50"))
    print(
        "[export_chess_question_set] Running with "
        f"N_EPISODES={n_eps}, steps_before_probe={steps_before}, include_fen={include_fen}, "
        f"fen_mode={fen_cfg.mode}, chat_context={chat_context}"
    )
    run = run_benchmark(
        env=env,
        serializer=serializer,
        probes=probes,
        lm=StubLM(mode="yes"),
        evaluators=evaluators,
        n_episodes=n_eps,
        steps_before_probe=steps_before,
        seed=benchmark_seed,
        verbose=False,
        include_obs=True,
    )

    out_dir = _default_out_dir()
    _write_outputs(run, out_dir)
    configure_observation_fen_display(None)


if __name__ == "__main__":
    main()
