"""Unified user-facing evaluation entry points for Grid, Chess, and InNav."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from halluworld.data import DATA_DIR, QuestionBankUnavailable, questions_dir
from halluworld.questions import read_bank

INNAV_PARITY_LEVELS = (
    "P1_dense_array",
    "P2_corridor_gauntlet",
    "P3_rotation_challenge",
)

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "xai": "XAI_API_KEY",
}


class EvaluationError(RuntimeError):
    """The requested evaluation cannot be started safely."""


def _bank_records(track: str, version: str = "v0.1") -> list:
    try:
        path = questions_dir(version) / f"{track}.jsonl.gz"
    except QuestionBankUnavailable as exc:
        raise EvaluationError(str(exc)) from exc
    if not path.exists():
        raise EvaluationError(f"no {track} question bank for version {version!r} at {path}")
    return list(read_bank(path))


def _known_levels(track: str, version: str = "v0.1") -> set[str]:
    return {record.task_or_level for record in _bank_records(track, version)}


def _bank_version(args) -> str:
    """Bank version for this run. Defaults to v0.1 for callers without the flag."""
    return getattr(args, "version", None) or "v0.1"


def _validate_levels(track: str, levels: list[str] | None,
                     version: str = "v0.1") -> list[str]:
    known = _known_levels(track, version)
    selected = sorted(known) if levels is None else list(dict.fromkeys(levels))
    unknown = set(selected) - known
    if unknown:
        raise EvaluationError(
            f"unknown {track} level(s): {', '.join(sorted(unknown))}"
        )
    return selected


def _require_provider_key(provider: str) -> None:
    key = PROVIDER_KEYS[provider]
    if not os.environ.get(key):
        raise EvaluationError(f"{key} must be set for provider={provider}")


def _effective_episodes(args) -> int:
    """Apply the optional per-level episode safety cap."""
    limit = getattr(args, "limit", None)
    return min(args.episodes, limit) if limit is not None else args.episodes


@contextmanager
def _temporary_environ(values: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _print_plan(track: str, args, levels: list[str], out: Path) -> None:
    print(f"track:     {track}")
    print(f"provider:  {args.provider}")
    print(f"model:     {args.model}")
    print(f"levels:    {len(levels)}")
    print(f"episodes:  {_effective_episodes(args)} per level")
    if getattr(args, "limit", None) is not None:
        print(f"limit:     {args.limit} episodes per level")
    print(f"output:    {out}")
    print("network:   disabled (dry-run)")


def _write_manifest(
    *, track: str, args, out: Path, config: dict, result_files: list[Path]
) -> None:
    from halluworld import manifest as manmod

    counts = {"result_files": len(result_files), "records": 0}
    for path in result_files:
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                counts["records"] += sum(1 for _ in csv.DictReader(handle))
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                counts["records"] += sum(1 for line in handle if line.strip())

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{now}_{track}_{manmod.config_digest(config)[:8]}"
    payload = manmod.build(
        run_id,
        config,
        track=track,
        question_bank={"version": _bank_version(args)},
        models=[{"requested": args.model, "provider": args.provider}],
        counts=counts,
    )
    manmod.write(payload, out / "manifest.json")


def _grid_bank_is_replayable(version: str) -> bool:
    """True when the bank stores askable questions rather than probe recipes.

    v0.1's grid bank was extracted from make_probes(), so regenerating
    reproduces it. An authored bank (v0.2) exists only as records -- make_probes
    can produce none of it -- so it has to be replayed or it is not being
    evaluated at all.
    """
    from halluworld.tracks.grid.replay import GridReplayError, load_grid_bank

    try:
        records = load_grid_bank(version)
    except GridReplayError:
        return False
    return bool(records) and all(r.context for r in records)


def run_grid(args) -> int:
    """Run the static Grid domain and optionally the small paired InNav subset.

    Replays the bank when it holds rendered questions, otherwise regenerates via
    perception.main(). --regenerate forces the latter.
    """
    version = _bank_version(args)
    levels = _validate_levels("grid", args.levels, version)
    out = Path(args.out).expanduser().resolve()
    replayable = not getattr(args, "regenerate", False) and _grid_bank_is_replayable(version)
    if args.dry_run:
        _print_plan("grid", args, levels, out)
        print("source:    %s" % (
            f"question bank {version} (replayed)" if replayable
            else "perception.make_probes (regenerated)"
        ))
        if args.include_innav:
            print(f"innav:    enabled ({', '.join(args.innav_levels)})")
        else:
            print("innav:    off")
        return 0

    if replayable:
        return _run_grid_replay(args, version, out, levels)

    _require_provider_key(args.provider)
    out.mkdir(parents=True, exist_ok=True)
    episodes = _effective_episodes(args)
    output_base = out / "results.csv"
    runner_args = [
        "--provider",
        args.provider,
        "--models",
        args.model,
        "--episodes",
        str(episodes),
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--serializer",
        args.serializer,
        "--levels",
        *levels,
        "--output",
        str(output_base),
    ]
    if args.reasoning_effort:
        runner_args.extend(["--reasoning-effort", args.reasoning_effort])
    if args.thinking_effort:
        runner_args.extend(["--thinking-effort", args.thinking_effort])
    if args.resume:
        runner_args.append("--resume")
    if args.verbose:
        runner_args.append("--verbose")

    from halluworld.tracks.grid import perception

    perception.main(runner_args)
    # Keep this in lockstep with perception.main's per-model filename rule.
    result_slug = args.model.replace("/", "_").replace("+", "_")
    result_path = out / f"results_{result_slug}.csv"
    result_files = [result_path] if result_path.is_file() else []
    config = {
        "provider": args.provider,
        "model": args.model,
        "levels": levels,
        "episodes": episodes,
        "episode_limit": getattr(args, "limit", None),
        "seed": args.seed,
        "serializer": args.serializer,
        "include_innav": args.include_innav,
    }
    _write_manifest(
        track="grid", args=args, out=out, config=config, result_files=result_files
    )

    if args.include_innav:
        innav_args = _innav_namespace_from_grid(args, out / "innav")
        return run_innav(innav_args)
    return 0


def _resume_state(args, out: Path):
    """Records already scored in a previous run of this same job.

    A partially completed run -- rate limited, out of credits, interrupted --
    leaves a mix of scored and unscored records. Re-asking the scored ones
    costs money for an answer already held, and --force would re-ask all of
    them, so resume keeps the scored ones and returns only the ids to skip.
    """
    from halluworld.results import read_jsonl

    path = out / "records.jsonl"
    if not getattr(args, "resume", False) or not path.is_file():
        return [], set()
    kept = [r for r in read_jsonl(path) if r.score is not None]
    mismatched = [
        r for r in kept
        if r.model_requested != args.model or r.provider != args.provider
    ]
    if mismatched:
        raise EvaluationError(
            "resume output in %s belongs to a different provider/model; "
            "choose a new --out" % out
        )
    return kept, {r.question_id for r in kept}


def _write_replay_outputs(track: str, records, args, out: Path, version: str,
                          levels: list[str]) -> int:
    """Write records + manifest and print the run summary."""
    from halluworld.results import write_csv, write_jsonl

    write_jsonl(records, out / "records.jsonl")
    write_csv(records, out / "records.csv")

    scored = [r for r in records if r.score is not None]
    excluded = len(records) - len(scored)
    accuracy = sum(r.score for r in scored) / len(scored) if scored else None
    print("replayed %d question(s) from bank %s" % (len(records), version))
    print("accuracy:  %s" % ("n/a (all excluded)" if accuracy is None else "%.4f" % accuracy))
    if excluded:
        print("excluded:  %d (provider-side failures, not scored as wrong)" % excluded)

    config = {
        "provider": args.provider,
        "model": args.model,
        "levels": levels,
        "episodes": _effective_episodes(args),
        "seed": args.seed,
        # Recorded because an answer is only "truncated" relative to the cap
        # its own run used, and a sweep that resumes part of a job mixes caps.
        # Without this, later analysis has to infer it from the token counts.
        "max_tokens": getattr(args, "max_tokens", None),
        "source": "question_bank_replay",
    }
    _write_manifest(
        track=track, args=args, out=out, config=config,
        result_files=[out / "records.jsonl"],
    )
    return 0


def _run_grid_replay(args, version: str, out: Path, levels: list[str]) -> int:
    from halluworld.tracks.grid.replay import GridReplayError, run_grid_replay

    _require_provider_key(args.provider)
    out.mkdir(parents=True, exist_ok=True)
    done, skip_ids = _resume_state(args, out)
    if skip_ids:
        print("resuming: %d already scored, re-asking the rest" % len(skip_ids))
    try:
        records = run_grid_replay(args, version, skip_ids=skip_ids or None)
    except GridReplayError as exc:
        raise EvaluationError(str(exc)) from exc
    merged = sorted(done + records, key=lambda r: r.question_id)
    return _write_replay_outputs("grid", merged, args, out, version, levels)


def _run_chess_replay(args, version: str, out: Path, levels: list[str]) -> int:
    """Ask the frozen bank's questions and write results in the shared schema."""
    from halluworld.tracks.chess.replay import ChessReplayError, run_chess_replay

    _require_provider_key(args.provider)
    out.mkdir(parents=True, exist_ok=True)
    done, skip_ids = _resume_state(args, out)
    if skip_ids:
        print("resuming: %d already scored, re-asking the rest" % len(skip_ids))
    try:
        records = run_chess_replay(args, version, out, skip_ids=skip_ids or None)
    except ChessReplayError as exc:
        raise EvaluationError(str(exc)) from exc
    merged = sorted(done + records, key=lambda r: r.question_id)
    return _write_replay_outputs("chess", merged, args, out, version, levels)


def _innav_namespace_from_grid(args, out: Path):
    from argparse import Namespace

    return Namespace(
        provider=args.provider,
        model=args.model,
        out=str(out),
        # Without this the paired InNav arm would silently fall back to the
        # v0.1 default while the grid arm ran against a different bank.
        version=_bank_version(args),
        levels=args.innav_levels,
        episodes=args.innav_episodes,
        limit=getattr(args, "limit", None),
        seed=args.seed,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        navigation_model=args.navigation_model,
        probe_timesteps=args.probe_timesteps,
        max_steps=args.max_steps,
        serializer=args.innav_serializer,
        trace_dir=None,
        dry_run=False,
        verbose=args.verbose,
    )


def run_innav(args) -> int:
    """Run paired InNav/CtrlStatic evaluation."""
    levels = _validate_levels("innav", args.levels, _bank_version(args))
    out = Path(args.out).expanduser().resolve()
    if args.dry_run:
        _print_plan("innav", args, levels, out)
        print(f"probe timesteps: {args.probe_timesteps}")
        print(f"navigation model: {args.navigation_model or args.model}")
        return 0

    _require_provider_key(args.provider)
    out.mkdir(parents=True, exist_ok=True)
    episodes = _effective_episodes(args)
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else out / "traces"
    output_path = out / "results.csv"
    runner_args = [
        "--provider",
        args.provider,
        "--models",
        args.model,
        "--levels",
        *levels,
        "--episodes",
        str(episodes),
        "--seed",
        str(args.seed),
        "--probe-timesteps",
        str(args.probe_timesteps),
        "--max-steps",
        str(args.max_steps),
        "--max-tokens",
        str(args.max_tokens),
        "--trace-dir",
        str(trace_dir),
        "--output",
        str(output_path),
    ]
    if args.reasoning_effort:
        runner_args.extend(["--reasoning-effort", args.reasoning_effort])
    if args.navigation_model:
        runner_args.extend(["--navigation-model", args.navigation_model])
    if args.serializer:
        runner_args.extend(["--serializer", args.serializer])

    from halluworld.tracks.innav import runner

    runner.main(runner_args)
    config = {
        "provider": args.provider,
        "model": args.model,
        "navigation_model": args.navigation_model or args.model,
        "levels": levels,
        "episodes": episodes,
        "episode_limit": getattr(args, "limit", None),
        "seed": args.seed,
        "probe_timesteps": args.probe_timesteps,
        "max_steps": args.max_steps,
    }
    _write_manifest(
        track="innav",
        args=args,
        out=out,
        config=config,
        result_files=[output_path],
    )
    return 0


def run_chess(args) -> int:
    """Score Chess against the frozen question bank.

    Default path replays the bank: the questions asked are exactly the frozen
    ones, in the observation condition --fen-mode names (v0.2 freezes both
    No-FEN and transpose). `--regenerate` runs the battery instead, which
    re-derives questions from a packaged config -- needed for conditions a
    version did not freeze (v0.1's fen-on and fen-transpose arms).
    """
    out = Path(args.out).expanduser().resolve()
    version = _bank_version(args)
    levels = _validate_levels("chess", args.levels, version)
    regenerate = getattr(args, "regenerate", False)
    if args.dry_run:
        _print_plan("chess", args, levels, out)
        print(f"fen mode:  {args.fen_mode}")
        print("source:    %s" % (
            "battery (regenerated from config)" if regenerate
            else f"question bank {version} (replayed)"
        ))
        return 0

    if not regenerate:
        return _run_chess_replay(args, version, out, levels)

    _require_provider_key(args.provider)
    from halluworld import config as cfgmod

    config_path = DATA_DIR / "configs" / "chess" / f"fen_{args.fen_mode}.yaml"
    config = cfgmod.load(config_path)
    config["N_EPISODES"] = _effective_episodes(args)
    config["LM_PROVIDER"] = args.provider
    model_key = {
        "openai": "OPENAI_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
        "baseten": "BASETEN_MODEL",
    }[args.provider]
    config[model_key] = args.model
    config["ENABLE_BENCHMARKS"] = ",".join(levels)
    if args.provider == "openai" and args.reasoning_effort:
        config["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    if args.provider == "anthropic" and args.thinking_effort:
        config["ANTHROPIC_THINKING_EFFORT"] = args.thinking_effort
    if args.provider == "openai":
        config["OPENAI_MAX_COMPLETION_TOKENS"] = args.max_tokens
    elif args.provider == "anthropic":
        config["ANTHROPIC_MAX_TOKENS"] = args.max_tokens
    else:
        config["BASETEN_MAX_TOKENS"] = args.max_tokens
    config["HALLUWORLD_RESULTS_DIR"] = str(out)
    if args.verbose:
        config["BENCHMARK_VERBOSE"] = "1"

    env = cfgmod.apply_to_environ(config, env={})
    out.mkdir(parents=True, exist_ok=True)
    from halluworld.tracks.chess import battery

    with _temporary_environ(env):
        battery.main()
    manifest_config = dict(config)
    manifest_config["episode_limit"] = getattr(args, "limit", None)
    _write_manifest(
        track="chess",
        args=args,
        out=out,
        config=manifest_config,
        result_files=[out / "results.jsonl"],
    )
    return 0
