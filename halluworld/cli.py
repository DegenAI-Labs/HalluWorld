"""The `halluworld` command line.

Replaces a scattering of loose scripts, each invoked by exporting some subset
of 73 environment variables and hoping the right ones were set.

    halluworld questions stats [--version v0.1]
    halluworld questions verify [--version v0.1]
    halluworld release check [--domain grid|chess|innav|terminal] [--version v0.1]
    halluworld eval grid --provider openai --model gpt-4o-mini --out results/grid
    halluworld eval chess --provider openai --model gpt-4o-mini --out results/chess
    halluworld eval innav --provider openai --model gpt-4o-mini --out results/innav
    halluworld eval terminal --provider openai --model gpt-4o-mini --out results/terminal
    halluworld terminal eval --provider openai --model gpt-4o-mini --out results/terminal
    halluworld config show <config.yaml>
    halluworld config from-env [--out FILE]
    halluworld config check <config.yaml>...
    halluworld run --config <config.yaml> [--set KEY=VALUE] [--dry-run]
    halluworld version
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from halluworld import __version__


def _bank_dir(version: str) -> Path:
    from halluworld.data import questions_dir
    return questions_dir(version)


# --------------------------------------------------------------------------- #
# questions                                                                    #
# --------------------------------------------------------------------------- #

def cmd_questions_stats(args) -> int:
    from halluworld.data import QuestionBankUnavailable

    try:
        path = _bank_dir(args.version) / "manifest.json"
    except QuestionBankUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    if not path.exists():
        print("no question bank at %s" % path, file=sys.stderr)
        return 1
    m = json.loads(path.read_text())
    print("HalluWorld question bank %s (schema %s)"
          % (m["bank_version"], m["schema_version"]))
    w = max([10] + [len(t) for t in m["tracks"]])
    print("%-*s %7s %8s %12s   %s" % (w, "track", "items", "levels", "fixed/gen", "tiers"))
    for track, e in sorted(m["tracks"].items()):
        tiers = " ".join("%s:%d" % (k, v)
                         for k, v in sorted(e["by_cognitive_tier"].items()))
        print("%-*s %7d %8d %12s   %s"
              % (w, track, e["count"], e["levels_or_tasks"],
                 "%d/%d" % (e["fixed"], e["generated"]), tiers))
    print("%-*s %7d" % (w, "total", m["total_questions"]))
    if m.get("trajectories"):
        t = m["trajectories"]
        print("\ntrajectories: %d traces, %d configs, %d worlds"
              % (t["count"], t["configs"], t["worlds"]))
    for name, note in (m.get("notes") or {}).items():
        print("\nnote [%s]:\n  %s" % (name, note))
    return 0


def cmd_questions_verify(args) -> int:
    from halluworld.data import QuestionBankUnavailable
    from halluworld.questions import verify_bank

    try:
        d = _bank_dir(args.version)
    except QuestionBankUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    path = d / "manifest.json"
    if not path.exists():
        print("no question bank at %s" % path, file=sys.stderr)
        return 1
    m = json.loads(path.read_text())
    ok = True
    for _, entry in sorted(m["tracks"].items()):
        good, msg = verify_bank(d / entry["file"], entry)
        print(("  OK   " if good else "  FAIL ") + msg)
        ok = ok and good
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# release                                                                      #
# --------------------------------------------------------------------------- #

def cmd_release_check(args) -> int:
    from halluworld.data import QuestionBankUnavailable
    from halluworld.release_contract import ReleaseContractError, check_release

    domains = args.domain or ["grid", "chess", "innav", "terminal"]
    try:
        reports = check_release(domains=domains, version=args.version)
    except (ReleaseContractError, QuestionBankUnavailable) as exc:
        print("  FAIL %s" % exc, file=sys.stderr)
        return 1
    for report in reports:
        unit = "tasks" if report.domain == "terminal" else "levels"
        print(
            "  OK   %-8s %d questions, %d %s -- %s"
            % (
                report.domain,
                report.questions,
                report.levels,
                unit,
                ", ".join(report.checks),
            )
        )
        if report.domain == "innav":
            print(
                "       %d trajectories (%d full, %d with unused endpoint omitted)"
                % (
                    report.details["trajectories"],
                    report.details["complete_action_sequences"],
                    report.details["endpoint_action_omitted"],
                )
            )
    return 0


# --------------------------------------------------------------------------- #
# eval                                                                         #
# --------------------------------------------------------------------------- #

def cmd_eval_grid(args) -> int:
    from halluworld.evaluation import EvaluationError, run_grid

    try:
        return run_grid(args)
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


def cmd_eval_chess(args) -> int:
    from halluworld.evaluation import EvaluationError, run_chess

    try:
        return run_chess(args)
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


def cmd_eval_innav(args) -> int:
    from halluworld.evaluation import EvaluationError, run_innav

    try:
        return run_innav(args)
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


def cmd_eval_terminal(args) -> int:
    from halluworld.tracks.terminal.evaluation import (
        TerminalEvaluationError,
        run_terminal,
    )

    try:
        return run_terminal(args)
    except TerminalEvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


def _add_eval_common(parser, *, episodes: int, max_tokens: int) -> None:
    parser.add_argument(
        "--provider", required=True,
        choices=("openai", "anthropic", "baseten", "xai")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--levels", nargs="+")
    parser.add_argument("--episodes", type=int, default=episodes)
    parser.add_argument(
        "--limit",
        type=lambda value: _positive_int(value, "--limit"),
        help="cap episodes per selected level (safety override for --episodes)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--version", default="v0.1",
                        help="question-bank version to evaluate against")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def _positive_int(value: str, flag: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag} expects an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{flag} must be at least 1")
    return parsed


def _nonnegative_int(value: str, flag: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag} expects an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{flag} must be at least 0")
    return parsed


def _add_terminal_eval_args(parser) -> None:
    parser.add_argument(
        "--provider", required=True,
        choices=("openai", "anthropic", "baseten", "xai")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--question-ids", nargs="+")
    parser.add_argument(
        "--probe-types",
        nargs="+",
        choices=("perceptual", "memory", "causal", "uncertainty", "cross-tier compound"),
    )
    parser.add_argument(
        "--limit",
        type=lambda value: _positive_int(value, "--limit"),
        help="evaluate only the first N selected frozen probes",
    )
    parser.add_argument(
        "--max-tokens",
        type=lambda value: _positive_int(value, "--max-tokens"),
        default=256,
    )
    parser.add_argument(
        "--max-context-chars",
        type=lambda value: _nonnegative_int(value, "--max-context-chars"),
        default=0,
        metavar="N",
        help="keep only the final N context characters (0 keeps the full frozen context)",
    )
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--thinking-effort", choices=("low", "medium", "high", "max"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--retries", type=lambda value: _positive_int(value, "--retries"), default=3
    )
    parser.add_argument(
        "--api-base",
        # Must default to None, not to Baseten's URL. This value is now passed
        # to the shared LM factory, which applies it to any OpenAI-compatible
        # provider -- so a Baseten default sent openai and xai runs to Baseten
        # and failed every request with Baseten's 403. The factory already
        # falls back to BASETEN_BASE_URL and then the serverless default when
        # --provider baseten is used and this is unset.
        default=None,
        help="override the API base for an OpenAI-compatible provider "
             "(baseten, xai); defaults per provider when unset",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="threads for bank replay (default 1). Independent questions, so "
             "this multiplies your request rate against the provider account's "
             "rate limit -- raise it until you start seeing rate_limit errors.",
    )
    parser.add_argument("--refusal-retries", type=int, default=0,
                   help="re-ask a refused prompt up to N more times (anthropic only; "
                        "opt-in, and disclose it -- it biases toward content the "
                        "model will engage with)")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(fn=cmd_eval_terminal)


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #

def cmd_config_show(args) -> int:
    from halluworld import config as cfgmod

    cfg = cfgmod.load(args.path)
    meta = cfg.pop("_meta", {})
    print("# %s" % args.path)
    if meta:
        print("# meta: %s" % meta)
    for key in sorted(cfg):
        print("%-34s %s" % (key, cfg[key]))
    print("\n%d key(s)" % len(cfg))
    return 0


def cmd_config_check(args) -> int:
    from halluworld import config as cfgmod

    bad = 0
    for path in args.paths:
        try:
            cfg = cfgmod.load(path)
            print("  OK   %-40s %d keys" % (path, len(cfg) - 1))
        except cfgmod.ConfigError as exc:
            print("  FAIL %s\n       %s" % (path, exc))
            bad += 1
    return 1 if bad else 0


def cmd_config_from_env(args) -> int:
    """Capture the currently-exported variables as a config file.

    The migration aid: rather than transcribing 73 shell exports by hand,
    source the old script and run this.
    """
    from halluworld import config as cfgmod

    captured = cfgmod.from_environ()
    if not captured:
        print("no HalluWorld config variables are set in this environment",
              file=sys.stderr)
        return 1
    text = cfgmod.to_yaml(captured)
    if args.out:
        Path(args.out).write_text(text)
        print("wrote %d key(s) to %s" % (len(captured), args.out))
    else:
        sys.stdout.write(text)
    return 0


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #

def cmd_run(args) -> int:
    from halluworld import config as cfgmod
    from halluworld import manifest as manmod

    cfg = cfgmod.load(args.config)
    for override in args.set or []:
        if "=" not in override:
            print("--set expects KEY=VALUE, got %r" % override, file=sys.stderr)
            return 2
        key, value = override.split("=", 1)
        if key not in cfgmod.CHESS_ENV_KEYS:
            print("--set: unknown key %r" % key, file=sys.stderr)
            return 2
        cfg[key] = value

    meta = cfg.get("_meta", {})
    track = meta.get("track", "chess")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = "%s_%s_%s" % (stamp, track, manmod.config_digest(cfg)[:8])

    if args.dry_run:
        print("run_id: %s" % run_id)
        print("track:  %s" % track)
        print("config: %s (%d keys)" % (args.config, len(cfg) - 1))
        env = cfgmod.apply_to_environ(cfg, env={})
        print("would export %d environment variable(s):" % len(env))
        for key in sorted(env):
            print("  %-34s %s" % (key, env[key]))
        missing = [k for k in cfgmod.FORBIDDEN_KEYS if k not in os.environ]
        if missing:
            print("\nnote: %s not set in this environment; a real run needs "
                  "whichever the provider requires." % ", ".join(missing))
        return 0

    cfgmod.apply_to_environ(cfg)
    out_dir = Path(args.out or "results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manmod.write(manmod.build(run_id, cfg, track=track), out_dir / "manifest.json")
    print("wrote %s" % (out_dir / "manifest.json"))

    if track == "chess":
        from halluworld.tracks.chess import battery
        return battery.main() or 0
    print("no runner wired for track %r yet" % track, file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="halluworld", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command")

    q = sub.add_parser("questions", help="inspect the frozen question bank")
    qs = q.add_subparsers(dest="sub")
    p = qs.add_parser("stats"); p.add_argument("--version", default="v0.1")
    p.set_defaults(fn=cmd_questions_stats)
    p = qs.add_parser("verify"); p.add_argument("--version", default="v0.1")
    p.set_defaults(fn=cmd_questions_verify)

    release = sub.add_parser("release", help="run offline release contract checks")
    release_sub = release.add_subparsers(dest="sub")
    p = release_sub.add_parser("check", help="verify artifacts against runtime wiring")
    p.add_argument("--version", default="v0.1")
    p.add_argument(
        "--domain", action="append", choices=("grid", "chess", "innav", "terminal")
    )
    p.set_defaults(fn=cmd_release_check)

    evaluate = sub.add_parser("eval", help="evaluate a model on a released domain")
    eval_sub = evaluate.add_subparsers(dest="domain")

    p = eval_sub.add_parser("grid", help="run the static Grid benchmark")
    _add_eval_common(p, episodes=1, max_tokens=256)
    p.add_argument("--serializer", choices=("symbolic", "grid", "memory"),
                   default="symbolic")
    p.add_argument("--api-base", help="OpenAI-compatible endpoint for --provider baseten")
    p.add_argument("--retries", type=int, default=3,
                   help="retries per request on rate-limit errors")
    p.add_argument("--refusal-retries", type=int, default=0,
                   help="re-ask a refused prompt up to N more times (anthropic only; "
                        "opt-in, and disclose it -- it biases toward content the "
                        "model will engage with)")
    p.add_argument("--workers", type=int, default=1,
                   help="threads for bank replay (default 1)")
    p.add_argument("--regenerate", action="store_true",
                   help="run perception.make_probes instead of replaying the bank")
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    p.add_argument("--thinking-effort", choices=("low", "medium", "high", "max"))
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--include-innav",
        action="store_true",
        help="also run the paired InNav subset (off by default)",
    )
    p.add_argument(
        "--innav-levels",
        nargs="+",
        default=["P1_dense_array", "P2_corridor_gauntlet", "P3_rotation_challenge"],
    )
    p.add_argument("--innav-episodes", type=int, default=5)
    p.add_argument("--innav-serializer", choices=("symbolic", "grid", "memory"))
    p.add_argument("--probe-timesteps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--navigation-model")
    p.set_defaults(fn=cmd_eval_grid)

    p = eval_sub.add_parser("chess", help="run the released standard-Chess battery")
    _add_eval_common(p, episodes=50, max_tokens=16384)
    p.add_argument("--fen-mode", choices=("off", "on", "transpose"), default="off",
                   help="FEN observation condition. Replay reads the bank's file for it "
                        "(chess.jsonl.gz for off, chess_fen_<mode>.jsonl.gz otherwise) and "
                        "fails if the version did not freeze that condition.")
    p.add_argument("--api-base", help="OpenAI-compatible endpoint for --provider baseten")
    p.add_argument("--retries", type=int, default=3,
                   help="retries per request on rate-limit errors")
    p.add_argument("--refusal-retries", type=int, default=0,
                   help="re-ask a refused prompt up to N more times (anthropic only; "
                        "opt-in, and disclose it -- it biases toward content the "
                        "model will engage with)")
    p.add_argument(
        "--workers", type=int, default=1,
        help="threads for bank replay (default 1). Independent questions, so "
             "this multiplies your request rate against the provider account's "
             "rate limit -- raise it until you start seeing rate_limit errors.",
    )
    p.add_argument("--resume", action="store_true",
                   help="keep questions already scored in --out and ask only the rest")
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="re-derive questions by running the battery instead of replaying "
             "the frozen bank. Needed for observation conditions a version did not "
             "freeze (v0.1 has only fen-off; v0.2 also freezes transpose); otherwise "
             "prefer the default, which guarantees the questions asked are the "
             "questions frozen.",
    )
    p.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh"))
    p.add_argument("--thinking-effort", choices=("low", "medium", "high", "max"))
    p.set_defaults(fn=cmd_eval_chess)

    p = eval_sub.add_parser("innav", help="run paired InNav/CtrlStatic evaluation")
    _add_eval_common(p, episodes=5, max_tokens=256)
    p.set_defaults(seed=42)
    p.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh"))
    p.add_argument("--navigation-model")
    p.add_argument("--probe-timesteps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--serializer", choices=("symbolic", "grid", "memory"))
    p.add_argument("--trace-dir")
    p.set_defaults(fn=cmd_eval_innav)

    p = eval_sub.add_parser("terminal", help="evaluate the frozen Terminal probe bank")
    _add_terminal_eval_args(p)

    terminal = sub.add_parser("terminal", help="Terminal evaluation and generation tools")
    terminal_sub = terminal.add_subparsers(dest="sub")
    p = terminal_sub.add_parser("eval", help="evaluate the frozen Terminal probe bank")
    _add_terminal_eval_args(p)

    c = sub.add_parser("config", help="inspect and migrate run configuration")
    cs = c.add_subparsers(dest="sub")
    p = cs.add_parser("show"); p.add_argument("path"); p.set_defaults(fn=cmd_config_show)
    p = cs.add_parser("check"); p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_config_check)
    p = cs.add_parser("from-env"); p.add_argument("--out")
    p.set_defaults(fn=cmd_config_from_env)

    r = sub.add_parser("run", help="run a benchmark from a config file")
    r.add_argument("--config", required=True)
    r.add_argument("--set", action="append", metavar="KEY=VALUE")
    r.add_argument("--out")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("version")
    p.set_defaults(fn=lambda a: (print("halluworld %s" % __version__), 0)[1])

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    return args.fn(args)
