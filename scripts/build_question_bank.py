#!/usr/bin/env python3
"""Build the frozen question bank from each track's live probe definitions.

Run from the repository root:

    python3 scripts/build_question_bank.py --version v0.1
    python3 scripts/build_question_bank.py --version v0.1 --check

`--check` rebuilds in memory and compares against the committed manifest
without writing, so CI can catch a bank that has drifted from its checksum.

How each track is extracted
---------------------------
terminal
    Already frozen on disk as 529 JSON files. Normalized into the shared
    schema; the free-text failure-mode labels are mapped onto the taxonomy.

grid
    make_probes() in tracks/grid/perception.py is 4775 lines holding 384
    FixedProbe literals plus 9 seeded generator probes across 33 levels. Rather
    than parse that, this *executes* make_probes for each level with a fixed
    seed and introspects the returned objects: FixedProbe exposes its question
    and ground truth as plain attributes and ignores `env` in generate(), so
    the literals come out exactly as written. Generator probes cannot be
    frozen as text -- they draw from the run RNG -- so they are recorded as a
    class plus arguments and their determinism rests on the seed instead.

chess
    Read from the question set exported on 2026-05-05 by
    tracks/chess/question_set.py under StubLM -- the artifact the reported
    chess numbers were measured against. Using the export rather than
    regenerating means the bank is exactly what was evaluated, and sidesteps
    needing the ~72-variable battery config to reproduce it.

innav
    Frozen from make_canonical_probes() plus the per-level navigation config in
    data/levels/*.innav.json. Every InNav probe is generated from the run RNG,
    so all records are kind="generated": the bank stores the class and its
    arguments, and determinism rests on the seed.

    IMPORTANT: InNav does NOT ask the same questions as the static gridworld
    benchmark, despite canonical_probes.py claiming to. See INNAV_PARITY below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from halluworld.questions import (  # noqa: E402
    SCHEMA_VERSION,
    QuestionRecord,
    verify_bank,
    write_bank,
)

BANK_ROOT = REPO / "halluworld" / "data" / "questions"

# Terminal's free-text labels -> the shared taxonomy.
# The three singleton labels at the tail of the distribution are NOT guessed
# into a bucket; they are marked unclassified and their original text is kept
# in extra.failure_mode_original, so the mapping stays auditable.
TERMINAL_FAILURE_MODES = {
    "stale memory": "stale_memory",
    "cross-tier reasoning": "cross_tier_reasoning",
    "uncertainty overclaim": "uncertainty_overclaim",
    "causal shortcut": "causal_shortcut",
    "version/API hallucination": "version_api_hallucination",
}

# Terminal's probe_type IS the cognitive tier, under different names.
TERMINAL_TIERS = {
    "perceptual": "P",
    "memory": "M",
    "causal": "C",
    "uncertainty": "U",
    "cross-tier compound": "X",
}

# Gridworld encodes the tier in the level-key prefix: P1_dense_array,
# M2_witness_stand, C6_flood_fire_escape, U1_fog_of_war, X7_zone_a.
GRID_TIER_PREFIXES = ("P", "M", "C", "U", "X")


def grid_tier(level_key: str) -> str:
    head = level_key[:1].upper()
    return head if head in GRID_TIER_PREFIXES else "unknown"


# --------------------------------------------------------------------------- #
# terminal                                                                     #
# --------------------------------------------------------------------------- #

TERMINAL_SOURCE = "terminal_20260504"


def build_terminal(source: str = TERMINAL_SOURCE, namespace: str | None = None) -> list[QuestionRecord]:
    from halluworld.data import PROBE_BANKS_DIR

    candidate = Path(source).expanduser()
    bank_dir = candidate if (candidate.is_absolute() or candidate.exists()) \
        else PROBE_BANKS_DIR / source
    if not bank_dir.is_dir():
        print("  WARN terminal probe directory not found at %s" % bank_dir)
        return []

    # `sequence` restarts at 1 in every generation, so on its own it collides
    # across banks: a held-out set would reuse v0.1's ids while holding
    # different questions. Non-default sources are namespaced by directory,
    # unless --terminal-namespace names the namespace explicitly -- which is
    # what lets the source folder be renamed (the Hugging Face dataset ships
    # them as raw/terminal) without changing a single question_id.
    if namespace is not None:
        prefix = "%s/" % namespace if namespace else ""
    else:
        prefix = "" if source == TERMINAL_SOURCE else "%s/" % bank_dir.name

    files = sorted(p for p in bank_dir.glob("*.json") if p.name != "summary.json")
    records = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        md = obj.get("metadata", {})
        raw_mode = md.get("failure_mode_target", "")
        mapped = TERMINAL_FAILURE_MODES.get(raw_mode, "unclassified")

        seq = md.get("sequence")
        records.append(QuestionRecord(
            question_id=("terminal/%s%06d" % (prefix, seq)) if seq is not None
                        else ("terminal/%s%s" % (prefix, path.stem)),
            track="terminal",
            suite="terminal_llm_generated",
            task_or_level=md.get("task_name", ""),
            kind="fixed",
            probe_type=md.get("probe_type", ""),
            question=obj.get("question", ""),
            ground_truth=obj.get("answer"),
            answer_schema=md.get("answer_schema", ""),
            context=obj.get("context", ""),
            cognitive_tier=TERMINAL_TIERS.get(md.get("probe_type", ""), "unknown"),
            failure_mode_target=mapped,
            difficulty=md.get("difficulty_score"),
            answerability=md.get("answerability_score"),
            provenance={
                "source_file": path.name,
                "generator_model": md.get("llm_probe_model", ""),
                "generator_effort": md.get("llm_probe_reasoning_effort")
                                    or md.get("llm_probe_thinking_effort", ""),
                "agent_model": md.get("model", ""),
                "run_id": md.get("run_id", ""),
                "trajectory_step_index": md.get("trajectory_step_index"),
                "golden_source": md.get("golden_source", ""),
                "golden_rationale": md.get("golden_command_or_heuristic", ""),
            },
            extra={
                "failure_mode_original": raw_mode,
                "injector": md.get("injector", ""),
                "context_scope": md.get("context_scope", ""),
                "trigger_command": md.get("trigger_command", ""),
                "session": md.get("session", ""),
                "locality_risk": md.get("locality_risk", ""),
                "usefulness": md.get("usefulness", ""),
                "difficulty_rationale": md.get("difficulty_rationale", ""),
            },
        ))
    return records


# --------------------------------------------------------------------------- #
# grid                                                                         #
# --------------------------------------------------------------------------- #

# make_probes is seeded from the run RNG. Freezing uses a fixed seed so the
# bank is reproducible; the golden test asserts this exact seed reproduces it.
GRID_SEED = 0


def _merge_rendered_prompts(records: list[QuestionRecord], csv_path: Path) -> int:
    """Fill empty contexts from the package's review CSV.

    The grid r4 hand-off splits the item across two files: the JSONL carries
    identity and provenance but an empty `context`, while the CSV carries the
    rendered model-visible `context` and the per-level `system_prompt`. Without
    the CSV half the questions cannot be asked at all, so the bank takes both.

    system_prompt has no field on QuestionRecord and lands in extra; grid uses
    ten different rule preambles, so it is per-record, not a constant.
    """
    import csv as csvmod

    csvmod.field_size_limit(10 ** 9)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        by_id = {row["question_id"]: row for row in csvmod.DictReader(handle)}

    merged = 0
    for record in records:
        row = by_id.get(record.question_id)
        if row is None:
            continue
        context = (row.get("context") or "").strip()
        system_prompt = (row.get("system_prompt") or "").strip()
        if context and not record.context:
            record.context = context
            merged += 1
        if system_prompt:
            record.extra = {**(record.extra or {}), "system_prompt": system_prompt}
    return merged


def _load_curated(path: Path, track: str) -> list[QuestionRecord]:
    """Read an externally authored question file in the shared record schema.

    Grid questions cannot be regenerated from a config the way chess can -- they
    are authored, so a new grid set arrives as a file rather than as a seed. The
    records are validated on the way in: QuestionRecord raises on a bad one, so a
    malformed hand-off fails the build instead of entering the bank.
    """
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row.pop("schema_version", None)
        try:
            records.append(QuestionRecord(**row))
        except Exception as exc:
            raise SystemExit("%s line %d: %s: %s" % (path, lineno, type(exc).__name__, exc))
    bad = sorted({r.track for r in records} - {track})
    if bad:
        raise SystemExit("%s: expected track=%s, found %s" % (path, track, ", ".join(bad)))

    # A record with no context cannot be asked. The hand-off ships the rendered
    # prompt in a sibling CSV, so pick it up rather than freezing unusable items.
    if any(not r.context for r in records):
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            merged = _merge_rendered_prompts(records, csv_path)
            print("  merged %d rendered context(s) from %s" % (merged, csv_path.name))

    empty = [r.question_id for r in records if not r.context]
    if empty:
        raise SystemExit(
            "%s: %d record(s) have no context and no sibling CSV supplies one "
            "(e.g. %s). Such a question cannot be asked."
            % (path, len(empty), empty[0])
        )
    return records


def build_grid(source: str | None = None) -> list[QuestionRecord]:
    import halluworld.tracks.grid.perception as perception
    from halluworld.tracks.grid.probes.visibility import FixedProbe

    if source:
        return _load_curated(Path(source).expanduser(), "grid")

    records = []
    for level_key in sorted(perception.LEVELS):
        try:
            probes = perception.make_probes(level_key, random.Random(GRID_SEED))
        except Exception as exc:  # a level whose generator needs a live env
            print("  WARN %-28s could not build: %s: %s"
                  % (level_key, type(exc).__name__, exc))
            continue

        for idx, probe in enumerate(probes, start=1):
            qid = "grid/%s/q%02d" % (level_key, idx)
            if isinstance(probe, FixedProbe):
                md = dict(getattr(probe, "_metadata", {}) or {})
                trap = md.get("trap", "") or md.get("violation", "")
                records.append(QuestionRecord(
                    question_id=qid,
                    track="grid",
                    suite="perception",
                    task_or_level=level_key,
                    kind="fixed",
                    probe_type=getattr(probe, "_probe_type", ""),
                    question=getattr(probe, "_question", ""),
                    ground_truth=getattr(probe, "_ground_truth", None),
                    cognitive_tier=grid_tier(level_key),
                    # Deliberately not classified: grid's `trap` prose is written
                    # for humans and does not map onto the failure-mode taxonomy
                    # without inventing labels. The prose is preserved below.
                    failure_mode_target="unclassified",
                    provenance={"source": "make_probes", "seed": GRID_SEED},
                    extra={"failure_mode_original": trap, **md},
                ))
            else:
                # Seeded generator: freeze the recipe, not the text.
                kwargs = {
                    k.lstrip("_"): v
                    for k, v in vars(probe).items()
                    if not k.startswith("__") and isinstance(v, (str, int, float, bool, type(None)))
                }
                records.append(QuestionRecord(
                    question_id=qid,
                    track="grid",
                    suite="perception",
                    task_or_level=level_key,
                    kind="generated",
                    probe_type=getattr(probe, "probe_type", type(probe).__name__),
                    probe_class=type(probe).__name__,
                    probe_kwargs=kwargs,
                    cognitive_tier=grid_tier(level_key),
                    provenance={"source": "make_probes", "seed": GRID_SEED},
                ))
    return records


# --------------------------------------------------------------------------- #
# chess                                                                        #
# --------------------------------------------------------------------------- #

# The battery's cognitive tiers, as documented in docs/CHESS.md. There is no U
# probe in this battery; that is a real gap in coverage, not an omission here.
CHESS_TIERS = {
    "chess_can_capture": "P",                  # capture legality on the visible board
    "chess_defended": "P",                     # defender graph on the visible board
    "chess_hanging": "P",                      # attacked-and-undefended, visible board
    "chess_hypothetical_in_check": "C",        # apply one move, read a rule-defined property
    "chess_after_move_undefended_count": "C",  # 1-2 ply rollout, then count a predicate
    "chess_hidden_side_capture_stats": "M",    # count captures over a long explicit move list
    "chess_san_legal_move": "X",               # SAN-only history: state + rules + emit legal move
}

CHESS_SOURCE = "chess_20260505_fen-off"


def _chess_source_dir(source: str) -> Path:
    """Resolve --chess-source: a path, or a name under PROBE_BANKS_DIR."""
    from halluworld.data import PROBE_BANKS_DIR

    candidate = Path(source).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROBE_BANKS_DIR / source


def build_chess(source: str = CHESS_SOURCE, source_name: str | None = None) -> list[QuestionRecord]:
    src_dir = _chess_source_dir(source)
    path = src_dir / "questions.jsonl"
    if not path.exists():
        print("  WARN chess question set not found at %s" % path)
        return []

    # A regenerated export (see scripts/build_v02_bank.sh) drops the env
    # config it was produced under next to questions.jsonl. When absent we are
    # reading the 2026-05-05 v0.1 export, whose facts are the defaults below.
    cfg_path = src_dir / "generation_config.json"
    gen_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    env_cfg = gen_cfg.get("env", {})

    # env_reset_id counts 0..N within a run, so on its own it collides across
    # banks: a regenerated set would reuse every v0.1 question_id while holding
    # different questions, and any result keyed by id would conflate the two.
    # Regenerated banks are namespaced by the seed that chose their positions.
    # v0.1 has no generation_config.json and keeps its frozen ids untouched.
    seed_ns = "s%s/" % env_cfg["BENCHMARK_SEED"] if env_cfg.get("BENCHMARK_SEED") else ""

    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        probe = row["probe"]
        md = dict(row.get("metadata") or {})
        # `model` is "stub" for every row -- the export runs under StubLM so no
        # API is called. It says nothing about the item, so it does not travel.
        md.pop("model", None)
        records.append(QuestionRecord(
            question_id="chess/%s%s/e%02d" % (seed_ns, probe, row["env_reset_id"]),
            track="chess",
            suite="chess_standard",
            task_or_level=probe,
            kind="fixed",
            probe_type=probe,
            question=row["question"],
            ground_truth=row.get("ground_truth"),
            context=row.get("user_prompt", ""),
            cognitive_tier=CHESS_TIERS.get(probe, "unknown"),
            # The battery has no per-probe failure-mode labels; see the grid
            # note above for why these are not invented.
            failure_mode_target="unclassified",
            provenance={
                "source": source_name or source,
                "exported": gen_cfg.get("exported", "2026-05-05"),
                "export_tool": "halluworld/tracks/chess/question_set.py (StubLM)",
                "fen_mode": env_cfg.get("OBSERVATION_FEN_MODE", "fen-off"),
                "env_seed": row.get("env_seed"),
                "env_reset_id": row.get("env_reset_id"),
                "benchmark_seed": int(env_cfg.get("BENCHMARK_SEED", 42)),
                # Full knob set for a regenerated bank, so "how v0.2 was made"
                # stays reproducible even when the instances stay private.
                **({"generation_env": env_cfg} if env_cfg else {}),
            },
            extra=md,
        ))
    return records


# --------------------------------------------------------------------------- #
# innav                                                                        #
# --------------------------------------------------------------------------- #

INNAV_SEED = 0

#: Levels where make_canonical_probes() genuinely reproduces the static
#: gridworld probe set. Everywhere else it substitutes a generic fallback --
#: see the parity note recorded in the manifest.
INNAV_PARITY_LEVELS = ("P1_dense_array", "P2_corridor_gauntlet", "P3_rotation_challenge")


def build_innav() -> list[QuestionRecord]:
    import random as _random

    import halluworld.tracks.grid.perception as perception
    from halluworld.data import LEVELS_DIR
    from halluworld.tracks.innav.canonical_probes import make_canonical_probes

    records = []
    for level_key in sorted(perception.LEVELS):
        probes = make_canonical_probes(level_key, _random.Random(INNAV_SEED))

        # Navigation config, when the level has one.
        cfg_path = LEVELS_DIR / ("%s.innav.json" % Path(perception.LEVELS[level_key]).stem)
        nav_cfg = {}
        if cfg_path.exists():
            nav_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        for idx, probe in enumerate(probes, start=1):
            kwargs = {
                k.lstrip("_"): v
                for k, v in vars(probe).items()
                if not k.startswith("__")
                and isinstance(v, (str, int, float, bool, type(None)))
            }
            records.append(QuestionRecord(
                question_id="innav/%s/q%02d" % (level_key, idx),
                track="innav",
                suite="innav_paired",
                task_or_level=level_key,
                kind="generated",
                probe_type=getattr(probe, "probe_type", type(probe).__name__),
                probe_class=type(probe).__name__,
                probe_kwargs=kwargs,
                cognitive_tier=grid_tier(level_key),
                failure_mode_target="unclassified",
                provenance={
                    "source": "make_canonical_probes",
                    "seed": INNAV_SEED,
                    "matches_static_benchmark": level_key in INNAV_PARITY_LEVELS,
                },
                extra={
                    # Each item is asked twice, once per arm, sharing a pair id
                    # at run time. Both arms get the SAME generated question.
                    "arms": ["innav", "static_control"],
                    "static_probe_location": nav_cfg.get("static_probe_location"),
                    "navigation_target_locations": nav_cfg.get("navigation_target_locations"),
                    "navigation_params": nav_cfg.get("navigation_params"),
                    "innav_start_position": nav_cfg.get("innav_start_position"),
                },
            ))
    return records


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #

BUILDERS = {
    "terminal": build_terminal,
    "grid": build_grid,
    "chess": build_chess,
    "innav": build_innav,
}

# Tracks that exist but cannot be frozen yet. Recorded explicitly in the
# manifest so a reader can tell "not yet built" from "deliberately empty".
PENDING = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--tracks", nargs="+", choices=sorted(BUILDERS),
                    help="tracks to build (default: all four)")
    ap.add_argument("--chess-source", default=CHESS_SOURCE,
                    help="directory holding the chess questions.jsonl export, "
                         "or a name under data/probe_banks/")
    ap.add_argument("--terminal-source", default=TERMINAL_SOURCE,
                    help="directory of extracted probe .json files, or a name "
                         "under data/probe_banks/")
    ap.add_argument("--grid-source",
                    help="curated grid questions .jsonl in the shared record "
                         "schema; grid is authored, not generated from a seed")
    ap.add_argument("--terminal-namespace", default=None,
                    help="question_id namespace for terminal records, overriding the "
                         "one derived from the source folder name ('' for none, as in v0.1)")
    ap.add_argument("--chess-source-name", default=None,
                    help="name recorded as provenance.source on chess records, overriding "
                         "--chess-source (v0.1 records carry 'chess_20260505_fen-off')")
    ap.add_argument("--chess-key", default="chess",
                    help="manifest key and file stem for the chess bank: 'chess' "
                         "(No-FEN) or 'chess_fen_<mode>' for another FEN condition "
                         "frozen alongside it, e.g. chess_fen_transpose")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare against the committed manifest")
    args = ap.parse_args()

    out_dir = BANK_ROOT / args.version
    manifest_path = out_dir / "manifest.json"
    selected = args.tracks or sorted(BUILDERS)

    built = {}
    for track in selected:
        fn = BUILDERS[track]
        print("building %s ..." % track)
        source = {"chess": args.chess_source, "grid": args.grid_source,
                  "terminal": args.terminal_source}.get(track)
        if track == "terminal":
            recs = fn(source, namespace=args.terminal_namespace)
        elif track == "chess":
            recs = fn(source, source_name=args.chess_source_name)
        elif track == "grid":
            recs = fn(source)
        else:
            recs = fn()
        built[track] = recs
        fixed = sum(1 for r in recs if r.kind == "fixed")
        print("  %s: %d records (%d fixed, %d generated)"
              % (track, len(recs), fixed, len(recs) - fixed))

    if args.chess_key != "chess":
        if not re.fullmatch(r"chess_fen_[a-z_]+", args.chess_key):
            print("--chess-key must be 'chess' or 'chess_fen_<mode>'")
            return 2
        if "chess" in built:
            built[args.chess_key] = built.pop("chess")

    if args.check:
        # Compare the REBUILT records against the published digests. The
        # manifest is found the way the harness finds it (local copy,
        # HALLUWORLD_QUESTIONS_DIR, or the Hugging Face dataset), so this works
        # from a fresh checkout. The frozen files themselves are not needed.
        from halluworld.data import QuestionBankUnavailable, questions_dir
        try:
            manifest = json.loads((questions_dir(args.version) / "manifest.json").read_text())
        except QuestionBankUnavailable as exc:
            print(exc)
            return 1
        ok = True
        for track, recs in built.items():
            entry = manifest["tracks"].get(track)
            if entry is None:
                print("  MISSING from manifest: %s" % track); ok = False; continue
            payload = "\n".join(
                r.to_json() for r in sorted(recs, key=lambda r: r.question_id)).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            if entry["count"] != len(recs) or digest != entry["sha256"]:
                print("  DRIFT %s: manifest %d records sha256 %s, rebuilt %d records sha256 %s"
                      % (track, entry["count"], entry["sha256"][:12], len(recs), digest[:12]))
                ok = False
            else:
                print("  OK   %s: %d records, rebuilt sha256 matches" % (track, len(recs)))
        return 0 if ok else 1

    # A subset build (--tracks) merges into any existing manifest rather than
    # dropping the tracks it did not build; a v0.2 bank is assembled over time
    # as each track's questions become available.
    existing = {}
    if manifest_path.exists() and args.tracks:
        existing = json.loads(manifest_path.read_text()).get("tracks", {})

    manifest = {
        "bank_version": args.version,
        "schema_version": SCHEMA_VERSION,
        "tracks": dict(existing),
        "pending": PENDING,
        "notes": {
            "innav_probe_generation": (
                "InNav probes are generated against whatever state the "
                "navigating agent reached, not drawn from a fixed list. The "
                "static gridworld benchmark uses FixedProbe on 30 of 33 levels, "
                "whose questions hardcode egocentric spatial references ('11 "
                "steps ahead and 3 to your right') that presuppose a known agent "
                "position -- so they cannot be reused once the agent has "
                "navigated elsewhere. make_canonical_probes therefore matches "
                "the static set on exactly the 3 levels where the static "
                "benchmark itself uses generated probes (%s). "
                "The paper's InNav result compares INNAV to CTRLSTATIC on "
                "identical trajectories, both arms receiving the same generated "
                "question, so it is unaffected. InNav numbers are NOT comparable "
                "to published static gridworld numbers. See trajectories.jsonl.gz "
                "-- the trajectory, not a question list, is InNav's frozen "
                "artifact." % ", ".join(INNAV_PARITY_LEVELS)
            ),
            "chess_no_uncertainty_probe": (
                "The chess battery has no U-tier probe. Grid and terminal both "
                "cover U, so cross-track uncertainty comparisons exclude chess."
            ),
        },
    }
    for track, recs in built.items():
        entry = write_bank(recs, out_dir / ("%s.jsonl.gz" % track))
        by_mode, by_tier = {}, {}
        for r in recs:
            by_mode[r.failure_mode_target] = by_mode.get(r.failure_mode_target, 0) + 1
            by_tier[r.cognitive_tier] = by_tier.get(r.cognitive_tier, 0) + 1
        entry["fixed"] = sum(1 for r in recs if r.kind == "fixed")
        entry["generated"] = sum(1 for r in recs if r.kind == "generated")
        entry["levels_or_tasks"] = len({r.task_or_level for r in recs})
        entry["by_cognitive_tier"] = dict(sorted(by_tier.items()))
        entry["by_failure_mode"] = dict(sorted(by_mode.items()))
        if track.startswith("chess"):
            entry["source"] = args.chess_source
        if track == "grid" and args.grid_source:
            entry["source"] = args.grid_source
        if track == "terminal" and args.terminal_source != TERMINAL_SOURCE:
            entry["source"] = args.terminal_source
        manifest["tracks"][track] = entry

    manifest["total_questions"] = sum(e["count"] for e in manifest["tracks"].values())
    conditions = sorted(k for k in manifest["tracks"] if k.startswith("chess_fen_"))
    if conditions:
        manifest["notes"]["chess_fen_conditions"] = (
            "%s ask the same questions about the same positions as chess.jsonl.gz "
            "(the No-FEN condition), differing only in the FEN line shown. Each is a "
            "reported condition, so total_questions counts every record; the number "
            "of distinct chess questions is the chess count alone. Select one with "
            "`halluworld eval chess --fen-mode <mode>`." % ", ".join(
                manifest["tracks"][k]["file"] for k in conditions))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
