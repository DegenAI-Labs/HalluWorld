#!/usr/bin/env python3
"""Migrate the legacy result CSVs into the unified ProbeRecord schema.

    python3 scripts/migrate_results.py --in results/legacy --out results/migrated/v1
    python3 scripts/migrate_results.py --in results/legacy --check

Adapters are keyed on frozenset(header), so column ORDER never matters and a
file cannot be matched to the wrong adapter by coincidence.

CONTRACT

  * An unrecognised header RAISES. It does not guess, and it does not skip
    quietly. A silently dropped file is a silently missing number.
  * Source files are never modified. Output goes only to --out.
  * Deterministic: rows are emitted in file order, files in sorted order, so
    re-running produces byte-identical output.
  * Aggregate tables (paired_diff summaries, the final results table, slope
    analyses) are recognised and deliberately NOT migrated -- they are derived
    products, not per-probe records, and re-deriving them from the migrated
    rows is the point.

WHAT CANNOT BE RECOVERED

The dynamics and inventory formats never stored the question text or the
per-probe ground truth in a form that identifies the item. Those fields are set
empty with parse_note="legacy_no_question" rather than reconstructed. Guessing
them would make the rows look complete while being unverifiable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from halluworld.results import ProbeRecord, summarize, write_csv, write_jsonl  # noqa: E402

GRID_CANON = {"model", "level", "episode", "seed", "serializer", "probe_type",
              "ground_truth", "response", "score", "prompt_tokens", "completion_tokens"}
GRID_NO_SER = GRID_CANON - {"serializer"}
GRID_PHASE = GRID_CANON | {"phase"}
INVENTORY = {"model", "correct", "score", "ground_truth", "source", "carrying",
             "asked_color", "asked_object", "n_trajectory_seen", "parse_note", "lm_response"}
DYNAMICS_V2 = {"model", "wind_hint_mode", "correct", "score", "has_wind", "wind_offset",
               "wind_naive_error", "parse_note", "action"}
DYNAMICS_V1 = DYNAMICS_V2 - {"wind_hint_mode"}
INNAV_FULL = {"model", "level", "episode", "seed", "navigation_model", "reached_goal",
              "steps_taken", "egocentric_accuracy", "controlled_static_accuracy", "n_probes",
              "timestep", "probe_type", "ground_truth", "egocentric_response",
              "egocentric_score", "controlled_static_response", "controlled_static_score"}
INNAV_WORLD = (INNAV_FULL - {"level"}) | {"world"}
INNAV_SHORT = {"model", "level", "episode", "seed", "reached_goal", "steps_taken",
               "egocentric_accuracy", "static_accuracy", "n_probes", "timestep",
               "probe_type", "ground_truth", "egocentric_response", "egocentric_score",
               "static_response", "static_score"}

#: Derived products. Recognised so they are skipped deliberately, not silently.
AGGREGATE_TABLES = [
    {"model", "paired_diff", "paired_diff_std", "ego_halluc", "static_halluc",
     "n_worlds", "cognitive_load_effect", "std_error", "ci_lower", "ci_upper"},
    {"model", "paired_diff", "ego_halluc", "static_halluc", "n_probes", "cognitive_load_effect"},
    {"Model", "Overall", "Perception", "Causal", "Memory", "Uncertainty", "X-Levels"},
    {"model", "ego_slope", "static_slope", "diff_slope", "ego_r2", "static_r2"},
    {"category", "cognitive_load_effect", "ego_halluc", "model", "n_worlds",
     "paired_diff", "paired_diff_std", "static_halluc", "std_error"},
    {"category", "cognitive_load_effect", "ego_halluc", "model", "n_probes",
     "paired_diff", "static_halluc"},
    {"Cognitive Load Effect (%)", "Controlled Static (%)", "Egocentric (%)", "Model"},
    {"Category", "Controlled Static (%)", "Egocentric (%)", "Model"},
]


def _num(value, cast=float):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _bool(value):
    if value in (None, ""):
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "t")


def _gt(value):
    """Ground truth was serialized inconsistently: bare strings, python bools,
    JSON dicts, JSON lists. Parse what parses; keep the string otherwise."""
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        low = str(value).strip().lower()
        if low in ("true", "false"):
            return low == "true"
        return value


def adapt_grid(rows, sid, has_serializer, has_phase):
    for i, row in enumerate(rows):
        extra = {}
        if has_phase and row.get("phase"):
            extra["phase"] = row["phase"]
        yield ProbeRecord(
            run_id=sid, track="grid", suite="perception",
            level=row.get("level", ""),
            variant=row.get("serializer", "") if has_serializer else "",
            serializer=row.get("serializer") if has_serializer else None,
            model=row.get("model", ""), model_requested=row.get("model", ""),
            episode=_num(row.get("episode"), int), trial=i,
            seed=_num(row.get("seed"), int),
            probe_type=row.get("probe_type", ""),
            ground_truth=_gt(row.get("ground_truth")),
            response=row.get("response", "") or "",
            score=_num(row.get("score")),
            is_correct=(None if _num(row.get("score")) is None
                        else _num(row.get("score")) >= 1.0),
            prompt_tokens=_num(row.get("prompt_tokens"), int),
            completion_tokens=_num(row.get("completion_tokens"), int),
            extra=extra,
        )


def adapt_inventory(rows, sid):
    for i, row in enumerate(rows):
        yield ProbeRecord(
            run_id=sid, track="grid", suite="inventory",
            level="inventory", probe_type="inventory",
            model=row.get("model", ""), model_requested=row.get("model", ""),
            trial=i,
            ground_truth=_gt(row.get("ground_truth")),
            response=row.get("lm_response", "") or "",
            is_correct=_bool(row.get("correct")),
            score=_num(row.get("score")),
            parse_note=row.get("parse_note", "") or "legacy_no_question",
            extra={k: row.get(k) for k in
                   ("source", "carrying", "asked_color", "asked_object", "n_trajectory_seen")},
        )


def adapt_dynamics(rows, sid, has_hint):
    for i, row in enumerate(rows):
        extra = {k: row.get(k) for k in ("has_wind", "wind_offset", "wind_naive_error", "action")}
        if has_hint:
            extra["wind_hint_mode"] = row.get("wind_hint_mode")
        yield ProbeRecord(
            run_id=sid, track="grid", suite="dynamics",
            level="dynamics", probe_type="dynamics",
            model=row.get("model", ""), model_requested=row.get("model", ""),
            trial=i,
            response=row.get("action", "") or "",
            is_correct=_bool(row.get("correct")),
            score=_num(row.get("score")),
            parse_note=row.get("parse_note", "") or "legacy_no_question",
            extra=extra,
        )


def adapt_innav(rows, sid, level_key, static_prefix, has_nav_model):
    """One legacy row -> two records, one per arm, sharing a pair_id.

    The legacy per-episode aggregates (egocentric_accuracy,
    *_static_accuracy) are NOT carried onto the rows: they are repeated on
    every probe row of an episode, and copying them here would let a
    groupby().mean() double-count them. They are recomputed from the rows.
    """
    for i, row in enumerate(rows):
        level = row.get(level_key, "")
        pair_id = "%s:%s:%s:%s:%s:%d" % (
            sid, level, row.get("episode", ""), row.get("seed", ""),
            row.get("timestep", ""), i)
        shared = dict(
            run_id=sid, track="innav", suite="innav_paired",
            level=level, pair_id=pair_id,
            model=row.get("model", ""), model_requested=row.get("model", ""),
            episode=_num(row.get("episode"), int), trial=i,
            seed=_num(row.get("seed"), int),
            timestep=_num(row.get("timestep"), int),
            probe_type=row.get("probe_type", ""),
            ground_truth=_gt(row.get("ground_truth")),
            extra={"reached_goal": row.get("reached_goal"),
                   "steps_taken": row.get("steps_taken"),
                   "n_probes": row.get("n_probes"),
                   **({"navigation_model": row.get("navigation_model")} if has_nav_model else {})},
        )
        for variant, resp_col, score_col in (
            ("innav", "egocentric_response", "egocentric_score"),
            ("static_control", "%s_response" % static_prefix, "%s_score" % static_prefix),
        ):
            score = _num(row.get(score_col))
            yield ProbeRecord(
                variant=variant,
                response=row.get(resp_col, "") or "",
                score=score,
                is_correct=(None if score is None else score >= 1.0),
                **shared,
            )


def source_id(path: Path, root: Path) -> str:
    """A stable id unique across the whole input tree.

    NOT path.stem: four stems collide across directories
    (probe_causal_gpt54_full.csv exists under both gridworld/ and
    egocentric_probe_files/), which silently merged their pair_ids and made
    1548 InNav pairs indistinguishable.
    """
    rel = path.relative_to(root).with_suffix("")
    return "/".join(rel.parts)


def migrate_file(path: Path, root: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = frozenset(reader.fieldnames or [])
        rows = list(reader)

    sid = source_id(path, root)

    if any(header == frozenset(t) for t in AGGREGATE_TABLES):
        return None, "aggregate"
    if header == frozenset(GRID_CANON):
        return list(adapt_grid(rows, sid, True, False)), "grid"
    if header == frozenset(GRID_NO_SER):
        return list(adapt_grid(rows, sid, False, False)), "grid"
    if header == frozenset(GRID_PHASE):
        return list(adapt_grid(rows, sid, True, True)), "grid"
    if header == frozenset(INVENTORY):
        return list(adapt_inventory(rows, sid)), "inventory"
    if header == frozenset(DYNAMICS_V2):
        return list(adapt_dynamics(rows, sid, True)), "dynamics"
    if header == frozenset(DYNAMICS_V1):
        return list(adapt_dynamics(rows, sid, False)), "dynamics"
    if header == frozenset(INNAV_FULL):
        return list(adapt_innav(rows, sid, "level", "controlled_static", True)), "innav"
    if header == frozenset(INNAV_WORLD):
        return list(adapt_innav(rows, sid, "world", "controlled_static", True)), "innav"
    if header == frozenset(INNAV_SHORT):
        return list(adapt_innav(rows, sid, "level", "static", False)), "innav"

    raise SystemExit(
        "unrecognised header in %s\n  columns: %s\n"
        "Refusing to guess. Add an adapter or exclude the file explicitly."
        % (path, sorted(header)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="results/legacy")
    ap.add_argument("--out", dest="dst", default="results/migrated/v1")
    ap.add_argument("--check", action="store_true",
                    help="migrate in memory and verify aggregates are preserved")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in src.rglob("*.csv"))
    if not files:
        print("no CSVs under %s" % src)
        return 1

    all_records, by_kind, skipped, drift = [], {}, [], []
    for path in files:
        records, kind = migrate_file(path, src)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if records is None:
            skipped.append(path)
            continue

        # Aggregate preservation: the mean of the migrated scores must equal
        # the mean of the source scores. This is the guard that migration did
        # not silently change a published number.
        if args.check:
            drift.extend(_check_means(path, records))
        all_records.extend(records)

    print("files: %d" % len(files))
    for kind, n in sorted(by_kind.items()):
        print("   %-12s %4d" % (kind, n))
    print("records: %d  (from %d migrated files)" % (len(all_records), len(files) - len(skipped)))
    print("skipped as aggregate tables: %d" % len(skipped))

    if args.check:
        if drift:
            print("\nAGGREGATE DRIFT in %d file(s):" % len(drift))
            for d in drift[:10]:
                print("   %s" % d)
            return 1
        print("aggregate preservation: OK (means match to 1e-9)")
        return 0

    dst = Path(args.dst)
    write_jsonl(all_records, dst / "records.jsonl")
    write_csv(all_records, dst / "records.csv")
    (dst / "summary.json").write_text(
        json.dumps(summarize(all_records), indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % dst)
    return 0


def _check_means(path: Path, records) -> list[str]:
    """Compare the migrated mean score against the source file's own."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    src_scores = []
    for row in rows:
        for col in ("score", "egocentric_score", "controlled_static_score", "static_score"):
            if col in row:
                v = _num(row[col])
                if v is not None:
                    src_scores.append(v)
    got = [r.score for r in records if r.score is not None]
    if not src_scores and not got:
        return []
    if len(src_scores) != len(got):
        return ["%s: %d source scores vs %d migrated" % (path.name, len(src_scores), len(got))]
    a, b = sum(src_scores) / len(src_scores), sum(got) / len(got)
    if abs(a - b) > 1e-9:
        return ["%s: mean %.12f -> %.12f" % (path.name, a, b)]
    return []


if __name__ == "__main__":
    sys.exit(main())
