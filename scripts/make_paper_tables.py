#!/usr/bin/env python3
"""Emit the paper's per-category tables from a sweep directory.

    scripts/make_paper_tables.py results/sweep_X [--version v0.2]

Cells are accuracy over the SCORED records in that cell. Excluded records
(provider failures, empty answers) are left out of both numerator and
denominator, so a cell's n is printed alongside -- a number resting on 300 of
347 questions should not look like one resting on all of them.

Cognitive tier comes from the frozen bank, joined on question_id, not from the
result record: the bank is what the paper cites.
"""

# No halluworld imports on purpose: the bank is gzipped JSONL and this only
# needs question_id -> cognitive_tier from it. Importing halluworld.questions
# for that pulled in a module the login shell's python 3.6 cannot parse, so
# the script failed before doing anything unless the venv was active.

import argparse
import collections
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TRACK_FILES = {"chess": "records.jsonl", "grid": "records.jsonl",
               "terminal": "results.jsonl"}

# Column order per the paper's table definitions.
LAYOUT = {
    "grid": ["Overall", "P", "M", "C", "U", "X"],
    "chess": ["Overall", "P", "C", "M", "X"],
    "terminal": ["Overall", "C", "X", "M", "P", "U"],
}


def _tiers(version, track):
    path = REPO / "halluworld" / "data" / "questions" / version / ("%s.jsonl.gz" % track)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {row["question_id"]: row["cognitive_tier"] for row in rows}


def _cells(records, tiers, columns):
    by_tier = collections.defaultdict(list)
    overall = []
    for record in records:
        if record.get("score") is None:
            continue
        overall.append(record["score"])
        tier = tiers.get(record["question_id"])
        if tier:
            by_tier[tier].append(record["score"])

    out = {}
    for column in columns:
        values = overall if column == "Overall" else by_tier.get(column, [])
        out[column] = (sum(values) / len(values), len(values)) if values else (None, 0)
    return out


def _job_cells(root, track, model, tiers, columns):
    """Cells for one model's job under `root`, or None if it has no results."""
    path = root / track / model / TRACK_FILES[track]
    if not path.is_file():
        return None, 0
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return _cells(records, tiers, columns), len(records)


def _fmt(cells, columns):
    return "  ".join(
        "%-14s" % ("--" if cells is None or cells[c][0] is None
                   else "%.4f (%d)" % cells[c])
        for c in columns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--version", default="v0.2")
    ap.add_argument("--fen-sweep",
                    help="sweep directory for the chess Incorrect-FEN condition; "
                         "its cells are appended as Table 4's second block, "
                         "paired by model")
    args = ap.parse_args()

    root = Path(args.sweep_dir)
    fen_root = Path(args.fen_sweep) if args.fen_sweep else None
    for track, columns in LAYOUT.items():
        tiers = _tiers(args.version, track)
        paired = track == "chess" and fen_root is not None
        header = columns + (columns if paired else [])
        print("// %s -- accuracy (n scored)%s" % (
            track, "   [No FEN | Incorrect FEN]" if paired else ""))
        print("//   %s" % "  ".join("%-14s" % c for c in ["model"] + header))
        rows = []
        for job in sorted((root / track).glob("*")):
            cells, total = _job_cells(root, track, job.name, tiers, columns)
            if cells is None:
                continue
            scored = cells["Overall"][1]
            flag = "" if scored == total else "   // %d of %d scored" % (scored, total)
            values = [cells[c][0] for c in columns]
            line = _fmt(cells, columns)
            if paired:
                # The two banks share question_ids, so the same tier map applies.
                fcells, ftotal = _job_cells(fen_root, track, job.name, tiers, columns)
                line += "  " + _fmt(fcells, columns)
                values += [None if fcells is None else fcells[c][0] for c in columns]
                if fcells is not None and fcells["Overall"][1] != ftotal:
                    flag += "   // incorrect-FEN: %d of %d scored" % (fcells["Overall"][1], ftotal)
            print("//   %-14s %s%s" % (job.name, line, flag))
            rows.append((job.name, values))

        print("  rows: [")
        for name, values in rows:
            cells = ", ".join("null" if v is None else "%.4f" % v for v in values)
            print("    ['%s', %s]," % (name, cells))
        print("  ],")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
