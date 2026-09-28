#!/usr/bin/env python3
"""Select the sweep jobs whose numbers a re-run would actually change.

    scripts/mark_for_rerun.py results/sweep_X                 # report only
    scripts/mark_for_rerun.py results/sweep_X --apply         # set them up to re-run

A job is worth re-running when its answers were cut off at the token cap, or
when questions went unanswered -- both make the reported accuracy rest on
something other than the model. A job that never hit the cap would return the
same answers at a higher one, so re-running it spends money to reproduce a
number already held.

Applying moves the affected job's result file aside (to .superseded-N) rather
than deleting it, so the old numbers stay inspectable. The sweep then sees
those jobs as unstarted and re-runs exactly them, skipping the rest.
"""

# No `from __future__ import annotations` and no halluworld imports on purpose:
# this reads JSON files and nothing else, so it should run under whatever
# python3 a login shell has -- the shebang picks up the system interpreter,
# which here is 3.6 and rejects that import outright.

import argparse
import json
from pathlib import Path

# OpenAILM floors reasoning models to 16000, so a 1024 request never bound
# them -- their completions are long by choice, not truncated.
FLOORED_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def _job_stats(path: Path, cap: int) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    model = path.parent.name
    scored = [r for r in records if r.get("score") is not None]
    floored = any(model.startswith(p) for p in FLOORED_PREFIXES)
    truncated = 0 if floored else sum(
        1 for r in scored if (r.get("completion_tokens") or 0) >= cap
    )
    return {
        "path": path,
        "track": path.parent.parent.name,
        "model": model,
        "total": len(records),
        "excluded": len(records) - len(scored),
        "truncated": truncated,
        "affected": (len(records) - len(scored) + truncated) / len(records) if records else 0.0,
    }


def _job_cap(records):
    """The cap this job actually ran under.

    A job resumed after the default changed holds answers from both caps, so
    the largest completion seen is the only evidence available in older files.
    Runs from now on record max_tokens in their manifest.
    """
    seen = max((r.get("completion_tokens") or 0) for r in records) if records else 0
    return 1024 if seen <= 1024 else 16000


def _mark_records(jobs, args):
    """Mark truncated records unscored so resume re-asks exactly them.

    Marking, not deleting. Deleting leaves a file that is shorter but entirely
    scored, and the sweep's skip check counts unscored records -- so the job
    reads as complete and the dropped questions are never re-asked. That
    silently shrank glm-5.3 from 347 chess answers to 341.
    """
    total = 0
    for job in jobs:
        path = job["path"]
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        cap = _job_cap(records)
        model = job["model"]
        if any(model.startswith(p) for p in FLOORED_PREFIXES):
            continue
        marked = 0
        for record in records:
            if record.get("score") is None:
                continue
            if (record.get("completion_tokens") or 0) < cap:
                continue
            record["score"] = None
            record["is_correct"] = None
            record["error"] = "empty_response"
            record["parse_note"] = "truncated at cap %d, marked to be re-asked" % cap
            marked += 1
        if not marked:
            continue
        total += marked
        print("  %-9s %-22s mark %2d truncated at cap %d (of %d)"
              % (job["track"], model, marked, cap, len(records)))
        if not args.apply:
            continue
        n = 1
        while path.with_suffix(path.suffix + ".pretrunc-%d" % n).exists():
            n += 1
        path.rename(path.with_suffix(path.suffix + ".pretrunc-%d" % n))
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    print("\n%d truncated answers marked to be re-asked" % total)
    if not args.apply:
        print("(report only; pass --apply with --records to rewrite)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--cap", type=int, default=1024,
                    help="the --max-tokens the sweep ran under (default 1024)")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="re-run jobs with at least this fraction affected (default 0.10)")
    ap.add_argument("--apply", action="store_true",
                    help="move the affected result files aside so the sweep re-runs them")
    ap.add_argument("--records", action="store_true",
                    help="drop only the affected RECORDS instead of whole jobs, so "
                         "resume re-asks those questions and keeps the rest. Use for "
                         "a handful of stragglers; re-asking a whole job for 7 "
                         "truncated answers pays for 224 already held.")
    args = ap.parse_args()

    root = Path(args.sweep_dir)
    paths = sorted(root.glob("*/*/records.jsonl")) + sorted(root.glob("*/*/results.jsonl"))
    paths = [p for p in paths if "_dryrun" not in p.parts and p.stat().st_size]
    if not paths:
        print("no results under %s" % root)
        return 1

    jobs = [_job_stats(p, args.cap) for p in paths]
    selected = [j for j in jobs if j["affected"] >= args.threshold]
    selected.sort(key=lambda j: -j["affected"])

    print("%-9s %-22s %6s %10s %10s %9s" %
          ("track", "model", "n", "truncated", "excluded", "affected"))
    for job in sorted(jobs, key=lambda j: -j["affected"]):
        mark = "  <- re-run" if job in selected else ""
        print("%-9s %-22s %6d %10d %10d %8.1f%%%s" %
              (job["track"], job["model"], job["total"], job["truncated"],
               job["excluded"], 100 * job["affected"], mark))

    questions = sum(j["total"] for j in selected)
    print()
    print("%d of %d jobs selected, %d questions" % (len(selected), len(jobs), questions))
    print("%d jobs untouched -- their answers never hit the cap, so a re-run "
          "would reproduce them" % (len(jobs) - len(selected)))

    if args.records:
        return _mark_records(jobs, args)

    if not args.apply:
        print("\n(report only; pass --apply to set the selected jobs up to re-run)")
        return 0

    for job in selected:
        target = job["path"]
        n = 1
        while target.with_suffix(target.suffix + ".superseded-%d" % n).exists():
            n += 1
        moved = target.with_suffix(target.suffix + ".superseded-%d" % n)
        target.rename(moved)
        print("  moved %s -> %s" % (target, moved.name))
    print("\nre-run the sweep on this directory; it will ask only these %d questions."
          % questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
