#!/usr/bin/env python3
"""Re-mark empty answers in existing results as excluded rather than wrong.

    scripts/rescore_empty_answers.py results/sweep_X            # report only
    scripts/rescore_empty_answers.py results/sweep_X --apply    # rewrite in place

chess and grid replay used to hand an empty response to the evaluator, which
returned "not correct" and scored it 0. Terminal has always excluded these, and
the project settled the principle when BadRequestError was fixed: score=None
means excluded, score=0 means the model answered and was wrong.

A model that spends its whole token budget on internal reasoning returns no
text. Scoring that 0 records a hallucination it never made, and it lands
hardest on exactly the models that reason most.

This needs no API calls: the responses are already in the records, so the
correction is a re-read of what was stored. The original file is kept as
.prescore-N.
"""

import argparse
import json
from pathlib import Path


def _is_refusal(record):
    """Strictly: zero output tokens AND the provider flagged a refusal.

    Mirrors halluworld.results.exclusion_reason. A refusal that emitted any
    text is not a refusal here -- it is graded as that text.
    """
    if (record.get("completion_tokens") or 0) != 0:
        return False
    extra = record.get("extra") or {}
    if extra.get("stop_reason") == "refusal":
        return True
    # Older records predate the structural marker; fall back to the text.
    return extra.get("stop_reason") is None and \
        str(record.get("response") or "").strip().startswith("[refusal]")


def _rescore(path, refusals=False):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = 0
    for record in records:
        if record.get("score") is None:
            continue
        empty = not (record.get("response") or "").strip()
        refused = refusals and _is_refusal(record)
        if not (empty or refused):
            continue
        record["score"] = None
        record["is_correct"] = None
        record["error"] = "refusal" if refused else "empty_response"
        record["parse_note"] = ("refusal: provider-flagged, 0 output tokens; excluded"
                                if refused else "empty response, re-marked as excluded")
        changed += 1
    return records, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--refusals", action="store_true",
                    help="also exclude refusals (strictly: 0 output tokens + provider "
                         "flag) with error=refusal. To RE-ASK them instead, run the "
                         "sweep afterwards with REFUSAL_RETRIES set -- resume re-asks "
                         "anything unscored.")
    args = ap.parse_args()

    root = Path(args.sweep_dir)
    paths = sorted(root.glob("*/*/records.jsonl")) + sorted(root.glob("*/*/results.jsonl"))
    paths = [p for p in paths if "_dryrun" not in p.parts and p.stat().st_size]

    total = 0
    for path in paths:
        records, changed = _rescore(path, refusals=args.refusals)
        if not changed:
            continue
        total += changed
        scored = [r for r in records if r.get("score") is not None]
        accuracy = sum(r["score"] for r in scored) / len(scored) if scored else None
        print("%-9s %-22s %3d re-marked  new accuracy %s (n=%d)"
              % (path.parent.parent.name, path.parent.name, changed,
                 "n/a" if accuracy is None else "%.4f" % accuracy, len(scored)))
        if not args.apply:
            continue
        n = 1
        while path.with_suffix(path.suffix + ".prescore-%d" % n).exists():
            n += 1
        path.rename(path.with_suffix(path.suffix + ".prescore-%d" % n))
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    print()
    print("%d %sanswers re-marked" % (total, "empty/refused " if args.refusals else "empty "))
    if not args.apply:
        print("(report only; pass --apply to rewrite, keeping the originals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
