#!/usr/bin/env python3
"""Pack the InNav navigation trajectories into a released artifact.

The paper commits to this: "We release all 250+ navigation trajectories
(5 episodes x 31 worlds x 1-3 serializers ...) used in our experiments,
enabling exact replication of our evaluation setup even in the egocentric
case."

Those trajectories were sitting in a gitignored results/ directory, so the
commitment was not being met. They are the InNav track's real frozen artifact:
because the agent navigates, an InNav probe is generated against whatever state
the agent actually reached, so there is no fixed question list to freeze. What
CAN be frozen -- and what makes a run replayable -- is the trajectory.

Packs 295 traces (25 MB of JSON) into one gzipped JSONL of about 0.8 MB.

    python3 scripts/build_trajectory_set.py [--check]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "results" / "legacy" / "navigation_traces"
OUT = REPO / "halluworld" / "data" / "questions" / "v0.1" / "trajectories.jsonl.gz"


def collect() -> tuple[list[dict], list[dict]]:
    """Returns (records, skipped). Skips are reported, never silent."""
    records, skipped = [], []
    for path in sorted(SOURCE.rglob("*.json")):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            skipped.append({"file": path.relative_to(SOURCE).as_posix(),
                            "bytes": path.stat().st_size,
                            "reason": "%s: %s" % (type(exc).__name__, exc)})
            print("  SKIP %s (%s, %d bytes)" % (path.name, type(exc).__name__,
                                                path.stat().st_size))
            continue
        rel = path.relative_to(SOURCE)
        # Directory name encodes world_model_serializer; loose files at the top
        # level predate that convention and keep whatever the trace itself says.
        config = rel.parts[0] if len(rel.parts) > 1 else ""
        records.append({
            "config": config,
            "file": rel.as_posix(),
            "seed": trace.get("seed"),
            "navigation_model": trace.get("navigation_model"),
            "probe_model": trace.get("probe_model"),
            "serializer": trace.get("serializer"),
            "reached_goal": trace.get("reached_goal"),
            "steps_taken": trace.get("steps_taken"),
            # Required for deterministic replay.  Older packed v0.1 records
            # omitted this field; release_contract.py can recover the action
            # prefix from messages/trajectory, but every new pack must retain
            # the source sequence directly.
            "action_sequence": trace.get("action_sequence"),
            "messages": trace.get("messages"),
            "trajectory": trace.get("trajectory"),
        })
    return records, skipped


def pack(records: list[dict]) -> bytes:
    ordered = sorted(records, key=lambda r: (r["config"], r["file"]))
    return "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False)
                     for r in ordered).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not SOURCE.is_dir():
        print("no trajectories at %s" % SOURCE)
        return 1

    records, skipped = collect()
    payload = pack(records)
    digest = hashlib.sha256(payload).hexdigest()

    worlds = {r["config"].split("_gpt")[0].split("_claude")[0]
              for r in records if r["config"]}
    print("traces:   %d" % len(records))
    print("configs:  %d" % len({r["config"] for r in records if r["config"]}))
    print("worlds:   %d" % len(worlds))
    print("raw:      %.1f MB" % (len(payload) / 1e6))
    print("sha256:   %s" % digest[:16])
    if skipped:
        print("skipped:  %d unreadable trace(s) -- recorded in the manifest" % len(skipped))

    if args.check:
        if not OUT.exists():
            print("MISSING: %s" % OUT)
            return 1
        with gzip.open(OUT, "rb") as fh:
            existing = fh.read()
        ok = hashlib.sha256(existing).hexdigest() == digest
        print("matches committed artifact:", ok)
        return 0 if ok else 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(OUT), mode="wb", mtime=0) as fh:
        fh.write(payload)
    print("gzipped:  %.2f MB -> %s" % (OUT.stat().st_size / 1e6, OUT))

    # Record it alongside the question banks.
    mpath = OUT.parent / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest["trajectories"] = {
        "file": OUT.name,
        "count": len(records),
        "configs": len({r["config"] for r in records if r["config"]}),
        "worlds": len(worlds),
        "sha256": digest,
        "skipped": skipped,
        "why": ("InNav's frozen artifact. Probes are generated against whatever "
                "state the navigating agent reached, so there is no fixed "
                "question list; the trajectory is what makes a run replayable. "
                "Released per the paper's commitment in Appendix J."),
    }
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("recorded in %s" % mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
