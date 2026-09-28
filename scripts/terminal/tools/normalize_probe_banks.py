#!/usr/bin/env python3
"""Normalize the two extracted probe banks for public release.

Two independent fixes, both idempotent:

1. RESTORE EVALUATION FIDELITY in extracted_llm_probes_20260504/ (the 529 bank).

   Commit b7d30c7 (2026-05-12) replaced "degenAI labs" with "exampleOrg labs"
   inside the ``context`` field -- the evidence text the model actually reads --
   of 5 string-file-search probes. The 11-model leaderboard in PROBE_RESULTS.md
   was produced on 2026-05-04/05, *eight days before* that edit, so the bank as
   published no longer matches the bank as evaluated.

   The edit was made for anonymity. The benchmark is now published under the
   authors' real names, so the anonymity motive is gone and the fidelity cost
   is not worth paying. This restores those 5 ``context`` fields to the text
   the models were actually scored on.

2. SCRUB MACHINE PATHS from extracted_llm_probes_opus_20260725/ (the 100 bank).

   The July Opus regeneration ran from unstripped sources and reintroduced
   absolute local paths into 7 metadata fields of all 100 probes
   (700 occurrences across two contributors' home directories). The 529 bank already
   stores these as bare basenames; this applies the same convention.

   Local machine paths do not belong in a published dataset regardless of
   anonymity, so unlike (1) this scrub is kept and extended.

Both banks are written back with json.dumps(indent=2), preserving each file's
existing trailing-newline convention (the 529 bank has one, the Opus bank does
not). That is byte-exact with how they are already stored, so the diff contains
only the fields this script intends to change.

Usage:
    python3 tools/normalize_probe_banks.py [--check]

    --check  report what would change and exit non-zero if anything would,
             without writing. Suitable for CI.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

BANK_529 = "extracted_llm_probes_20260504"
BANK_OPUS = "extracted_llm_probes_opus_20260725"

# Commit that made the anonymizing edits; its parent holds the pre-edit text.
SCRUB_COMMIT = "b7d30c7"

# Metadata fields that hold filesystem paths and must be reduced to basenames.
PATH_FIELDS = (
    "runtime_probe_path",
    "context_path",
    "pre_context_path",
    "post_context_path",
    "selected_context_path",
    "llm_probe_prompt_path",
    "llm_probe_response_path",
)

# The 5 probes whose `context` was altered by SCRUB_COMMIT.
CONTEXT_ALTERED = (
    "000440_string-file-search_step-0014_llm_generated_runtime-agent-14-2.json",
    "000441_string-file-search_step-0014_llm_generated_runtime-agent-14-3.json",
    "000442_string-file-search_step-0014_llm_generated_runtime-agent-14-4.json",
    "000443_string-file-search_step-0016_llm_generated_runtime-agent-16-0.json",
    "000444_string-file-search_step-0016_llm_generated_runtime-agent-16-1.json",
)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj, trailing_newline):
    """Write in the bank's existing on-disk format.

    Both banks use json.dumps(indent=2), but they differ on the trailing
    newline: the 529 bank has one, the Opus bank does not. Preserving each
    file's existing convention keeps the diff to the lines we actually mean
    to change.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2))
        if trailing_newline:
            fh.write("\n")


def ends_with_newline(path):
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            return False
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def git_show(rev, path):
    out = subprocess.run(
        ["git", "show", "{}:{}".format(rev, path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
    )
    if out.returncode != 0:
        raise RuntimeError("git show {}:{} failed: {}".format(rev, path, out.stderr.strip()))
    return out.stdout


def restore_contexts(check):
    """Fix 1: restore the 5 pre-anonymization `context` fields in the 529 bank."""
    changed = []
    for name in CONTEXT_ALTERED:
        path = os.path.join(BANK_529, name)
        if not os.path.exists(path):
            print("  WARN missing, skipped: {}".format(name))
            continue
        current = read_json(path)
        original = json.loads(git_show(SCRUB_COMMIT + "^", path))
        if current.get("context") == original.get("context"):
            continue  # already restored -- idempotent
        changed.append(name)
        if not check:
            nl = ends_with_newline(path)
            current["context"] = original["context"]
            write_json(path, current, nl)
    return changed


def scrub_paths(check):
    """Fix 2: reduce the 7 path metadata fields to basenames in the Opus bank."""
    changed = []
    files = sorted(
        f for f in glob.glob(os.path.join(BANK_OPUS, "*.json"))
        if not f.endswith("summary.json")
    )
    for path in files:
        nl = ends_with_newline(path)
        obj = read_json(path)
        meta = obj.get("metadata", {})
        hits = 0
        for field in PATH_FIELDS:
            val = meta.get(field)
            if isinstance(val, str) and ("/" in val):
                base = os.path.basename(val)
                if base != val:
                    hits += 1
                    if not check:
                        meta[field] = base
        if hits:
            changed.append((os.path.basename(path), hits))
            if not check:
                write_json(path, obj, nl)
    return changed


def verify_clean():
    """Assert no machine paths survive anywhere in either bank."""
    bad = []
    for bank in (BANK_529, BANK_OPUS):
        for path in glob.glob(os.path.join(bank, "**", "*.json"), recursive=True):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if "/Users/" in text or "/home/" in text:
                bad.append(path)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report changes without writing; exit 1 if any are needed")
    args = ap.parse_args()

    if not os.path.isdir(BANK_529) or not os.path.isdir(BANK_OPUS):
        sys.exit("error: run from the repository root (probe bank directories not found)")

    print("1. Restoring evaluation fidelity in {}/".format(BANK_529))
    restored = restore_contexts(args.check)
    if restored:
        print("   {} probe context(s) {}:".format(
            len(restored), "would be restored" if args.check else "restored"))
        for n in restored:
            print("     {}".format(n))
    else:
        print("   already consistent with the evaluated bank -- nothing to do")

    print("2. Scrubbing machine paths from {}/".format(BANK_OPUS))
    scrubbed = scrub_paths(args.check)
    if scrubbed:
        total = sum(n for _, n in scrubbed)
        print("   {} field(s) across {} file(s) {}".format(
            total, len(scrubbed), "would be scrubbed" if args.check else "scrubbed"))
    else:
        print("   no machine paths present -- nothing to do")

    if not args.check:
        bad = verify_clean()
        if bad:
            print("\nFAILED: machine paths still present in {} file(s):".format(len(bad)))
            for p in bad[:10]:
                print("  {}".format(p))
            return 1
        print("\nverified: no /Users/ or /home/ paths remain in either bank")
        return 0

    return 1 if (restored or scrubbed) else 0


if __name__ == "__main__":
    sys.exit(main())
