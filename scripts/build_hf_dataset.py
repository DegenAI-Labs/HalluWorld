#!/usr/bin/env python3
"""Stage the frozen question bank as a Hugging Face dataset repository.

The bank is distributed as a *gated* Hugging Face dataset rather than from the
public source repository, so the items are one click-through away from anyone
who wants to evaluate a model but are not sitting in a GitHub tree that gets
scraped into training corpora. This script produces the directory that gets
uploaded; it never uploads unless asked.

    python scripts/build_hf_dataset.py --out ../halluworld-questions-hf
    python scripts/build_hf_dataset.py --out ../halluworld-questions-hf --check
    python scripts/build_hf_dataset.py --out ../halluworld-questions-hf \
        --upload DegenAI-Labs/HalluWorld --gated manual

Layout of the staged directory
------------------------------
One folder per bank version, each self-contained:

    README.md                          dataset card: gating prompt, viewer configs, schema
    v0.1/README.md                     what the folder holds
    v0.1/questions/                    manifest.json + <track>.jsonl.gz + trajectories.jsonl.gz
    v0.1/data/                         flattened views for `load_dataset` / the viewer
    v0.1/configs/chess/                the chess run configs behind the v0.1 results
    v0.1/raw/terminal/                 source probes the terminal bank is built from
    v0.1/raw/chess/                    the chess export the chess bank is built from

With ``--heldout DIR`` (a local copy of the held-out material) it also stages:

    v0.2/README.md                     how to rebuild v0.2 from the public repo
    v0.2/questions/                    the frozen held-out bank: chess (No-FEN and
                                       incorrect-FEN conditions), grid, terminal
    v0.2/configs/                      chess generation configs (the held-out seeds)
    v0.2/raw/grid/                     the authored grid v0.2 questions
    v0.2/raw/terminal/                 source probes the terminal v0.2 bank is built from

Raw folders use plain track names. The builder's --terminal-namespace and
--chess-source-name flags supply the original source names, which are baked
into question ids and provenance, so a rebuild stays byte-identical.

`halluworld.data.hub_bank_path` maps a bank version to its folder here
(`v0.2` -> `v0.2/questions`). The held-out set
lives in the same gated dataset; the public GitHub repo holds v0.1 only.

Two copies of the v0.1 questions are deliberate. `questions/` is what the
harness downloads and verifies against the manifest's SHA-256 digests, so it
has to be the exact bytes `scripts/build_question_bank.py` wrote. `data/` exists because
`ground_truth` is `Any` in the record schema (bool, int, str, list, or null
depending on the probe), and Arrow refuses a column whose type changes between
rows. The views JSON-encode `ground_truth` and the nested dicts so every column
has one type, which is what the dataset viewer and `datasets.load_dataset`
need.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from halluworld.data import PROBE_BANKS_DIR, hub_bank_path, questions_dir  # noqa: E402

DEFAULT_REPO_ID = "DegenAI-Labs/HalluWorld"
TRACKS = ("grid", "chess", "innav", "terminal")

#: Raw generation inputs that travel with the bank so it can be rebuilt. The
#: Opus regeneration (`terminal_opus_20260725`) is v0.2 seed material and is
#: deliberately not listed.
RAW_SOURCES = {"terminal": "terminal_20260504", "chess": "chess_20260505_fen-off"}

#: Columns whose values are not scalars in the record schema. They are
#: JSON-encoded in the flattened views so each column has a single Arrow type.
JSON_COLUMNS = ("ground_truth", "probe_kwargs", "provenance", "extra")
TRAJECTORY_JSON_COLUMNS = ("messages", "trajectory")

#: Declared Arrow types for the views, written into the card's `dataset_info`.
#: Declaring them matters for the `all` config: a column that is entirely null
#: in one track (grid has no difficulty scores) is inferred as `null` from that
#: file alone, and Arrow refuses to cast another track's integers into it.
QUESTION_FEATURES = (
    ("question_id", "string"), ("track", "string"), ("suite", "string"),
    ("task_or_level", "string"), ("kind", "string"), ("probe_type", "string"),
    ("question", "string"), ("ground_truth", "string"), ("ground_truth_type", "string"),
    ("answer_schema", "string"), ("context", "string"), ("probe_class", "string"),
    ("probe_kwargs", "string"), ("cognitive_tier", "string"),
    ("failure_mode_target", "string"), ("difficulty", "int64"),
    ("answerability", "int64"), ("provenance", "string"), ("extra", "string"),
    ("schema_version", "string"),
)
TRAJECTORY_FEATURES = (
    ("config", "string"), ("file", "string"), ("seed", "int64"),
    ("steps_taken", "int64"), ("reached_goal", "bool"),
    ("navigation_model", "string"), ("probe_model", "string"),
    ("serializer", "string"), ("messages", "string"), ("trajectory", "string"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _flatten_question(record: dict) -> dict:
    out = dict(record)
    gt = record.get("ground_truth")
    out["ground_truth_type"] = "null" if gt is None else type(gt).__name__
    for key in JSON_COLUMNS:
        out[key] = json.dumps(record.get(key), sort_keys=True, ensure_ascii=False)
    return out


def _flatten_trajectory(record: dict) -> dict:
    out = dict(record)
    for key in TRAJECTORY_JSON_COLUMNS:
        out[key] = json.dumps(record.get(key), sort_keys=True, ensure_ascii=False)
    return out


def _write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
            n += 1
    return n


def _heldout_section(heldout: dict | None) -> str:
    if not heldout:
        return ""
    return f"""## Held-out v0.2

The held-out set lives in this gated dataset too, but not in the public GitHub repo, which holds
v0.1 only. `v0.2/questions/` is the frozen bank ({heldout.get("v0.2", "?")} records): chess in two
reported conditions, No-FEN (`chess.jsonl.gz`) and incorrect-FEN (`chess_fen_transpose.jsonl.gz`,
the same questions with a corrupted FEN line), plus grid and terminal, all under one manifest.

```bash
halluworld eval chess --provider openai --model gpt-4o-mini --version v0.2 --out results/chess_v0.2
halluworld eval chess --provider openai --model gpt-4o-mini --version v0.2 --fen-mode transpose \\
    --out results/chess_v0.2_transpose
```

The generation configs (`v0.2/configs/`), the authored grid source (`v0.2/raw/grid/`) and the
terminal source probes (`v0.2/raw/terminal/`) are here as well. [v0.2/README.md](v0.2/README.md) explains how to copy them into a checkout of the public
repo and rebuild v0.2 from scratch. Please report v0.2 results without reposting its items.

"""


def _heldout_tree(heldout: dict | None) -> str:
    if not heldout:
        return ""
    return (
        "v0.2/                      held out (see below)\n"
        "  README.md                how to rebuild v0.2 from the public repo\n"
        "  questions/               the frozen v0.2 bank, both chess conditions\n"
        "  configs/                 chess generation configs\n"
        "  raw/grid/                the authored grid questions\n"
        "  raw/terminal/            the terminal source probes\n"
    )


def _v01_readme(manifest: dict, repo_id: str) -> str:
    return f"""# HalluWorld v0.1

The public release: {manifest["total_questions"]:,} frozen questions across grid, chess, InNav and
terminal, plus {manifest.get("trajectories", {}).get("count", 0)} InNav navigation trajectories.
These are the questions behind the published results.

| Path | What it is |
|---|---|
| `questions/` | the canonical frozen bank: `manifest.json`, one `<track>.jsonl.gz` per track, and `trajectories.jsonl.gz`. The harness downloads and checksums this folder. |
| `data/` | the same records flattened to one type per column, for `datasets.load_dataset` and the viewer |
| `configs/chess/` | the chess run configs behind the published chess results (also packaged in the harness) |
| `raw/terminal/` | the source probes `scripts/build_question_bank.py` builds the terminal bank from |
| `raw/chess/` | the chess export the chess bank is built from |

Rebuild all four tracks from a checkout of the public repo and compare against the published
digests. `--check` hashes the rebuilt records; nothing is written.

```bash
hf download {repo_id} --repo-type dataset --local-dir ../halluworld-hf --include "v0.1/*"
python scripts/build_question_bank.py --version v0.1 --check \\
    --terminal-source ../halluworld-hf/v0.1/raw/terminal --terminal-namespace "" \\
    --chess-source ../halluworld-hf/v0.1/raw/chess --chess-source-name chess_20260505_fen-off
```

The two name flags restore the original source names, which are part of the question ids and
provenance. Without them the check reports drift.

Grid and InNav need no inputs from here: grid is extracted from the probe definitions in the
public code, and InNav regenerates from its seeds.
"""


def _card(manifest: dict, repo_id: str, version: str, heldout: dict | None = None) -> str:
    tracks = manifest["tracks"]
    total = manifest["total_questions"]
    traj = manifest.get("trajectories") or {}

    configs = []
    for track in TRACKS:
        configs.append(
            f"- config_name: {track}\n  data_files:\n"
            f"  - split: test\n    path: {version}/data/{track}.jsonl"
        )
    configs.append(
        "- config_name: trajectories\n  data_files:\n"
        f"  - split: test\n    path: {version}/data/trajectories.jsonl"
    )
    all_paths = "\n".join(f"    - {version}/data/{t}.jsonl" for t in TRACKS)
    configs.append(
        "- config_name: all\n  default: true\n  data_files:\n"
        f"  - split: test\n    path:\n{all_paths}"
    )
    configs_yaml = "\n".join(configs)

    def info(name: str, features, n: int) -> str:
        feats = "\n".join(f"  - name: {col}\n    dtype: {dtype}" for col, dtype in features)
        return (
            f"- config_name: {name}\n  features:\n{feats}\n"
            f"  splits:\n  - name: test\n    num_examples: {n}"
        )

    infos = [info(t, QUESTION_FEATURES, tracks[t]["count"]) for t in TRACKS]
    infos.append(info("all", QUESTION_FEATURES, total))
    if traj:
        infos.append(info("trajectories", TRAJECTORY_FEATURES, traj["count"]))
    dataset_info_yaml = "\n".join(infos)

    rows = "\n".join(
        "| {t} | {c} | {l} | {tiers} |".format(
            t=track,
            c=tracks[track]["count"],
            l=tracks[track]["levels_or_tasks"],
            tiers=" ".join(f"{k}:{v}" for k, v in sorted(tracks[track]["by_cognitive_tier"].items())),
        )
        for track in TRACKS
    )
    qdir = hub_bank_path(version)
    checksums = "\n".join(
        f"| `{qdir}/{tracks[t]['file']}` | {tracks[t]['count']} | `{tracks[t]['sha256']}` |"
        for t in TRACKS
    )
    if traj:
        checksums += f"\n| `{qdir}/{traj['file']}` | {traj['count']} | `{traj['sha256']}` |"

    return f"""---
license: cc-by-4.0
pretty_name: HalluWorld
language:
- en
task_categories:
- question-answering
tags:
- hallucination
- benchmark
- world-models
- language-model-agents
- gridworld
- chess
- terminal
size_categories:
- 1K<n<10K
extra_gated_heading: Request access to the HalluWorld question bank
extra_gated_prompt: >-
  HalluWorld is a frozen benchmark. Its value depends on the items staying out
  of training corpora, so the bank is distributed behind this gate rather than
  from the public source repository. The data is licensed CC BY 4.0; by
  requesting access you acknowledge the requests below, which are asked of you
  as a benchmark user rather than imposed as license terms.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use:
    type: select
    options:
    - Evaluating models
    - Benchmark research
    - Other
  I will not repost the items publicly or include them in model training data: checkbox
  I will report results on the full frozen bank or state clearly which subset was used: checkbox
extra_gated_button_content: Request access
configs:
{configs_yaml}
dataset_info:
{dataset_info_yaml}
---

# HalluWorld

The frozen question bank behind [HalluWorld: A Controlled Benchmark for Hallucination via
Reference World Models](https://arxiv.org/abs/2605.19341). **{total:,} items** across four tracks,
plus **{traj.get('count', 0)} navigation trajectories** for the in-navigation track. Code, docs,
and the evaluation harness live at
[github.com/DegenAI-Labs/HalluWorld](https://github.com/DegenAI-Labs/HalluWorld).

| Track | Items | Levels / tasks | Cognitive tiers |
|---|---:|---:|---|
{rows}

Tiers: **P** perceptual, **M** memory, **C** causal, **U** uncertainty, **X** compound. The chess
battery has no U-tier probe by design.

## Using it with the harness

The harness fetches this repository on first use. Accept the gate, log in once, and every
`halluworld` command finds the bank on its own:

```bash
pip install halluworld
hf auth login                    # or export HF_TOKEN=...
halluworld questions verify      # downloads {qdir}/ and checks every checksum
halluworld eval terminal --provider openai --model gpt-4o-mini --out results/terminal
```

Air-gapped machines can point the harness at a local copy instead:

```bash
hf download {repo_id} --repo-type dataset --local-dir ./halluworld-hf
export HALLUWORLD_QUESTIONS_DIR=./halluworld-hf
```

## Loading it as a dataset

```python
from datasets import load_dataset

terminal = load_dataset("{repo_id}", "terminal", split="test")
everything = load_dataset("{repo_id}", split="test")   # the `all` config
```

## Layout

Each version is one self-contained folder:

```
README.md                  this card
{version}/
  README.md
  questions/               manifest.json + the frozen <track>.jsonl.gz files + trajectories
  data/                    flattened views of the same records, for load_dataset and the viewer
  configs/chess/           the chess run configs behind the published results
  raw/                     generation inputs the terminal and chess banks are rebuilt from
{_heldout_tree(heldout)}```

The `{qdir}/` files are the release artifact: their digests are what
`halluworld questions verify` checks and what published numbers trace back to. The `{version}/data/` views
hold identical content but JSON-encode the fields whose type varies between rows
(`ground_truth`, `probe_kwargs`, `provenance`, `extra`; `messages` and `trajectory` for traces) so
every column has one Arrow type. `ground_truth_type` records what to decode it back to.

| File | Records | SHA-256 (uncompressed) |
|---|---:|---|
{checksums}

{_heldout_section(heldout)}## Record schema

Every item shares one schema, defined in `halluworld/questions.py`:

| Field | Meaning |
|---|---|
| `question_id` | stable, namespaced: `terminal/000498`, `grid/P4_harder_array/q07` |
| `track`, `suite`, `task_or_level` | where the item comes from |
| `kind` | `fixed`: question and ground truth are frozen literals. `generated`: rebuilt at run time from `probe_class` + `probe_kwargs` under a fixed seed (all InNav items, 9 grid items) |
| `question`, `ground_truth`, `answer_schema` | the item; `ground_truth` is a bool, int, str, list, or null depending on the probe |
| `context` | the observation the model is shown, when it is frozen with the item (terminal, chess) |
| `cognitive_tier` | P / M / C / U / X, the cross-track comparison axis |
| `failure_mode_target` | terminal-only fine-grained label; `unclassified` elsewhere |
| `difficulty`, `answerability` | 1-5 generator scores where available |
| `provenance`, `extra` | track-specific payload |

Terminal `context` fields are verbatim tmux pane captures from agent runs inside Terminal-Bench
task containers, so they contain fragments of those tasks' files and command output. HalluWorld's
CC BY 4.0 grant covers its questions, answers, labels, and curation; it does not relicense
third-party material appearing inside captured contexts.

## Citation

```bibtex
@article{{liu2026halluworld,
  title   = {{HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models}},
  author  = {{Liu, Emmy and Gangal, Varun and Yu, Michael and Tao, Zhuofu and
             Singh, Karan and Kumar, Sachin and Feng, Steven Y.}},
  journal = {{arXiv preprint arXiv:2605.19341}},
  year    = {{2026}}
}}
```
"""


def _stage_heldout(out: Path, heldout: Path, repo_id: str) -> dict:
    """Copy the held-out v0.2 material in, scrubbing local paths from manifests."""
    staged = {}
    for name in ("v0.2",):
        src = heldout / "frozen" / name
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        dest = out / hub_bank_path(name)
        dest.mkdir(parents=True)
        for entry in manifest["tracks"].values():
            shutil.copy2(src / entry["file"], dest / entry["file"])
            ok_digest = hashlib.sha256(gzip.open(src / entry["file"], "rb").read()).hexdigest()
            if ok_digest != entry["sha256"]:
                raise SystemExit(f"{name}/{entry['file']}: sha256 mismatch against manifest")
            # Machine-local build paths do not belong in a shared dataset. The
            # manifest is not itself checksummed, so this changes no digest.
            if entry.get("source"):
                entry["source"] = Path(entry["source"]).name
        (dest / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged[name] = manifest["total_questions"]
    shutil.copytree(heldout / "configs" / "v0.2", out / "v0.2" / "configs")
    shutil.copytree(heldout / "grid", out / "v0.2" / "raw" / "grid")
    # The original folder name (extracted_llm_probes_heldout_20260908) is the
    # question_id namespace; the README's rebuild passes it via
    # --terminal-namespace, so the folder itself can carry a plain name.
    (src,) = sorted((heldout / "raw_terminal").iterdir())
    shutil.copytree(src, out / "v0.2" / "raw" / "terminal")
    (out / "v0.2" / "README.md").write_text(_regen_doc(repo_id), encoding="utf-8")
    return staged


def _regen_doc(repo_id: str) -> str:
    return f"""# Rebuilding HalluWorld v0.2 (held out)

v0.2 is the held-out set. Its frozen bank is `questions/`, one manifest over four files:

| File | Records | What it is |
|---|---:|---|
| `chess.jsonl.gz` | 347 | chess, No-FEN condition |
| `chess_fen_transpose.jsonl.gz` | 347 | the same chess questions with a corrupted FEN line (incorrect-FEN condition) |
| `grid.jsonl.gz` | 231 | authored grid questions |
| `terminal.jsonl.gz` | 200 | terminal probes |
 The public GitHub repo has all the
code needed to rebuild them but deliberately none of the inputs, which live here:

| Path | What it is |
|---|---|
| `configs/chess.env` | chess generation config: seeds, Lichess rating band, difficulty knobs |
| `configs/chess_fen_transpose.env` | same positions and questions, incorrect-FEN condition |
| `raw/grid/r4-new-questions.jsonl` | the 231 authored grid v0.2 questions |
| `raw/grid/r4-new-questions.csv` | rendered model-visible contexts, merged in by the builder |
| `raw/grid/manifest.json` | coauthor-review metadata, including one ground-truth correction |
| `raw/terminal/` | the 200 terminal source probes plus the generator's `summary.json` |

Please keep these files out of the public repo and out of anything public.

## Evaluate on the frozen bank

No rebuild is needed for this; the harness downloads `v0.2/questions/` on demand:

```bash
halluworld questions verify --version v0.2
halluworld eval chess --provider openai --model MODEL --version v0.2 --out results/chess_v0.2
halluworld eval chess --provider openai --model MODEL --version v0.2 --fen-mode transpose --out results/chess_v0.2_transpose
```

## Rebuild from the public repo

From the root of a checkout, with `pip install -e '.[chess,hf]'`. The bank is written to
`halluworld/data/questions/v0.2/`, which is gitignored, so nothing below can be committed by
accident. Keep the configs outside the checkout, as below.

```bash
hf download {repo_id} --repo-type dataset --local-dir ../halluworld-hf --include "v0.2/*"
H=../halluworld-hf/v0.2

# chess + grid (chess runs under a stub model: no API calls)
TRACKS="chess grid" GRID_SOURCE=$H/raw/grid/r4-new-questions.jsonl \\
    scripts/build_v02_bank.sh $H/configs/chess.env

# incorrect-FEN chess, added to the same v0.2 bank as chess_fen_transpose.jsonl.gz
CHESS_BANK_KEY=chess_fen_transpose scripts/build_v02_bank.sh $H/configs/chess_fen_transpose.env

# terminal; the namespace flag restores the source name baked into every question_id
python scripts/build_question_bank.py --version v0.2 --tracks terminal \\
    --terminal-source $H/raw/terminal --terminal-namespace extracted_llm_probes_heldout_20260908
```

The terminal probes were generated by GPT-5.4 from agent trajectories in the v0.1 Terminal-Bench
run, and extracted from its runtime-probe output. The probe files here are the extraction's output,
also kept in the private `DegenAI-Labs/HalluWorld-heldout-testset` GitHub repo. The generator's
prompt and response logs (`llm_io/`) stay there and are not needed to rebuild. Two metadata fields
that held a contributor's local paths (`llm_probe_prompt_path`, `llm_probe_response_path`) are
reduced to bare file names, as in v0.1; no other byte differs.

## Checking a rebuild

Grid and terminal rebuild byte-identical to `questions/grid.jsonl.gz` and
`questions/terminal.jsonl.gz` (checked 2026-09-27).

No-FEN chess rebuilds with identical ids, questions, contexts and answers. Incorrect-FEN chess
rebuilds with identical ids, questions and answers, but 340 of its 347 contexts show a different
corrupted FEN line. Every seed in `configs/chess_fen_transpose.env` was recorded and does
reproduce; the pair of pieces swapped, however, was also keyed on Python's built-in `hash()` of
the FEN, which is salted with a fresh random key each time the interpreter starts
(`PYTHONHASHSEED` was unset). That key was never recorded and cannot be recovered. The harness now
uses a stable hash, so rebuilds agree with each other but not with the frozen file, which stays
canonical.

Both chess files' SHA-256 also differ because each record's
`provenance` stores the export date and local export path. Compare without those two fields:

```python
import gzip, json

def load(path):
    out = {{}}
    for line in gzip.open(path, "rt"):
        if line.strip():
            record = json.loads(line)
            for key in ("exported", "source"):
                record["provenance"].pop(key, None)
            out[record["question_id"]] = record
    return out

assert load("../halluworld-hf/v0.2/questions/chess.jsonl.gz") == load("halluworld/data/questions/v0.2/chess.jsonl.gz")

# Incorrect-FEN: compare everything but the context, whose FEN line differs (see above).
def strip_context(records):
    return {{k: {{f: v for f, v in r.items() if f != "context"}} for k, r in records.items()}}

assert strip_context(load("../halluworld-hf/v0.2/questions/chess_fen_transpose.jsonl.gz")) == \\
    strip_context(load("halluworld/data/questions/v0.2/chess_fen_transpose.jsonl.gz"))
```

## Caveats

- **Terminal overlaps v0.1 at the evidence level.** 198 of the 200 probes ask a new question over
  a terminal pane that also appears in v0.1: held out at the question level, not the observation
  level.
- **One grid answer was corrected in review:** `grid/v0.2/C4_forking_paths/b9a86a12bff6`, True to
  False.
- **The review package's scorer warning is historical.** Grid replay now takes the last integer
  in a reply, not the first.
"""


def stage(out: Path, version: str, repo_id: str, include_raw: bool,
          heldout: Path | None = None) -> dict:
    src = questions_dir(version)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bank_version") != version:
        raise SystemExit(f"{src}/manifest.json declares bank_version={manifest.get('bank_version')!r}")

    if out.exists():
        shutil.rmtree(out)
    canonical = out / hub_bank_path(version)
    canonical.mkdir(parents=True)
    views = out / version / "data"
    views.mkdir(parents=True)

    report = {"version": version, "canonical": {}, "views": {}, "raw": {}}

    shutil.copy2(src / "manifest.json", canonical / "manifest.json")
    entries = {t: manifest["tracks"][t] for t in TRACKS}
    if manifest.get("trajectories"):
        entries["trajectories"] = manifest["trajectories"]
    for name, entry in entries.items():
        path = src / entry["file"]
        shutil.copy2(path, canonical / entry["file"])
        # The manifest hashes uncompressed bytes; recheck here so a stale or
        # half-rebuilt bank never gets staged for upload.
        with gzip.open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"{path}: sha256 mismatch against manifest; rebuild the bank first")
        report["canonical"][entry["file"]] = digest

        flatten = _flatten_trajectory if name == "trajectories" else _flatten_question
        n = _write_jsonl(views / f"{name}.jsonl", (flatten(r) for r in _iter_jsonl_gz(path)))
        if n != entry["count"]:
            raise SystemExit(f"{name}: wrote {n} view rows, manifest says {entry['count']}")
        report["views"][f"{name}.jsonl"] = n

    if include_raw:
        for name, source in RAW_SOURCES.items():
            src_dir = PROBE_BANKS_DIR / source
            if not src_dir.is_dir():
                print(f"  WARN raw source missing, skipping: {src_dir}")
                continue
            dest = out / version / "raw" / name
            shutil.copytree(src_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
            report["raw"][name] = sum(1 for p in dest.rglob("*") if p.is_file())

    from halluworld.data import DATA_DIR
    chess_configs = DATA_DIR / "configs" / "chess"
    if chess_configs.is_dir():
        shutil.copytree(chess_configs, out / version / "configs" / "chess")
    (out / version / "README.md").write_text(_v01_readme(manifest, repo_id), encoding="utf-8")

    if heldout is not None:
        report["heldout"] = _stage_heldout(out, heldout, repo_id)
    (out / "README.md").write_text(
        _card(manifest, repo_id, version, heldout=report.get("heldout")), encoding="utf-8")
    (out / ".gitattributes").write_text(
        "*.gz filter=lfs diff=lfs merge=lfs -text\n"
        "*.jsonl filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    return report


def check(out: Path, version: str) -> None:
    """Load every config from the staged directory exactly as the Hub will.

    `load_dataset(<dir>)` reads the card's `configs` and `dataset_info`, so
    this exercises the YAML, the declared features, and the files together.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("--check needs `pip install datasets`") from exc

    manifest = json.loads((out / hub_bank_path(version) / "manifest.json").read_text(encoding="utf-8"))
    expected = {t: manifest["tracks"][t]["count"] for t in TRACKS}
    expected["all"] = manifest["total_questions"]
    if manifest.get("trajectories"):
        expected["trajectories"] = manifest["trajectories"]["count"]
    for name, n in expected.items():
        ds = load_dataset(str(out), name=name, split="test")
        if ds.num_rows != n:
            raise SystemExit(f"{name}: {ds.num_rows} rows loaded, manifest says {n}")
        null_typed = [c for c, f in ds.features.items() if getattr(f, "dtype", "") == "null"]
        if null_typed:
            raise SystemExit(f"{name}: columns inferred as null: {null_typed}")
        print(f"  OK   {name:<12} {ds.num_rows:>5} rows, {len(ds.features)} columns")


def upload(out: Path, repo_id: str, gated: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    # A new repo is created private. Making it public is the release step and
    # is left to a human: `hf repos settings <id> --repo-type dataset --public`
    # or the web UI. An existing repo keeps whatever visibility it has.
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if gated is not None:
        api.update_repo_settings(repo_id, repo_type="dataset", gated=gated)
    # The staged directory is the whole dataset: anything on the Hub that it
    # no longer contains (an old layout, a renamed file) is removed in the
    # same commit.
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish frozen question bank",
        delete_patterns=["*"],
    )
    print(f"uploaded {out} -> https://huggingface.co/datasets/{repo_id} (gated={gated})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory to stage (replaced if it exists)")
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="dataset id written into the card")
    ap.add_argument("--no-raw", action="store_true", help="omit raw/ generation inputs")
    ap.add_argument("--heldout", type=Path, metavar="DIR",
                    help="also stage held-out v0.2 material from DIR (frozen/, configs/, grid/)")
    ap.add_argument("--check", action="store_true", help="after staging, load every view with `datasets`")
    ap.add_argument("--upload", metavar="REPO_ID", help="upload the staged directory to this dataset repo")
    ap.add_argument("--gated", choices=("auto", "manual"), default=None,
                    help="set the access-request mode on upload (default: leave the Hub setting alone)")
    args = ap.parse_args()

    out = args.out.resolve()
    repo_id = args.upload or args.repo_id
    report = stage(out, args.version, repo_id, include_raw=not args.no_raw,
                   heldout=args.heldout.resolve() if args.heldout else None)
    print(f"staged {args.version} at {out}")
    for name, digest in report["canonical"].items():
        print(f"  {name:<28} sha256 {digest[:12]}")
    for name, n in report["views"].items():
        print(f"  {args.version}/data/{name:<20} {n} rows")
    for name, n in report["raw"].items():
        print(f"  {args.version}/raw/{name:<24} {n} files")
    for name, n in report.get("heldout", {}).items():
        print(f"  {hub_bank_path(name) + '/':<28} {n} held-out items")
    if args.check:
        print("checking views with datasets ...")
        check(out, args.version)
    if args.upload:
        upload(out, args.upload, args.gated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
