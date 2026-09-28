# The frozen question bank

The set of items every model is evaluated on, frozen so a published number traces back to the exact
question that produced it.

## What is in v0.1

| Track | Items | Coverage | Kind |
|---|---|---|---|
| terminal | 529 | 110 tasks | fixed |
| grid | 443 | 33 levels | 434 fixed / 9 generated |
| chess | 350 | 7 probes | fixed |
| innav | 117 | 33 levels | generated |
| **total** | **1,439** | | |

Plus **294 navigation trajectories** (58 configs, 32 worlds), which are the InNav track's frozen
artifact — see [INNAV.md](INNAV.md) for why it has trajectories rather than a question list.

## Where it lives

The bank is **not in this repository**. It is distributed as a gated Hugging Face dataset,
[`DegenAI-Labs/HalluWorld`](https://huggingface.co/datasets/DegenAI-Labs/HalluWorld),
so the items are one click-through away from anyone evaluating a model but are not sitting in a
GitHub tree that gets scraped into training corpora. Request access there, log in once with
`hf auth login` (or export `HF_TOKEN`), and the harness fetches `v0.1/questions/` into the Hugging Face cache
the first time anything needs it. Files are gzipped JSONL with a `manifest.json` carrying per-track
counts and SHA-256 digests of the uncompressed bytes.

```bash
halluworld questions stats     # counts, tiers, and the standing caveats
halluworld questions verify    # download if needed, then check every file against its checksum
halluworld release check       # also check all four domain runtimes
```

`halluworld.data.questions_dir(version)` is the one resolver every loader goes through. It checks,
in order: `HALLUWORLD_QUESTIONS_DIR` (a local copy, for air-gapped machines and CI caches; either
a full download of the dataset or a bank folder itself), then `halluworld/data/questions/<version>/` in a
source checkout, then the dataset above (`HALLUWORLD_HF_REPO` overrides the id). The dataset also
carries flattened `v0.1/data/*.jsonl` views for `datasets.load_dataset` and the viewer;
`ground_truth` varies in type between rows, so the views JSON-encode it and the nested dicts.

The distinction matters: a checksum can pass while a runner has renamed a level or stopped routing
a probe type to the correct evaluator. See [RELEASE.md](RELEASE.md) for the domain-specific gates.

## Record schema

Defined in `halluworld/questions.py`. The fields that carry the design:

**`cognitive_tier`** — P / M / C / U / X. The cross-track comparison axis, and the reason a single
bank is worth building. It is *derived*, not assigned: gridworld encodes it in the level-key prefix
(`P1_dense_array`), terminal in `probe_type` (`perceptual`), chess in its documented battery.

**`kind`** — `fixed` means question text and ground truth are literals frozen here. `generated`
means the probe is built from a seeded RNG at run time, so the bank stores the class and its
arguments and freezing rests on the seed. A consumer that just wants to ask questions reads
`question`/`ground_truth`; one that wants to rebuild the bank reads `probe_class`/`probe_kwargs`.

**`failure_mode_target`** — deliberately **terminal-only**. Terminal's generator assigned explicit
per-probe labels. Gridworld has free-text `trap` prose written for humans ("one-shot trigger; gate
stays open; model's continuous prior says 'closed'") which does not map onto that taxonomy without
inventing a classification, so grid records are `unclassified` and keep the original prose in
`extra.failure_mode_original`. Slice by `cognitive_tier` for cross-track work.

## Rebuilding

```bash
python3 scripts/build_question_bank.py --version v0.1           # rebuild
python3 scripts/build_question_bank.py --version v0.1 --check   # CI: detect drift
python3 scripts/build_trajectory_set.py                         # repack trajectories
```

The builder writes to `halluworld/data/questions/<version>/`, which is gitignored. Terminal and
chess are built from raw generation inputs that are also not in this repository: they ship as
`v0.1/raw/terminal/` and `v0.1/raw/chess/` in the Hugging Face dataset and are
read from `halluworld/data/probe_banks/` locally, or from anywhere via `--terminal-source` and
`--chess-source`.

### Publishing to Hugging Face

```bash
python3 scripts/build_hf_dataset.py --out ../halluworld-questions-hf --check
python3 scripts/build_hf_dataset.py --out ../halluworld-questions-hf \
    --upload DegenAI-Labs/HalluWorld --gated manual
```

The dataset has one self-contained folder per version: `v0.1/` holds `questions/`, `data/`,
`configs/` and `raw/`, and `v0.2/` holds the held-out equivalents. The first command stages it
(canonical bank files copied byte-for-byte and re-hashed against the manifest, flattened views, raw
inputs, and the dataset card with
the gating prompt) and proves every view loads under Arrow. The second uploads it and turns on
manual access approval.

Builds are byte-reproducible: records sort by `question_id` and gzip is written with `mtime=0`, so
rebuilding identical content yields identical files.

The grid bank is *extracted* from `make_probes()` in `tracks/grid/perception.py` — 4,775 lines
holding 384 `FixedProbe` literals. The builder executes that function per level with a fixed seed
and introspects the results rather than parsing it.
`tests/test_question_bank.py::test_grid_bank_is_a_faithful_extraction` re-runs it across all 33
levels and asserts the frozen text is identical, which is what makes the extraction safe.

## Versioning, and the held-out set

`v0.1` is released, behind the gate above: the questions used in the reported experiments.

A later version is intended to be **held out** — harder items, not released. For those, only the
manifest is published, so set membership and size can be verified without revealing the items. The
schema and builder are version-agnostic by design, so a held-out set is the same pipeline with a
different output directory and a different distribution policy.

### Building a held-out bank

The v0.2 generation configs and the authored grid source are not in this repository, because they
carry the held-out seeds and items. They live in the gated `DegenAI-Labs/HalluWorld` dataset
in its `v0.2/` folder next to the frozen v0.2 bank, and `v0.2/README.md` there has the full recipe. Evaluating on the
frozen bank needs none of this: `--version v0.2` downloads it like v0.1.

v0.2 freezes both reported chess conditions in one manifest: `chess.jsonl.gz` (No-FEN) and
`chess_fen_transpose.jsonl.gz` (the same questions shown with a corrupted FEN line). Select the
second with `halluworld eval chess --version v0.2 --fen-mode transpose`. v0.1 froze only the No-FEN
condition, so its fen-on and fen-transpose arms need `--regenerate`.

To build a new held-out bank, point `build_v02_bank.sh` at a config file kept outside the
checkout. The v0.2 configs themselves are in the dataset:

```bash
hf download DegenAI-Labs/HalluWorld --repo-type dataset --local-dir ../halluworld-hf --include "v0.2/*"
cp ../halluworld-hf/v0.2/configs/chess.env ../my-bank.env
$EDITOR ../my-bank.env                                   # new seeds for a new held-out set
BANK_VERSION=v0.3 scripts/build_v02_bank.sh ../my-bank.env   # generate, freeze, check contamination
halluworld eval chess --provider openai --model MODEL --version v0.3 --out results/chess_v0.3
```

`halluworld/data/questions/` is gitignored in its entirety, so neither the released bank nor a
held-out one can reach the public repo by an absent-minded `git add`.

**Not every track can be regenerated from a config file.** What each one needs:

| Track | Regenerating for a held-out set |
|---|---|
| chess | Config-driven. 81 environment knobs; the seed selects the positions, so a new seed yields new questions. This is what `build_v02_bank.sh` drives. |
| grid | **Not possible from config.** 434 of grid's 443 records are `FixedProbe` literals hardcoded in `perception.py`, and the other 9 freeze as class+kwargs recipes — every seed produces a byte-identical bank. A grid v0.2 requires newly authored levels or probe literals. Once they exist in `perception.LEVELS`, `TRACKS="grid chess" scripts/build_v02_bank.sh` picks them up unchanged. |
| innav | Derives its probes from grid, so it inherits grid's situation. |
| terminal | Generated by the terminal-bench runtime injectors during an agent run, not by this script. See `docs/TERMINAL.md`. The extracted v0.2 probes ship in the dataset's `v0.2/raw/`; `build_question_bank.py --tracks terminal --terminal-source` builds the bank from them. |

Regenerated chess banks namespace their `question_id` by the generating seed
(`chess/s<seed>/chess_can_capture/e00`). Without that, `env_reset_id` restarts at 0 each run and
a new bank would silently reuse v0.1's ids while holding different questions — so any result keyed
by `question_id` would conflate the two. v0.1's ids are unchanged.

The build reports how much of the new bank's question text also appears in v0.1. Some overlap is
expected — short templated yes/no questions ("Can white legally capture the black knight on f6?")
collide by chance across position samples — but a high rate means weaker contamination protection,
and an identical bank fails the build outright.
