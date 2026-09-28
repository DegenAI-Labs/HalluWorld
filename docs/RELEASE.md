# Release contract

The release gate for Grid, Chess, InNav, and Terminal is one offline command:

```bash
pip install 'halluworld[chess]'
halluworld release check
```

It is stronger than `halluworld questions verify`. Verification checks the frozen bytes against
their manifest; the release command also imports the installed runtime and proves that those bytes
still mean what the benchmark code expects.

## Required checks

| Domain | Frozen unit | Runtime contract |
|---|---|---|
| Grid | 443 probe definitions across 33 levels | Every packaged level exists; seeded `make_probes` produces the same ordered probe classes; all 434 fixed questions and answers are identical. |
| Chess | 350 complete prompt/answer pairs across 7 probe types | Every probe type maps to the evaluator used by the battery, and every golden answer self-grades to 1.0. |
| InNav | 117 generated-probe templates plus 294 navigation traces | All 33 level/config pairs exist; seeded `make_canonical_probes` matches the bank; every trace can reconstruct every timestep selected by the released probing policy. |
| Terminal | 529 complete probes across 110 retained tasks | Every probe has frozen context and a schema; every golden answer self-grades; structured prefixes and partial lists are rejected. |

The command makes no provider requests and needs no API key. A failure is release-blocking: do not
publish a wheel or report new benchmark numbers until the bank/runtime mismatch is understood.

Run one domain while developing with repeated `--domain` options:

```bash
halluworld release check --domain grid
halluworld release check --domain chess --domain innav
halluworld release check --domain terminal
```

## InNav v0.1 trace compatibility

The v0.1 trajectory pack contains two historical record shapes. Eighty-four records contain their
complete action sequence. In 210 records, the saved dialogue snapshot omits only the final
assistant action. The InNav policy deliberately skips trajectory endpoints, and the checker proves
that every selected probe state precedes that omitted action and remains reconstructable. New
trajectory packs retain the source `action_sequence` directly; the compatibility extraction exists
only so the published v0.1 bytes and checksum do not change silently.

## Package boundary

The wheel contains the level/config files needed by these checks but **not** the question bank.
The bank is a gated Hugging Face dataset that `halluworld.data.questions_dir` downloads on first
use, or reads from `HALLUWORLD_QUESTIONS_DIR`; see [QUESTIONS.md](QUESTIONS.md). Raw terminal
probe-generation inputs, retained Terminal-Bench task fixtures, and alternate experimental banks
remain local or dataset artifacts; they are not evaluation runtime data and are not in the wheel.

## CI sequence

```bash
python -m pytest
python scripts/build_question_bank.py --version v0.1 --check \
    --terminal-source ../halluworld-hf/v0.1/raw/terminal --terminal-namespace "" \
    --chess-source ../halluworld-hf/v0.1/raw/chess --chess-source-name chess_20260505_fen-off
python -m build
python -m venv /tmp/halluworld-wheel-smoke
/tmp/halluworld-wheel-smoke/bin/pip install 'python-chess>=1.11' dist/halluworld-0.1.0-py3-none-any.whl
export HF_TOKEN=...   # an account that has accepted the dataset gate; or HALLUWORLD_QUESTIONS_DIR=...
/tmp/halluworld-wheel-smoke/bin/halluworld release check
/tmp/halluworld-wheel-smoke/bin/halluworld eval grid --provider openai --model gpt-4o-mini --out /tmp/grid --dry-run
/tmp/halluworld-wheel-smoke/bin/halluworld eval chess --provider openai --model gpt-4o-mini --out /tmp/chess --dry-run
/tmp/halluworld-wheel-smoke/bin/halluworld eval innav --provider openai --model gpt-4o-mini --out /tmp/innav --dry-run
/tmp/halluworld-wheel-smoke/bin/halluworld terminal eval --provider openai --model gpt-4o-mini --out /tmp/terminal --dry-run --limit 3
```

The last command must be run outside the source checkout so an omitted wheel file cannot be found
accidentally through the working tree.
