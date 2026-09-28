# Evaluating a model

HalluWorld exposes Grid, Chess, InNav, and Terminal through one installed CLI. Install the Chess extra if
you will run or release-check Chess:

```bash
pip install 'halluworld[chess]'
export OPENAI_API_KEY=...
```

Use `ANTHROPIC_API_KEY` or `BASETEN_API_KEY` with the corresponding provider. Provider selection is
explicit and does not depend on the spelling of the model name.

The question bank is not in the package. It is a gated Hugging Face dataset: request access at
[DegenAI-Labs/HalluWorld](https://huggingface.co/datasets/DegenAI-Labs/HalluWorld),
run `hf auth login` once (or export `HF_TOKEN`), and the first command below downloads it into the
Hugging Face cache. On a machine without network access, point `HALLUWORLD_QUESTIONS_DIR` at a
local copy instead. See [QUESTIONS.md](QUESTIONS.md).

## Target commands

```bash
halluworld eval grid \
  --provider openai \
  --model gpt-4o-mini \
  --out results/grid

halluworld eval chess \
  --provider openai \
  --model gpt-4o-mini \
  --out results/chess

halluworld eval innav \
  --provider openai \
  --model gpt-4o-mini \
  --out results/innav

halluworld eval terminal \
  --provider openai \
  --model gpt-4o-mini \
  --out results/terminal
```

These commands make network requests and can incur provider charges. Append `--dry-run` first to
check provider, model, domain selection, episode or probe count, and output path without reading an
API key or calling the network. Grid, Chess, and InNav do not create output in dry-run mode;
Terminal writes canonical excluded records (`error="dry_run"`) so its complete output wiring can be
validated offline.

Use `--limit N` as a safety cap on episodes per selected level. It applies after `--episodes`, so
`--episodes 50 --limit 2` runs two episodes per level. An episode can contain multiple provider
requests because a level may contain several probes; the flag intentionally limits episodes, not
HTTP requests.

Terminal has frozen probes rather than episodes, so its `--limit N` selects the first N probes after
task, question-ID, and probe-type filters. This makes `--limit 1` one logical provider request;
rate-limit retries may add attempts unless `--retries 1` is also set.

A minimal live Grid smoke test is:

```bash
halluworld eval grid \
  --provider openai \
  --model gpt-4o-mini \
  --out results/smoke \
  --levels P1_dense_array \
  --limit 1
```

## Grid with the paired InNav subset

Grid runs only its static benchmark by default. To follow it with the paired InNav parity subset:

```bash
halluworld eval grid \
  --provider openai \
  --model gpt-4o-mini \
  --out results/grid \
  --include-innav
```

The opt-in subset contains `P1_dense_array`, `P2_corridor_gauntlet`, and
`P3_rotation_challenge`: the three levels where both Grid and InNav generate canonical probes.
Override it with `--innav-levels`, and control its sample count separately with
`--innav-episodes`. Its results go under `results/grid/innav/`; the static Grid result is unchanged.

## Selection and outputs

Use `--levels LEVEL [LEVEL ...]` to select frozen level or probe identifiers. An unknown identifier
fails before any provider request. Without it, the command runs the domain's full released
selection. The main artifacts are:

| Command | Primary output |
|---|---|
| `eval grid` | `results_<model>.csv` and `manifest.json` |
| `eval chess` | `results.jsonl` and `manifest.json` |
| `eval innav` | `results.csv`, per-level CSV files, traces, and `manifest.json` |
| `eval terminal` | canonical `results.jsonl`, `summary.json`, and `manifest.json` |

Useful controls include `--episodes`, `--limit`, `--seed`, and `--max-tokens`. Run
`halluworld eval <domain> --help` for domain-specific controls such as serializer, FEN mode,
reasoning effort, trace reuse, or a separate navigation model.

The manifest records the requested provider/model, resolved selection and configuration, package
versions, config digest, and result counts. Preserve the provider's response metadata alongside the
run if an endpoint exposes a dated model snapshot; the CLI cannot guarantee that every provider
returns one.
