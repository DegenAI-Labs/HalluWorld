# Reproducing HalluWorld

What is reproducible, what is not, and why.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[chess,dev]'
pytest -q                          # offline; no API keys needed, none spent
```

The test suite is offline by design: an autouse fixture in `tests/conftest.py` strips every provider
key unless a test is explicitly marked `live_api`, so an accidental network test fails with a
missing-key error rather than a bill.

## Running

```bash
halluworld run --config halluworld/data/configs/chess/fen_off.yaml
```

Configuration is YAML, validated on load. Unknown keys are an **error** — `SAN_MIN_PLYS` was
silently ignored for as long as configuration was environment variables, because a variable nobody
set just returns its default. API keys cannot appear in a config file at all.

Every run writes `manifest.json`: git SHA and whether the tree was dirty, package versions, the
fully resolved config and its SHA-256, the question-bank version, and the model snapshot the API
returned.

## What is genuinely reproducible

- **The question bank.** Frozen, checksummed, byte-reproducible to rebuild. Every model sees
  identical items.
- **Seeding.** Every stochastic component is explicitly seeded, and per-episode seeds derive from
  one master RNG recorded in each result row.
- **Grid probe extraction.** A golden test asserts the frozen bank is identical to what the probe
  definitions produce.
- **Result migration.** Aggregate-preserving to 1e-9.

## What is not, and why

**Model endpoints drift.** A lockfile pins libraries; nothing pins a model. `gpt-5.5` today is not
necessarily `gpt-5.5` in six months. The manifest's recorded snapshot (`models[].reported` — what
the API returned, not the alias requested) is the only meaningful model-side pin, which is why it is
recorded.

**Reasoning models are not deterministic.** Temperature being forced to `None` for `o*`/`gpt-5*`
does not make them so; backend nondeterminism and reasoning-path variation cause run-to-run drift.
Expect to average over replicates.

**Episode index is not a pure function of the master seed.** A probe that cannot generate for a
position — a game-over chess position, an unreachable object — consumes a seed without emitting an
episode. So `episode 19` in one run and `episode 19` in another need not share a seed. Seeds
themselves reproduce exactly; the *mapping from index to seed* also depends on how many resets were
skipped, which depends on env and probe logic. Every result row records its actual `seed`; use that,
not the index.

**Historical model outputs are not part of this release.** The v0.1 questions are, and they are
the paper's items, so a new run is directly comparable. See [RESULTS.md](RESULTS.md#what-is-published).

## Cost

The bank is 1,439 items. A full four-track sweep of one reasoning model is on the order of a few
thousand API calls; the terminal track's contexts are large (a 60 KB tmux capture is typical), so
budget input tokens accordingly. Use `--dry-run` and `provider: stub` to validate a config before
spending anything.
