# Results

## Schema

One row is **one probe trial**: one question, asked of one model, about one environment state.
Episodes, levels, and runs are aggregation keys over that. 30 canonical columns, defined in
`halluworld/results.py`.

Three things worth knowing before you aggregate:

**`score=None` means excluded; `score=0.0` means wrong.** An API failure or a probe that could not
generate is not a wrong answer. `summarize()` keeps excluded trials out of the mean and reports them
separately, so an outage lowers *n* rather than the score.

The `error` field distinguishes `rate_limit`, `bad_request`, `empty_response`, `api_error`, and
other exclusions. Terminal dry-runs deliberately emit `error="dry_run"` records to exercise the
canonical writer without counting those records as scored trials.

**InNav trials are two rows**, one per arm, sharing a `pair_id`. The paired contrast is a self-join:

```python
df.pivot_table(index="pair_id", columns="variant", values="score")
```

**Track-specific fields live in the JSON `extra` column.** The union across tracks is 27+ fields
while any row populates three to six; `pd.json_normalize` recovers the wide view. Nothing you group
by lives in `extra`.

JSONL is canonical and CSV is a regenerable view — questions and responses contain embedded
newlines, and CSV round-tripping those is what produced the corrupted files this project previously
had to recover from.

## Migrating legacy results

Eight incompatible per-probe CSV layouts predate this schema.

```bash
python3 scripts/migrate_results.py --in results/legacy --out results/migrated/v1
python3 scripts/migrate_results.py --in results/legacy --check   # verify aggregates preserved
```

`--check` compares every source file's mean score against the mean of its migrated records and fails
on any drift beyond 1e-9. Adapters key on `frozenset(header)`, so column order cannot matter, and an
**unrecognized header raises** rather than skipping — a silently dropped file is a silently missing
number.

Fields that cannot be recovered are left empty with `parse_note="legacy_no_question"`. The dynamics
and inventory formats never stored question text or an item-identifying ground truth; reconstructing
them would make rows look complete while being unverifiable.

## What is published

The **question banks and trajectories** are the gated Hugging Face dataset
[`DegenAI-Labs/HalluWorld`](https://huggingface.co/datasets/DegenAI-Labs/HalluWorld); see
[QUESTIONS.md](QUESTIONS.md). The v0.1 bank is exactly the set of items behind the paper's numbers.

The historical per-response model outputs behind the paper's tables are **not** part of this
release. They were recovered for Grid, Chess and Terminal and can be released on request. InNav's
historical outputs were never recovered.
