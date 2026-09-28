# Provenance and licensing boundaries

This document records where the released HalluWorld benchmark material came
from. It complements `LICENSE`; it does not replace the license text for any
component.

## HalluWorld material

The HalluWorld authors created the benchmark questions, golden answers,
cognitive-tier labels, curation decisions, level definitions, and released
navigation trajectories. That material, and HalluWorld source code outside the
vendored Terminal-Bench tree, is licensed CC BY 4.0 as described in `LICENSE`
unless a file says otherwise. The question bank itself is distributed as a
gated Hugging Face dataset rather than from this repository; see
`docs/QUESTIONS.md`.

The terminal authoring snapshot at `halluworld/data/terminal_tasks/` contains
50 task packages. Forty-seven also appear in Terminal-Bench and retain the
upstream task metadata naming Michael Yu as author. Three do not appear in the
vendored Terminal-Bench task set and were authored by the HalluWorld Terminal
Team:

- `deep-scavenger-hunt`
- `string-scavenger-hunt`
- `tensor-metadata`

These 50 authoring packages are not the source of the frozen terminal question
bank and are not referenced by its 529 records.

## Terminal-Bench material

`external/terminal-bench/` is a vendored and modified Terminal-Bench v0.2.18
distribution. Its own code and task distribution are provided under Apache
License 2.0; the full text and upstream notice are retained in that directory.
The pre-release repository was squashed, so an exact source commit cannot be
recovered from its Git history. A [public archival repository][tb-archive]
named `laude-institute__terminal-bench.6f1292df` preserves a nearby historical
snapshot, but the vendored tree also contains later upstream and HalluWorld
changes and is not represented as byte-identical to that snapshot.

The released terminal bank has 529 records spanning exactly 110 Terminal-Bench
tasks. Only those task directories are redistributed here. Their canonical,
sorted allowlist is `external/terminal-bench/USED_TASKS.txt`; it is generated
from the distinct `task_or_level` values in the v0.1 terminal bank
(`terminal.jsonl.gz`).

The HalluWorld runtime-probe changes inside the vendored package are marked in
their source headers. They consist of the runtime-probe package plus its
integration into `terminal_bench/terminal/tmux_session.py` and are distributed
under Apache-2.0 with the surrounding fork.

## Captured terminal contexts

Terminal questions were generated from agent trajectories inside the retained
Terminal-Bench task environments. Their contexts can contain filenames,
commands, terminal output, and excerpts emitted from task fixtures or tools.
The HalluWorld CC BY 4.0 license covers HalluWorld's questions, golden answers,
labels, and selection/curation. It does not relicense third-party expression
that may appear inside a captured terminal context; such material retains its
applicable original terms.

Task containers can also fetch or transform additional software at build or
run time. Those dependencies are not vendored merely because a Dockerfile
mentions them, and users remain responsible for their upstream terms. One
retained package, `make-doom-for-mips`, includes a patch to GPL-2.0-licensed
DoomGeneric and fetches the source at a pinned commit. Its GPL-2.0 license text
is included alongside the task as `LICENSE.doomgeneric`.

## Other tracks

- Gridworld and InNav build on MiniGrid and Gymnasium (Apache-2.0).
- Chess uses the optional python-chess dependency (GPL-3.0-or-later) at run
  time. It is not bundled into HalluWorld's MIT-licensed source distribution.
- Lichess puzzle positions, when enabled, come from the CC0 Lichess puzzle
  database. The exact dump used for the paper remains a reproducibility record
  to recover, as noted in `LICENSE`.

[tb-archive]: https://github.com/behavioral-data/laude-institute__terminal-bench.6f1292df
