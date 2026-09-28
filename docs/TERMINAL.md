# HalluWorld-Terminal

Probes injected into a Docker + tmux agent session while the agent works, with ground truth computed
by running out-of-band commands inside the live container.

See [PROBE_METHODS.md](PROBE_METHODS.md) for the methodology and
[PROBE_RESULTS.md](PROBE_RESULTS.md) for the 11-model leaderboard.

## Evaluate the released bank

The installed wheel contains all 529 released probe records, referencing 110 retained
Terminal-Bench tasks. The task fixtures themselves remain source-checkout-only. Evaluation needs an
LM provider but does not need Docker, tmux, the raw probe directory, or Terminal-Bench:

```bash
export OPENAI_API_KEY=...
halluworld terminal eval \
  --provider openai \
  --model gpt-4o-mini \
  --out results/terminal/gpt-4o-mini
```

`halluworld eval terminal` is an equivalent spelling. Anthropic and Baseten use
`ANTHROPIC_API_KEY` and `BASETEN_API_KEY`. Provider selection is explicit.

Terminal contexts average roughly 50,000 characters, so a full 529-probe run can be expensive.
Inspect three records without a provider call:

```bash
halluworld terminal eval \
  --provider openai \
  --model gpt-4o-mini \
  --out /tmp/halluworld-terminal-dry \
  --dry-run \
  --limit 3
```

Unlike the episode cap used by Grid, Chess, and InNav, Terminal's `--limit` counts frozen probes:
one selected probe is one logical provider query. Filter first with `--tasks`, `--question-ids`, or
`--probe-types`; use `--resume` to retain scored records and retry excluded API failures.

The canonical artifact is `results.jsonl`, with one `halluworld.results.ProbeRecord` per attempted
probe. `score=None` means the trial was excluded (`rate_limit`, `bad_request`, `empty_response`, or
another API error), never silently scored wrong. `summary.json` excludes those records from its
accuracy, and `manifest.json` records the frozen-bank version and resolved run configuration.

The production grader is conservative: exact normalized equality is accepted; structured answers
must include every declared key with its complete value. Commas, pipes, semicolons, arrows, equals
signs, paths, spaces, and angle brackets are preserved rather than truncated.

## Advanced: generate trajectories and probes

The Docker/tmux machinery is a source-checkout-only authoring workflow. It lives behind the
`terminal-gen` extra (including its Python 3.12 requirement) and is intentionally absent from the
wheel:

```bash
pip install -e '.[terminal-gen]'
python3 halluworld/tracks/terminal/run_tb_task.py \
  acl-permissions-inheritance \
  --n-concurrent 1
```

---

The raw released probes are not in the public source tree. They ship as `v0.1/raw/terminal/` in
the gated Hugging Face dataset next to the frozen bank, and live under
`halluworld/data/probe_banks/terminal_20260504/` (gitignored) in a working checkout; the bank
builder reads either via `--terminal-source`. They are inputs to bank construction, not the
supported evaluation interface. The older
scripts under `scripts/terminal/tools/evaluate*_probe*.py` are retained for historical artifact
compatibility; new runs should use the installed CLI above.

### Extract a newly generated bank

```bash
python3 scripts/terminal/tools/extract_run_probes.py \
  --run-dir runs/all-tasks__20260504_002958 \
  --out-dir runs/all-tasks__20260504_002958/extracted_probes \
  --llm-out-dir halluworld/data/probe_banks/terminal_20260504
```

### Audit newly generated probes

```bash
python3 scripts/terminal/tools/audit_probe_eval_failures.py \
  --probe-dir runs/all-tasks__20260504_002958/extracted_probes \
  --result-dir runs/all-tasks__20260504_002958/gpt-5.4_probe_eval \
  --export-probe-dir runs/all-tasks__20260504_002958/likely_genuine_model_error_probes \
  --overwrite-export
```

## How the pieces fit together

The terminal track wraps a patched Terminal-Bench checkout in `external/terminal-bench` and adds a probe pipeline around the normal task runs:

- `external/terminal-bench/USED_TASKS.txt` lists the 110 retained upstream tasks referenced by the frozen bank; unused upstream task packages are not redistributed.
- `run_tb_task.py` launches Terminal-Bench, defaults runtime probes on, then post-processes each trial into a `trajectory.json`.
- `external/terminal-bench/terminal_bench/probes/runtime_injectors.py` generates runtime probes while the agent is acting in the terminal.
- `scripts/terminal/tools/extract_run_probes.py` turns runtime probes into standalone JSON examples.
- `scripts/terminal/tools/audit_probe_eval_failures.py` filters generation artifacts for review.
- `halluworld terminal eval` evaluates the released normalized bank.
- `runs/` contains generated run artifacts and can get large.

Runtime probes are enabled by default in `run_tb_task.py` via `TB_RUNTIME_PROBES=1`. The patched `TmuxSession` observes commands in the agent session, captures pre/post pane context, asks `RuntimeProbeManager` for candidate probes, and writes JSON files under each trial's `runtime_probes/agent/` directory. Post-processing attaches those probes to the matching steps in `trajectory.json`.

The 50 task packages under `halluworld/data/terminal_tasks/` are a separate
authoring snapshot and are not referenced by the released 529-probe bank. Of
those, 47 also appear in the retained Terminal-Bench distribution and preserve
the upstream author credit to Michael Yu. The three HalluWorld-only packages
(`deep-scavenger-hunt`, `string-scavenger-hunt`, and `tensor-metadata`) are
credited to the HalluWorld Terminal Team. See [PROVENANCE.md](PROVENANCE.md).

## Adding runtime probe injectors

Add new runtime injectors in `external/terminal-bench/terminal_bench/probes/runtime_injectors.py`.

The normal pattern is:

1. Decide whether the probe is active/perceptual or passive/history-based.
   - Add command-state probes to `_active_perceptual_probes()`.
   - Add prior-context, memory, uncertainty, or causal probes to `_passive_memory_probes()`.
2. Add a small helper method that returns one probe dictionary. `_generate_probes()` fills in shared metadata such as `probe_id`, `session`, `trigger_step_index`, `trigger_command`, cwd, context paths, `defer_model_answer`, `q`, and `a`.
3. Include these required fields in the returned dictionary:
   - `injector`: stable slug used in filenames and analysis.
   - `probe_type`: category such as `perceptual`, `memory`, `uncertainty`, `causal`, or `cross-tier compound`.
   - `context_scope`: usually `through_trigger` or `last_10_commands`.
   - `question`: the model-facing probe question.
   - `answer_schema`: exact answer format the probe answerer should use.
   - `golden`: the expected answer in that schema.
   - `golden_source`: where the answer came from, such as `container_command`, `trajectory_text`, or `command_heuristic`.
   - `golden_command_or_heuristic`: a short explanation or command for traceability.
4. Use `CommandRecord` fields where possible: `normalized_command`, `cwd_before`, `cwd_after`, `observation`, `program`, `ports`, and `urls`.
5. If the golden answer depends on live container state, use `_container_shell(...)` and quote user-controlled command parts with `shlex.quote(...)`. Pass `cwd=record.cwd_before` when path interpretation should match the agent's working directory.
6. Keep questions narrow and answer schemas machine-checkable. Prefer `key=value` answers over free text.
7. Respect `TB_RUNTIME_PROBE_MAX_PER_COMMAND`; `_generate_probes()` truncates candidate probes per command, so put the most valuable probes earlier in the candidate list.

Example skeleton:

```python
def _active_perceptual_probes(self, record: CommandRecord) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    # Existing candidates...
    if should_trigger_my_probe(record.normalized_command):
        probes.append(self._my_new_probe(record))
    return probes

def _my_new_probe(self, record: CommandRecord) -> dict[str, Any]:
    exit_code, stdout, _ = self._container_shell(
        "some safe shell command",
        cwd=record.cwd_before,
    )
    answer = "my_key=yes" if exit_code == 0 else "my_key=no"
    return {
        "injector": "my_new_probe",
        "probe_type": "perceptual",
        "context_scope": "through_trigger",
        "question": "Ask one precise question. Answer as `my_key=<yes_or_no>`.",
        "answer_schema": "my_key=<yes_or_no>",
        "golden": answer,
        "golden_source": "container_command",
        "golden_command_or_heuristic": "some safe shell command",
    }
```

After adding an injector, do at least a syntax check:

```bash
python3 -m py_compile external/terminal-bench/terminal_bench/probes/runtime_injectors.py
```

For an end-to-end smoke test, run one small task and inspect the generated probe JSON:

```bash
python3 halluworld/tracks/terminal/run_tb_task.py acl-permissions-inheritance --n-concurrent 1
find runs -path '*/runtime_probes/agent/*.json' | tail
```

## Onboarding notes

Before running tasks, make sure Docker is running and `OPENAI_API_KEY` is available in the environment. `run_tb_task.py` looks for the Terminal-Bench CLI at `external/terminal-bench/.venv/bin/tb`, then on `PATH`, and finally honors `TB_CMD` if you need to point at a specific command.

Useful environment variables:

- `TB_CMD`: override the Terminal-Bench command used by `run_tb_task.py`.
- `TB_RUNTIME_PROBES`: set to `0`/unset to disable runtime probe generation outside this wrapper.
- `TB_RUNTIME_PROBE_MAX_PER_COMMAND`: cap generated runtime probes per observed command; default is `6`.
- `TB_RUNTIME_PROBE_LLM_MAX_OUTPUT_TOKENS`: cap LLM probe-generation output; default is `32000`.
- `TB_RUNTIME_PROBE_LLM_TIMEOUT_SEC`: per-attempt LLM probe-generation timeout; default is `900`.
- `TB_RUNTIME_PROBE_LLM_RETRIES`: retry count for LLM probe generation; default is `5`.
- `TB_LEGACY_HEURISTIC_PROBES`: set to `1` to re-enable old command-level `pip`/`python`/`cat` heuristic probes in trajectories.
- `TB_ANSWER_RUNTIME_PROBES`: set to `1` to ask the run model to answer deferred runtime probes during post-processing.
- `--eval-probes`: opt in to the old post-processing LLM probe evaluator. It is skipped by default.

Generated artifacts to know:

- `commands.txt`: raw observed tmux commands for a trial.
- `command_contexts/agent/*.txt`: pane snapshots before commands.
- `probe_contexts/agent/*.txt`: pane snapshots after commands used by runtime probes.
- `runtime_probes/agent/*.json`: raw runtime probe records.
- `trajectory.json`: normalized run record with steps, probes, and artifact paths.
- `probe_convs/`: prompts/answers used when deferred probes are answered by a model.
- `probes/`: judge records created by the built-in trajectory probe evaluator.

When debugging a bad probe, start from the extracted JSON, follow `metadata.runtime_probe_path` back to the raw runtime probe, then inspect `metadata.selected_context_path` or the trial's `trajectory.json` step for the command and context that produced it.
