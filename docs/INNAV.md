# In-Navigation (InNav)

Probing an agent about the world *while it is acting in that world*, against a paired control that
sees the same states without the navigation context.

## The design

1. A navigation model drives an agent through a gridworld until a sufficiency validator fires
   (enough steps taken, agent stable, near a target location).
2. The trajectory is recorded.
3. At selected timesteps the trajectory is replayed and the probe model is asked about world state
   **twice**:
   - **InNav** — as a follow-up turn inside the ongoing navigation dialogue.
   - **CtrlStatic** — the same state, the same question, no navigation context.

Both arms receive the *same* generated question: `engine.py` calls `probe.generate(env)` once and
asks the result twice. That is what makes the contrast a controlled paired comparison rather than
two separate evaluations.

The paper isolates two competing effects with it: **epistemic grounding** (acting may anchor the
world model) versus **cognitive load** (navigating may degrade state tracking). A single
high-performance navigation model generates trajectories for all probe models, so navigation ability
is not confounded with reasoning ability.

## Why there is no fixed question list

InNav is the one track whose questions cannot be frozen as text, and the reason is structural.

The static gridworld benchmark uses `FixedProbe` on 30 of its 33 levels. Those questions hardcode
egocentric spatial references — *"What color is the key that is 11 steps ahead and 3 steps to your
right?"* — which presuppose a known agent position. The static benchmark can assume that: fixed
start, fixed observation.

InNav cannot. The agent navigates, its path depends on the model, and probes are asked against
whatever state it actually reached. "11 steps ahead" refers to nothing in particular once the agent
is somewhere else. So InNav generates probes from the reached state, and
`make_canonical_probes` matches the static set on exactly the three levels where the static
benchmark *itself* uses generated probes (`P1_dense_array`, `P2_corridor_gauntlet`,
`P3_rotation_challenge`). That equivalence is asserted by
`tests/test_question_bank.py::test_innav_matches_static_only_where_static_is_also_generated`.

**What is frozen instead is the trajectory.** 294 traces across 58 configs and 32 worlds ship as
`trajectories.jsonl.gz` in the v0.1 bank, which is what makes an InNav run replayable
and lets others fork a trajectory or add probes to an existing one.

The v0.1 pack combines two historical trace formats. In 210 dialogue-shaped records the saved
snapshot omitted the final assistant action. The probing policy skips endpoints, so every state
that was actually selected for a probe remains reconstructable; `halluworld release check --domain
innav` asserts this for all 294 records. New packs preserve `action_sequence` directly. See
[RELEASE.md](RELEASE.md) for the exact compatibility contract.

## ⚠️ Comparability limit

**InNav numbers are not comparable to published static gridworld numbers.** On 30 of 33 levels the
two were measured on different questions, for the reason above.

The comparison the paper makes — InNav versus CtrlStatic on identical trajectories — is unaffected
and sound, because both arms share a question. Do not put InNav results in a table beside static
gridworld results and read across.

## Running it

Level configs live at `halluworld/data/levels/*.innav.json` and carry the static probe location,
navigation targets, and sufficiency parameters. Run the standalone paired evaluation with:

```bash
halluworld eval innav \
  --provider openai \
  --model gpt-4o-mini \
  --out results/innav
```

Alternatively, add `--include-innav` to `halluworld eval grid`. That opt-in runs the P1/P2/P3
parity subset and writes it to the Grid output directory's `innav/` child. It is off by default.
The underlying runner is `halluworld/tracks/innav/runner.py`; the engine is `engine.py`.
