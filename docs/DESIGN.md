# HalluWorld — Design Notes

## Core idea

Hallucination = **inaccurate world modeling, observable to the user** (Liu et al., 2025).

Two axes of variation:
- **Reference world model** — what information the LM receives (`Serializer` + `Probe`)
- **Conflict policy** — what counts as wrong (`Evaluator`)

By independently swapping each axis we can isolate *which component* of world modeling fails — something most NLP benchmarks cannot do because ground truth, presentation, and scoring are all entangled.

---

## Code structure

```
halluworld/
├── envs/               # MiniGrid environment definitions
│   └── simple_env.py   # 8×8 room with key, locked door, ball, goal
│
├── serializers/        # env state → LM input string
│   ├── base.py         # Serializer ABC  →  serialize(env) -> str
│   └── symbolic.py     # SymbolicSerializer: agent FOV as labeled text grid
│                       # also exports visible_objects(env) used by all probes
│
├── probes/             # (question, ground_truth) grounded in live env state
│   ├── base.py         # Probe ABC  →  generate(env) -> ProbeResult
│   └── visibility.py   # PresenceProbe, LocationProbe
│
├── lm/                 # language model interface
│   ├── base.py         # LM ABC  →  query(system, user) -> LMResponse
│   ├── stub.py         # StubLM  — fixed / random / callable, for dry runs
│   └── openai_lm.py    # OpenAILM — swap in for real eval
│
├── evaluators/         # LMResponse × ProbeResult → EvalResult
│   ├── base.py         # Evaluator ABC  →  evaluate(response, probe) -> EvalResult
│   └── exact_match.py  # PresenceEvaluator (yes/no), LocationEvaluator (partial credit)
│
└── benchmark.py        # run_benchmark(...) → BenchmarkRun(.accuracy, .summary())
```

**Invariant:** `Probe.generate(env)` returns `(question, ground_truth)` derived from the *same* env state snapshot — they are always consistent by construction.

**To add a new probe type:** subclass `Probe` in `probes/`, add a companion `Evaluator` in `evaluators/`. Nothing else changes.

---

## Implemented probes

| Probe | Question | Ground truth type |
|---|---|---|
| `PresenceProbe` | Is there a [color] [object] in your view? | `bool` |
| `LocationProbe` | Where is the [color] [object] relative to you? | `{steps_ahead, lateral}` |

---

## Future probe types

### Perceptual
- **`CountProbe`** — "How many objects are in your view?" Tests quantity hallucination
- **`AttributeProbe`** — "What color is the door?" / "Is it open or locked?" Tests property binding (right object, wrong attribute — known VLM failure mode)
- **`BoundaryProbe`** — "Is there a wall directly to your left?" Tests spatial boundary awareness vs. room-structure priors

### Memory (multi-step)
- **`MemoryProbe`** — observe state S₁, take steps that move object out of FOV, ask about S₁. Tests retained world model vs. fresh perception
- **`ChangeProbe`** — "Has anything changed since your last observation?" after taking an action. Tests update fidelity

### Causal / Action-conditional
- **`OutcomeProbe`** — "If you move forward, what will you be adjacent to?" Tests forward simulation from current observation
- **`ActionProbe`** — "What should you do to pick up the red key?" Tests planning grounded in visible layout
- **`CounterfactualProbe`** — "Could you open this door right now?" Depends on whether agent holds the key. Tests precondition understanding

### Belief / Uncertainty
- **`ConfidenceProbe`** — ask about something outside FOV (always absent, LM can't know). Does it say "I don't know" or hallucinate? Tests uncertainty calibration
- **`OcclusionProbe`** — ask about objects visible 2 steps ago but now occluded (`see_through_walls=False`). Pure memory hallucination

---

## Experimental design ideas

| Variable | How to vary it | Hypothesis |
|---|---|---|
| FOV size | `agent_view_size` 7→5→3 | Smaller context → higher hallucination rate |
| Object density | More objects placed in `_gen_grid` | Working-memory limit: error rate jumps above N objects |
| Foil plausibility | Contextually consistent vs. implausible foils | LMs hallucinate *likely* objects more (prior leakage) |
| Probe difficulty | Presence → Location → Attribute → Causal → Memory | Maps model capability ceiling |
| Serializer format | Symbolic grid vs. NL description | Presentation changes rate independent of content |
| Context length | Last K observations prepended | More context helps (memory) or hurts (overwriting)? |
| Model size | GPT-4o-mini vs. GPT-4o vs. larger | Does hallucination scale smoothly or show thresholds? |

---

## Key claim

> We independently vary what the LM is shown (**serializer**), what it is asked (**probe**), and what counts as wrong (**evaluator**), with ground truth held constant. This isolates *which component* of world modeling fails — perceptual, spatial, causal, or memory.
