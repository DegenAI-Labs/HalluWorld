# The HalluWorld benchmark

The level taxonomy, what each tier isolates, and the HalluWorld-Hard subset.

Companion documents: [CHESS.md](CHESS.md) for the chess battery, [INNAV.md](INNAV.md) for
in-navigation probing, [TERMINAL.md](TERMINAL.md) for the terminal track, and
[LEVEL_DESIGNS.md](LEVEL_DESIGNS.md) for the full per-level specification.

> **Coverage note.** The chess battery has no U-tier (uncertainty) probe, while grid and terminal
> both do. Cross-track uncertainty comparisons therefore exclude chess. This is asserted by a test
> so that adding one later is a visible change.

---

## Part 1 — All Levels

### Tier P — Perception (static single-observation scene)

| ID | Name | Serializers | Probe focus |
|---|---|---|---|
| P1 | Dense Array | Symbolic, Grid | Counting accuracy under object density |
| P2 | Corridor Gauntlet | Symbolic, Grid | Egocentric spatial ordering along a narrow path |
| P3 | Rotation Challenge | Symbolic, Grid | Egocentric location + allocentric compass direction |
| P4a | Harder Array | Symbolic, Grid | Dense multi-type array; compound counting |
| P4b | Delta Perception | Symbolic, Grid | Change detection between two observations |
| P5 | Object Permanence | Symbolic, Grid | Out-of-FOV reasoning about a moved object |

### Tier M — Memory (multi-observation, sequential snapshots)

| ID | Name | Serializers | Probe focus |
|---|---|---|---|
| M1_3 | River Field (3 steps) | Memory | Object position via river physics, short horizon |
| M1_6 | River Field (6 steps) | Memory | Object position via river physics, mid horizon |
| M1_9 | River Field (9 steps) | Memory | Object position via river physics, long horizon |
| M2 | Witness Stand | Symbolic, Grid | Source attribution across 5 sequential observations |
| M3 | Incident Report | Symbolic, Grid | Change detection: color swap, door state, removal |
| M4 | Unreliable Narrator | Memory | Separate what agent sees vs. what the sign claims |

### Tier C — Causal (physical rules; dynamic interactions)

| ID | Name | Serializers | Probe focus |
|---|---|---|---|
| C1a | Persistent Chain | Memory | 3-step key-gate-goal chain; permanent trigger |
| C1a_nb | Persistent Chain (no board) | Memory | Same, but rule not stated — must be inferred |
| C1b | Continuous Chain | Memory | Same layout as C1a; gates only open while held |
| C1b_nb | Continuous Chain (no board) | Memory | Same, but rule not stated — must be inferred |
| C2 | Fire Crossing | Memory | Wet-ball extinguishes fire; dry ball does not |
| C3 | Flood Room | Memory | Step-precise flood schedule; timed path planning |
| C4 | Forking Paths | Memory | Multi-path + explicit red herring (no boat) |
| C5a | Adversarial Board | Memory | Notice board *lies* about the persistence rule |
| C6 | Flood-Fire Escape | Memory | Flood extinguishes fire; timed boulder action window |

### Tier U — Uncertainty (epistemic calibration)

| ID | Name | Serializers | Probe focus |
|---|---|---|---|
| U1 | Fog of War | Symbolic | Hub vs. sealed-room content; "cannot determine" |
| U2_low | Oracle (20% reliable) | Symbolic | Bayesian: low-reliability signpost claims |
| U2_mid | Oracle (50% reliable) | Symbolic | Bayesian: uncertain signpost |
| U2_high | Oracle (80% reliable) | Symbolic | Bayesian: high-reliability signpost |
| U4 | Amnesiac | Symbolic | Evidence-based vs. rule-based vs. unanswerable |

### Tier X — Cross-Tier Compound (multi-room tours)

| ID | Name | Serializers | Probe focus |
|---|---|---|---|
| X1 | Facility Tour (3 zones) | Symbolic | Baseline cross-zone intrusion |
| X2 | Facility Tour+ (5 zones) | Symbolic, Grid | Adds lying signpost about a prior zone |
| X3 | Facility Tour++ (7 zones) | Symbolic, Grid | Recency bias; early zones degrade |
| X4 | Facility Tour (Compound Witness) | Symbolic, Grid | Witness Stand sub-sequence embedded mid-tour |
| X5 | Cascading Testimony (7 zones) | Symbolic, Grid | Later signposts contradict earlier observations |
| X6 | Return Visit (5 zones + revisit) | Symbolic, Grid | Room modified on return; does model update? |
| X7 | Dragon Keep (8 zones + NPCs) | Symbolic, Grid | Full adversarial: unreliable NPCs, backtracking, narrative assimilation |

---

## Part 2 — Importance to Paper Narrative

**Hardest tier by model:** C > M > X > P > U (mean hallucination rates: 27.6%, 14.5%, 13.1%, 10.7%, 3.2%)

### Primary levels (core paper claims)

**C1a + C1b** are the most important levels in the benchmark. They share an *identical layout* and differ only in one bit: whether the pressure plate trigger is permanent or continuous. Hallucination on the two levels is nearly symmetric across models — the same models that fail C1a (continuous bias) tend to pass C1b, and vice versa. This is the clearest evidence of a **directional bias** in model world representations, not general causal confusion.

**C1a_nb / C1b_nb** (no-board variants) strip the hint away and force the model to infer the rule from observation. Harder by ~5–10pp — useful for measuring how much models rely on explicit verbal cues vs. mechanical reasoning.

**C5a** directly pits a lying notice board against annotation-derived mechanics. Paired with M4 (lying narrator), it lets us decompose *trust in text* vs. *trust in structural inference*.

**C6** is the compositional causality test: flood extinguishes fire, which opens a timed window for a boulder action. High failure rates (30–61%) even for strong models.

### Secondary levels (supporting claims)

- **P2 / P3** — grid-vs-symbolic serializer gap is largest here; drives the "format matters" finding
- **M1 (river)** — physics-based state tracking without direct observation; mid/late steps are HalluWorld-Hard
- **M2 / M3** — source attribution and change detection; M3 shows removal harder than addition universally
- **X5 / X6 / X7** — recency bias, update failures, narrative assimilation at scale
- **U1 / U2** — calibration; U is the easiest tier, which is itself a notable finding (models handle explicit uncertainty well)

### Headline numbers

| Claim | Stat |
|---|---|
| Hardest tier | C-tier: **27.6%** mean hallucination |
| Easiest tier | U-tier: **3.2%** — explicit uncertainty well-handled |
| Thinking hurts memory (sonnet) | base 16.6% → thinking **22.2%** (+5.6pp on Memory) |
| Thinking hurts memory (opus) | base 15.5% → thinking **18.8%** (+3.3pp on Memory) |
| Grid vs. symbolic gap (gpt-4o-mini) | **+17.4pp** — largest in field |
| Grid vs. symbolic gap (Claude) | **~+1.5pp** — nearly flat |
| C-tier count probe failure rate | **60.6%** — worst tier × probe type combo |
| Kimi-K2: overall vs. C-tier | 9.3% overall / **47.4% C-tier** — largest tier spike of any model |

---

## Part 3 — HalluWorld-Hard Subset

**Definition:** 12 (level, serializer) pairs where ≥5 models score ≥20% hallucination. All 15 complete models have data for all 12 pairs.

| Hard pair | Hallucination trap |
|---|---|
| C1a_noboard / Memory | No notice board — must infer persistence from mechanics |
| C1a_persistent_chain / Memory | Full causal chain with continuous-bias attractor |
| C1b_continuous_chain / Memory | Full causal chain with persistent-bias attractor |
| C1b_noboard / Memory | No board — must infer continuity from mechanics |
| C5a_adversarial_board / Memory | Notice board directly contradicts correct mechanic |
| C6_flood_fire_escape / Memory | Two-mechanic composition + lying notice board |
| M1_river_6 / Memory | River physics at step 6 — positions maximally ambiguous |
| M1_river_9 / Memory | River physics at step 9 — one object has dried |
| P1_dense_array / Grid | Dense object array under ASCII grid parsing |
| P2_corridor_gauntlet / Grid | Narrow corridor; egocentric spatial parsing under ASCII |
| P4_harder_array / Grid | Densest P-tier variant under ASCII grid parsing |
| X5_facility_tour / Grid | 7-zone tour with lying signposts under ASCII |

**Key pattern:** All 12 hard pairs are either MemorySerializer (causal chain levels) or GridSerializer (perception levels). No symbolic-serializer pair qualifies — confirming serialization format is a first-order difficulty driver independent of level content.

**Hard-subset hallucination rates (all 15 complete models, 12/12 pairs each):**

| Model | Hard halluc |
|---|---|
| moonshotai_Kimi-K2.6 | 53.9% |
| gpt-4o-mini | 39.7% |
| qwen-3-30b-instruct | 37.8% |
| deepseek-ai_DeepSeek-V3-0324 | 34.6% |
| gpt-5.4-mini | 34.1% |
| zai-org_GLM-5 | 33.5% |
| gpt-4o | 31.1% |
| o4-mini | 26.3% |
| o3-mini | 25.8% |
| o3 | 21.0% |
| claude-sonnet-4-6_thinking | 20.8% |
| claude-sonnet-4-6 | 20.0% |
| claude-opus-4-6 | 19.5% |
| claude-opus-4-6_thinking | 19.3% |
| gpt-5.5 | 18.6% |

---

## Part 4 — Navigation Trace Prioritization

> **Context:** Navigation traces are collected by running an LM through the gridworld step-by-step. Most models navigate erratically; gpt-5.4-mini is used because it produces the most realistic traces. Collection is expensive, so we want to prioritize which levels to cover first.

> **Note on X* series:** The X* (Facility Tour / cross-tier compound) levels predate door implementation, so navigation may not behave correctly in those environments. Stick to P* levels for now — they are self-contained, door-free, and well-suited for navigation trace collection.

### Recommended priority order

**Tier 1 — Run these first:**

- **P2 · Corridor Gauntlet** — Best starting point. The main failure mode we observe is misreading egocentric spatial offsets from the ASCII grid (lateral positions, forward ordering). A navigation trace that actually traverses the corridor provides grounded spatial context that static probing can't. Three probe types (location, order, between) all test directly navigation-relevant spatial skills.

- **P3 · Rotation Challenge** — Only level with allocentric (compass-direction) probes. Hardest probe type overall (28.3% hallucination). Run this if orientation-tracking is in scope.

**Tier 2 — Good coverage, lower urgency:**

- **P1 · Dense Array** — Simple open room; easy to navigate. Useful as a low-difficulty baseline to anchor the trace dataset.

- **P4a / P4b · Harder Array / Delta Perception** — Harder counting and change-detection variants. Navigation traces here test whether the model actively attends to the denser half of the grid.

- **P5 · Object Permanence** — Object moves out of FOV mid-episode. Navigation trace can show whether the model seeks out the object vs. ignores its prior position.

**Not recommended (for now):**

- **X* series** — No door support yet; navigation behavior is unreliable
- **M / C / U tiers** — Failure modes are memory/causal/epistemic reasoning; navigation traces are not the right tool to probe those
