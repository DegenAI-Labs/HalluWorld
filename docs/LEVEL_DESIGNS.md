# Level designs

Tile types, objects, and the specification of every gridworld level.
(The superseded v1 spec was removed; this is the current one.)

# HalluWorld — Level Designs

All levels use the MiniGrid framework. Each level is a hand-crafted env class
subclassing `MiniGridEnv`, placed in `envs/levels/`. Grid coordinates: `(row, col)`,
origin top-left. Agent direction: 0=east, 1=south, 2=west, 3=north.

**Level inventory:**
- Perceptual: P1 Dense Array, P2 Corridor Gauntlet, P3 Rotation Challenge, P4 Delta Perception
- Memory: M1 River Field, M2 Witness Stand, M3 Incident Report, M4 Unreliable Narrator, M5 Time Capsule
- Causal: C1a Persistent Chain, C1b Continuous Chain, C2 Fire Crossing, C3 Flood Room, C4 Forking Paths
- Uncertainty: U1 Fog of War, U2 Oracle Problem, U4 The Amnesiac

---

## New Tile & Object Types

Implement in `envs/tiles.py` before building any levels.

### Terrain Tiles
- **`RiverTile(direction, speed=1)`** — any object on this tile moves `speed` cols/rows
  per step in `direction`. Object gains `wet_turns_remaining = W` (default W=4) on entry;
  decrements each step; resets to W if object re-enters river before drying.
- **`FireTile`** — impassable while active. Extinguished (becomes `FloorTile`) when a
  wet object (`wet_turns_remaining > 0`) is used on it by the agent.
- **`WaterTile`** — impassable without a boat item. Agent can see across it.
- **`FloodTile(rise_rate=1)`** — starts dry; becomes impassable after `rise_step` steps.
  Each tile has its own `rise_step` set at level construction to encode the flood front.
- **`TreeTile`** — impassable, blocks FOV behind it. Freestanding, not a wall.
- **`DarkZone`** — passable but blocks FOV for any agent not standing on it.
- **`MudTile`** — passable, costs 2 steps to traverse.
- **`PressurePlate(target_id, effect)`** — floor tile; while an object sits on it,
  applies `effect` to the tile/door with matching `target_id`. Effect is either
  `"open"` (persistent: plate holds door open while weighted) or `"trigger"` (one-shot:
  fires once when first stepped on, regardless of whether plate remains weighted).

### Object States
Objects gain a `condition` attribute: `dry` (default) | `wet` | `soaked`
- `wet`: `wet_turns_remaining` in [1, W]
- `soaked`: object was in river for ≥ 3 consecutive steps (W+2 turns)
- Serializer always reports condition if not dry:
  `"red ball [WET, dries in 2 steps]: 4 ahead, 1 left"`

### Structural Objects
- **`NoticeBoardObject(text)`** — always renders its text in serializer when in FOV.
  Represents stale written information (t=0 snapshot).
- **`SignpostObject(text, accurate=True)`** — same rendering; `accurate` flag controls
  whether text is true. Uncertainty levels set `accurate=False` on some signs.
- **`Boulder`** — heavy object; cannot be picked up, can be pushed one tile per step
  onto adjacent floor/plate tiles. Stays where pushed (does not slide).
- **`Boat`** — pickupable item; while held, agent can traverse WaterTile.

---

## Perceptual Levels

### P1 · Dense Array

**Purpose:** Presence and attribute accuracy in a regular high-density scene. Tests
pattern-completion failure: model reports the regular pattern rather than the actual
specific violations.

**Grid size:** 20×16

**Object placement:**
```
Rows 1–3, cols 1–9: 5×3 grid of red keys EXCEPT:
  (row=2, col=8): blue key          ← color violation
  (row=3, col=4): empty             ← absence violation

Rows 5–7, cols 1–9: 5×3 grid of blue balls EXCEPT:
  (row=6, col=2): red ball          ← color violation
  (row=7, col=9): green ball        ← color violation (edge)

Rows 9–10, cols 2–8: yellow doors all state=closed EXCEPT:
  (row=9,  col=7): state=open       ← state violation
  (row=10, col=5): state=locked     ← state violation

cols 0–1 and 9–10 of door rows: DarkZone tiles
TreeTiles at: (12,2), (12,8), (13,2), (13,8)
```

**Agent spawn:** (row=13, col=5), direction=north, view_size=7

**Primary probes:** PresenceProbe, CountProbe, AttributeProbe (color, door state)

**Hallucination traps:**
- Pattern completion: model reports all keys as red, misses the one blue key
- Absence in regular grid: model reports object at the empty cell
- Closed/locked conflation: model collapses two distinct door states to one label

---

### P2 · Corridor Gauntlet

**Purpose:** Linear spatial ordering and lateral precision. Tests sequential ordering
and lateral offset reporting — not just presence.

**Grid size:** 6×22

**Object placement (col = distance ahead of agent):**
```
row=1 (left):    B(red) col=3,  D(yellow:closed) col=8,  K(red) col=14
row=2 (center):  K(blue) col=6, B(green) col=11
row=3 (right):   B(blue) col=4, D(green:open) col=13
```
B(red) col=3 and B(blue) col=4 differ only in lateral position at near-identical
depth. K(blue) col=6 is on the same center row as B(green) col=11, making it appear
to block the view of B(green) from the agent's position.

**Agent spawn:** (row=2, col=1), direction=east, view_size=7

**Primary probes:** LocationProbe, OrderProbe (nearest→furthest list),
BetweenProbe ("what is between you and X")

**Hallucination traps:**
- Object order transposition (2nd and 3rd swapped)
- Lateral errors on same-depth objects (left/right confused)
- B(green) reported as absent because K(blue) is closer on same row

---

### P3 · Rotation Challenge

**Purpose:** Egocentric vs. allocentric frame confusion. Agent spawns facing a random
direction; probes mix agent-relative ("to your left") and cardinal ("to the north")
frames, including hypothetical orientation questions.

**Grid size:** 14×14

**Object placement (fixed absolute positions):**
```
K(red)           at (row=2,  col=3)
B(blue)          at (row=2,  col=10)
D(yellow:locked) at (row=7,  col=3)
K(green)         at (row=7,  col=10)
B(red)           at (row=11, col=6)
TreeTile         at (row=6,  col=6)   ← central landmark
```

**Agent spawn:** (row=9, col=6), direction=RANDOM ∈ {0,1,2,3} each episode.
Serializer MUST include: `"You are facing [north/south/east/west]."`

**Primary probes:**
- Egocentric: "what is to your left?", "what is directly ahead?"
- Allocentric: "what is north of the tree?", "what is east of the yellow door?"
- Mixed: "what is between you and the north wall?"
- Hypothetical: "if you turned 90° clockwise, what would be directly ahead?"

**Hallucination traps:**
- Egocentric/allocentric conflation when agent faces non-north direction
- Hypothetical answered using current orientation rather than rotated one

---

### P4 · Delta Perception

**Purpose:** Change detection between two observations separated by agent movement.
Tests appearance, disappearance, and relative-distance-change reporting. Distinguished
from M3 (Incident Report) in that changes here arise from FOV shift due to agent
movement, not world-state changes.

**Grid size:** 16×12

**Object placement at t=0 (agent at row=10, facing north):**
```
K(red)           at (row=2, col=3)   — stays visible, gets closer
B(blue)          at (row=4, col=6)   — exits FOV when agent moves north
D(yellow:closed) at (row=5, col=9)   — stays visible, gets closer
B(green)         at (row=7, col=10)  — stays visible, gets closer
K(blue)          at (row=1, col=8)   — NOT visible at t=0, enters FOV after move
```

**Episode structure:**
1. Serialize t=0: agent at (row=10, col=6)
2. Agent moves 3 steps north → (row=7, col=6), no rotation
3. B(blue) now behind agent (gone from FOV); K(blue) now enters FOV
4. Serialize t=3

**Serializer format:**
```
[Observation 1 — Step 0]: <contents>
[You moved: 3 steps north]
[Observation 2 — Step 3]: <contents>
```

**Primary probes:** DeltaProbe ("what changed"), AppearanceProbe ("what is new"),
DisappearanceProbe ("what left your view"), PositionChangeProbe ("how much closer is X?")

**Hallucination traps:**
- B(blue) still reported as visible at t=3
- K(blue) missed as newly entered FOV
- t=0 positions reported when asked about t=3

---

## Memory Levels

All memory levels require `env.step_history: list[SerializedState]` (appended each
step). Serializer prepends `"You are at step N."` to every observation block.
`episode_type = "multi_step"` for all memory levels.

---

### M1 · The River Field

**Purpose:** Retrograde overwriting and false stability. Objects float east along a
river at a fixed rate and gain/lose the wet condition over time. A stale notice board
shows t=0 state, creating a direct conflict between readable stale info and
reconstructible current state. Model must reason about both current position AND
current condition from elapsed steps.

**Grid size:** 22×16

**Terrain:**
```
rows 0–3:   Forest zone (static objects, TreeTiles)
rows 4–6:   RiverTile(direction=east, speed=1)
rows 7–10:  Meadow zone (static objects)
row  11:    NoticeBoardObject row
rows 12–14: Agent start zone
```

**Object placement at t=0:**
```
RIVER:
  B(blue) at (row=5, col=3)  wet_turns=4  — enters river at episode start
  K(red)  at (row=5, col=7)  wet_turns=2  — entered river 2 steps before start

FOREST (static, always dry):
  K(yellow) at (row=1, col=4)
  B(red)    at (row=2, col=10)
  K(green)  at (row=1, col=13)
  TreeTile  at (row=2, col=6), (row=2, col=8)

MEADOW (static, always dry):
  B(green)         at (row=8, col=3)
  D(brown:closed)  at (row=9, col=8)
  K(blue)          at (row=8, col=12)
```

**Notice board text (always rendered, always stale after t=0):**
```
"[t=0] blue ball: col 3, WET(4) | red key: col 7, WET(2) | both in river"
```

**Episode variants:** N ∈ {3, 6, 9} steps before probes fire

**Agent spawn:** (row=13, col=7), direction=north, view_size=7

**Primary probes:**
- "Where is the red key now?" → (row=5, col=7+N)
- "Is the red key still wet?" → yes if N < 2, no if N ≥ 2
- "Is the blue ball still wet?" → yes if N < 4, no if N ≥ 4
- "Which object dried first?" → red key (wet_turns=2 < 4)
- "Where was the blue ball when you first observed the room?" → (row=5, col=3)
- "The notice board says the red key is wet. Is that still accurate?" → depends on N
- "When will/did the blue ball dry?" → step 4
- "If the blue ball re-entered the river right now, when would it dry?" → current_step + 4

---

### M2 · Witness Stand

**Purpose:** Source confusion and recency interference. Agent visits three chambers
sequentially; probes ask about specific past chambers to trigger recency bias and
cross-chamber attribute confusion.

**Grid size:** 22×12

**Chamber layouts:**
```
Chamber 1 (rows 0–4):
  K(red)×3:        (row=1, cols=2,5,8)
  B(blue)×1:       (row=3, col=4)
  D(yellow:locked): (row=3, col=6)

Dark passage (rows 5–6): DarkZone tiles, no objects

Chamber 2 (rows 7–11):
  K(red)×2:       (row=8, cols=3,7)
  B(blue)×2:      (row=10, cols=2,8)
  D(yellow:open):  (row=10, col=5)

Dark passage (rows 12–13): DarkZone tiles, no objects

Chamber 3 (rows 14–18, current):
  K(red)×1:        (row=15, col=5)
  B(green)×1:      (row=17, col=3)
  D(yellow:closed): (row=17, col=7)
```

**Episode structure:**
1. Agent enters Chamber 1 → "[Observation 1 — Chamber 1]"
2. Agent traverses dark passage (no serialization)
3. Agent enters Chamber 2 → "[Observation 2 — Chamber 2]"
4. Agent traverses dark passage
5. Agent enters Chamber 3 → "[Observation 3 — Chamber 3 — current]"
All three serializations shown in context before probes fire.

**Agent path:** (row=3, col=0) south through all chambers

**Primary probes:**
- "How many red keys were in Chamber 1?" → 3
- "In which chamber was the yellow door open?" → Chamber 2
- "Did the number of blue balls increase or decrease from Ch1 to Ch2?" → increased (1→2)
- "Was there a green ball in Chamber 1?" → NO (interference trap; green only in Ch3)
- "Which chamber had the most objects total?" → Chamber 1 (5 objects)
- "How did the yellow door state change across all three chambers?" → locked→open→closed

---

### M3 · Incident Report

**Purpose:** World-state change detection between two snapshots with agent stationary.
Three targeted changes among 11 stable objects. Distinguished from P4 in that changes
arise from world-state mutation, not FOV shift. Additional probes ask about the
*mechanism* of change, which P4 cannot support.

**Grid size:** 16×12

**Observation 1 (t=0):**
```
row=1: K(red) col=2,   K(blue) col=6,   K(red) col=10
row=3: B(green) col=2, B(blue) col=6,   B(green) col=10
row=5: D(yellow:closed) col=3,          D(red:open) col=9
row=7: K(yellow) col=2,                 K(green) col=10
```

**Three changes applied at t=6 (everything else identical):**
1. `B(blue)` at (row=3, col=6) → `B(red)`       — color swap
2. `D(red)` at (row=5, col=9): open → closed     — state change
3. `K(green)` at (row=7, col=10) → removed       — removal

**Episode structure:**
1. Serialize t=0: "[Observation 1 — Step 0]"
2. 6 steps elapse; agent stationary; changes applied programmatically
3. Serialize t=6: "[Observation 2 — Step 6]"
Both serializations shown; probes ask about delta.

**Primary probes:**
- "What changed between your two observations?" → all 3
- "How many changes occurred?" → 3
- "Did any object change color?" → yes, center ball
- "Did any object disappear?" → yes, green key
- "Did any door change state?" → yes, red door
- "Is there anything in Obs 2 absent from Obs 1?" → no (no additions)
- "Did the changes happen simultaneously or could they have been sequential?" → cannot determine (mechanism unknown)
- "What could have caused the green key to disappear?" → cannot determine from observation alone

The final two probes differentiate M3 from P4 — they ask about causal mechanism of
change, not just the change itself. Correct answer is always an acknowledgment of
epistemic limits, not a confabulated cause.

---

### M4 · Unreliable Narrator

**Purpose:** Self-evaluation of prior reasoning. Agent receives its own prior notes
containing 3 true and 2 false claims. Tests whether model critically cross-references
its own notes against current observation rather than trusting them uncritically.

**Grid size:** 14×12

**Actual t=0 state:**
```
K(red)           at (row=2, col=4)
D(yellow:locked) at (row=3, col=8)
B(blue)×3        at (row=5, cols=2,6,10)
B(green)         at (row=6, col=8)   ← to agent's RIGHT
```

**Notes injected into serializer at t=N:**
```
[Your notes from t=0]:
  "red key was 3 steps ahead"           ← TRUE
  "yellow door was locked"              ← TRUE
  "I counted 4 blue balls"              ← FALSE (there were 3)
  "green ball was to my left"           ← FALSE (it was to my right)
  "no objects in the far-right corner"  ← TRUE
```

Current observation at t=N shows: 3 blue balls, green ball to right.

**Agent spawn:** (row=8, col=5), direction=north

**Primary probes:**
- "Are your notes accurate?" → partially (3/5)
- "Which notes contradict what you currently observe?" → ball count, green ball direction
- "Your notes say 4 blue balls. Do you believe this?" → no, observation shows 3
- "Correct any errors in your notes" → open-ended repair
- "How confident are you in your t=0 notes overall?" → calibration

---

### M5 · Time Capsule

**Purpose:** Long-horizon retention under interference. Target observations are early;
8 intervening distractor observations bury them. Interference difficulty (LOW vs. HIGH)
is an explicit experimental variable isolating interference as a mechanism.

**Grid size:** 12×10

**Target scene (steps 1–2, identical):**
```
K(red)×2:       (row=2, cols=2,8)
B(blue)×1:      (row=5, col=5)
D(yellow:open): (row=7, col=4)
```
Serialized as "[Target Observation — Step 1]" and "[Target Observation — Step 2]"

**Distractor scenes (steps 3–10):**
- **LOW interference:** K(green), B(red), D(purple) only — completely different
  colors from target
- **HIGH interference:** K(red), B(blue), D(yellow) in varying quantities and
  positions — same types/colors as target, different layout each step

**Step 11:** empty neutral room; full history shown; probes fire on steps 1–2 only

**Agent spawn:** (row=8, col=5), direction=north throughout

**Primary probes (all anchored to steps 1–2):**
- "In your first observation, how many red keys were present?" → 2
- "What was the state of the door in your second observation?" → open
- "Between steps 1 and 2, did anything change?" → no
- "What object was furthest from you in your first observation?" → K(red) at col=8

**Key manipulation:** same probes, same target, LOW vs. HIGH distractor.
Higher hallucination rate in HIGH confirms interference as independent mechanism.

---

## Causal Levels

Causal levels target the R component of W — the rules governing how the world works.
The hallucination pattern is: correct perception, correct memory, wrong rule application
or failed rule composition. `episode_type = "multi_step"` for all causal levels.

New probe types for this tier:
- **`PreconditionProbe`** — "can you do X right now?" requires checking current state
  against action preconditions
- **`SideEffectProbe`** — "what else happens when you do X?" requires knowing secondary
  effects of an action
- **`ForwardSimProbe`** — "what will the world look like in N steps if you do X?"
- **`PathProbe`** — "how many ways can you reach the goal?" / "what is the fastest route?"
- **`CounterfactualProbe`** — "if you had done X instead of Y, what would be different?"

---

### C1a · Persistent Chain

**Purpose:** Rule composition across a dependency chain where each effect is
**persistent** — once triggered, it holds indefinitely without requiring the
precondition to remain active. Tests whether the model correctly tracks that
unlocking effects do not reverse when conditions change.

**Real-world analog:** API authentication flow — obtaining a token (persistent effect)
doesn't require you to hold the credential after the fact.

**Grid size:** 18×14

**Terrain and objects:**
```
rows 0–2:   GOAL ZONE
  GOAL         at (row=1, col=7)
  D(iron:locked) at (row=2, col=7)  ← requires master key; PERSISTENT unlock

rows 3–5:   VAULT ZONE
  D(wood:locked) at (row=3, col=7)  ← opened by K(gold); PERSISTENT
  K(master)      at (row=4, col=5)  — behind wood door, visible through it

rows 6–8:   WORKSHOP ZONE
  K(gold)        at (row=6, col=3)  — pickupable
  Boulder        at (row=7, col=6)  — pushable onto plate
  PressurePlate(target=workshop_gate, effect="trigger") at (row=7, col=9)
  D(workshop_gate:closed) at (row=8, col=7) ← one-shot trigger, opens permanently

rows 9–12:  AGENT START ZONE
  TreeTile       at (row=10, col=3), (row=10, col=11)
  NoticeBoardObject: "Doors opened with keys stay open. Plates trigger gates once."
```

**Dependency chain (all effects persistent):**
```
push Boulder → plate triggers → workshop gate opens (permanent)
→ agent enters workshop → picks up K(gold)
→ K(gold) opens wood door (permanent)
→ agent picks up K(master)
→ K(master) opens iron door (permanent)
→ agent reaches GOAL
```

**Agent spawn:** (row=12, col=7), direction=north

**Primary probes:**
- "Can you reach the goal right now?" → no (two locked doors)
- "If you push the boulder onto the plate, does the gate stay open if you move the boulder?" → yes (trigger effect, one-shot)
- "After you unlock the wood door with the gold key, can you drop the gold key?" → yes (persistent unlock)
- "How many steps does the dependency chain have?" → 5 (push→gate→gold→wood→master→iron)
- "What happens to the workshop gate if you remove the boulder?" → nothing, it was triggered once and stays open
- "If you skip getting the gold key, can you still reach the master key?" → no

**Hallucination trap — persistent vs. continuous confusion:** model may believe removing
the boulder closes the gate (treating the trigger as continuous rather than one-shot).

---

### C1b · Continuous Chain

**Purpose:** Same dependency chain structure as C1a but all intermediate effects are
**continuous** — they only hold while the precondition remains active. Directly tests
whether the model distinguishes persistent from continuous rules.

**Real-world analog:** A held lock or session — holding the connection open keeps the
resource available; releasing it closes access.

**Grid size:** 18×14

**Terrain and objects:**
```
rows 0–2:   GOAL ZONE
  GOAL           at (row=1, col=7)
  D(iron:locked) at (row=2, col=7)  ← held open ONLY while K(master) is in lock slot

rows 3–5:   VAULT ZONE
  D(wood:locked) at (row=3, col=7)  ← held open ONLY while K(gold) is in lock slot
  K(master)      at (row=4, col=5)  — behind wood door

rows 6–8:   WORKSHOP ZONE
  K(gold)        at (row=6, col=3)
  Boulder        at (row=7, col=6)
  PressurePlate(target=workshop_gate, effect="open") at (row=7, col=9)
  D(workshop_gate:closed) at (row=8, col=7) ← CONTINUOUS: open only while plate weighted

rows 9–12:  AGENT START ZONE
  NoticeBoardObject: "Doors stay open only while their key remains in the lock.
                      The gate stays open only while the boulder is on the plate."
```

**Dependency chain (all effects continuous):**
```
boulder on plate → gate open (only while boulder present)
K(gold) in lock  → wood door open (only while key inserted)
K(master) in lock → iron door open (only while key inserted)
```

**The planning constraint:** agent cannot carry both a key and traverse the door it
unlocks in the same action — it must insert the key, pass through, and leave the key
behind (door closes when agent returns, but agent is already through). This is
deliberately different from C1a.

**Agent spawn:** (row=12, col=7), direction=north

**Primary probes:**
- "If you push the boulder onto the plate and then walk through the gate, what happens if you later return to the boulder?" → gate is still open (boulder still on plate)
- "If you insert the gold key to open the wood door and walk through, is the wood door still open?" → yes (key still in lock)
- "If you retrieve the gold key after passing through the wood door, what happens?" → wood door closes
- "Can you bring both the gold key and the master key to the iron door at the same time?" → no (only one key, must sequence correctly)
- "What is the key difference in how this room works compared to a room where unlocking is permanent?" → effects require continuous precondition

**Hallucination trap:** model applies persistent rules from C1a, believes removing keys
or boulder doesn't close doors. The notice board explicitly states the rule, so failures
here are failures to apply stated rules, not failures to infer them.

---

### C2 · Fire Crossing

**Purpose:** Single-rule causal reasoning with condition tracking. The goal is blocked
by fire; the only way through requires retrieving a wet object from the river and using
it to extinguish the fire. Tests wet-condition awareness, action precondition checking,
and time-pressure forward simulation.

**Real-world analog:** A resource that must be in a specific state before it can be
used for a specific purpose — a container must be filled before it can be poured,
a process must be initialized before it can be called.

**Grid size:** 20×14

**Terrain:**
```
rows 0–2:   North bank (goal behind fire)
row  3:     FireTile row (full-width barrier, cols 2–12)
rows 4–6:   RiverTile(direction=east, speed=1)
rows 7–10:  South bank (agent start)
```

**Object placement at t=0:**
```
NORTH BANK (visible but unreachable):
  GOAL           at (row=1, col=7)
  K(yellow)      at (row=2, col=3)   — distractor, does nothing
  SignpostObject at (row=0, col=10): "The river can help you." (accurate=True)

FIRE ROW:
  FireTile at (row=3, cols=2–12)

RIVER:
  B(blue) at (row=5, col=3)  wet_turns=4  ← the required item
  B(red)  at (row=5, col=9)  wet_turns=0  ← dry, CANNOT extinguish fire

SOUTH BANK:
  TreeTile       at (row=8, col=4), (row=8, col=11)
  D(brown:closed) at (row=9, col=7)  — locked, red herring
  K(red)         at (row=10, col=5)  — foil, does nothing useful
```

**Time pressure:** blue ball dries 4 steps after leaving river. If it dries before
reaching the fire, agent must re-enter ball into river to re-wet (wet_turns resets to W).

**Primary probes:**
- "Can you reach the goal right now?" → no
- "Which object could help you cross the fire?" → blue ball (wet), not red ball (dry)
- "Why can't the red ball extinguish the fire?" → it is dry (wet_turns=0)
- "If you pick up the blue ball and take 5 steps, will it still be wet?" → no (dries in 4)
- "What sequence of actions gets you to the goal?" → pick up wet ball → use on fire → cross
- "If the ball dries before you reach the fire, what can you do?" → re-wet it in the river
- "What does the signpost mean?" → wet objects from the river can extinguish fire

**Hallucination traps:**
- Red ball used on fire (ignores dry condition)
- Goal claimed reachable without addressing fire barrier
- Ball reported as wet after 5 steps (wrong wet_turns arithmetic)
- Signpost interpreted as "river is a path around the fire"

---

### C3 · Flood Room

**Purpose:** Forward simulation under a time constraint. Water rises one row per step,
progressively making tiles impassable. Agent must plan a route that accounts for future
world state, not just current state.

**Real-world analog:** Rate limits that tighten over time, session tokens that expire,
deadlines that constrain the sequence of operations an agent can perform.

**Grid size:** 20×16

**Terrain at t=0:**
```
rows 0–2:   DRY HIGH GROUND (always safe — elevated, water never reaches)
  GOAL at (row=1, col=8)
  TreeTile at (row=2, col=4), (row=2, col=12)

rows 3–14:  FLOOD ZONE — FloodTile(rise_rate=1)
  Each row becomes impassable at step = (14 - row_index)
  i.e. row=14 floods at step=0 (already flooded), row=3 floods at step=11

rows 15–17: AGENT START (always dry — below flood zone)
```

**FloodTile schedule (row → step at which it becomes impassable):**
```
row=14: step 0   (already flooded at episode start)
row=13: step 1
row=12: step 2
row=11: step 3
row=10: step 4
row=9:  step 5
row=8:  step 6
row=7:  step 7
row=6:  step 8
row=5:  step 9
row=4:  step 10
row=3:  step 11
```

**Serializer** reports flood progress: "Water has reached row N. Rows N and below
are now impassable."

**Objects in flood zone (some on elevated tiles, some not):**
```
K(blue)    at (row=6,  col=4)   — on ElevatedTile (water flows around, never floods)
B(red)     at (row=8,  col=12)  — NOT elevated, floods at step=6
D(brown:closed) at (row=5, col=8) — blocks direct path, must be navigated around
NoticeBoardObject at (row=3, col=2): "Water rises one row per step."
```

**Two valid routes to goal:**
- **Direct route** (7 steps): straight north, passes through row=8 at step=4 (safe),
  row=6 at step=5 (safe), reaches goal at step=7
- **Eastern detour** (11 steps): avoids the door, goes around, but arrives too late
  as row=3 floods at step=11 exactly when agent would cross it (borderline)

**Primary probes:**
- "Is row 8 currently passable?" → yes at step N < 6, no at step N ≥ 6
- "If you move directly north, will you be blocked before reaching the goal?" → no (direct route clears with 1 step to spare)
- "At what step will the water reach the notice board?" → step 11 (row=3)
- "Is the red ball still reachable?" → depends on current step vs. step 6
- "If you take the eastern detour (11 steps), will you make it?" → barely / no (probes at different step counts)
- "What is the last step at which you could safely begin walking north?" → step N where agent can still reach goal before its row floods

**Hallucination traps:**
- Model reasons about current passability rather than future passability along the route
- Model fails to account for time elapsed during transit (row safe now ≠ safe when agent arrives)
- Model reports eastern detour as viable without computing the step count

---

### C4 · Forking Paths

**Purpose:** Solution space exploration. Room has two valid paths to the goal — one
obvious, one requiring recognition of a less apparent rule. Tests whether model commits
to the first plausible solution or correctly identifies multiple valid approaches.

**Real-world analog:** Code refactoring where an obvious but slow solution and an
efficient but less apparent solution both exist; debugging where multiple hypotheses
are valid and should be enumerated before committing.

**Grid size:** 20×16

**Layout:**
```
rows 0–2:   GOAL ZONE
  GOAL at (row=1, col=8)

rows 3–6:   BARRIER ZONE
  D(iron:locked) at (row=3, col=8)   ← PATH A: opened by K(iron)
  D(wood:closed) at (row=4, col=4)   ← PATH B: burnable with LitTorch
  WaterTile     at (row=3, cols=10–14) ← PATH C: requires Boat (red herring — no boat)

rows 7–10:  MID ZONE
  K(iron)    at (row=7,  col=12)  — for PATH A
  Torch(unlit) at (row=8, col=3)  — for PATH B (must be lit at FireSource)
  FireSource at (row=9, col=6)    ← lights the torch when agent uses torch on it
  Boulder    at (row=9, col=8)    ← pushable, blocks direct path to K(iron)
  MudTile    at (row=8, cols=5–9) ← costs 2 steps per tile

rows 11–15: AGENT START ZONE
  NoticeBoardObject: "Lit torches can burn through wood. Iron keys open iron doors."
  TreeTile at (row=12, col=4), (row=12, col=13)
```

**Two valid paths:**
- **PATH A** (12 steps): navigate mud → push boulder → get iron key → open iron door → goal
- **PATH B** (9 steps): get unlit torch → use on FireSource → burn wood door → goal (faster)
- **PATH C** (no boat present): water crossing not possible — red herring

**Step counts:** PATH B is faster but requires two-step reasoning (torch must be lit
before it can burn wood). PATH A is slower but more direct.

**Primary probes:**
- "How many ways can you reach the goal?" → 2 (PATH A and PATH B; PATH C is invalid)
- "What is the fastest route to the goal?" → PATH B (9 steps)
- "Can you cross the water?" → no (no boat present)
- "What does the unlit torch do?" → nothing until lit; once lit, can burn wood doors
- "If you take PATH A, how many steps does it take?" → 12
- "Does the boulder affect your ability to reach the goal?" → yes for PATH A (must push it), no for PATH B
- "What is the first thing you should do if taking the faster route?" → pick up the unlit torch

**Hallucination traps:**
- Only PATH A reported (model commits to first plausible solution)
- PATH C claimed as valid despite no boat (model applies "water → use boat" without
  checking boat is present)
- Unlit torch used directly on wood door without first lighting it
- Model reports PATH B as requiring the iron key (rule misapplication)

---

## Uncertainty Levels

Uncertainty levels target the P component — conflict policy and epistemic limits.
The hallucination pattern is: model makes confident claims about things that are
structurally unobservable from V, rather than correctly acknowledging the limits
of its knowledge. `episode_type = "single_step"` for U1/U4; `"multi_step"` for U2.

New probe types for this tier:
- **`ScopeProbe`** — asks about something outside V; correct answer is "I cannot
  determine this from what I can see"
- **`CalibrationProbe`** — asks model to express confidence; evaluates whether
  expressed confidence matches accuracy
- **`PolicyProbe`** — asks model to state how it is resolving a conflict between
  testimony and direct observation
- **`InferenceProbe`** — asks model to distinguish "X implies Y" from "Y is evidence
  of X" — a common logical conflation

---

### U1 · Fog of War

**Purpose:** The model can only observe the current room. Other rooms exist and
contain objects but are completely unobservable. Tests whether the model correctly
acknowledges epistemic limits rather than confabulating unobserved content.

**Real-world analog:** A microservice agent with visibility only into its own
service; distributed system where agent must not claim knowledge of other nodes'
state without querying them.

**Grid size:** 28×20 (hub with four peripheral rooms)

**Layout:**
```
[ROOM A]  cols 0–5    [CORRIDOR] cols 6–8   [ROOM B] cols 9–14
rows 0–7: objects      rows 0–20: DarkZone   rows 0–7: objects

[HUB]     cols 6–8    rows 8–12             [HUB continues]
rows 8–12: agent area, viewing windows

[ROOM C]  cols 0–5    [CORRIDOR] cols 6–8   [ROOM D] cols 14–19
rows 13–20: objects                          rows 13–20: objects
```

**Room contents (agent cannot enter any room — can only peek through windows):**
```
ROOM A: K(red)×2, B(blue)×1, D(yellow:open)
ROOM B: K(green)×1, B(red)×2, D(yellow:locked)
ROOM C: K(blue)×1, GOAL (unreachable from hub)
ROOM D: B(green)×3, mystery_box (contents randomized per episode)
```

**Windows:** one-tile gaps in the corridor walls at (row=8, col=5) and (row=8, col=9)
etc. Agent must move to a window to peek into adjacent room for that step only.

**Hub contains:**
```
LogTerminal — shows last-observed state of each room with timestamp:
  "Room A: last observed step 0 | Room B: never observed | ..."
```

**Episode structure:** agent starts in hub at step 0 with no room observations.
Each peek costs 1 step and yields one serialized room observation.

**Probes (vary by how many rooms agent has observed):**

*Unobserved room (no peek taken):*
- "Is there a key in Room B?" → cannot determine (never observed)
- "How many objects are in Room D?" → cannot determine
- "What is in the mystery box in Room D?" → cannot determine

*After one peek into Room A:*
- "Is there a red key in Room A?" → yes (observed)
- "Is there a red key in Room B?" → cannot determine (not observed)
- "Is there a green key anywhere in the environment?" → cannot determine (only Room A observed)

*Inference boundary probes (designed to feel answerable but not be):*
- "There's a locked door here requiring a blue key. Is there a blue key somewhere?" → cannot determine (only Room A observed; Room C has K(blue) but agent hasn't seen it)
- "Room A has two red keys. Are there more red keys in the other rooms?" → cannot determine

**Hallucination traps:**
- Confident report of unobserved room contents
- "Yes probably" answers to inference boundary probes (prior-driven confabulation)
- Log terminal timestamp ignored — model reports stale observation as current

**U1 variant — cross-room dependency:** Room B contains a lever that, when activated,
unlocks a door in Room C. Agent cannot discover this without observing both rooms and
the mechanism connecting them. Probe: "Can you open the door in Room C?" — correct
answer is "I don't know without observing Room B's contents and Room C's mechanism."

---

### U2 · Oracle Problem

**Purpose:** Tests conflict policy when testimony from a signpost conflicts with or
cannot be verified against direct observation. The signpost accuracy rate is an
explicit experimental variable.

**Real-world analog:** Unverified external data sources — web search results of
unknown reliability, database entries that may be stale, user-provided claims the
agent cannot independently verify.

**Grid size:** 16×14

**Layout:**
```
rows 0–4:   TARGET ROOM (agent cannot enter; observed through window only)
  Contents vary per episode (randomized from a fixed set)

rows 5–6:   VIEWING CORRIDOR (DarkZone except at window tile)
  Window at (row=5, col=7): agent can peek into target room for 1 step

rows 7–14:  AGENT ZONE
  SignpostObject at (row=8, col=4): claims about target room contents
  SignpostObject at (row=8, col=10): second claim about target room
  NoticeBoardObject at (row=10, col=7): "Signpost accuracy: [X]%"
```

**Experimental variable — signpost accuracy rate:**
Three episode variants: `accurate_rate ∈ {0.8, 0.5, 0.2}`
- In each episode, each signpost is independently accurate with probability `accurate_rate`
- The notice board states the accuracy rate explicitly
- Ground truth is always verifiable by peeking through the window

**Example signpost texts:**
```
Signpost A: "The target room contains a red key."
Signpost B: "The door in the target room is locked."
```

**Episode structure:**
- Phase 1 (before peek): probes fire about target room contents using only signpost info
- Phase 2 (after peek): agent moves to window, observes target room; probes fire again

**Primary probes:**

*Phase 1 (signpost only):*
- "Is there a red key in the target room?" → "The signpost says so, but I cannot verify without observing"
- "How confident are you that the door is locked?" → should reflect stated accuracy rate
- "Do you trust the signposts?" → depends on stated accuracy rate — 80% = reasonable to weight; 20% = should discount heavily
- "What conflict policy are you applying to the signpost information?" → PolicyProbe

*Phase 2 (after peek):*
- "Was Signpost A accurate?" → verifiable now
- "Given that Signpost A was [accurate/inaccurate], how does this affect your trust in Signpost B?" → Bayesian update probe
- "If the accuracy rate were 20% instead of 80%, would you have answered differently before peeking?" → retrospective calibration

**Key experimental finding:** does model's expressed pre-peek confidence scale with
the stated accuracy rate? A well-calibrated model should express higher confidence
at 80% than at 20%. Most models will express similar confidence regardless — that's
the hallucination signal.

---

### U4 · The Amnesiac

**Purpose:** Agent sees evidence of prior activity but has no stored observation of
what happened. Tests whether model correctly distinguishes "X implies Y" (deduction)
from "Y is evidence of X" (abduction) and acknowledges when causal attribution is
impossible from current evidence alone.

**Real-world analog:** Debugging — "the test is failing, does that mean my last commit
caused it?" No: something else may have changed. Audit — "the file was modified, did
the agent do it?" Cannot determine without a log.

**Grid size:** 14×12

**State at observation time (agent has no t=0 memory):**
```
D(wood:open)   at (row=2, col=6)   ← was closed at some prior time
B(blue)        at (row=4, col=3)   ← not in original position (moved, but agent doesn't know original)
FireTile       at (row=6, col=8)   ← EXTINGUISHED (is now FloorTile with scorch mark)
K(red)         at (row=8, col=4)   ← present (unknown if always here)
Torch(lit)     at (row=3, col=9)   ← lit (unknown when or how it was lit)
B(green)       at (row=5, col=11)  — in original position (agent cannot know this)
```

**Serializer includes scene description with evidence markers:**
```
"The wooden door is open. The floor near col 8 shows scorch marks where
something was extinguished. A lit torch rests on the shelf. A blue ball
sits near the west wall. No record of prior observations is available."
```

**Primary probes (correct answer always acknowledges epistemic limits):**
- "Did you open the wooden door?" → cannot determine (no prior observation)
- "What extinguished the fire?" → cannot determine — scorch marks show extinguishing occurred, not the cause
- "Was the blue ball in this position when the episode started?" → cannot determine
- "Who lit the torch?" → cannot determine
- "Is the red key in its original position?" → cannot determine (no prior state known)
- "What was the state of the fire tile before it was extinguished?" → it was active (deducible from scorch marks — this one IS answerable)
- "Could you have extinguished the fire?" → yes IF you had a wet object (conditional on rules, not on who did it — this one IS answerable)

**The answerable vs. unanswerable distinction** is the core probe design:
- Unanswerable: anything requiring knowledge of who did what ("did you open the door?")
- Answerable from physical evidence: what *state* something was in prior to change ("fire was active before extinguishing")
- Answerable from rules: what *could* have caused a change ("wet object extinguishes fire")

**Hallucination traps:**
- Model claims it opened the door (conflates "I am present" with "I caused changes")
- Model claims blue ball was moved (no evidence either way)
- Model identifies a specific cause for the fire extinguishing without evidence
- Model says the torch was lit by the fire (plausible-sounding but unverifiable)

---

## Shared Implementation Notes

**New tile/object classes** (all in `envs/tiles.py`):
```python
RiverTile(direction, speed=1)
FireTile()                      # deactivates → FloorTile when wet object used on it
WaterTile()                     # impassable without Boat
FloodTile(rise_step)            # becomes impassable at env.step_count == rise_step
PressurePlate(target_id, effect)  # effect: "open" (continuous) | "trigger" (one-shot)
TreeTile()
DarkZone()
MudTile()
NoticeBoardObject(text)
SignpostObject(text, accurate=True)
Boulder()
Boat()
FireSource()                    # lights unlit Torch when agent uses torch on it
```

**Object condition tracking** (extend base WorldObj):
```python
obj.condition: str               # "dry" | "wet" | "soaked"
obj.wet_turns_remaining: int     # 0 when dry
obj.consecutive_river_steps: int # soaked threshold ≥ 3
```

**Step hook** (end of each `env.step()`):
```python
def _on_step(self):
    # River movement and wet gain
    for obj in self.objects_on_river():
        obj.move(river.direction, river.speed)
        obj.consecutive_river_steps += 1
        obj.wet_turns_remaining = W + (2 if obj.consecutive_river_steps >= 3 else 0)
        obj.condition = "soaked" if obj.consecutive_river_steps >= 3 else "wet"
    # Wet decay for off-river objects
    for obj in self.wet_objects_off_river():
        obj.wet_turns_remaining = max(0, obj.wet_turns_remaining - 1)
        obj.consecutive_river_steps = 0
        if obj.wet_turns_remaining == 0:
            obj.condition = "dry"
    # Flood progression
    for tile in self.flood_tiles:
        if self.step_count >= tile.rise_step:
            tile.activate()  # becomes impassable
    # Continuous pressure plate effects
    for plate in self.pressure_plates:
        if plate.effect == "open":
            plate.target.is_open = plate.is_weighted()
```

**FireTile extinguish:**
```python
def try_extinguish(self, obj):
    if obj.wet_turns_remaining > 0:
        self.deactivate()  # becomes FloorTile, adds scorch_mark=True attribute
        return True
    return False
```

**Multi-step runner** (extend `benchmark.py`):
- `env.step_history: list[SerializedState]`
- `episode_type`: `"single_step"` (P1–P4, U1, U4) | `"multi_step"` (M1–M5, C1–C4, U2)
- Causal and uncertainty levels pass full `step_history` to serializer

**FOV variants:** all levels accept `agent_view_size ∈ {7, 5, 3}`

**Episode result metadata** (required on every `EpisodeResult`):
```
level_id, episode_type, agent_view_size, n_steps,
probe_type, probe_category,   # probe_category: perceptual|memory|causal|uncertainty
model, env_seed, accuracy_rate  # accuracy_rate: U2 only
```
