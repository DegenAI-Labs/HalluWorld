# Chess Benchmark Prompts and Results

This document describes how the chess hallucination benchmark is currently prompted, what question types are used, and the reported model results.

## 1) Prompt Structure Used by the Benchmark

Each benchmark trial builds a prompt in two parts:

- **System prompt** (`SYSTEM_PROMPT` in `halluworld/benchmark.py`)
- **User prompt** (`## Current observation` + `## Question`)

### 1.1 System Prompt (Current)

```text
You are an agent navigating a gridworld. Answer questions about your current observation concisely and accurately. Base your answers only on what you are told you can see. Do not infer or guess about things outside your view.
```

Note: this text is inherited from the original gridworld benchmark and is currently reused for chess runs.

### 1.2 User Prompt Template

```text
## Current observation
{serialized_chess_observation}

## Question
{probe_question}
```

---

## 2) Chess Observation Format (What the Model Sees)

The chess serializer (`halluworld/serializers/chess.py`) renders:

- A short instruction header
- An 8x8 coordinate-labeled board grid
- Side to move
- FEN string
- Legal UCI move list (optional but currently on)

### 2.1 Concrete Observation Example

```text
You are evaluating a chess position.
Coordinates follow standard chess notation: files a-h, ranks 1-8.
Board ('.' means empty):
8 r n b q k b n r
7 p p p p p p p p
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 P P P P P P P P
1 R N B Q K B N R
  a b c d e f g h
Side to move: white
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Legal moves (UCI): a2a3, a2a4, b1a3, b1c3, b2b3, b2b4, c2c3, c2c4, d2d3, d2d4, e2e3, e2e4, f2f3, f2f4, g1f3, g1h3, g2g3, g2g4, h2h3, h2h4
```

---

## 3) Question Types (Chess Probes)

The chess probes are organized along the four axes of the HalluWorld
world-model definition (V = perceptual, H = memory/persistence, dynamics,
P = belief/uncertainty):

| Axis            | Probe                              | Question shape                                                        | Evaluator                  |
| --------------- | ---------------------------------- | --------------------------------------------------------------------- | -------------------------- |
| Perceptual (V)  | `chess_piece_presence`             | Is there a `<piece>` on `<sq>`?                                       | `ChessYesNoEvaluator`      |
| Perceptual (V)  | `chess_attacker`                   | Is `<sq>` attacked by `<color>`?                                      | `ChessYesNoEvaluator`      |
| Perceptual (V)  | `chess_defended`                   | Is the piece on `<sq>` defended by its own side?                      | `ChessYesNoEvaluator`      |
| Perceptual (V)  | `chess_pinned`                     | Is the piece on `<sq>` pinned to its own king?                        | `ChessYesNoEvaluator`      |
| Perceptual (V)  | `chess_piece_count`                | How many `<color>` `<piece_type>`s are on the board?                  | `ChessIntegerEvaluator`    |
| Memory (H)      | `chess_history_readout`            | After these moves, what is on `<sq>`? (board hidden)                  | `ChessPieceNameEvaluator`  |
| Memory (H)      | `chess_capture_count`              | How many captures occurred during these moves? (board hidden)         | `ChessIntegerEvaluator`    |
| Memory (H)      | `chess_castling_rights_history`    | Does `<color>` still have `<side>` castling rights? (board hidden)    | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_move_legality`              | Is `<uci>` legal right now?                                           | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_hypothetical_in_check`      | If `<color>` plays `<uci>`, would the side to move be in check?       | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_hypothetical_is_capture`    | If `<color>` plays `<uci>`, would it be a capture?                    | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_hypothetical_en_passant_available` | …would en passant be legal next move?                          | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_hypothetical_checkmate`     | …would the position be checkmate?                                     | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_en_passant`                 | (After a forced double-push setup) is en passant legal?               | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_castling_legality`          | Can `<color>` legally castle `<kingside\|queenside>` right now?       | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_mate_in_one`                | Does the side to move have a mate-in-1?                               | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_hanging`                    | Is the piece on `<sq>` hanging (attacked + undefended)?               | `ChessYesNoEvaluator`      |
| Dynamics        | `chess_can_capture`                | Can the side to move legally capture the piece on `<sq>`?             | `ChessYesNoEvaluator`      |
| Dynamics / H    | `chess_san_legal_move`             | SAN-only (optional FEN line); one legal UCI reply                     | `ChessLegalUciSetEvaluator` |
| Belief (P)      | `chess_hidden_square`              | Question about a square that may be hidden in the rendered board      | `ChessYesNoIDKEvaluator`   |
| Belief (P)      | `chess_conflicting_prompt`         | Yes/no question with a deliberately false claim injected in the prompt| `ChessYesNoEvaluator`      |
| Policy          | `chess_best_move`                  | What is the best UCI move? (requires Stockfish)                       | `ChessBestMoveEvaluator`   |

`chess_san_legal_move` also sets `observation_override`: SAN movetext (no UCI
hints, no grid). By default it includes ``Starting FEN:`` plus ``Moves played:``.
With ``observation_mode="san_only"`` it omits the FEN line and replays from the
standard start so the SAN line is a full prefix from move 1 (``replay_from``
must be ``"startpos"``). The evaluator accepts any reply whose first UCI token
is legal in the resulting position. The probe supports capture-biased replays
and retries until `min_plies` is reached without ending the game (see
`ChessSanLegalContinuationProbe` in `halluworld/probes/chess.py`).

Memory (H) and `chess_hidden_square` probes set `ProbeResult.observation_override`,
which the benchmark loop substitutes for the default serializer output:

* History probes render `Starting FEN` + a UCI/SAN move list with no board grid.
* `chess_hidden_square` renders the standard board with one or more squares
  shown as `?`, dropping the FEN and legal-move lines so the mask cannot be
  bypassed.

The original two probes (`chess_piece_presence`, `chess_move_legality`) are
unchanged; their formats remain as documented below.

### 3.1 Piece Presence Question Format

Template:

```text
Is there a {piece_name} on square {square}?
Answer with exactly 'yes' or 'no'.
```

Examples:

```text
Is there a white queen on square d1?
Answer with exactly 'yes' or 'no'.
```

```text
Is there a black knight on square e4?
Answer with exactly 'yes' or 'no'.
```

### 3.2 Move Legality Question Format

Template:

```text
Is the move {uci_move} legal for the side to move in this position?
Answer with exactly 'yes' or 'no'.
```

Examples:

```text
Is the move e2e4 legal for the side to move in this position?
Answer with exactly 'yes' or 'no'.
```

```text
Is the move e1e8 legal for the side to move in this position?
Answer with exactly 'yes' or 'no'.
```

---

## 4) How Scoring Works

Each probe is paired with an evaluator. Yes/no probes use exact string matching
on the parsed `yes` / `no` token; integer probes parse the first integer in the
response; piece-name probes match a canonical `<color> <piece>` label or
`empty`; `chess_san_legal_move` parses the first UCI token and checks membership
in the engine legal set; the `chess_hidden_square` probe uses a 3-way evaluator that adds
`idk` and treats over-confident wrong answers and abstaining-when-known as
equally incorrect.

- Parsed model answer compared to ground truth
- **Correct match** -> score `1.0`
- **Mismatch/ambiguous/unparseable** -> score `0.0`

Reported metrics:

- **accuracy**: fraction of correct responses
- **hallucination_rate**: `1 - mean_score`
- **mean_score**: mean per-item score (same as accuracy here, since scores are binary)

## 4.1) Lichess Puzzle Caching

`load_lichess_puzzles(...)` (and the legacy `load_lichess_puzzle_fens(...)`
wrapper) cache their filtered/shuffled output on disk under
`~/.cache/halluworld/` (override with `HALLUWORLD_CACHE` or the `cache_dir`
argument). The cache key is a hash of `(num_samples, seed, min_rating,
max_rating, themes)`, so identical calls skip the (slow) Hugging Face download
+ filter pass. Pass `force_refresh=True` to bypass the cache. Records include
`fen`, `moves`, `themes`, `rating`, and `puzzle_id` so probe code can filter
positions (e.g. by the `enPassant` or `castling` themes) without re-scanning
the dataset.

---

## 5) Reported Results

## 5.1 Non-puzzle Chess Runs

### GPT-4o-mini


| probe                | n   | accuracy | hallucination_rate |
| -------------------- | --- | -------- | ------------------ |
| chess_move_legality  | 100 | 0.83     | 0.17               |
| chess_piece_presence | 100 | 0.79     | 0.21               |
| ALL                  | 200 | 0.81     | 0.19               |


### GPT-3.5-Turbo


| probe                | n   | accuracy | hallucination_rate |
| -------------------- | --- | -------- | ------------------ |
| chess_move_legality  | 100 | 0.67     | 0.33               |
| chess_piece_presence | 100 | 0.67     | 0.33               |
| ALL                  | 200 | 0.67     | 0.33               |


## 5.2 Chess Puzzle Runs (`Lichess/chess-puzzles`)

### GPT-4o-mini


| probe                | n   | accuracy | hallucination_rate |
| -------------------- | --- | -------- | ------------------ |
| chess_move_legality  | 100 | 0.80     | 0.20               |
| chess_piece_presence | 100 | 0.59     | 0.41               |
| ALL                  | 200 | 0.70     | 0.30               |


### GPT-3.5-Turbo


| probe                | n   | accuracy | hallucination_rate |
| -------------------- | --- | -------- | ------------------ |
| chess_move_legality  | 100 | 0.55     | 0.45               |
| chess_piece_presence | 100 | 0.53     | 0.47               |
| ALL                  | 200 | 0.54     | 0.46               |


## 5.3 Full Probe Battery — Non-puzzle Chess Run

Run with `python -m halluworld.tracks.chess.battery` (default settings: 20 episodes per
probe, seed 42, default FEN pool, `gpt-4o-mini`). Probes grouped by axis;
within each axis ordered as in the table in §3.

### GPT-4o-mini

**Perceptual (V)**

| probe                  |  n | accuracy | hallucination_rate |
| ---------------------- | -: | -------: | -----------------: |
| chess_piece_presence   | 20 |     0.70 |               0.30 |
| chess_attacker         | 20 |     0.70 |               0.30 |
| chess_defended         | 20 |     0.35 |               0.65 |
| chess_pinned           | 20 |     0.95 |               0.05 |
| chess_piece_count      | 20 |     0.80 |               0.20 |

**Memory (H)**

| probe                              |  n | accuracy | hallucination_rate |
| ---------------------------------- | -: | -------: | -----------------: |
| chess_history_readout              | 20 |     0.40 |               0.60 |
| chess_capture_count                | 20 |     0.45 |               0.55 |
| chess_castling_rights_history      | 20 |     0.90 |               0.10 |

**Dynamics**

| probe                                     |  n | accuracy | hallucination_rate |
| ----------------------------------------- | -: | -------: | -----------------: |
| chess_move_legality                       | 20 |     0.75 |               0.25 |
| chess_hypothetical_in_check               | 20 |     0.95 |               0.05 |
| chess_hypothetical_is_capture             | 20 |     0.55 |               0.45 |
| chess_hypothetical_en_passant_available   | 20 |     0.85 |               0.15 |
| chess_hypothetical_checkmate              | 20 |     0.75 |               0.25 |
| chess_en_passant                          | 20 |     0.85 |               0.15 |
| chess_castling_legality                   | 20 |     0.85 |               0.15 |
| chess_mate_in_one                         | 20 |     0.35 |               0.65 |
| chess_hanging                             | 20 |     0.75 |               0.25 |
| chess_can_capture                         | 20 |     0.65 |               0.35 |

**Belief (P)**

| probe                       |  n | accuracy | hallucination_rate |
| --------------------------- | -: | -------: | -----------------: |
| chess_hidden_square         | 20 |     0.50 |               0.50 |
| chess_conflicting_prompt    | 20 |     0.55 |               0.45 |

**Aggregate**

| probe |   n | accuracy | hallucination_rate |
| ----- | --: | -------: | -----------------: |
| ALL   | 400 |     0.68 |               0.32 |

## 5.4 Model Comparison — Full Probe Battery (`gpt-4o` vs `gpt-4o-mini`)

Run with `python -m halluworld.tracks.chess.battery` on the default non-puzzle FEN pool.
The two models were run independently and at different sample budgets, so
per-probe `n` differs (shown in the table). Probes with low `n` (e.g.
`chess_en_passant`, which can only fire when the side to move has a legal
double pawn push) should be read with appropriate caution.

### What each axis is testing

- **Perceptual (V) — use/understanding of the rendered observation.** Can the
  model read off facts that are explicitly present in the board grid + FEN +
  legal-move list it was shown? Includes piece presence, attacks, defense,
  pins, and material counts. Failures here are pure read-out hallucinations.
- **Memory / persistence (H) — tracking state across an interaction history.**
  The board grid is hidden and the model is given only the starting FEN and a
  list of moves played. It must mentally apply the moves to answer the
  question. Tests whether the model maintains an internal `H` rather than
  parroting `V`.
- **Causal / dynamics — reasoning about rule-governed outcomes.** Asks
  whether some property holds *now* (mate-in-1, hanging, can-capture,
  castling legal) or *after* a hypothetical move (would this be a check, a
  capture, expose en passant, deliver mate). Tests whether the model can
  compose movement rules with the position rather than memorize patterns.
- **Belief / uncertainty (P) — calibration when the world is partially
  observed or the prompt lies.** `chess_hidden_square` masks parts of the
  rendered board and expects `idk` when the answer depends on a hidden
  square; `chess_conflicting_prompt` injects a false claim into the question
  and rewards anchoring to the (true) board over the (false) assertion.

### Per-probe accuracy

| axis      | probe                                     | gpt-4o (n) | gpt-4o (acc) | gpt-4o-mini (n) | gpt-4o-mini (acc) | Δ (4o − mini) |
| --------- | ----------------------------------------- | ---------: | -----------: | --------------: | ----------------: | ------------: |
| V         | chess_piece_presence                      |         10 |         0.90 |             100 |              0.61 |        +0.29  |
| V         | chess_attacker                            |         10 |         0.50 |             100 |              0.49 |        +0.01  |
| V         | chess_defended                            |         10 |         0.30 |             100 |              0.57 |        −0.27  |
| V         | chess_pinned                              |         10 |         0.80 |             100 |              0.76 |        +0.04  |
| V         | chess_piece_count                         |         10 |         0.90 |             100 |              0.87 |        +0.03  |
| H         | chess_history_readout                     |         10 |         0.90 |             100 |              0.63 |        +0.27  |
| H         | chess_capture_count                       |         10 |         0.60 |             100 |              0.11 |        +0.49  |
| H         | chess_castling_rights_history             |         10 |         1.00 |             100 |              0.98 |        +0.02  |
| Dynamics  | chess_move_legality                       |         10 |         0.90 |             100 |              0.79 |        +0.11  |
| Dynamics  | chess_hypothetical_in_check               |         10 |         0.60 |             100 |              0.48 |        +0.12  |
| Dynamics  | chess_hypothetical_is_capture             |         10 |         0.60 |             100 |              0.47 |        +0.13  |
| Dynamics  | chess_hypothetical_en_passant_available   |         10 |         1.00 |             100 |              0.76 |        +0.24  |
| Dynamics  | chess_hypothetical_checkmate              |         10 |         0.90 |             100 |              0.37 |        +0.53  |
| Dynamics  | chess_en_passant                          |          5 |         1.00 |              66 |              0.86 |        +0.14  |
| Dynamics  | chess_castling_legality                   |         10 |         1.00 |             100 |              0.94 |        +0.06  |
| Dynamics  | chess_mate_in_one                         |         10 |         0.90 |             100 |              0.21 |        +0.69  |
| Dynamics  | chess_hanging                             |         10 |         0.70 |             100 |              0.56 |        +0.14  |
| Dynamics  | chess_can_capture                         |         10 |         0.40 |             100 |              0.70 |        −0.30  |
| P         | chess_hidden_square                       |         10 |         0.90 |             100 |              0.50 |        +0.40  |
| P         | chess_conflicting_prompt                  |         10 |         0.90 |             100 |              0.47 |        +0.43  |
| **ALL**   | (aggregate)                               |    **195** |     **0.78** |        **1966** |          **0.60** |    **+0.18**  |

### Per-axis aggregates

Macro-averaged accuracy across the probes within each axis (equal weight per
probe, ignoring per-probe `n` differences):

| axis      | gpt-4o (macro acc) | gpt-4o-mini (macro acc) | Δ      |
| --------- | -----------------: | ----------------------: | -----: |
| V         |               0.68 |                    0.66 | +0.02  |
| H         |               0.83 |                    0.57 | +0.26  |
| Dynamics  |               0.80 |                    0.61 | +0.19  |
| P         |               0.90 |                    0.49 | +0.41  |

### Headline observations

- **`gpt-4o` is meaningfully ahead overall** (0.78 vs 0.60), and the gap is
  largest on the two axes the original two-probe benchmark didn't cover at
  all: **belief (P)** (+0.41) and **memory (H)** (+0.26). The smaller model
  is essentially at chance on `chess_hidden_square` (0.50) and
  `chess_conflicting_prompt` (0.47), suggesting it neither abstains under
  uncertainty nor anchors to V when the prompt lies — the qualitative
  failure modes the P axis was designed to surface.
- **The two largest single-probe gaps are `chess_mate_in_one` (+0.69) and
  `chess_hypothetical_checkmate` (+0.53).** Both are dynamics probes that
  require the model to *push a move and re-evaluate* — exactly the
  composition step where weaker models tend to hallucinate. `gpt-4o-mini`
  scores 0.21 on `chess_mate_in_one`, which is below the natural class-prior
  baseline for the probe.
- **`gpt-4o-mini` collapses on `chess_capture_count` (0.11)** despite doing
  well on the simpler `chess_castling_rights_history` (0.98). Both probes
  hide the board and show only a UCI move list, but `capture_count` requires
  the model to mentally replay each move and check `is_capture` cumulatively,
  whereas castling rights only needs it to notice king/rook moves. This
  isolates *quantitative* memory tracking as a specific failure mode, not
  history-tracking generally.
- **Two probes invert** (mini > 4o): `chess_defended` (−0.27) and
  `chess_can_capture` (−0.30). With n=10 for `gpt-4o` these are noisy, but
  worth flagging as candidates for re-running at higher `n`.

---

## 6) Quick Interpretation

- GPT-4o-mini outperforms GPT-3.5-Turbo on both non-puzzle and puzzle runs.
- Move-legality judgments appear easier than piece-presence checks for both models.
- Puzzle positions remain harder than the baseline non-puzzle setup (especially for piece-presence).
- On the full probe battery (§5.3), the easiest probes for `gpt-4o-mini` are
  `chess_pinned` and `chess_hypothetical_in_check` (0.95 each); the hardest are
  `chess_defended` (0.35) and `chess_mate_in_one` (0.35). Memory probes that
  hide the board (`chess_history_readout`, `chess_capture_count`) sit
  noticeably below their perceptual analogues, consistent with weak `H`.
  Belief probes (`chess_hidden_square`, `chess_conflicting_prompt`) are near
  chance, suggesting the model neither abstains under genuine uncertainty nor
  reliably anchors to V when the prompt asserts a falsehood.

