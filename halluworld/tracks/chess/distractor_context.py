"""Synthetic past-game blocks + kibitz to prepend before the real chess observation."""

from __future__ import annotations

import random

import chess

# A few FENs so “past games” are not always from the initial array.
_START_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/pp1ppp1p/5p2/2p5/6P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 4",
    "r1bq1rk1/pppp1ppp/2n2n2/4p3/1b1PP3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 4 6",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 3 11",
    "8/8/8/3B4/8/8/8/4K3 w - - 0 1",
    "8/5k2/8/3B4/8/8/8/4K3 w - - 0 1",
    "r3k2r/pppq1ppp/2n1bn2/3p4/3P4/2N1PN2/PPQ2PPP/R3KB1R w KQkq - 2 10",
]

_GAME_LABELS = [
    "online blitz",
    "club evening",
    "rapid",
    "correspondence",
    "post-mortem replay",
    "skittles",
    "team board",
    "park simul row 3",
    "titled Tuesday",
    "OTB Swiss R7",
    "board 47 (hall)",
    "stream delay +2m",
    "increment death",
    "45+30 classical",
]

_MOVE_COMMENTARY = [
    "Time trouble hit right after the opening.",
    "Both players looked happier with the structure than the clock.",
    "Someone said the rooks were “talking” across the fifth rank.",
    "The post-game started before the scoresheets were signed.",
    "Engine later shrugged at the whole middlegame plan.",
    "Crowd noise made half the moves unreadable on the stream.",
    "White spent ages on a knight hop everyone had already predicted.",
    "Black’s king looked drafty for most of the slice we saw.",
    "Someone premoved three times in a row and immediately regretted it.",
    "The arbiters whispered about illegal move claims that went nowhere.",
    "A phone buzzed; both players flinched like it was a takeback offer.",
    "The increment meant nobody trusted anyone’s flag story.",
    "Table next door ended in a fist-bump draw that stole the room’s attention.",
    "One coach wrote “??” on a napkin and slid it under the table.",
    "The DGT cable looked suspiciously loose the whole round.",
    "Half the audience was still arguing about yesterday’s tie-break rules.",
    "Someone tried to reconstruct the order of captures from memory and gave up.",
    "The king walk looked heroic on the demo board and silly on the live board.",
]

_KIBITZ = [
    "Notebook margin: “remember the …6 break” — rest smudged.",
    "Friend texted a diagram that did not match anyone’s memory.",
    "TD walked past twice; nobody adjusted the pieces.",
    "Coffee arrived during the sharpest tactics; opinions differ on what happened next.",
    "Two spectators argued whether the scoresheet said Bc4 or Bb5.",
    "Discord clip title promised “CRAZY BISHOP TRAP” but the audio was just wind.",
    "Someone’s phone autocorrected Nf3 to “Nf3!!!” in the group chat forever.",
    "The pairing sheet had a handwritten arrow nobody could parse.",
    "A junior loudly recited opening theory from the wrong decade.",
    "Lost scoresheet rumor: half the room thinks the game started 1…c5.",
]

_RUMOR_TEMPLATES = [
    "Side chatter insisted someone had tried {uci} in a skittles copy.",
    "A kid claimed the sheet showed {uci}; nobody verified it.",
    "Half the bar thought {uci} had been played; the other half rolled their eyes.",
    "A mod pinned “{uci}??” in chat before deleting it two seconds later.",
    "The relay operator swore they saw {uci} on the feed; the PGN never updated.",
    "Someone’s cousin’s coach said {uci} was “basically forced” in the skittles line.",
]

_STACK_SEPARATOR = (
    "\n\n──────────── unrelated chess clutter (different boards / eras) ────────────\n\n"
)

_POST_OBS_HEADER = "\n\nBelow the diagram:\n"

_LOBBY_LINES = [
    "chat: anyone else lagging on board 12?",
    "chat: bro just hung a queen 💀 (wrong board)",
    "chat: engine says +0.3 which means we're all clueless",
    "chat: rematch??",
    "chat: TD said touch-move but the clip is ambiguous",
    "chat: I had the same position in a puzzle and picked the wrong rook",
    "chat: is this that Berlin thing or not",
    "chat: my lichess elo is fake anyway",
    "chat: who pressed the clock twice",
    "chat: pretty sure that was illegal but nobody wants drama",
    "chat: link to previous round PGN?",
    "chat: stream delay is killing me",
]


def _random_uci(rng: random.Random) -> str:
    a, b = rng.sample(range(64), 2)
    return chess.Move(a, b).uci()


def _chat_noise_segment(
    rng: random.Random,
    *,
    min_games: int,
    max_games: int,
    past_san_min_plies: int,
    past_san_max_plies: int,
    rumor_rate: float,
    lobby_lines: int,
) -> str:
    lo_g = min(min_games, max_games)
    hi_g = max(min_games, max_games)
    n_games = rng.randint(lo_g, hi_g)
    lines: list[str] = ["Past games:", ""]

    span_lo = min(past_san_min_plies, past_san_max_plies)
    span_hi = max(past_san_min_plies, past_san_max_plies)

    for i in range(n_games):
        label = rng.choice(_GAME_LABELS)
        fen = rng.choice(_START_FENS)
        san_line = _random_legal_line(rng, fen, min_plies=span_lo, max_plies=span_hi)
        lines.append(f"Game {i + 1} ({label}): {san_line}")
        lines.append(rng.choice(_MOVE_COMMENTARY))
        if rng.random() < rumor_rate:
            lines.append(rng.choice(_RUMOR_TEMPLATES).format(uci=_random_uci(rng)))
        lines.append("")

    lines.append(rng.choice(_KIBITZ))
    lines.append("")

    if lobby_lines > 0:
        lines.append("Lobby / stream (unrelated boards):")
        lines.append("")
        for _ in range(lobby_lines):
            lines.append(rng.choice(_LOBBY_LINES))
        lines.append("")

    return "\n".join(lines)


def _random_legal_line(
    rng: random.Random,
    start_fen: str,
    *,
    min_plies: int,
    max_plies: int,
) -> str:
    board = chess.Board(start_fen)
    moves: list[chess.Move] = []
    hi = max(min_plies, max_plies)
    lo = min(min_plies, max_plies)
    n = rng.randint(lo, hi)
    for _ in range(n):
        leg = list(board.legal_moves)
        if not leg:
            break
        moves.append(rng.choice(leg))
        board.push(moves[-1])
    root = chess.Board(start_fen)
    if not moves:
        return "(fragment too short to replay)"
    try:
        return root.variation_san(moves)
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return " ".join(m.uci() for m in moves)


def prefix_chess_observation_with_chat_context(
    observation: str,
    rng: random.Random,
    *,
    min_games: int = 2,
    max_games: int = 4,
    past_san_min_plies: int = 4,
    past_san_max_plies: int = 22,
    rumor_rate: float = 0.45,
    lobby_lines: int = 0,
    prefix_stack: int = 1,
    post_observation_lines: int = 0,
) -> str:
    """Prepend plausible past games + commentary, then ``Current game:`` + real observation.

    Past lines are syntactically chess-like but unrelated to the benchmark position.
    No “ignore this” instructions — only section headers.

    ``lobby_lines``: if >0, append that many synthetic chat/kibitz lines before
    ``Current game:`` (unrelated noise).

    ``prefix_stack``: repeat the entire noise block (fresh random draws) this many
    times before ``Current game:``, separated by a visible divider.

    ``post_observation_lines``: append this many extra noise lines *after* the real
    observation (still inside the observation blob the model receives before the question).
    """
    stack = max(1, int(prefix_stack))
    segments = [
        _chat_noise_segment(
            rng,
            min_games=min_games,
            max_games=max_games,
            past_san_min_plies=past_san_min_plies,
            past_san_max_plies=past_san_max_plies,
            rumor_rate=rumor_rate,
            lobby_lines=lobby_lines,
        )
        for _ in range(stack)
    ]
    head = _STACK_SEPARATOR.join(segments)
    out = head + "\n\nCurrent game:\n\n" + observation

    n_tail = max(0, int(post_observation_lines))
    if n_tail > 0:
        out += _POST_OBS_HEADER
        tail_pool = _LOBBY_LINES + _MOVE_COMMENTARY + _KIBITZ
        for _ in range(n_tail):
            if rng.random() < 0.35:
                out += rng.choice(_RUMOR_TEMPLATES).format(uci=_random_uci(rng)) + "\n"
            else:
                out += rng.choice(tail_pool) + "\n"

    return out
