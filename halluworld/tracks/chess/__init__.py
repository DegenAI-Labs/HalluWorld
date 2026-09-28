"""Chess track: chess-variant environments, probes, serializers, evaluators.

Everything specific to the chess track lives under this package:

    rules/          board rule engines (atomic, old-bishop), Stockfish, FEN pools
    envs/           the three chess environments
    probes/         standard, atomic, and old-bishop probe classes
    serializers.py  board / FEN / SAN-history rendering
    evaluators.py   yes-no, integer, legal-UCI-set, piece-name graders
    battery.py      the runnable probe battery
    question_set.py frozen question-set export (StubLM, no API calls)

Importing this package requires ``python-chess``, which is an optional extra:

    pip install halluworld[chess]

That is deliberate. python-chess is GPL-3.0-or-later, so it is kept out of the
core dependency set; see LICENSE. Because chess code
now lives entirely under this package, the optionality is structural -- the
gridworld and InNav tracks simply never import it, and no try/except guard is
needed anywhere.
"""

from halluworld.tracks.chess.evaluators import (
    ChessBestMoveEvaluator,
    ChessIntegerEvaluator,
    ChessLegalUciSetEvaluator,
    ChessPieceNameEvaluator,
    ChessYesNoEvaluator,
    ChessYesNoIDKEvaluator,
    ChessYesNoStrictEvaluator,
)
from halluworld.tracks.chess.envs.atomic_chess_env import (
    AtomicChessEnv,
    make_atomic_chess_env,
)
from halluworld.tracks.chess.envs.chess_env import (
    ChessEnv,
    load_lichess_puzzle_fens,
    load_lichess_puzzles,
    make_chess_env,
)
from halluworld.tracks.chess.envs.old_bishop_chess_env import (
    OldBishopChessEnv,
    make_old_bishop_chess_env,
)
from halluworld.tracks.chess.probes.atomic import AtomicHypotheticalPieceAfterUciProbe
from halluworld.tracks.chess.probes.old_bishop import (
    OldBishopBishopAttackSquareProbe,
    OldBishopFideIllegalBishopUciProbe,
    OldBishopLegalAnyUciProbe,
    OldBishopLegalBishopUciProbe,
    OldBishopLegalMoveProbe,
    OldBishopLegalQuietUciProbe,
    OldBishopQueenLongSlideLegalProbe,
    OldBishopRulesFactProbe,
)
from halluworld.tracks.chess.probes.standard import (
    ChessAfterMoveUndefendedCountProbe,
    ChessAttackerProbe,
    ChessBestMoveProbe,
    ChessCanCaptureProbe,
    ChessCaptureCountProbe,
    ChessCastlingLegalityProbe,
    ChessCastlingRightsHistoryProbe,
    ChessConflictingPromptProbe,
    ChessDefendedProbe,
    ChessEnPassantProbe,
    ChessHangingProbe,
    ChessHiddenSideCaptureStatsProbe,
    ChessHiddenSquareProbe,
    ChessHistoryReadoutProbe,
    ChessHypotheticalProbe,
    ChessLegalMoveProbe,
    ChessMateInOneProbe,
    ChessMateStalemateConfusionProbe,
    ChessPieceCountProbe,
    ChessPiecePresenceProbe,
    ChessPinProbe,
    ChessSanLegalContinuationProbe,
    ChessTwoStepHangingProbe,
)
from halluworld.tracks.chess.serializers import ChessSerializer

__all__ = [
    # environments
    "ChessEnv",
    "make_chess_env",
    "load_lichess_puzzle_fens",
    "load_lichess_puzzles",
    "AtomicChessEnv",
    "make_atomic_chess_env",
    "OldBishopChessEnv",
    "make_old_bishop_chess_env",
    # serializer
    "ChessSerializer",
    # standard probes
    "ChessAfterMoveUndefendedCountProbe",
    "ChessAttackerProbe",
    "ChessBestMoveProbe",
    "ChessCanCaptureProbe",
    "ChessCaptureCountProbe",
    "ChessCastlingLegalityProbe",
    "ChessCastlingRightsHistoryProbe",
    "ChessConflictingPromptProbe",
    "ChessDefendedProbe",
    "ChessEnPassantProbe",
    "ChessHangingProbe",
    "ChessHiddenSideCaptureStatsProbe",
    "ChessHiddenSquareProbe",
    "ChessHistoryReadoutProbe",
    "ChessHypotheticalProbe",
    "ChessLegalMoveProbe",
    "ChessMateInOneProbe",
    "ChessMateStalemateConfusionProbe",
    "ChessPieceCountProbe",
    "ChessPiecePresenceProbe",
    "ChessPinProbe",
    "ChessSanLegalContinuationProbe",
    "ChessTwoStepHangingProbe",
    # variant probes
    "AtomicHypotheticalPieceAfterUciProbe",
    "OldBishopBishopAttackSquareProbe",
    "OldBishopFideIllegalBishopUciProbe",
    "OldBishopLegalAnyUciProbe",
    "OldBishopLegalBishopUciProbe",
    "OldBishopLegalMoveProbe",
    "OldBishopLegalQuietUciProbe",
    "OldBishopQueenLongSlideLegalProbe",
    "OldBishopRulesFactProbe",
    # evaluators
    "ChessBestMoveEvaluator",
    "ChessIntegerEvaluator",
    "ChessLegalUciSetEvaluator",
    "ChessPieceNameEvaluator",
    "ChessYesNoEvaluator",
    "ChessYesNoIDKEvaluator",
    "ChessYesNoStrictEvaluator",
]
