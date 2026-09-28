"""Unit tests for the new chess probes, evaluators, serializer helpers, and
the Lichess puzzle cache layer."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import chess

from halluworld.tracks.chess.envs.chess_env import ChessEnv, _coerce_record, load_lichess_puzzles
from halluworld.tracks.chess import ChessIntegerEvaluator, ChessLegalUciSetEvaluator, ChessPieceNameEvaluator, ChessYesNoEvaluator, ChessYesNoIDKEvaluator
from halluworld.lm.base import LMResponse
from halluworld.tracks.chess import ChessAfterMoveUndefendedCountProbe, ChessAttackerProbe, ChessCanCaptureProbe, ChessCaptureCountProbe, ChessCastlingLegalityProbe, ChessCastlingRightsHistoryProbe, ChessConflictingPromptProbe, ChessDefendedProbe, ChessEnPassantProbe, ChessHangingProbe, ChessHiddenSideCaptureStatsProbe, ChessHiddenSquareProbe, ChessHistoryReadoutProbe, ChessHypotheticalProbe, ChessMateInOneProbe, ChessMateStalemateConfusionProbe, ChessPieceCountProbe, ChessPinProbe, ChessSanLegalContinuationProbe, ChessTwoStepHangingProbe
from halluworld.tracks.chess.probes.standard import _is_hanging, _newly_undefended_after_hypothetical_moves, _undefended_piece_count
from halluworld.tracks.chess.serializers import FenDisplayConfig, board_to_grid_text, configure_observation_fen_display, derive_display_fen, parse_piece_name, render_full, render_history


def _env_for(fen: str) -> ChessEnv:
    env = ChessEnv(fens=[fen])
    env.reset()
    return env


def _resp(text: str) -> LMResponse:
    return LMResponse(text=text, model="test")


# --------------------------------------------------------------------------- #
# Perceptual probes                                                           #
# --------------------------------------------------------------------------- #


class PerceptualProbeTests(unittest.TestCase):
    def test_attacker_probe_known_position(self):
        # Simple endgame: white king e1, white knight on f3 attacks e5/g5/etc.
        env = _env_for("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")
        rng = random.Random(0)
        # Force positive-only behaviour via positive_rate=1.0
        probe = ChessAttackerProbe(positive_rate=1.0, rng=rng)
        for _ in range(20):
            res = probe.generate(env)
            self.assertIsInstance(res.ground_truth, bool)
            sq = chess.parse_square(res.metadata["square"])
            color = chess.WHITE if res.metadata["attacker_color"] == "white" else chess.BLACK
            self.assertEqual(res.ground_truth, env.board.is_attacked_by(color, sq))

    def test_pin_probe_constructed_pin(self):
        # Black king on e8, black rook on e7, white queen on e1: rook is pinned.
        env = _env_for("4k3/4r3/8/8/8/8/8/4Q1K1 b - - 0 1")
        rng = random.Random(0)
        probe = ChessPinProbe(positive_rate=1.0, rng=rng)
        # Try several times — when probe picks the rook, gt should be True.
        seen_pinned_true = False
        for _ in range(40):
            res = probe.generate(env)
            if res.metadata["square"] == "e7":
                self.assertTrue(res.ground_truth)
                seen_pinned_true = True
                break
        self.assertTrue(seen_pinned_true, "Rook on e7 should be reported as pinned at least once")

    def test_piece_count_probe(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessPieceCountProbe(rng=random.Random(0))
        for _ in range(20):
            res = probe.generate(env)
            color = chess.WHITE if res.metadata["color"] == "white" else chess.BLACK
            pt_name = res.metadata["piece_type"]
            pt = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
                  "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}[pt_name]
            expected = len(env.board.pieces(pt, color))
            self.assertEqual(res.ground_truth, expected)

    def test_defended_probe_structure(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessDefendedProbe(rng=random.Random(0))
        res = probe.generate(env)
        self.assertEqual(res.probe_type, "chess_defended")
        self.assertIn(res.ground_truth, (True, False))


# --------------------------------------------------------------------------- #
# Memory probes                                                               #
# --------------------------------------------------------------------------- #


class MemoryProbeTests(unittest.TestCase):
    def test_history_readout_uses_override_and_correct_gt(self):
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(7)
        probe = ChessHistoryReadoutProbe(n_plies=4, rng=rng)
        res = probe.generate(env)
        self.assertIsNotNone(res.observation_override)
        self.assertNotIn("Board ('.' means empty):", res.observation_override or "")
        self.assertIn("Starting FEN:", res.observation_override or "")

        # Recompute the ground truth by replaying the same UCI list.
        board = chess.Board(chess.STARTING_FEN)
        for uci in res.metadata["moves_uci"]:
            board.push(chess.Move.from_uci(uci))
        sq = chess.parse_square(res.metadata["square"])
        piece = board.piece_at(sq)
        expected = "empty" if piece is None else {
            "P": "white pawn", "N": "white knight", "B": "white bishop",
            "R": "white rook", "Q": "white queen", "K": "white king",
            "p": "black pawn", "n": "black knight", "b": "black bishop",
            "r": "black rook", "q": "black queen", "k": "black king",
        }[piece.symbol()]
        self.assertEqual(res.ground_truth, expected)

    def test_capture_count_probe_matches_replay(self):
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(11)
        probe = ChessCaptureCountProbe(n_plies=10, rng=rng)
        res = probe.generate(env)
        board = chess.Board(chess.STARTING_FEN)
        captures = 0
        for uci in res.metadata["moves_uci"]:
            mv = chess.Move.from_uci(uci)
            if board.is_capture(mv):
                captures += 1
            board.push(mv)
        self.assertEqual(res.ground_truth, captures)

    def test_castling_rights_history_probe(self):
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(3)
        probe = ChessCastlingRightsHistoryProbe(n_plies=4, rng=rng)
        res = probe.generate(env)
        board = chess.Board(chess.STARTING_FEN)
        for uci in res.metadata["moves_uci"]:
            board.push(chess.Move.from_uci(uci))
        color = chess.WHITE if res.metadata["color"] == "white" else chess.BLACK
        if res.metadata["side"] == "kingside":
            expected = board.has_kingside_castling_rights(color)
        else:
            expected = board.has_queenside_castling_rights(color)
        self.assertEqual(res.ground_truth, expected)

    def test_hidden_side_capture_stats_matches_replay(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessHiddenSideCaptureStatsProbe(
            min_plies=25, max_plies=30, rng=random.Random(21), capture_bias=0.35
        )
        for _ in range(40):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            board = chess.Board(chess.STARTING_FEN)
            white_caps = 0
            black_lost = 0
            for uci in res.metadata["moves_uci"]:
                mv = chess.Move.from_uci(uci)
                if board.is_capture(mv):
                    if board.turn == chess.WHITE:
                        white_caps += 1
                    if board.is_en_passant(mv):
                        ep_sq = mv.to_square + (-8 if board.turn == chess.WHITE else 8)
                        victim = board.piece_at(ep_sq)
                    else:
                        victim = board.piece_at(mv.to_square)
                    if victim is not None and victim.color == chess.BLACK:
                        black_lost += 1
                board.push(mv)
            kind = res.metadata.get("question_kind")
            if kind == "white_captures":
                self.assertEqual(res.ground_truth, white_caps)
            else:
                self.assertEqual(res.ground_truth, black_lost)
            return
        self.fail("hidden side capture probe kept skipping")

    def test_after_move_undefended_count_matches_helper(self):
        env = _env_for("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        probe = ChessAfterMoveUndefendedCountProbe(
            rng=random.Random(5), mode="post", hypothesis_plies=1, max_tries=60
        )
        for _ in range(50):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            uci = res.metadata["setup_uci"]
            assert uci is not None
            b = chess.Board(env.board.fen())
            color = b.turn
            b.push(chess.Move.from_uci(uci))
            self.assertEqual(res.ground_truth, _undefended_piece_count(b, color))
            return
        self.fail("after-move undefended probe skipped unexpectedly")

    def test_after_move_undefended_delta_one_ply_matches_helper(self):
        env = _env_for("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        probe = ChessAfterMoveUndefendedCountProbe(
            rng=random.Random(5), mode="delta", hypothesis_plies=1, max_tries=60
        )
        for _ in range(50):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            uci = res.metadata["setup_uci"]
            assert uci is not None
            b0 = chess.Board(env.board.fen())
            color = b0.turn
            mv = chess.Move.from_uci(uci)
            expected = _newly_undefended_after_hypothetical_moves(b0, [mv], color)
            self.assertEqual(res.ground_truth, expected)
            return
        self.fail("after-move delta probe skipped unexpectedly")

    def test_after_move_undefended_delta_two_ply_matches_helper(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessAfterMoveUndefendedCountProbe(
            rng=random.Random(9), mode="delta", hypothesis_plies=2, max_tries=80
        )
        for _ in range(80):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            ucis = res.metadata["moves_uci"]
            self.assertEqual(len(ucis), 2)
            b0 = chess.Board(chess.STARTING_FEN)
            color = b0.turn
            moves = [chess.Move.from_uci(ucis[0]), chess.Move.from_uci(ucis[1])]
            expected = _newly_undefended_after_hypothetical_moves(b0, moves, color)
            self.assertEqual(res.ground_truth, expected)
            return
        self.fail("after-move two-ply delta probe kept skipping")

    def test_after_move_undefended_post_two_ply_matches_helper(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessAfterMoveUndefendedCountProbe(
            rng=random.Random(11), mode="post", hypothesis_plies=2, max_tries=80
        )
        for _ in range(80):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            ucis = res.metadata["moves_uci"]
            b0 = chess.Board(chess.STARTING_FEN)
            color = b0.turn
            b2 = b0.copy(stack=False)
            b2.push_uci(ucis[0])
            b2.push_uci(ucis[1])
            self.assertEqual(res.ground_truth, _undefended_piece_count(b2, color))
            return
        self.fail("after-move post two-ply probe kept skipping")

    def test_two_step_hanging_ground_truth(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessTwoStepHangingProbe(rng=random.Random(8), max_tries=80)
        for _ in range(60):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            b = chess.Board(chess.STARTING_FEN)
            u1, u2 = res.metadata["moves_uci"]
            b.push(chess.Move.from_uci(u1))
            b.push(chess.Move.from_uci(u2))
            sq = chess.parse_square(res.metadata["square"])
            self.assertEqual(res.ground_truth, _is_hanging(b, sq))
            return
        self.fail("two-step hanging probe kept skipping")

    def test_mate_stalemate_confusion_ground_truth_is_no(self):
        probe = ChessMateStalemateConfusionProbe(
            rng=random.Random(0), max_attempts=200, max_plies=160, capture_bias=0.42
        )
        env = _env_for(chess.STARTING_FEN)
        for _ in range(80):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            self.assertIs(res.ground_truth, False)
            self.assertIn(res.metadata["true_terminal"], ("checkmate", "stalemate"))
            b = chess.Board()
            for uci in res.metadata["moves_uci"]:
                b.push_uci(uci)
            if res.metadata["true_terminal"] == "checkmate":
                self.assertTrue(b.is_checkmate())
            else:
                self.assertTrue(b.is_stalemate())
            self.assertNotIn("side to move has no legal moves", res.question)
            return
        self.fail("mate/stalemate confusion probe kept skipping")


# --------------------------------------------------------------------------- #
# Dynamics probes                                                             #
# --------------------------------------------------------------------------- #


class DynamicsProbeTests(unittest.TestCase):
    def test_castling_into_check_is_illegal(self):
        # White king e1, white rook h1, black rook on f8 attacks f1.
        # Castling rights present (KQ) but kingside castling moves king through
        # f1 which is attacked → illegal. Queenside also illegal because b1/c1
        # are not attacked but castling rights only permit it if the path is
        # clear — let's leave queens/knights out and rely on f8 rook.
        fen = "4k1r1/8/8/8/8/8/8/4K2R w K - 0 1"
        env = _env_for(fen)
        # Force a kingside castling question.
        rng = random.Random(0)
        for _ in range(40):
            res = ChessCastlingLegalityProbe(positive_rate=0.0, rng=rng).generate(env)
            if res.metadata["direction"] == "kingside":
                self.assertFalse(res.ground_truth, "kingside castling through check must be illegal")
                return
        self.fail("never sampled kingside castling — RNG bug?")

    def test_castling_legal_clean_position(self):
        fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
        env = _env_for(fen)
        # python-chess: white can castle kingside legally here.
        rng = random.Random(0)
        for _ in range(40):
            res = ChessCastlingLegalityProbe(positive_rate=1.0, rng=rng).generate(env)
            if res.metadata["direction"] == "kingside":
                self.assertTrue(res.ground_truth)
                return
        self.fail("never sampled kingside castling")

    def test_en_passant_probe_finds_ep_setup(self):
        # White pawn on e5, black pawn on d7: black playing d7d5 enables EP
        # for white's e5 pawn. Set side-to-move to black so the probe finds
        # the double push.
        fen = "4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1"
        env = _env_for(fen)
        rng = random.Random(0)
        probe = ChessEnPassantProbe(positive_rate=1.0, rng=rng)
        res = probe.generate(env)
        self.assertFalse(res.metadata.get("skipped", False))
        self.assertEqual(res.metadata["setup_move"], "d7d5")
        self.assertTrue(res.ground_truth)
        # Observation override should describe the post-setup position.
        self.assertIn("Hypothetical position after black plays d7d5", res.observation_override or "")

    def test_mate_in_one_back_rank(self):
        # Classic back-rank mate: white queen on e1, black king on g8 with
        # f7/g7/h7 pawns and white kingside structure. Qe1-e8# covers f8
        # and h8 along rank 8, and the king is unable to capture from g8.
        fen = "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1"
        env = _env_for(fen)
        res = ChessMateInOneProbe(rng=random.Random(0)).generate(env)
        self.assertTrue(res.ground_truth)
        self.assertEqual(res.metadata["mate_move"], "e1e8")

    def test_hanging_probe_isolated_piece(self):
        # White knight on e4 attacked by black pawn on f5, undefended → hanging.
        fen = "4k3/8/8/5p2/4N3/8/8/4K3 w - - 0 1"
        env = _env_for(fen)
        rng = random.Random(0)
        probe = ChessHangingProbe(positive_rate=1.0, rng=rng)
        for _ in range(20):
            res = probe.generate(env)
            if res.metadata["square"] == "e4":
                self.assertTrue(res.ground_truth)
                return
        self.fail("never sampled e4 knight")

    def test_can_capture_probe(self):
        # White rook on a1; black queen on a8 — rook can capture queen.
        fen = "q3k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
        env = _env_for(fen)
        rng = random.Random(0)
        probe = ChessCanCaptureProbe(positive_rate=1.0, rng=rng)
        res = probe.generate(env)
        self.assertTrue(res.ground_truth)
        self.assertEqual(res.metadata["square"], "a8")

    def test_can_capture_ghost_negative_implies_no_legal_capture(self):
        # Position where a pseudo-legal capture to an opponent square exists but is illegal.
        fen = "r1b2k1r/p2p3p/n7/1p2PpPn/qb2Pp2/2pQ1K2/P1P4P/RN3BNR w - - 0 21"
        env = _env_for(fen)
        probe = ChessCanCaptureProbe(
            positive_rate=0.0,
            prefer_ghost_captures=True,
            rng=random.Random(0),
        )
        res = probe.generate(env)
        self.assertFalse(res.metadata.get("skipped", False))
        self.assertTrue(res.metadata.get("ghost_negative"))
        self.assertFalse(res.ground_truth)

    def test_hypothetical_in_check(self):
        # Constructed: white queen on h5, black king on e8 — Qh5xe8+? No, e8 is
        # not a queen-attack square. Instead use a line: after Qh5-e2 black king
        # is not in check; after Qh5-h8 it is. We just need the probe to be
        # consistent with python-chess on an arbitrary setup move.
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(0)
        probe = ChessHypotheticalProbe(predicate="in_check", rng=rng)
        for _ in range(10):
            res = probe.generate(env)
            board = chess.Board(chess.STARTING_FEN)
            board.push(chess.Move.from_uci(res.metadata["setup_move"]))
            self.assertEqual(res.ground_truth, board.is_check())


# --------------------------------------------------------------------------- #
# Belief probes + IDK evaluator                                               #
# --------------------------------------------------------------------------- #


class BeliefProbeTests(unittest.TestCase):
    def test_hidden_square_probe_idk_when_target_hidden(self):
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(0)
        # Force questions to always target the hidden square.
        probe = ChessHiddenSquareProbe(n_hidden=1, unmasked_rate=0.0, rng=rng)
        res = probe.generate(env)
        self.assertEqual(res.ground_truth, "idk")
        self.assertTrue(res.metadata["asked_about_hidden"])
        self.assertIn("?", res.observation_override or "")

    def test_hidden_square_probe_known_when_target_visible(self):
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(0)
        probe = ChessHiddenSquareProbe(n_hidden=1, unmasked_rate=1.0, rng=rng)
        res = probe.generate(env)
        self.assertIn(res.ground_truth, (True, False))

    def test_yes_no_idk_evaluator_scoring(self):
        ev = ChessYesNoIDKEvaluator()
        from halluworld.probe import ProbeResult
        # ground truth idk, model says idk → 1.0
        pr = ProbeResult("chess_hidden_square", "Q", "idk")
        self.assertEqual(ev.evaluate(_resp("idk"), pr).score, 1.0)
        # gt idk, model says yes → 0.0 (overconfident)
        self.assertEqual(ev.evaluate(_resp("yes"), pr).score, 0.0)
        # gt True, model says yes → 1.0
        pr2 = ProbeResult("chess_hidden_square", "Q", True)
        self.assertEqual(ev.evaluate(_resp("yes"), pr2).score, 1.0)
        # gt True, model says idk → 0.0 (abstain when known)
        self.assertEqual(ev.evaluate(_resp("idk"), pr2).score, 0.0)
        # gt True, model says no → 0.0
        self.assertEqual(ev.evaluate(_resp("no"), pr2).score, 0.0)

    def test_conflicting_prompt_probe_grounded_in_board(self):
        env = _env_for(chess.STARTING_FEN)
        rng = random.Random(0)
        probe = ChessConflictingPromptProbe(rng=rng, lie_kind="piece")
        for _ in range(10):
            res = probe.generate(env)
            sq = chess.parse_square(res.metadata["square"])
            actual = env.board.piece_at(sq)
            actual_name = {
                "P": "white pawn", "N": "white knight", "B": "white bishop",
                "R": "white rook", "Q": "white queen", "K": "white king",
                "p": "black pawn", "n": "black knight", "b": "black bishop",
                "r": "black rook", "q": "black queen", "k": "black king",
            }.get(actual.symbol() if actual else "", "empty")
            expected_truth = (res.metadata["asked_label"] == actual_name)
            self.assertEqual(res.ground_truth, expected_truth)

    def test_conflicting_prompt_fen_transpose_injects_bogus_fen(self):
        fen = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1"
        env = _env_for(fen)
        probe = ChessConflictingPromptProbe(rng=random.Random(3), lie_kind="fen_transpose")
        for _ in range(30):
            res = probe.generate(env)
            bogus = res.metadata.get("bogus_fen")
            if bogus and bogus != res.metadata["true_fen"]:
                self.assertIn("FEN:", res.question)
                self.assertIn(bogus, res.question)
                self.assertIn("diagram", res.question.lower())
                return
        self.fail("expected a differing bogus FEN from transpose")

    def test_derive_display_fen_flip_turn(self):
        fen = chess.Board().fen()
        flipped = derive_display_fen(fen, FenDisplayConfig(mode="flip_turn", seed=0))
        self.assertNotEqual(flipped.split()[1], fen.split()[1])

    def test_render_history_respects_global_fen_display(self):
        configure_observation_fen_display(None)
        try:
            configure_observation_fen_display(FenDisplayConfig(mode="startpos", seed=0))
            text = render_history(
                "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1",
                ["e2e4"],
            )
            self.assertIn(f"Starting FEN: {chess.STARTING_FEN}", text)
        finally:
            configure_observation_fen_display(None)


class SanLegalContinuationProbeTests(unittest.TestCase):
    def test_san_legal_move_metadata_matches_engine(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessSanLegalContinuationProbe(n_plies=14, min_plies=4, rng=random.Random(7))
        seen = 0
        for _ in range(50):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            seen += 1
            b = chess.Board(chess.STARTING_FEN)
            for u in res.metadata["moves_uci"]:
                b.push_uci(u)
            legal_engine = sorted(m.uci() for m in b.legal_moves)
            self.assertEqual(res.metadata["legal_ucis"], legal_engine)
            self.assertEqual(res.metadata.get("reply_constraint"), "any")
            self.assertIn("Moves played:", res.observation_override or "")
            self.assertNotIn("Board ('", res.observation_override or "")
        self.assertGreater(seen, 0)

    def test_san_legal_move_reply_constraint_non_capture(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessSanLegalContinuationProbe(
            n_plies=14,
            min_plies=4,
            rng=random.Random(7),
            reply_constraint="non_capture",
        )
        seen = 0
        for _ in range(80):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            seen += 1
            self.assertEqual(res.metadata.get("reply_constraint"), "non_capture")
            b = chess.Board(chess.STARTING_FEN)
            for u in res.metadata["moves_uci"]:
                b.push_uci(u)
            expected = sorted(
                m.uci() for m in b.legal_moves if not b.is_capture(m)
            )
            self.assertEqual(res.metadata["legal_ucis"], expected)
            self.assertIn("must not capture", res.question.lower())
            for uci in res.metadata["legal_ucis"]:
                mv = chess.Move.from_uci(uci)
                self.assertFalse(b.is_capture(mv))
        self.assertGreater(seen, 0)

    def test_san_legal_move_reply_constraint_quiet_non_promotion(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessSanLegalContinuationProbe(
            n_plies=14,
            min_plies=4,
            rng=random.Random(11),
            reply_constraint="quiet_non_promotion",
        )
        seen = 0
        for _ in range(120):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            seen += 1
            self.assertEqual(res.metadata.get("reply_constraint"), "quiet_non_promotion")
            b = chess.Board(chess.STARTING_FEN)
            for u in res.metadata["moves_uci"]:
                b.push_uci(u)
            quiet_np = []
            for m in b.legal_moves:
                if b.is_capture(m):
                    continue
                tmp = b.copy(stack=False)
                tmp.push(m)
                if tmp.is_check():
                    continue
                if m.promotion is not None:
                    continue
                quiet_np.append(m)
            expected = sorted(m.uci() for m in quiet_np)
            self.assertEqual(res.metadata["legal_ucis"], expected)
            self.assertIn("promotion", res.question.lower())
        self.assertGreater(seen, 0)

    def test_san_omit_side_to_move_startpos(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessSanLegalContinuationProbe(
            n_plies=12,
            min_plies=4,
            rng=random.Random(3),
            observation_mode="san_only",
            replay_from="startpos",
            omit_side_to_move=True,
        )
        for _ in range(50):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            obs = res.observation_override or ""
            self.assertNotIn("It is now", obs)
            self.assertTrue(res.metadata.get("omit_side_to_move"))
            return
        self.fail("never got a non-skipped san_only sample")

    def test_san_max_constrained_replies_retries(self):
        env = _env_for(chess.STARTING_FEN)
        probe = ChessSanLegalContinuationProbe(
            n_plies=20,
            min_plies=6,
            rng=random.Random(0),
            reply_constraint="any",
            max_constrained_replies=2,
            max_replay_attempts=40,
        )
        seen = 0
        for _ in range(60):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            seen += 1
            self.assertLessEqual(len(res.metadata["legal_ucis"]), 2)
        self.assertGreater(seen, 0)

    def test_san_only_requires_startpos_constructor(self):
        with self.assertRaises(ValueError):
            ChessSanLegalContinuationProbe(observation_mode="san_only", replay_from="env")

    def test_san_prepend_board_grid_includes_ascii_board(self):
        configure_observation_fen_display(None)
        env = _env_for(chess.STARTING_FEN)
        probe = ChessSanLegalContinuationProbe(
            n_plies=12,
            min_plies=4,
            rng=random.Random(2),
            observation_mode="san_only",
            replay_from="startpos",
            prepend_terminal_board_grid=True,
        )
        for _ in range(40):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            obs = res.observation_override or ""
            self.assertIn("Board ('", obs)
            self.assertIn("FEN:", obs)
            self.assertTrue(res.metadata.get("prepend_terminal_board_grid"))
            return
        self.fail("never got non-skipped san_only with grid prepend")

    def test_san_only_observation_has_moves_no_starting_fen_line(self):
        env = _env_for("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        probe = ChessSanLegalContinuationProbe(
            n_plies=10,
            min_plies=4,
            rng=random.Random(2),
            observation_mode="san_only",
            replay_from="startpos",
        )
        for _ in range(40):
            res = probe.generate(env)
            if res.metadata.get("skipped"):
                continue
            obs = res.observation_override or ""
            self.assertNotIn("Starting FEN:", obs)
            self.assertIn("Moves played:", obs)
            self.assertEqual(res.metadata.get("replay_from"), "startpos")
            self.assertEqual(res.metadata.get("observation_mode"), "san_only")
            self.assertTrue(str(res.metadata.get("san_line", "")).lstrip().startswith("1."))
            return
        self.fail("never got a non-skipped san_only sample")

    def test_legal_uci_set_evaluator(self):
        from halluworld.probe import ProbeResult

        ev = ChessLegalUciSetEvaluator()
        pr = ProbeResult(
            probe_type="chess_san_legal_move",
            question="dummy",
            ground_truth=["e2e3", "e2e4"],
            metadata={"legal_ucis": ["e2e3", "e2e4"]},
        )
        self.assertEqual(ev.evaluate(_resp("e2e4"), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("My move: e2e4"), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("e2e5"), pr).score, 0.0)


# --------------------------------------------------------------------------- #
# Evaluators                                                                  #
# --------------------------------------------------------------------------- #


class EvaluatorTests(unittest.TestCase):
    def test_integer_evaluator(self):
        from halluworld.probe import ProbeResult
        ev = ChessIntegerEvaluator()
        pr = ProbeResult("chess_piece_count", "Q", 8)
        self.assertEqual(ev.evaluate(_resp("8"), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("the answer is 8 pawns"), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("seven"), pr).score, 0.0)
        self.assertEqual(ev.evaluate(_resp("9"), pr).score, 0.0)

    def test_piece_name_evaluator(self):
        from halluworld.probe import ProbeResult
        ev = ChessPieceNameEvaluator()
        pr = ProbeResult("chess_history_readout", "Q", "white knight")
        self.assertEqual(ev.evaluate(_resp("white knight"), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("It's a white knight."), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("white pawn"), pr).score, 0.0)
        pr2 = ProbeResult("chess_history_readout", "Q", "empty")
        self.assertEqual(ev.evaluate(_resp("empty"), pr2).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("nothing is there"), pr2).score, 1.0)

    def test_yes_no_evaluator_unchanged(self):
        from halluworld.probe import ProbeResult
        ev = ChessYesNoEvaluator()
        pr = ProbeResult("chess_attacker", "Q", True)
        self.assertEqual(ev.evaluate(_resp("yes"), pr).score, 1.0)
        self.assertEqual(ev.evaluate(_resp("no"), pr).score, 0.0)


# --------------------------------------------------------------------------- #
# Serializer helpers                                                          #
# --------------------------------------------------------------------------- #


class SerializerHelperTests(unittest.TestCase):
    def test_render_history_includes_san(self):
        text = render_history(chess.STARTING_FEN, ["e2e4", "e7e5"])
        self.assertIn("Starting FEN:", text)
        self.assertIn("e2e4", text)
        self.assertIn("e2e4 (e4)", text)  # SAN annotation
        self.assertNotIn("Board (", text)

    def test_render_history_companion_board_is_moves_only(self):
        text = render_history(
            chess.STARTING_FEN,
            ["e2e4", "e7e5"],
            companion_board=True,
        )
        self.assertIn("Moves played (UCI with SAN):", text)
        self.assertIn("e2e4 (e4)", text)
        self.assertNotIn("move history alone", text.lower())
        self.assertNotIn("Starting FEN:", text)
        self.assertNotIn("Side to move:", text)

    def test_render_full_with_mask_drops_fen(self):
        board = chess.Board()
        text = render_full(board, mask={chess.E4, chess.D4}, include_legal_moves=True)
        self.assertIn("?", text)
        self.assertNotIn("FEN:", text)
        self.assertNotIn("Legal moves", text)

    def test_board_to_grid_text_mask_marker(self):
        board = chess.Board()
        text = board_to_grid_text(board, mask={chess.A1})
        # a1 originally white rook 'R' — should now be '?'
        # The first rank line is the last printed line above coordinate axis.
        rank1 = text.splitlines()[-2]
        self.assertTrue(rank1.startswith("1 "))
        self.assertIn("?", rank1)

    def test_parse_piece_name(self):
        self.assertEqual(parse_piece_name("white knight"), "white knight")
        self.assertEqual(parse_piece_name("It's a black queen there."), "black queen")
        self.assertEqual(parse_piece_name("empty"), "empty")
        self.assertIsNone(parse_piece_name("???"))


# --------------------------------------------------------------------------- #
# Lichess cache                                                               #
# --------------------------------------------------------------------------- #


class LichessCacheTests(unittest.TestCase):
    def test_coerce_record_normalizes_columns(self):
        rec = _coerce_record({
            "FEN": chess.STARTING_FEN,
            "Moves": "e2e4 e7e5",
            "Themes": "opening short",
            "Rating": "1500",
            "PuzzleId": "abc123",
        })
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["fen"], chess.STARTING_FEN)
        self.assertEqual(rec["moves"], ["e2e4", "e7e5"])
        self.assertEqual(rec["themes"], ["opening", "short"])
        self.assertEqual(rec["rating"], 1500)
        self.assertEqual(rec["puzzle_id"], "abc123")

    def test_coerce_record_rejects_invalid(self):
        self.assertIsNone(_coerce_record({"FEN": "this is not a fen"}))
        self.assertIsNone(_coerce_record({}))

    def test_load_lichess_puzzles_uses_cache(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            # Pre-populate the expected cache file. If the cache works, the
            # function returns this content without ever calling `datasets`.
            payload = [
                {"fen": chess.STARTING_FEN, "moves": ["e2e4"],
                 "themes": [], "rating": 1500, "puzzle_id": "test"},
            ]
            # Compute the same key the function uses.
            from halluworld.tracks.chess.envs.chess_env import _cache_key
            key = _cache_key({
                "n": 1, "seed": 42, "min": None, "max": None,
                "themes": None, "schema": 2,
            })
            (cache_dir / f"lichess_puzzles_{key}.json").write_text(json.dumps(payload))

            out = load_lichess_puzzles(num_samples=1, seed=42, cache_dir=str(cache_dir))
            self.assertEqual(out, payload)


if __name__ == "__main__":
    unittest.main()


def test_fen_transpose_is_identical_across_processes():
    """The swapped pair must not depend on the per-process str hash salt."""
    import subprocess
    import sys

    code = (
        "from halluworld.tracks.chess.serializers import FenDisplayConfig, derive_display_fen;"
        "print(derive_display_fen('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',"
        " FenDisplayConfig(mode='transpose', seed=0)))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**__import__("os").environ, "PYTHONHASHSEED": str(salt)}).stdout
        for salt in (1, 2, 3)
    }
    assert len(outs) == 1, outs
