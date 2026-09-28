import unittest

import chess

from halluworld.tracks.chess.rules.old_bishop_board import OldBishopBoard, old_bishop_attack_mask
from halluworld.tracks.chess.envs.old_bishop_chess_env import make_old_bishop_chess_env
from halluworld.tracks.chess.evaluators import ChessLegalUciSetEvaluator
from halluworld.lm.base import LMResponse
from halluworld.tracks.chess.probes.old_bishop import (
    OldBishopFideIllegalBishopUciProbe,
    OldBishopLegalAnyUciProbe,
    _fide_legal_variant_illegal_bishop_ucis,
)


class TestOldBishopBoard(unittest.TestCase):
    def test_long_diagonal_bishop_not_pseudo_legal(self):
        fen = "8/8/8/3B4/8/8/8/4K3 w - - 0 1"
        ob = OldBishopBoard(fen)
        std = chess.Board(fen)
        mv = chess.Move.from_uci("d5a8")
        self.assertTrue(std.is_pseudo_legal(mv))
        self.assertFalse(ob.is_pseudo_legal(mv))

    def test_attack_mask(self):
        d5 = chess.parse_square("d5")
        m = old_bishop_attack_mask(d5, chess.Board("8/8/8/3B4/8/8/8/4K3 w - - 0 1").occupied)
        self.assertTrue(chess.BB_SQUARES[chess.parse_square("c6")] & m)
        self.assertFalse(chess.BB_SQUARES[chess.parse_square("a8")] & m)

    def test_env(self):
        env = make_old_bishop_chess_env(seed=0)
        env.reset(seed=2)
        self.assertIsInstance(env.board, OldBishopBoard)

    def test_foil_uci_nonempty_open_diagonal(self):
        b = OldBishopBoard("8/8/8/3B4/8/8/8/4K3 w - - 0 1")
        foils = _fide_legal_variant_illegal_bishop_ucis(b)
        self.assertTrue(foils)

    def test_any_uci_probe_and_evaluator(self):
        import random

        env = make_old_bishop_chess_env(fens=[chess.STARTING_FEN], seed=0)
        env.reset(seed=0)
        pr = OldBishopLegalAnyUciProbe(rng=random.Random(0)).generate(env)
        self.assertIsInstance(pr.ground_truth, list)
        self.assertGreater(len(pr.ground_truth), 0)
        ev = ChessLegalUciSetEvaluator()
        good = ev.evaluate(LMResponse(text=pr.ground_truth[0]), pr)
        self.assertTrue(good.correct)
        bad = ev.evaluate(LMResponse(text="d5a8"), pr)
        self.assertFalse(bad.correct)

    def test_fide_illegal_probe(self):
        import random

        env = make_old_bishop_chess_env(
            fens=["8/8/8/3B4/8/8/8/4K3 w - - 0 1"],
            seed=0,
        )
        env.reset(seed=0)
        pr = OldBishopFideIllegalBishopUciProbe(rng=random.Random(1)).generate(env)
        self.assertIsInstance(pr.metadata.get("legal_ucis"), list)
        self.assertGreater(len(pr.metadata["legal_ucis"]), 0)
        u = pr.metadata["legal_ucis"][0]
        self.assertTrue(ChessLegalUciSetEvaluator().evaluate(LMResponse(text=u), pr).correct)


if __name__ == "__main__":
    unittest.main()
