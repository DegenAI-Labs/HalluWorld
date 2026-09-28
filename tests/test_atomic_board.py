import unittest

import chess

from halluworld.tracks.chess.rules.atomic_board import AtomicBoard, blast_ring_mask
from halluworld.tracks.chess.rules.atomic_hf_fens import _fen_from_movetext
from halluworld.tracks.chess.envs.atomic_chess_env import make_atomic_chess_env
from halluworld.tracks.chess.evaluators import ChessPieceNameEvaluator
from halluworld.lm.base import LMResponse
from halluworld.tracks.chess.probes.atomic import AtomicHypotheticalPieceAfterUciProbe
from halluworld.tracks.chess.serializers import ChessSerializer, render_atomic_san_history_observation


class TestAtomicBoard(unittest.TestCase):
    def test_blast_ring_mask(self):
        e4 = chess.parse_square("e4")
        m = blast_ring_mask(e4)
        self.assertTrue(m & chess.BB_SQUARES[e4])
        self.assertTrue(m & chess.BB_SQUARES[chess.parse_square("d3")])

    def test_variant_board_is_atomic(self):
        b = AtomicBoard()
        self.assertTrue(b.is_legal(chess.Move.from_uci("e2e4")))

    def test_env_tracks_history(self):
        env = make_atomic_chess_env(fens=[chess.STARTING_FEN], seed=0)
        env.reset(seed=1)
        self.assertEqual(env._episode_root_fen, env.board.fen())
        self.assertEqual(env._episode_moves_uci, [])
        env.step(None)
        self.assertEqual(len(env._episode_moves_uci), 1)

    def test_hypothetical_probe(self):
        import random

        env = make_atomic_chess_env(fens=[chess.STARTING_FEN], seed=0)
        env.reset(seed=0)
        pr = AtomicHypotheticalPieceAfterUciProbe(
            rng=random.Random(0),
            hypothesis_plies_min=1,
            hypothesis_plies_max=1,
        ).generate(env)
        self.assertIsInstance(pr.ground_truth, str)
        self.assertTrue(pr.ground_truth)
        self.assertIn("hypothesis_ucis", pr.metadata)
        self.assertEqual(pr.metadata.get("hypothesis_plies"), 1)
        ev = ChessPieceNameEvaluator()
        self.assertTrue(ev.evaluate(LMResponse(text=pr.ground_truth), pr).correct)

    def test_hypothetical_multi_ply(self):
        import random

        env = make_atomic_chess_env(fens=[chess.STARTING_FEN], seed=0)
        env.reset(seed=0)
        pr = AtomicHypotheticalPieceAfterUciProbe(
            rng=random.Random(1),
            hypothesis_plies_min=2,
            hypothesis_plies_max=2,
            max_move_tries=80,
        ).generate(env)
        self.assertFalse(pr.metadata.get("skipped"))
        self.assertGreaterEqual(len(pr.metadata["hypothesis_ucis"]), 2)

    def test_san_history_observation(self):
        env = make_atomic_chess_env(fens=[chess.STARTING_FEN], seed=0)
        env.reset(seed=0)
        for _ in range(3):
            env.step(None)
        text = render_atomic_san_history_observation(
            env, include_legal_moves=False, max_legal_moves=30, include_fen=True
        )
        self.assertIn("move list alone", text)
        self.assertIn("Moves played from that FEN", text)
        self.assertNotIn("Board ('.' means empty)", text)
        ser = ChessSerializer(
            include_legal_moves=False,
            include_fen=True,
            observation_mode="san_history",
        )
        out = ser.serialize(env)
        self.assertIn("SAN", out)

    def test_fen_from_movetext(self):
        import random

        mt = "1. e3 e5 2. Qf3 f6 3. Bb5 c6"
        fen = _fen_from_movetext(mt, random.Random(0), min_plies=2, max_plies=6)
        self.assertIsInstance(fen, str)
        b = AtomicBoard(fen)
        self.assertFalse(b.is_variant_end())


class TestAtomicHfFens(unittest.TestCase):
    @unittest.skip("requires datasets + network; run manually with pip install datasets")
    def test_load_hf_pool_smoke(self):
        from halluworld.tracks.chess.rules.atomic_hf_fens import load_atomic_hf_fen_pool

        fens = load_atomic_hf_fen_pool(target=5, max_scan=5000, seed=1, min_plies=4, max_plies=20)
        self.assertGreaterEqual(len(fens), 1)


if __name__ == "__main__":
    unittest.main()
