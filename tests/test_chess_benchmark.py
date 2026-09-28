from __future__ import annotations

import random
import unittest

from halluworld.benchmark import run_benchmark
from halluworld.tracks.chess.distractor_context import prefix_chess_observation_with_chat_context
from halluworld.tracks.chess import make_chess_env
from halluworld.tracks.chess import ChessBestMoveEvaluator, ChessIntegerEvaluator, ChessYesNoEvaluator
from halluworld.lm import StubLM
from halluworld.lm.base import LMResponse
from halluworld.tracks.chess import ChessBestMoveProbe, ChessHiddenSideCaptureStatsProbe, ChessLegalMoveProbe, ChessPiecePresenceProbe
from halluworld.tracks.chess import ChessSerializer


class _AlwaysBadRequestLM:
    """Simulates a provider that fails every call with a token-budget error.

    Mirrors what OpenAILM.query() returns for a reasoning model whose
    max_completion_tokens was too small: blank text, LMResponse.error set,
    never a raised exception (the run must be able to continue).
    """

    def query(self, system, user, max_tokens=None):
        return LMResponse(text="", model="fake-reasoning-model", error="bad_request")


class _HalfBadRequestLM:
    """Fails every other call; answers "yes" (always correct here) otherwise."""

    def __init__(self):
        self.calls = 0

    def query(self, system, user, max_tokens=None):
        self.calls += 1
        if self.calls % 2 == 0:
            return LMResponse(text="", model="fake-reasoning-model", error="bad_request")
        return LMResponse(text="yes", model="fake-reasoning-model")


class _FakeStockfish:
    def __init__(self, move: str):
        self.move = move

    def best_move_uci(self, _board) -> str:
        return self.move

    def close(self) -> None:
        return None


class ChessBenchmarkTests(unittest.TestCase):
    def test_serializer_includes_board_and_fen(self):
        env = make_chess_env(fens=["8/8/8/8/8/8/8/K6k w - - 0 1"], seed=1)
        env.reset(seed=1)
        text = ChessSerializer(include_legal_moves=True).serialize(env)
        self.assertIn("Board ('.' means empty):", text)
        self.assertIn("FEN:", text)
        self.assertIn("Side to move: white", text)

    def test_run_benchmark_chess_yes_no_probes(self):
        env = make_chess_env(seed=7)
        lm = StubLM(mode="yes")
        results = run_benchmark(
            env=env,
            serializer=ChessSerializer(include_legal_moves=False),
            probes=[
                ChessPiecePresenceProbe(positive_rate=1.0),
                ChessLegalMoveProbe(legal_rate=1.0),
            ],
            lm=lm,
            evaluators=[ChessYesNoEvaluator(), ChessYesNoEvaluator()],
            n_episodes=4,
            steps_before_probe=0,
            seed=11,
            verbose=False,
        )
        self.assertEqual(len(results.results), 8)
        self.assertEqual(results.accuracy("chess_piece_presence"), 1.0)
        self.assertEqual(results.accuracy("chess_move_legality"), 1.0)

    def test_distractor_context_reaches_the_user_prompt(self):
        """Distractor chat must survive into the prompt the model actually sees.

        run_benchmark used to take a ``user_observation_transform`` callback.
        That parameter was removed in favour of ``include_obs``, and the
        supported way to post-process observation text is now to wrap the
        serializer -- see _TransformedObservationSerializer in
        examples/chess_full_probes.py, which this mirrors.

        The old version of this test also asserted on a
        ``chat_context_prefix`` metadata key. Nothing sets that key any more;
        it was written by the removed callback, and no producer for it exists
        anywhere in the tree. Asserting on it would only re-test the absence.
        The content assertions below are the part that carries meaning: they
        check that the noise block, the section headers, and the real board
        all land in the user turn, in that order.
        """
        class _TransformedSerializer:
            def __init__(self, base, transform_fn, rng_seed):
                self._base = base
                self._transform_fn = transform_fn
                self._rng = random.Random(rng_seed)

            def serialize(self, env):
                return self._transform_fn(self._base.serialize(env), self._rng)

        env = make_chess_env(seed=7)
        lm = StubLM(mode="yes")
        results = run_benchmark(
            env=env,
            serializer=_TransformedSerializer(
                ChessSerializer(include_legal_moves=False),
                prefix_chess_observation_with_chat_context,
                rng_seed=101,
            ),
            probes=[ChessPiecePresenceProbe(positive_rate=1.0)],
            lm=lm,
            evaluators=[ChessYesNoEvaluator()],
            n_episodes=1,
            steps_before_probe=0,
            seed=101,
            verbose=False,
        )
        self.assertEqual(len(results.results), 1)
        user = results.results[0].messages[1]["content"]
        self.assertIn("Past games:", user)
        self.assertIn("Current game:", user)
        self.assertIn("Board ('.' means empty):", user)
        # The real position must come after the distractor block, not before.
        self.assertLess(user.index("Past games:"), user.index("Current game:"))

    def test_run_benchmark_prepends_serializer_when_probe_requests_it(self):
        env = make_chess_env(seed=1)
        env.reset(seed=1)
        lm = StubLM(mode="fixed", response="0")
        results = run_benchmark(
            env=env,
            serializer=ChessSerializer(include_legal_moves=False, include_fen=True),
            probes=[
                ChessHiddenSideCaptureStatsProbe(
                    min_plies=6,
                    max_plies=10,
                    rng=random.Random(0),
                    capture_bias=0.5,
                )
            ],
            lm=lm,
            evaluators=[ChessIntegerEvaluator()],
            n_episodes=1,
            steps_before_probe=0,
            seed=1,
            verbose=False,
        )
        self.assertEqual(len(results.results), 1)
        user = results.results[0].messages[1]["content"]
        self.assertIn("\n\n---\n\n", user)
        self.assertIn("Board ('", user)
        self.assertIn("Moves played", user)
        after_sep = user.split("\n\n---\n\n", 1)[1]
        self.assertNotIn("move history alone", after_sep.lower())
        self.assertNotIn("Starting FEN:", after_sep)
        self.assertNotIn("Side to move:", after_sep)

    def test_bad_request_is_excluded_not_scored_zero(self):
        """A provider-side failure must not be scored as a wrong answer.

        Before this, OpenAILM swallowed a token-budget BadRequestError and
        returned a blank LMResponse that the evaluator then scored 0 -- an
        infrastructure failure silently became a measured hallucination.
        """
        env = make_chess_env(seed=7)
        results = run_benchmark(
            env=env,
            serializer=ChessSerializer(include_legal_moves=False),
            probes=[ChessPiecePresenceProbe(positive_rate=1.0)],
            lm=_AlwaysBadRequestLM(),
            evaluators=[ChessYesNoEvaluator()],
            n_episodes=3,
            steps_before_probe=0,
            seed=101,
            verbose=False,
        )
        self.assertEqual(len(results.results), 3)
        for r in results.results:
            self.assertIsNone(r.score)
            self.assertIsNone(r.is_correct)
            self.assertEqual(r.metadata.get("error"), "bad_request")

        # Excluded trials must not appear in any denominator: an all-excluded
        # run has no scoreable trials, which is not the same as a 0% accurate one.
        self.assertIsNone(results.accuracy())
        self.assertIsNone(results.hallucination_rate())
        self.assertIsNone(results.mean_score())
        summary = results.summary()
        row = summary[summary["probe"] == "ALL"].iloc[0]
        self.assertEqual(row["excluded"], 3)
        self.assertIsNone(row["accuracy"])
        self.assertIsNone(row["hallucination_rate"])

    def test_bad_request_is_excluded_from_a_mixed_run(self):
        """Excluded trials are dropped from the denominator, not averaged in as zero."""
        env = make_chess_env(seed=7)
        results = run_benchmark(
            env=env,
            serializer=ChessSerializer(include_legal_moves=False),
            probes=[ChessPiecePresenceProbe(positive_rate=1.0)],
            lm=_HalfBadRequestLM(),
            evaluators=[ChessYesNoEvaluator()],
            n_episodes=4,
            steps_before_probe=0,
            seed=101,
            verbose=False,
        )
        self.assertEqual(len(results.results), 4)
        excluded = [r for r in results.results if r.score is None]
        scored = [r for r in results.results if r.score is not None]
        self.assertEqual(len(excluded), 2)
        self.assertEqual(len(scored), 2)
        # Every scored trial answered "yes" correctly; if the excluded ones
        # were silently averaged in as 0, accuracy would read 0.5, not 1.0.
        self.assertEqual(results.accuracy(), 1.0)
        self.assertEqual(results.hallucination_rate(), 0.0)
        summary = results.summary()
        row = summary[summary["probe"] == "ALL"].iloc[0]
        self.assertEqual(row["excluded"], 2)
        self.assertEqual(row["accuracy"], 1.0)

    def test_best_move_probe_with_injected_helper(self):
        # Use a non-trivial position; bare K vs K triggers insufficient-material
        # game-over and the probe correctly skips, defeating the test intent.
        env = make_chess_env(fens=["4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"], seed=2)
        env.reset(seed=2)
        probe = ChessBestMoveProbe(helper=_FakeStockfish("a1a2"))
        probe_result = probe.generate(env)
        self.assertEqual(probe_result.ground_truth, "a1a2")

        evaluator = ChessBestMoveEvaluator()
        lm = StubLM(mode="fixed", response="a1a2")
        eval_result = evaluator.evaluate(lm.query("", probe_result.question), probe_result)
        self.assertTrue(eval_result.correct)
        self.assertEqual(eval_result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
