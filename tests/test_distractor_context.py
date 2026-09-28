from __future__ import annotations

import random
import unittest

from halluworld.tracks.chess.distractor_context import prefix_chess_observation_with_chat_context


class DistractorContextTests(unittest.TestCase):
    def test_prefix_contains_sections_and_body(self) -> None:
        rng = random.Random(0)
        body = "REAL_OBSERVATION_MARKER_XYZ"
        out = prefix_chess_observation_with_chat_context(body, rng)
        self.assertIn("Past games:", out)
        self.assertIn("Current game:", out)
        self.assertIn(body, out)
        self.assertLess(out.index("Past games:"), out.index("Current game:"))
        self.assertLess(out.index("Current game:"), out.index(body))

    def test_prefix_rumor_rate_controls_density(self) -> None:
        rng = random.Random(1)
        body = "X"
        dense = prefix_chess_observation_with_chat_context(
            body, rng, rumor_rate=1.0, min_games=5, max_games=5
        )
        rumorish = sum(
            1
            for line in dense.splitlines()
            if any(
                k in line
                for k in (
                    "claimed",
                    "insisted",
                    "thought",
                    "pinned",
                    "swore",
                    "chatter",
                    "cousin",
                )
            )
        )
        self.assertGreaterEqual(rumorish, 1)
        self.assertGreaterEqual(dense.count("Game "), 5)

    def test_lobby_section_when_requested(self) -> None:
        rng = random.Random(2)
        out = prefix_chess_observation_with_chat_context("Z", rng, lobby_lines=3, min_games=1, max_games=1)
        self.assertIn("Lobby / stream", out)
        self.assertGreaterEqual(out.count("chat:"), 3)

    def test_prefix_stack_and_post_obs(self) -> None:
        rng = random.Random(3)
        out = prefix_chess_observation_with_chat_context(
            "MARK",
            rng,
            min_games=1,
            max_games=1,
            past_san_min_plies=3,
            past_san_max_plies=5,
            prefix_stack=2,
            post_observation_lines=4,
        )
        self.assertEqual(out.count("unrelated chess clutter"), 1)
        self.assertIn("MARK", out)
        self.assertIn("Below the diagram", out)


if __name__ == "__main__":
    unittest.main()
