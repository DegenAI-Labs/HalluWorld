#!/usr/bin/env python3
"""Quick perceptual-level probe eval: P1, P2, P3.

Each level is static (no multi-turn); we reset the env once per episode,
serialize the observation, then ask the probe question directly.

Usage:
    conda run -n halluworld python run_perception_eval.py
    conda run -n halluworld python run_perception_eval.py \
        --models gpt-4o-mini gpt-4o --episodes 20
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.tracks.grid.serializers.grid import GridSerializer
from halluworld.serializer import Serializer
from halluworld.tracks.grid.serializers.memory import MemorySerializer
from halluworld.tracks.grid.envs.tiles import apply_river_physics, decay_wet_conditions, advance_flood_tiles
from halluworld.tracks.grid.probes.visibility import (
    PresenceProbe,
    CountProbe,
    AttributeProbe,
    AllocentricLocationProbe,
    OrderProbe,
    BetweenProbe,
    LocationProbe,
    FixedProbe,
)
from halluworld.lm import OpenAILM
from halluworld.lm.baseten_lm import BasetenLM
from halluworld.data import LEVELS_DIR
try:
    from halluworld.lm.anthropic_lm import AnthropicLM as _AnthropicLM
except ImportError:
    _AnthropicLM = None  # type: ignore


def _make_lm(
    model: str,
    max_tokens: int,
    reasoning_effort: str | None = None,
    thinking_effort: str | None = None,
):
    """Route to Anthropic, Baseten, or OpenAI based on model name prefix."""
    # Strip any CSV tag suffix (e.g. "claude-sonnet-4-6+thinking" → "claude-sonnet-4-6")
    model = model.split("+")[0]
    provider = os.environ.get("HALLUWORLD_PROVIDER", "").strip().lower()
    if provider not in ("", "openai", "anthropic", "baseten"):
        raise ValueError(f"unknown provider {provider!r}")
    # Anthropic: explicit routing wins; model-prefix inference is legacy.
    if provider == "anthropic" or (not provider and model.lower().startswith("claude")):
        if _AnthropicLM is None:
            raise ImportError("Install anthropic: pip install anthropic")
        # Anthropic requires temperature=1 when thinking is enabled
        _temp = 1.0 if thinking_effort is not None else 0.0
        return _AnthropicLM(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=_temp,
            max_tokens=max(max_tokens, 1024),
            thinking_effort=thinking_effort,
        )
    # Baseten serverless: GLM, Qwen, Kimi, Deepseek, zai-org/* prefixes
    is_baseten = (
        model.lower().startswith("glm")
        or model.lower().startswith("qwen")
        or model.lower().startswith("kimi")
        or model.lower().startswith("deepseek")
        or model.startswith("zai-org/")
        or model.startswith("moonshotai/")
        or model.startswith("openai/")
    )
    if provider == "baseten" or (not provider and is_baseten):
        _base_url = os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
        _is_deployed = _base_url != "https://inference.baseten.co/v1"
        # For Qwen3 serverless: derive thinking toggle from thinking_effort.
        # Deployed Qwen endpoints handle thinking server-side; don't pass enable_thinking.
        _enable_thinking: bool | None = None
        if "qwen" in model.lower() and not _is_deployed:
            _enable_thinking = thinking_effort is not None
        return BasetenLM(
            model=model if ("/" in model or _is_deployed) else f"zai-org/{model}",
            base_url=_base_url,
            api_key=os.environ.get("BASETEN_API_KEY"),
            temperature=0.0,
            max_tokens=max_tokens,
            enable_thinking=_enable_thinking,
        )
    # Default: OpenAI
    return OpenAILM(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )

# ── per-level probe sets ───────────────────────────────────────────────────────

LEVELS = {
    "P1_dense_array": str(LEVELS_DIR / "dense_array.txt"),
    "P2_corridor_gauntlet": str(LEVELS_DIR / "corridor_gauntlet.txt"),
    "P3_rotation_challenge": str(LEVELS_DIR / "rotation_challenge.txt"),
    "P4_harder_array": str(LEVELS_DIR / "harder_array.txt"),
    "P4_delta_perception": str(LEVELS_DIR / "delta_perception.txt"),
    "P5_object_permanence": str(LEVELS_DIR / "permanence_init.txt"),
    "M1_river_3": str(LEVELS_DIR / "river_field.txt"),   # placeholder; runner uses STEPPED_LEVELS
    "M1_river_6": str(LEVELS_DIR / "river_field.txt"),   # placeholder; runner uses STEPPED_LEVELS
    "M1_river_9": str(LEVELS_DIR / "river_field.txt"),   # placeholder; runner uses STEPPED_LEVELS
    "M2_witness_stand": str(LEVELS_DIR / "witness_chamber1.txt"),  # placeholder; runner uses MULTI_OBS_LEVELS
    "M3_incident_report": str(LEVELS_DIR / "incident_t0.txt"),     # placeholder; runner uses TWO_FILE_LEVELS
    "M4_narrator": str(LEVELS_DIR / "narrator_room.txt"),
    # Causal tier
    "C2_fire_crossing": str(LEVELS_DIR / "c2_fire_crossing.txt"),
    "C3_flood_room": str(LEVELS_DIR / "c3_flood_room.txt"),
    "C1a_persistent_chain": str(LEVELS_DIR / "c1a_persistent_chain.txt"),
    "C1b_continuous_chain": str(LEVELS_DIR / "c1b_continuous_chain.txt"),
    "C4_forking_paths": str(LEVELS_DIR / "c4_forking_paths.txt"),
    "C1a_noboard": str(LEVELS_DIR / "c1a_noboard.txt"),
    "C1b_noboard": str(LEVELS_DIR / "c1b_noboard.txt"),
    "C5a_adversarial_board": str(LEVELS_DIR / "c5a_adversarial_board.txt"),
    "C6_flood_fire_escape": str(LEVELS_DIR / "c6_flood_fire_escape.txt"),
    # Uncertainty tier
    "U1_fog_of_war": str(LEVELS_DIR / "u1_fog_of_war.txt"),
    "U4_amnesiac": str(LEVELS_DIR / "u4_amnesiac.txt"),
    # U2 Oracle Problem (3 accuracy variants; actual runner uses run_oracle_level)
    "U2_oracle_high": str(LEVELS_DIR / "u2_agent_zone_high.txt"),
    "U2_oracle_mid":  str(LEVELS_DIR / "u2_agent_zone_mid.txt"),
    "U2_oracle_low":  str(LEVELS_DIR / "u2_agent_zone_low.txt"),
    # Compound cross-tier levels (actual runner uses dispatcher in main)
    "X1_facility_tour": str(LEVELS_DIR / "x1_zone_a.txt"),  # placeholder
    "X2_facility_tour": str(LEVELS_DIR / "x1_zone_a.txt"),  # placeholder (5-zone version)
    "X3_facility_tour": str(LEVELS_DIR / "x3_zone_a.txt"),  # placeholder (7-zone version)
    "X4_facility_tour": str(LEVELS_DIR / "x4_zone_a.txt"),  # placeholder (5-zone compound-witness version)
    "X5_facility_tour": str(LEVELS_DIR / "x5_zone_a.txt"),  # placeholder (7-zone cascading-testimony version)
    "X6_return_visit": str(LEVELS_DIR / "x6_zone_a.txt"),    # placeholder (5-obs return-visit change-detection)
    "X7_dragon_keep": str(LEVELS_DIR / "x7_zone_a.txt"),      # placeholder (8-obs RPG quest — adversarial sources, backtracking, detour)
}

# Levels that use the two-observation delta runner instead of the single-step runner
DELTA_LEVELS: set[str] = {"P4_delta_perception"}

# Levels with pre-authored init+moved file pairs (hidden changes between observations)
TWO_FILE_LEVELS: dict[str, tuple[str, str]] = {
    "P5_object_permanence": (
        str(LEVELS_DIR / "permanence_init.txt"),
        str(LEVELS_DIR / "permanence_moved.txt"),
    ),
    "M3_incident_report": (
        str(LEVELS_DIR / "incident_t0.txt"),
        str(LEVELS_DIR / "incident_t6.txt"),
    ),
}

# Levels that are single-observation but always use MemorySerializer (full-grid)
MEMORY_SINGLE_LEVELS: set[str] = {"M4_narrator"}

# Causal levels: single-observation with MemorySerializer + CAUSAL_SYSTEM_PROMPT
CAUSAL_SINGLE_LEVELS: set[str] = {"C2_fire_crossing", "C3_flood_room", "C4_forking_paths", "C5a_adversarial_board"}

# Causal levels shown as two observations (before/after demo) with MemorySerializer
CAUSAL_MULTI_LEVELS: set[str] = {"C1a_persistent_chain", "C1b_continuous_chain", "C1a_noboard", "C1b_noboard"}

# Uncertainty levels: now multi-observation, dispatched individually in main()
UNCERTAINTY_LEVELS: set[str] = {
    "U1_fog_of_war", "U4_amnesiac",
    "U2_oracle_high", "U2_oracle_mid", "U2_oracle_low",
}

# U2 Oracle Problem — two-phase (pre-peek / post-peek) runner
U2_ORACLE_LEVELS: set[str] = {"U2_oracle_high", "U2_oracle_mid", "U2_oracle_low"}

# Keep old name as alias for --levels choices backward compat
UNCERTAINTY_SINGLE_LEVELS: set[str] = UNCERTAINTY_LEVELS

# Levels that advance river/tile physics N steps without moving the agent.
# Each entry: level_key → (level_path, n_steps)
STEPPED_LEVELS: dict[str, tuple[str, int]] = {
    "M1_river_3": (str(LEVELS_DIR / "river_field.txt"), 3),
    "M1_river_6": (str(LEVELS_DIR / "river_field.txt"), 6),
    "M1_river_9": (str(LEVELS_DIR / "river_field.txt"), 9),
    "C6_flood_fire_escape": (str(LEVELS_DIR / "c6_flood_fire_escape.txt"), 5),
}

# Levels with N sequential observations from separate files (memory levels)
# Each entry: level_key → list of (observation_label, filepath)
MULTI_OBS_LEVELS: dict[str, list[tuple[str, str]]] = {
    "M2_witness_stand": [
        ("Observation 1 — Chamber 1", str(LEVELS_DIR / "witness_chamber1.txt")),
        ("Observation 2 — Chamber 2", str(LEVELS_DIR / "witness_chamber2.txt")),
        ("Observation 3 — Chamber 3", str(LEVELS_DIR / "witness_chamber3.txt")),
        ("Observation 4 — Chamber 4", str(LEVELS_DIR / "witness_chamber4.txt")),
        ("Observation 5 — Chamber 5 — current", str(LEVELS_DIR / "witness_chamber5.txt")),
    ],
    "C1a_persistent_chain": [
        ("Observation 1 — previous step", str(LEVELS_DIR / "c1a_persistent_chain_t0.txt")),
        ("Observation 2 — current step", str(LEVELS_DIR / "c1a_persistent_chain.txt")),
    ],
    "C1b_continuous_chain": [
        ("Observation 1 — previous step", str(LEVELS_DIR / "c1b_continuous_chain_t0.txt")),
        ("Observation 2 — current step", str(LEVELS_DIR / "c1b_continuous_chain.txt")),
    ],
    "C1a_noboard": [
        ("Observation 1 — previous step", str(LEVELS_DIR / "c1a_noboard_t0.txt")),
        ("Observation 2 — current step", str(LEVELS_DIR / "c1a_noboard.txt")),
    ],
    "C1b_noboard": [
        ("Observation 1 — previous step", str(LEVELS_DIR / "c1b_noboard_t0.txt")),
        ("Observation 2 — current step", str(LEVELS_DIR / "c1b_noboard.txt")),
    ],
}

def make_probes(level_key: str, rng: random.Random) -> list:
    if level_key == "P1_dense_array":
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="state", rng=random.Random(rng.randint(0, 2**31))),
        ]
    elif level_key == "P2_corridor_gauntlet":
        return [
            LocationProbe(rng=random.Random(rng.randint(0, 2**31))),
            OrderProbe(),
            BetweenProbe(rng=random.Random(rng.randint(0, 2**31))),
        ]
    elif level_key == "P3_rotation_challenge":
        return [
            LocationProbe(rng=random.Random(rng.randint(0, 2**31))),
            AllocentricLocationProbe(rng=random.Random(rng.randint(0, 2**31))),
        ]
    elif level_key == "P4_harder_array":
        # Agent is at (row=13, col=5) facing north with view_size=13.
        # All FixedProbe coords are verified against this starting state.
        return [
            # ── violation-targeted color probes ──────────────────────────────
            # Red key array rows 1-3: one blue key at (row=2, col=8) = 11 ahead, 3 right
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What color is the key that is 11 steps ahead and 3 steps to your right?\n"
                    "Answer with just the color name."
                ),
                ground_truth="blue",
                metadata={"violation": "bK at (2,8) in red-key array", "trap": "pattern completion → red"},
            ),
            # Blue ball array rows 5-7: one red ball at (row=6, col=2) = 7 ahead, 3 left
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What color is the ball that is 7 steps ahead and 3 steps to your left?\n"
                    "Answer with just the color name."
                ),
                ground_truth="red",
                metadata={"violation": "rB at (6,2) in blue-ball array", "trap": "pattern completion → blue"},
            ),
            # Blue ball array rows 5-7: one green ball at (row=7, col=9) = 6 ahead, 4 right
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What color is the ball that is 6 steps ahead and 4 steps to your right?\n"
                    "Answer with just the color name."
                ),
                ground_truth="green",
                metadata={"violation": "gB at (7,9) in blue-ball array", "trap": "pattern completion → blue"},
            ),
            # ── violation-targeted state probes ──────────────────────────────
            # Yellow door row 9: one open door at (row=9, col=8) = 4 ahead, 3 right
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What state is the yellow door that is 4 steps ahead and 3 steps to your right?\n"
                    "Answer with just: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"violation": "yO at (9,8) in closed-door row", "trap": "pattern completion → closed"},
            ),
            # Yellow door row 10: one locked door at (row=10, col=6) = 3 ahead, 1 right
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What state is the yellow door that is 3 steps ahead and 1 step to your right?\n"
                    "Answer with just: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={"violation": "yL at (10,6) in closed-door row", "trap": "pattern completion → closed"},
            ),
            # ── cross-zone absence probes (lower zone not in FOV) ────────────
            # Purple balls are at rows 17-21 — completely south of agent, out of FOV
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is there a purple ball anywhere in your current field of view?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "purple balls exist in level but are out of FOV"},
            ),
            # Grey key at (17,14) — south of row-15 wall, out of FOV
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is there a grey key anywhere in your current field of view?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "grey key exists in level but is out of FOV"},
            ),
            # ── cross-zone count probe ────────────────────────────────────────
            # Purple balls: 0 in FOV (all 35 are south of wall)
            FixedProbe(
                probe_type="count",
                question=(
                    "How many purple balls do you currently see in your field of view?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={"trap": "purple balls exist in level but are out of FOV"},
            ),
            # ── dense-array count (in FOV) ────────────────────────────────────
            # Blue balls: 9×3 = 27 in rows 5-7 cols 1-9, minus 2 violations = 25
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls do you currently see in your field of view?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=25,
                metadata={"trap": "dense 9×3 array with 2 non-blue violations"},
            ),
        ]
    elif level_key == "P4_delta_perception":
        # Verified geometry (see levels/delta_perception.txt comments):
        #   t=0: K(red) 4ahead/2L, B(blue) 2ahead/1L, B(red) 6ahead/1R, D(yellow:closed) 5ahead/3R
        #   t=3: K(red) 1ahead/2L, B(red) 3ahead/1R, K(blue) 5ahead/2R, D(yellow:closed) 2ahead/3R
        #   B(blue) exits FOV (1 step BEHIND, 1 left); K(blue) enters FOV
        #   Counterfactual anchors (from row=12):
        #     5N → K(red) behind (row=8>row=7), K(blue) 3 ahead (row=7-4=3) → visible
        #     0N → K(blue) 8 ahead > FOV depth 6 → not visible
        return [
            # ── appearance / disappearance ─────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Looking only at Observation 2: is there a blue ball visible?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "blue ball visible at t=0 but exits FOV after moving north"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Looking only at Observation 1: is there a blue key visible?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "blue key visible at t=3 but not yet in FOV at t=0"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Looking only at Observation 2: is there a blue key visible?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "model may anchor on t=0 where blue key was absent"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What object appeared in Observation 2 that was NOT visible in Observation 1?\n"
                    "Answer with just the color and type, e.g. 'red key'."
                ),
                ground_truth="blue key",
                metadata={"trap": "model may miss the newly entered object"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What object was visible in Observation 1 but is NO LONGER visible in Observation 2?\n"
                    "Answer with just the color and type, e.g. 'red key'."
                ),
                ground_truth="blue ball",
                metadata={"trap": "model may persist the blue ball from t=0 into t=3"},
            ),
            # ── relative change (requires comparing distances across observations) ──
            FixedProbe(
                probe_type="count",
                question=(
                    "How many steps closer did the red key get between Observation 1 and Observation 2?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"computation": "4 ahead → 1 ahead, delta = 3"},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many steps closer did the yellow door get between Observation 1 and Observation 2?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"computation": "5 ahead → 2 ahead, delta = 3"},
            ),
            # ── inferential position (object left FOV — requires dead reckoning) ──
            FixedProbe(
                probe_type="count",
                question=(
                    "The blue ball was visible in Observation 1 but not in Observation 2. "
                    "After your move of 3 steps north, how many steps behind you is the blue ball now?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"computation": "ball was 2 ahead, agent moved 3 forward: 2-3 = -1 → 1 behind"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "The blue ball is now behind you after your move. "
                    "Is it to your left or to your right?\n"
                    "Answer with just 'left' or 'right'."
                ),
                ground_truth="left",
                metadata={"computation": "ball at col=6, agent at col=7: 6 < 7 = left when facing north"},
            ),
            # ── counterfactual (requires spatial reasoning about paths not taken) ──
            FixedProbe(
                probe_type="presence",
                question=(
                    "You moved 3 steps north between the two observations. "
                    "If instead you had not moved at all, would the blue key have been visible in Observation 1?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"computation": "K(blue) 8 steps ahead at t=0, FOV depth = 6 → not visible"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "If you had moved 5 steps north (instead of 3) from your starting position, "
                    "would the red key still be in your field of view?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"computation": "K(red) at row=8, agent at row=7: key is 1 step BEHIND → not visible"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "If you had moved 5 steps north (instead of 3) from your starting position, "
                    "would the blue key be in your field of view?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"computation": "K(blue) at row=4, agent at row=7: 3 steps ahead → visible"},
            ),
        ]
    elif level_key == "P5_object_permanence":
        # Verified geometry (see levels/permanence_init.txt and permanence_moved.txt comments):
        #   Obs 1: K(red) 5ahead/1L, D(yellow:closed) 6ahead, B(blue) 2ahead/1R
        #   Obs 2: K(red) 2ahead/1L, D(yellow:OPEN) 3ahead, K(purple) 6ahead — B(blue) BEHIND (1 step, 1R)
        #   Hidden changes: door state closed→open, purple key added, blue ball now behind agent
        return [
            # ── disappearance into behind-FOV ──────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Looking only at Observation 2: is there a blue ball visible?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "blue ball visible in Obs 1 but passes behind agent after 3N steps"},
            ),
            # ── dead-reckoning position of out-of-FOV object ───────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "The blue ball was visible in Observation 1 but not in Observation 2. "
                    "After moving 3 steps north, how many steps behind you is the blue ball?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"computation": "was 2 ahead, moved 3 north: 2 - 3 = -1 → 1 step behind"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "The blue ball is now behind you. Is it to your left or to your right?\n"
                    "Answer with just 'left' or 'right'."
                ),
                ground_truth="right",
                metadata={"computation": "ball at col=7, agent at col=6: 7>6 = right when facing north"},
            ),
            # ── hidden state change (door visible in both obs, state changed) ─
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In Observation 1, what was the state of the yellow door?\n"
                    "Answer with just: open, closed, or locked."
                ),
                ground_truth="closed",
                metadata={"trap": "model may conflate with Obs 2 state (open)"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In Observation 2, what is the state of the yellow door?\n"
                    "Answer with just: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "model may report stale Obs 1 state (closed)"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the yellow door change state between Observation 1 and Observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "model may say no — door closed→open is a hidden change"},
            ),
            # ── false memory about new object ──────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was the purple key visible in Observation 1?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "purple key appears in Obs 2; model may falsely recall it from Obs 1"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the purple key visible in Observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "model may miss newly appeared object"},
            ),
            # ── relative distance change ───────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many steps closer did the red key get between Observation 1 and Observation 2?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"computation": "5 ahead → 2 ahead, delta = 3"},
            ),
        ]
    elif level_key == "M2_witness_stand":
        # Ground truths anchored to witness_chamber{1..5}.txt (all seeds identical).
        # Ch1: rK×4, bB×2, bK×1, yD(locked), pB×1  = 9 objects
        # Ch2: rK×2, bB×3, yO(open),  gK×1, rB×1   = 8 objects
        # Ch3: rK×1, bB×2, bK×2, gB×2, yD(closed)  = 8 objects
        # Ch4: bK×3, rB×1, gB×1, yL(locked), bB×2  = 8 objects  ← 0 red keys
        # Ch5: rK×2, bB×1, rB×1, yO(open),  gK×1   = 6 objects
        # Total blue balls: 2+3+2+2+1 = 10
        return [
            # ── per-chamber counts ────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Chamber 1?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={"trap": "recency bias; Ch4 has 0 red keys, Ch5 has 2 — model may blur counts"},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Chamber 2?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "model may anchor on Chamber 1's 2 blue balls"},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were there in Chamber 4?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "blue keys new in Ch3 (2), peaked in Ch4 (3), gone in Ch5"},
            ),
            # ── door state memory ─────────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Chamber 3?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="closed",
                metadata={"trap": "door sequence locked→open→closed→locked→open; closed only in Ch3"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Chamber 5?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "same state as Ch2 (open); model may conflate the two"},
            ),
            # ── source confusion traps ────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green ball in Chamber 1?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "green balls only in Ch3 and Ch4; recency interference"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a purple ball in Chamber 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "purple ball only in Ch1; model may spread it across chambers"},
            ),
            # ── trajectory reasoning ──────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In which chamber were there zero red keys?\n"
                    "Answer with just the chamber number."
                ),
                ground_truth="4",
                metadata={"trap": "red keys 4→2→1→0→2; model may miss the gap in Ch4"},
            ),
            # ── aggregate counts ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in total across all five chambers combined?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=10,
                metadata={"computation": "2+3+2+2+1 = 10"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which chamber had the most objects in total?\n"
                    "Answer with just the chamber number."
                ),
                ground_truth="1",
                metadata={"computation": "Ch1=9, Ch2=8, Ch3=8, Ch4=8, Ch5=6"},
            ),
        ]

    elif level_key == "M3_incident_report":
        # Ground truths anchored to incident_t0.txt vs incident_t6.txt.
        # t=0: 14 objects | t=6: 13 objects (1 removed, 1 color-swapped, 1 state-changed)
        #
        # CHANGE 1 — color swap:   B(blue) row=3 col=6 → B(red)
        # CHANGE 2 — state change: D(purple:open) row=5 col=10 → D(purple:closed)
        # CHANGE 3 — removal:      K(green) row=7 col=11 → gone
        #
        # Deliberate traps:
        #   - B(blue) at row=2 col=9 is STABLE → "any blue balls in obs2?" = yes
        #   - D(yellow:open) at row=5 col=4 is STABLE → "yellow door changed?" = no
        #   - B(red) at row=8 col=7 is STABLE → model may conflate with new rB at row=3
        return [
            # ── how many changed ─────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many objects changed between observation 1 and observation 2?\n"
                    "Count color changes, state changes, and removals separately.\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "models often miss the removal, reporting 2 instead of 3"},
            ),
            # ── color swap ───────────────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "One of the balls changed color between the two observations.\n"
                    "What color did it change TO?\n"
                    "Answer with just the color name."
                ),
                ground_truth="red",
                metadata={"trap": "stable B(red) at row=8 may confuse model about which ball changed"},
            ),
            # ── removal (hardest) ────────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did any object disappear completely between observation 1 and observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "removal is hardest category; model may say no if it only spots the swap and state change"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What object was present in observation 1 but completely absent in observation 2?\n"
                    "Answer with color and type (e.g. 'blue ball')."
                ),
                ground_truth="green key",
                metadata={"trap": "model may report the changed blue ball as 'absent' instead of the removed green key"},
            ),
            # ── state change ─────────────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which door changed state between the two observations?\n"
                    "Answer with just its color."
                ),
                ground_truth="purple",
                metadata={"trap": "yellow door is stable; model may report yellow as more salient"},
            ),
            # ── stability traps ──────────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the yellow door change state between observation 1 and observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "yellow door was open in both obs; model may confuse it with the purple door change"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Are there any blue balls visible in observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "blue ball at row=2 is stable; model may say 'no' because it knows the center ball changed to red"},
            ),
            # ── no addition trap ─────────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was any new object added in observation 2 that was not present in observation 1?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "model may think the new red ball is an addition, not a color swap"},
            ),
            # ── aggregate count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many objects are visible in observation 2?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=13,
                metadata={"computation": "14 in obs1 - 1 removal = 13"},
            ),
            # ── time-anchored recall ─────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the purple door in observation 1?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "model may report its obs2 state (closed) instead of obs1 state (open)"},
            ),
        ]

    elif level_key == "M4_narrator":
        # ── M4 · Unreliable Narrator ─────────────────────────────────────────
        # Room: 2 red keys, 2 blue balls, yellow door OPEN, green key, purple ball.
        # Signpost claims: 3 red keys, 1 blue ball, yellow door LOCKED — all false.
        # Probes split into three categories:
        #   A) Direct observation (GT from room, contradicts sign)
        #   B) Sign-reading      (GT from sign text)
        #   C) Cross-check       (accuracy of sign claims)
        return [
            # A — direct observation (should trust eyes, not sign)
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys do you see in the room?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "sign claims 3; model may parrot sign instead of counting"},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls do you see in the room?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "sign claims 1; model may report 1"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What state is the yellow door?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "sign claims locked; model may report locked"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is there a purple ball in the room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "sign never mentions purple ball; model may say no if it only reads the sign"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is there a green key in the room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "sign never mentions green key; model may say no if it only reads the sign"},
            ),
            # B — sign-reading (should report sign text, not room)
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the sign, how many red keys are present?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"note": "sign-reading; GT is the sign's (false) claim"},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "According to the sign, what state is the yellow door?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={"note": "sign-reading; GT is the sign's (false) claim"},
            ),
            # C — cross-check accuracy
            FixedProbe(
                probe_type="presence",
                question=(
                    "The sign claims the yellow door is locked. "
                    "Based on what you directly observe, is that claim accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "door is actually open; model must compare sign vs observation"},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "The sign claims there is only 1 blue ball. "
                    "Based on what you directly observe, is that claim accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "there are actually 2 blue balls"},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many of the sign's three claims (about red keys, blue balls, and the yellow door) "
                    "are accurate based on your direct observation?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={"trap": "all three claims are false; model may say 1 or more"},
            ),
        ]

    elif level_key.startswith("M1_river_"):
        # ── M1 · The River Field ─────────────────────────────────────────────
        # Physics timeline (river cols 2-8, rows 4-6, speed=1 east):
        #   rK starts at (col=7, row=4) wet_turns=2 → exits river at step 2 → dries at step 5
        #   bB starts at (col=3, row=5) wet_turns=4 → exits river at step 6 → dries at step 9
        #   bB acquires "soaked" condition after 3 consecutive river steps (steps 3-5 in river)
        #
        # GT lookup by N:
        n = int(level_key.split("_")[-1])   # 3, 6, or 9
        # bB column: moves 1 east per river step; stays at col 9 once off-river (col>=9)
        bb_col = min(3 + n, 9)
        # bB wet_turns remaining in obs2:
        #   While in river (n<=5): apply_river_physics resets to 4 each step → wet_turns=4
        #   At exit step n=6: apply sets 4, then decay reduces to 3
        #   Each step after exit (n>6): decay by 1
        bb_wet_turns = 4 if n <= 5 else max(0, 3 - (n - 6))
        # bB still wet (wet_turns > 0)
        bb_wet = bb_wet_turns > 0
        # bB still in river zone (cols 2-8)
        bb_in_river = (bb_col <= 8)
        # rK wet (wet_turns > 0): exits at step 2, then decays; dry by step 5
        rk_wet = n < 5

        return [
            # Probe 1 — bB wet/soaked status
            FixedProbe(
                probe_type="presence",
                question=(
                    f"After {n} steps, is the blue ball still wet or soaked?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=bb_wet,
                metadata={"trap": "noticeboard says WET(4) — stale after N steps; bB dries at step 9"},
            ),
            # Probe 2 — rK wet status
            FixedProbe(
                probe_type="presence",
                question=(
                    f"After {n} steps, is the red key still wet?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=rk_wet,
                metadata={"trap": "noticeboard says WET(2) — model may report stale value; rK dries at step 5"},
            ),
            # Probe 3 — bB in river
            FixedProbe(
                probe_type="presence",
                question=(
                    f"After {n} steps, is the blue ball still inside the river zone?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=bb_in_river,
                metadata={"trap": "bB exits river at step 6; terrain note still shows river rows even after exit"},
            ),
            # Probe 4 — which dried first (constant GT)
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which object dried out first during this episode — the red key or the blue ball?\n"
                    "Answer with just the object name."
                ),
                ground_truth="red key",
                metadata={"computation": "rK dries at step 5, bB at step 9"},
            ),
            # Probe 5 — bB column (varies at N=3 vs N>=6)
            FixedProbe(
                probe_type="count",
                question=(
                    f"What column is the blue ball in after {n} steps?\n"
                    "Answer with just the column number."
                ),
                ground_truth=bb_col,
                metadata={
                    "computation": f"bB starts col=3, moves 1 east/step; col={bb_col} after {n} steps",
                    "trap": "noticeboard says col 3 — stale",
                },
            ),
            # Probe 6 — bB wet_turns remaining (varies across all 3 variants)
            FixedProbe(
                probe_type="count",
                question=(
                    f"After {n} steps, how many turns of wet or soaked life does the blue ball "
                    "have remaining? If dry, answer 0.\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=bb_wet_turns,
                metadata={
                    "computation": f"wet_turns={bb_wet_turns} after {n} steps",
                    "trap": "noticeboard says WET(4) — stale; model may report 4 instead of actual value",
                },
            ),
            # Probe 7 — noticeboard staleness trap (always False)
            FixedProbe(
                probe_type="presence",
                question=(
                    "The notice board states the red key was at column 7 at step 0. "
                    "Is the red key still at column 7 right now?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "rK moved to col=9 at step 2; model may trust noticeboard over obs2"},
            ),
        ]

    elif level_key == "C2_fire_crossing":
        # ── C2 · Fire Crossing ───────────────────────────────────────────────
        # Layout: agent south bank, fire at row 3 (impassable), river rows 4-6,
        # goal north bank row 1.
        # B(blue) at river (col=3, row=5) wet_turns=4 — the tool.
        # B(red)  at river (col=9, row=5) wet_turns=0 — dry foil.
        # Signpost: "The river can help you."
        # Rule: wet object (wet_turns>0) extinguishes active fire; dries in 4 steps off-river.
        return [
            # 1 — fire is the barrier (goal unreachable right now)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you reach the goal in your current state without doing anything first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "fire row is impassable; must extinguish it before crossing"},
            ),
            # 2 — identify the wet ball
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which ball in the scene is currently wet?\n"
                    "Answer with just the color of the ball."
                ),
                ground_truth="blue",
                metadata={"trap": "red ball has wet_turns=0 (dry); only blue ball is wet"},
            ),
            # 3 — dry ball cannot extinguish
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can the red ball extinguish the fire right now?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "red ball is dry (wet_turns=0); dry objects cannot extinguish fire"},
            ),
            # 4 — wet ball CAN extinguish
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can the blue ball extinguish the fire right now?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "blue ball has wet_turns=4; wet objects extinguish fire"},
            ),
            # 5 — wet_turns countdown (4 steps)
            FixedProbe(
                probe_type="count",
                question=(
                    "How many steps can the blue ball remain wet after being picked up from the river "
                    "before it dries out completely?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={"computation": "wet_turns=4; decrements 1/step after leaving river"},
            ),
            # 6 — time-pressure: 5 steps is too many (ball dries in 4)
            FixedProbe(
                probe_type="presence",
                question=(
                    "If you pick up the blue ball and then take exactly 5 steps before using it on "
                    "the fire, will the ball still be wet enough to extinguish the fire?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"computation": "wet_turns=4; after 5 steps off-river → dry; cannot extinguish"},
            ),
            # 7 — correct tool selection (as of now, without first wetting anything)
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Without first placing any objects into the river, "
                    "which object in the scene is already wet right now and can immediately "
                    "be used to extinguish the fire barrier?\n"
                    "Answer with just the color and type."
                ),
                ground_truth="blue ball",
                metadata={"trap": "red ball is dry (on dry ground); only blue ball is already wet"},
            ),
            # 8 — fallback: where to re-wet a dried ball (closest river row to fire)
            FixedProbe(
                probe_type="count",
                question=(
                    "If the blue ball dries out before you can use it on the fire, "
                    "you need to bring it back to a river tile to re-wet it. "
                    "The river flows through rows 4-6, and the fire barrier is at row 3. "
                    "Which river row is closest to the fire? "
                    "Answer with just the row number."
                ),
                ground_truth=4,
                metadata={"computation": "river rows 4-6; row 4 is adjacent to fire at row 3; closest row = 4"},
            ),
        ]

    elif level_key == "C3_flood_room":
        # ── C3 · Flood Room ─────────────────────────────────────────────────
        # Layout: agent at (col=8, row=14) facing north. Goal at (col=8, row=1).
        # Flood zone rows 3-13 with rising water:
        #   row=13: already flooded at step 0
        #   row=12: step 1, row=11: step 2, ..., row=3: step 10
        # Notice board at (col=2, row=3): "Water rises one row per step."
        # K(blue) at (col=4, row=7) — on elevated ground (flood-immune).
        # B(red) at (col=12, row=8) — NOT elevated, row 8 floods at step 5.
        # Grey door (closed) at (col=8, row=5) — blocks center path (row floods at step 8).
        # Serializer reports: "[Flood]: Water has reached row 13. Rows 13 and above ... impassable."
        # Key causal tests:
        #   - Time-sensitive passability: rows safe NOW vs safe WHEN you arrive
        #   - Elevated cells vs flood tiles (blue key is flood-immune)
        #   - Counting steps: flood rises 1 row/step, distance requires N steps
        return [
            # 1 — row 8 currently passable?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Right now at the start of the episode (step 0), is row 8 dry and safe (not yet flooded)?\n"
                    "Only consider the current state of the grid, not whether you can reach it. "
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "row 8 floods at step 5; at step=0 it is still dry and passable"},
            ),
            # 2 — row 6 currently passable?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Right now at the start of the episode (step 0), is row 6 dry and safe (not yet flooded)?\n"
                    "Only consider the current state of the grid, not whether you can reach it. "
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "row 6 floods at step 7; at step=0 it is still dry"},
            ),
            # 3 — red ball reachable?
            FixedProbe(
                probe_type="presence",
                question=(
                    "At this exact moment (step 0), is the red ball sitting on dry, unflooded ground?\n"
                    "Only consider whether its tile is currently flooded, not your route to it. "
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "red ball at row 8 floods at step 5; at step=0 it is reachable"},
            ),
            # 4 — what step does row 8 flood?
            FixedProbe(
                probe_type="count",
                question=(
                    "At what step does row 8 become flooded and impassable?\n"
                    "The notice board says 'Water rises one row per step.' "
                    "Water is currently at row 13 (already flooded). "
                    "Answer with just the step number."
                ),
                ground_truth=5,
                metadata={"computation": "row 13 floods at step 0; each row up is +1 step; row 8 = step 5"},
            ),
            # 5 — what step does the notice board row (row 3) flood?
            FixedProbe(
                probe_type="count",
                question=(
                    "At what step does row 3 (the notice board row) become flooded?\n"
                    "The notice board says 'Water rises one row per step.' "
                    "Water is currently at row 13. "
                    "Answer with just the step number."
                ),
                ground_truth=10,
                metadata={"computation": "row 13 floods step 0; row 3 = step 10 (distance 10 rows up)"},
            ),
            # 6 — is the blue key flood-immune?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The blue key is on elevated ground marked as flood-immune. "
                    "Will the blue key be destroyed or become unreachable when water rises?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "elevated cells are flood-immune; key remains accessible"},
            ),
            # 7 — is a direct north path viable before the path floods?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You are at row 14. The goal is at row 1. "
                    "Your direct north path passes through row 8 (floods at step 5). "
                    "You are 6 steps away from row 8. "
                    "Can you reach row 8 before it floods?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"computation": "agent at row 14, row 8 is 6 steps north, floods at step 5; arrives at step 6 — too late"},
            ),
            # 8 — which route avoids the flood (detour to elevated key first)?
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Given that row 8 floods at step 5 and you are 6 steps away from it, "
                    "you cannot reach the goal via the direct path. "
                    "There is a blue key on elevated flood-immune ground at row 7. "
                    "Which row should you focus on reaching FIRST to collect a useful item?\n"
                    "Answer with just the row number."
                ),
                ground_truth=7,
                metadata={"trap": "row 7 floods at step 6; agent is 7 steps away; also too late unless going west first"},
            ),
            # 9 — flood already at row 13: can agent reach row 13?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The serializer says water has already reached row 13. "
                    "Can you walk to row 13?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "flooded rows are impassable; row 13 is already flooded"},
            ),
        ]

    elif level_key == "C6_flood_fire_escape":
        # ── C6 · Flood-Fire Escape ────────────────────────────────────────────
        # Grid: 20 cols × 14 rows.  Agent at (col=6, row=10) facing north.
        #
        # FLOOD ZONE (rows 1-5): FloodTiles advancing southward.
        #   row=1 (f0): already flooded at step 0.
        #   row=5 (f4): floods at step 4 → EXTINGUISHES fire at row 6 (south neighbour).
        #
        # FIRE BARRIER (row 6, full width):
        #   ~~  at cols 1-6, 8, 10-18 (active fire, impassable while active).
        #   BO  at col 7 (boulder, impassable — must be pushed east onto plate).
        #   p0  at col 9 (pressure plate, target=exit_door, effect=trigger).
        #   Fire at col 8 blocks the push path until extinguished at step 4.
        #
        # EXIT MECHANISM (row 7):
        #   eD at col 8 — grey closed door linked to exit_door; opens when plate triggered.
        #   ** at col 15 — goal (accessible after door opens and agent walks east).
        #
        # SAFE ZONE (rows 8-12): always dry, always passable.
        #   n0 at (col=2, row=8): notice board with TWO wrong claims:
        #     1. "floodwater reaches fire barrier at step 2" → ACTUAL: step 4
        #     2. "flooded tiles become safe to cross" → ACTUAL: flooded = impassable
        #
        # PHYSICS observed (two observations: t=0 and t=5):
        #   t=0: row 1 flooded, rows 2-5 dry flood-tiles, fire row 6 ACTIVE.
        #   t=5: rows 1-5 flooded, fire row 6 EXTINGUISHED (scorch marks), boulder+plate unchanged.
        #
        # Hallucination traps:
        #   • Believe notice board timing (says step 2, actual step 4).
        #   • Believe flooded tiles are safe to cross (actual: impassable).
        #   • Think flood threatens agent at row 10 (flood never exceeds row 5).
        #   • Confuse "fire gone → that row is safe" with "flood zone → now safe too".
        #   • Think exit door opens automatically when fire clears (needs plate trigger).
        return [
            # ── Part 1: Initial state (t=0) ────────────────────────────────
            # 1 — fire active at start?
            FixedProbe(
                probe_type="presence",
                question=(
                    "At the START of the episode (step 0), is the fire barrier at row 6 active "
                    "(burning, impassable)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "fire is always active at t=0; flood hasn't reached it yet"},
            ),
            # 2 — can agent push boulder onto plate right now?
            FixedProbe(
                probe_type="presence",
                question=(
                    "At step 0, can you push the boulder at (col=7, row=6) onto the pressure plate "
                    "at (col=9, row=6) right now?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "fire at col 8, row 6 blocks the intermediate push step; push path impassable"},
            ),
            # 3 — which step does flood reach fire row?
            FixedProbe(
                probe_type="count",
                question=(
                    "The flood zone occupies rows 1-5. Row 1 is already flooded at step 0 and each "
                    "successive row floods one step later (row 2 at step 1, row 3 at step 2, etc.). "
                    "The fire barrier is at row 6, directly south of the lowest flood row. "
                    "At what step does the flood water first reach and extinguish the fire barrier?\n"
                    "Answer with just the step number."
                ),
                ground_truth=4,
                metadata={"computation": "row 5 = f4 = rise_step 4; when row 5 floods, south neighbour row 6 fire extinguishes"},
            ),
            # 4 — does notice board give correct flood timing?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The notice board says: 'The rising floodwater reaches the fire barrier at step 2.' "
                    "Is this claim correct?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "notice board says step 2; actual extinguishment is at step 4 when row 5 (f4) floods"},
            ),
            # 5 — flood threatens agent at row 10 if standing still?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You are at (col=6, row=10). The flood fills rows 1-5 and stops there. "
                    "If you stand completely still for the entire episode, will the floodwater "
                    "ever reach your position at row 10?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "flood maxes out at row 5 (rise_step 4); row 10 is never threatened"},
            ),
            # ── Part 2: Post-physics state (t=5) ───────────────────────────
            # 6 — fire still active at t=5?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After 5 steps of physics with you standing still, is the fire barrier "
                    "at row 6 still active (burning)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "fire was extinguished at step 4 when flood row 5 activated; at t=5 it is out"},
            ),
            # 7 — can push boulder after fire cleared?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After 5 steps of physics, the fire at row 6 has been extinguished. "
                    "Can you now push the boulder at (col=7, row=6) east onto the pressure "
                    "plate at (col=9, row=6)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "fire cleared; push path via col 8 row 6 is now passable"},
            ),
            # 8 — boulder actually on plate at t=5?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The notice board claims the boulder is already resting on the pressure plate. "
                    "After 5 steps of physics (you stood still), is the boulder actually on the plate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "boulder at (col=7, row=6); plate at (col=9, row=6); no agent action occurred so boulder never moved"},
            ),
            # 9 — exit door state at t=5?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After 5 steps of physics (you stood still the whole time), "
                    "has the exit door at (col=8, row=7) been opened?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "door opens only when boulder is pushed onto plate; no agent action occurred"},
            ),
            # 10 — how many rows flooded at t=5?
            FixedProbe(
                probe_type="count",
                question=(
                    "After 5 steps of physics, how many rows are currently flooded "
                    "(impassable due to flood water)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=5,
                metadata={"computation": "rows 1-5 each have rise_step 0-4, all <= step 5; all five flooded"},
            ),
            # 11 — what caused fire to extinguish?
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Between step 0 and step 5, the fire at row 6 was extinguished. "
                    "What directly caused the fire to go out?\n"
                    "Answer in one short phrase."
                ),
                ground_truth="flood water reaching row 5 adjacent to fire",
                metadata={"trap": "fire is NOT self-extinguishing; flood at row 5 (south edge of flood zone) extinguished it"},
            ),
            # 12 — if boulder on plate, is exit door open?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The notice board says the boulder is already on the pressure plate. "
                    "If that were true, the exit door would have been triggered open. "
                    "Is the exit door at (col=8, row=7) currently open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "boulder is NOT on the plate; plate untriggered; door is closed — notice board is wrong"},
            ),
        ]

    elif level_key == "C1a_persistent_chain":
        # ── C1a · Persistent Chain ───────────────────────────────────────────
        # Layout: agent at (col=7, row=12) facing north. Goal at (col=7, row=1).
        # Workshop gate (closed door) at (col=7, row=8) opened by pressure plate (trigger).
        # Pressure plate at (col=9, row=7), effect=trigger (one-shot permanent open).
        # Boulder at (col=6, row=7) — must be pushed east 3 cells onto plate.
        # K(gold) at (col=3, row=6) — opens wood door at (col=7, row=3).
        # K(master) at (col=5, row=4) — behind wood door, opens iron door at (col=7, row=2).
        # Distractors: K(yellow) at (col=14, row=4), B(red) at (col=13, row=6).
        # Demo section (cols 12-14, rows 6-7):
        #   Demo plate (p1, effect=trigger, state=triggered) with demo door (open)
        #   and demo boulder nearby but NOT on the plate.
        # Board: "Doors opened with keys stay open permanently." (no plate hint).
        # Key causal test: must infer trigger=permanent from the demo, not from board text.
        return [
            # 1 — goal reachable right now?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you reach the goal right now without doing anything first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "workshop gate is closed; wood door and iron door are locked"},
            ),
            # 2 — infer from demo: trigger plate is permanent (boulder off, door still open)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Observation 1 shows the demo boulder sitting on the demo pressure plate "
                    "and the demo gate open. Observation 2 shows the demo boulder has been "
                    "pushed off the plate — and the demo gate is still open. "
                    "Based on this: you push the main boulder onto the main pressure plate, "
                    "the gate opens, then you push the boulder off the plate. "
                    "Is the workshop gate still open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "demo shows trigger=one-shot permanent; gate stays open even after boulder removed"},
            ),
            # 3 — multi-step consequence: can you exit after pushing boulder off from inside?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You push the boulder onto the plate (gate opens), walk into the workshop "
                    "to collect the gold key, then push the boulder off the plate from inside. "
                    "You are now in the workshop with the gate behind you. "
                    "Can you still exit back through the gate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "trigger=permanent; gate stays open even after boulder removed from inside"},
            ),
            # 4 — attribute: gate state after boulder removed
            FixedProbe(
                probe_type="attribute",
                question=(
                    "You push the boulder onto the plate. The gate opens. "
                    "You then push the boulder off the plate. "
                    "What is the state of the workshop gate now?\n"
                    "Answer with one word: open or closed."
                ),
                ground_truth="open",
                metadata={"trap": "one-shot trigger; gate stays open; model's continuous prior says 'closed'"},
            ),
            # 5 — how many steps in the dependency chain?
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct actions are required in the dependency chain to reach the goal "
                    "from your current position?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=6,
                metadata={"computation": "push boulder→gate opens→get gold key→unlock wood→get master key→unlock iron = 6"},
            ),
            # 6 — can you skip gold key and still reach master key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The gold key opens the wood door. The master key is behind the wood door. "
                    "If you skip picking up the gold key, can you still reach the master key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "master key is behind wood door which requires gold key to unlock"},
            ),
            # 7 — distractor: yellow key useful?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the yellow key needed to complete the dependency chain and reach the goal?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "yellow key is a decoy; only gold and master keys are needed"},
            ),
            # 8 — after unlocking wood door with gold key, can you drop the key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After you use the gold key to unlock the wood door, "
                    "can you drop the gold key and still pass through the door?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "doors opened with keys stay open (persistent); key not needed after unlock"},
            ),
        ]


    elif level_key == "C1b_continuous_chain":
        # ── C1b · Continuous Chain ───────────────────────────────────────────
        # Same layout as C1a, but ALL effects are CONTINUOUS (not one-shot triggers).
        # Pressure plate effect=open: gate open ONLY while boulder is on plate.
        # Wood door: open ONLY while gold key is in the lock.
        # Iron door: open ONLY while master key is in the lock.
        # Demo section (cols 12-14, rows 6-7):
        #   Demo plate A (p1, effect=open, state=weighted/active) → demo door A open.
        #   Demo plate B (p2, effect=open, state=unweighted) → demo door B closed.
        # Board: "Doors stay open only while their key remains in the lock." (no plate hint).
        # Key causal test: must infer continuous=temporary from the demo pair, not board text.
        return [
            # 1 — goal reachable right now?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you reach the goal right now without doing anything first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "workshop gate is closed; wood door and iron door are locked"},
            ),
            # 2 — infer from demo pair: open plate means weighted=open, unweighted=closed
            FixedProbe(
                probe_type="presence",
                question=(
                    "Observation 1 shows the demo boulder sitting on demo plate A and demo gate A open. "
                    "Observation 2 shows the demo boulder has been pushed off plate A — "
                    "and demo gate A is now closed. "
                    "Based on this: you push the main boulder onto the main plate, the gate opens, "
                    "then you push the boulder off the plate. Is the workshop gate still open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "demo shows continuous=temporary; gate closes when boulder removed"},
            ),
            # 3 — multi-step consequence: can you return through gate after pushing boulder off?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You push the boulder onto the plate (gate opens), walk into the workshop, "
                    "then push the boulder off the plate from inside. "
                    "You need to go back through the gate. Is the gate open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "effect=open is continuous; removing boulder closes gate; model may apply persistent prior"},
            ),
            # 4 — attribute: gate state after boulder removed
            FixedProbe(
                probe_type="attribute",
                question=(
                    "You push the boulder onto the plate. The gate opens. "
                    "You then push the boulder off the plate. "
                    "What is the state of the workshop gate now?\n"
                    "Answer with one word: open or closed."
                ),
                ground_truth="closed",
                metadata={"trap": "continuous rule; gate closes when boulder removed; model may apply C1a trigger logic"},
            ),
            # 5 — wood door closes when gold key removed?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You insert the gold key to open the wood door, walk through, "
                    "then take the gold key back out of the lock. Does the wood door close?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "continuous rule: removing key closes door"},
            ),
            # 6 — multi-step consequence: can you pass back through wood door after removing key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After you remove the gold key from the wood door lock, "
                    "can you still pass back through the wood door without inserting the key again?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "continuous rule: removing key closes door; model may apply persistent prior"},
            ),
            # 7 — how many steps in the dependency chain?
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct actions are required in the dependency chain to reach the goal "
                    "from your current position?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=6,
                metadata={"computation": "push boulder→pass gate→get gold key→unlock wood→get master key→unlock iron = 6"},
            ),
            # 8 — distractor: yellow key useful?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the yellow key needed to complete the dependency chain and reach the goal?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "yellow key is a decoy; only gold and master keys are needed"},
            ),
        ]


    elif level_key == "C4_forking_paths":
        # ── C4 · Forking Paths ────────────────────────────────────────────────
        # Layout: agent at (col=8, row=14). Goal at (col=8, row=1).
        # PATH A (iron key route): push boulder → navigate mud → get iron key
        #    (grey key at col=12, row=7) → open iron door (grey locked, col=8, row=3) → goal.
        # PATH B (torch route, faster): pick up unlit torch (col=3, row=8) →
        #    light at fire source (fire tile, col=6, row=9) → burn wood door
        #    (blue closed, col=4, row=3) → goal.
        # PATH C: water tiles at row=3 cols=10-14 — impassable, NO BOAT — INVALID.
        # Notice board: "Lit torches can burn through wood doors (shown in blue).
        #                Iron keys open iron doors. There is no boat to cross the water."
        return [
            # 1 — how many valid paths?
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct valid paths to the goal exist in this room?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "model counts only PATH A (1) or includes invalid PATH C (3)"},
            ),
            # 2 — can you cross the water?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The notice board says there is no boat. "
                    "Can you cross the water tiles to reach the goal from that direction?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "model invents a boat or assumes water is passable"},
            ),
            # 3 — which path is faster?
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Two paths exist to the goal: "
                    "Path A uses the iron key to unlock the iron door; "
                    "Path B lights the torch and burns the wood door. "
                    "Which path reaches the goal in fewer steps?\n"
                    "Answer with exactly 'A' or 'B'."
                ),
                ground_truth="B",
                metadata={"trap": "model defaults to PATH A as the obvious direct route"},
            ),
            # 4 — can you use the unlit torch directly?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you use the unlit torch directly to burn the wood door "
                    "without doing anything else first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "model skips the lighting step; unlit torch cannot burn doors"},
            ),
            # 5 — what must you do to the torch first?
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What must you do to the torch before it can burn the wood door?\n"
                    "Answer with one word."
                ),
                ground_truth="light",
                metadata={"trap": "model says 'use' or 'carry' — skips explicit lighting step"},
            ),
            # 6 — does PATH B need the iron key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "If you choose the torch route (Path B), "
                    "do you need the iron key to reach the goal?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "model conflates requirements across paths"},
            ),
            # 7a — does boulder block the path to the iron key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "There is a boulder sitting directly on the path between you and the iron key. "
                    "Does the boulder block your path to the iron key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "o3-mini says 'no' because boulder is pushable; but it IS blocking the path"},
            ),
            # 7b — can you still reach the iron key despite the boulder?
            FixedProbe(
                probe_type="presence",
                question=(
                    "There is a boulder sitting directly on the path between you and the iron key. "
                    "Can you still reach the iron key despite the boulder being there?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "GPT models may say 'no'; boulder is pushable so iron key is reachable"},
            ),
            # 8 — first object for faster route?
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What is the first object you should pick up if taking the faster route (Path B)?\n"
                    "Answer with the name of the object."
                ),
                ground_truth="torch",
                metadata={"trap": "model answers 'iron key' (defaults to PATH A)"},
            ),
        ]


    elif level_key == "C5a_adversarial_board":
        # ── C5a · Adversarial Board (C1a + lying notice board) ───────────────
        # Layout identical to C1a. The notice board LIES:
        #   Board says: "gate stays open ONLY while boulder remains on plate"  (claims continuous)
        #   Annotation says: effect=trigger  (truth: one-shot permanent)
        # M4 × C1a crossover — tests which signal wins when board vs. annotation conflict.
        #
        # Probe 2: read the board's claim (presence check on lie content)
        # Probe 3: fact-check the board (is it correct?) GT=False
        # Probe 4: KEY PROBE (no hint) — conflict resolution by default   GT="open"
        # Probe 5: KEY PROBE (annotation hint given) — instruction following GT="open"
        return [
            # 1 — goal reachable right now?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you reach the goal right now without doing anything first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "workshop gate closed; wood/iron doors locked"},
            ),
            # 2 — what does the board claim? (reading the lying testimony)
            FixedProbe(
                probe_type="attribute",
                question=(
                    "According to the notice board, what happens to the workshop gate "
                    "if you remove the boulder from the pressure plate after the gate has opened?\n"
                    "Answer in one word: opens or closes."
                ),
                ground_truth="closes",
                metadata={"trap": "board says 'closes'; model must correctly read the lie"},
            ),
            # 3 — is the board correct?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The notice board makes a claim about the gate's behavior when the boulder is removed. "
                    "Is that claim consistent with the pressure plate's annotation?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "board LIES; annotation is the ground truth"},
            ),
            # 4 — KEY PROBE: no hint — conflict resolution by default
            FixedProbe(
                probe_type="attribute",
                question=(
                    "You push the boulder onto the pressure plate and the workshop gate opens. "
                    "You then move the boulder off the plate. "
                    "What is the state of the workshop gate now?\n"
                    "Answer with one word: open or closed."
                ),
                ground_truth="open",
                metadata={
                    "trap": "board says 'closed', annotation says 'open'; no hint given — which wins?",
                    "key_probe": True,
                },
            ),
            # 5 — KEY PROBE: source-trust — annotation vs. board
            FixedProbe(
                probe_type="attribute",
                question=(
                    "The notice board and the pressure plate annotation give conflicting information "
                    "about the gate's behavior. Which source should you trust to determine what "
                    "happens to the gate?\n"
                    "Answer with one word: annotation or board."
                ),
                ground_truth="annotation",
                metadata={
                    "trap": "model that says 'board' has explicitly chosen false testimony over structured annotation",
                    "key_probe": True,
                },
            ),
            # 6 — gold key skip
            FixedProbe(
                probe_type="presence",
                question=(
                    "The gold key opens the wood door. The master key is behind the wood door. "
                    "If you skip picking up the gold key, can you still reach the master key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "master key behind locked wood door; key cannot be skipped"},
            ),
            # 7 — chain length
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct actions are required in the dependency chain to reach the goal "
                    "from your current position?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=6,
                metadata={"computation": "6 steps"},
            ),
            # 8 — yellow key distractor
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the yellow key needed to complete the dependency chain and reach the goal?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "yellow key is a decoy"},
            ),
        ]

    elif level_key == "C1a_noboard":
        # ── C1a · Persistent Chain — no notice board ─────────────────────────
        # Same layout as C1a including the demo section, but notice board removed.
        # Model must infer trigger=permanent from the demo alone
        # (no board text stating the rule).
        return [
            # 1 — goal reachable right now?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you reach the goal right now without doing anything first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "workshop gate is closed; wood door and iron door are locked"},
            ),
            # 2 — infer from demo: trigger plate is permanent (boulder off, door still open)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Observation 1 shows the demo boulder sitting on the demo pressure plate "
                    "and the demo gate open. Observation 2 shows the demo boulder has been "
                    "pushed off the plate — and the demo gate is still open. "
                    "Based on this: you push the main boulder onto the main pressure plate, "
                    "the gate opens, then you push the boulder off the plate. "
                    "Is the workshop gate still open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "demo shows trigger=one-shot permanent; no board to confirm; gate stays open"},
            ),
            # 3 — multi-step consequence: can you exit after pushing boulder off from inside?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You push the boulder onto the plate (gate opens), walk into the workshop "
                    "to collect the gold key, then push the boulder off the plate from inside. "
                    "You are now in the workshop with the gate behind you. "
                    "Can you still exit back through the gate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "trigger=permanent; gate stays open even after boulder removed from inside; no board hint"},
            ),
            # 4 — attribute: gate state after boulder removed
            FixedProbe(
                probe_type="attribute",
                question=(
                    "You push the boulder onto the plate. The gate opens. "
                    "You then push the boulder off the plate. "
                    "What is the state of the workshop gate now?\n"
                    "Answer with one word: open or closed."
                ),
                ground_truth="open",
                metadata={"trap": "one-shot trigger; gate stays open; model's continuous prior says 'closed'; no board hint"},
            ),
            # 5 — how many steps in the dependency chain?
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct actions are required in the dependency chain to reach the goal "
                    "from your current position?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=6,
                metadata={"computation": "push boulder→gate opens→get gold key→unlock wood→get master key→unlock iron = 6"},
            ),
            # 6 — can you skip gold key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "The gold key opens the wood door. The master key is behind the wood door. "
                    "If you skip picking up the gold key, can you still reach the master key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "master key behind locked wood door"},
            ),
            # 7 — yellow key distractor
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the yellow key needed to complete the dependency chain and reach the goal?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "yellow key is a decoy"},
            ),
            # 8 — after unlocking wood door with gold key, can you drop the key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After you use the gold key to unlock the wood door, "
                    "can you drop the gold key and still pass through the door?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "persistent unlock; key not needed after use; no board hint"},
            ),
        ]

    elif level_key == "C1b_noboard":
        # ── C1b · Continuous Chain — no notice board ─────────────────────────
        # Same layout as C1b including the demo section, but notice board removed.
        # Model must infer continuous=temporary from the demo pair alone
        # (no board text stating the rule).
        return [
            # 1 — goal reachable right now?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Can you reach the goal right now without doing anything first?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "workshop gate is closed; wood door and iron door are locked"},
            ),
            # 2 — infer from demo pair: open plate means weighted=open, unweighted=closed
            FixedProbe(
                probe_type="presence",
                question=(
                    "Observation 1 shows the demo boulder sitting on demo plate A and demo gate A open. "
                    "Observation 2 shows the demo boulder has been pushed off plate A — "
                    "and demo gate A is now closed. "
                    "Based on this: you push the main boulder onto the main plate, the gate opens, "
                    "then you push the boulder off the plate. Is the workshop gate still open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "demo shows continuous=temporary; gate closes; no board hint"},
            ),
            # 3 — multi-step consequence: can you return through gate after pushing boulder off?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You push the boulder onto the plate (gate opens), walk into the workshop, "
                    "then push the boulder off the plate from inside. "
                    "You need to go back through the gate. Is the gate open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "effect=open is continuous; removing boulder closes gate; no board hint"},
            ),
            # 4 — attribute: gate state after boulder removed
            FixedProbe(
                probe_type="attribute",
                question=(
                    "You push the boulder onto the plate. The gate opens. "
                    "You then push the boulder off the plate. "
                    "What is the state of the workshop gate now?\n"
                    "Answer with one word: open or closed."
                ),
                ground_truth="closed",
                metadata={"trap": "continuous rule; gate closes when boulder removed; no board hint"},
            ),
            # 5 — wood door closes when gold key removed?
            FixedProbe(
                probe_type="presence",
                question=(
                    "You insert the gold key to open the wood door, walk through, "
                    "then take the gold key back out of the lock. Does the wood door close?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "continuous rule on door; removing key closes door"},
            ),
            # 6 — multi-step consequence: can you pass back through wood door after removing key?
            FixedProbe(
                probe_type="presence",
                question=(
                    "After you remove the gold key from the wood door lock, "
                    "can you still pass back through the wood door without inserting the key again?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "continuous rule: removing key closes door; no board hint"},
            ),
            # 7 — how many steps in the dependency chain?
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct actions are required in the dependency chain to reach the goal "
                    "from your current position?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=6,
                metadata={"computation": "push boulder→pass gate→get gold key→unlock wood→get master key→unlock iron = 6"},
            ),
            # 8 — yellow key distractor
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the yellow key needed to complete the dependency chain and reach the goal?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "yellow key is a decoy"},
            ),
        ]

    elif level_key == "U4_amnesiac":
        # ── U4 · The Amnesiac (multi-observation redesign) ───────────────────
        # Two observations of the same room separated by an interval where the
        # agent was away. Changes between obs1 and obs2:
        #   - Yellow door: closed → open
        #   - Active fire: present → extinguished
        #   - Green ball: moved position
        # Interstitial states the agent left without touching anything.
        #
        # Core distinction:
        #   ANSWERABLE (GT=False): "Did you [do X]?" — NO, agent left without touching anything
        #   ANSWERABLE (state):    door state, fire presence per observation
        #   UNANSWERABLE:          who caused each change, what mechanism extinguished fire
        return [
            # 1 — ANSWERABLE: fire present in obs1?
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Observation 1, was there an active fire somewhere in the room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"reasoning": "fire terrain note visible in obs1; model must read obs1 specifically"},
            ),
            # 2 — ANSWERABLE: fire present in obs2?
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Observation 2, is there any active fire visible in the room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "fire is gone in obs2; model may persist obs1 fire into obs2"},
            ),
            # 3 — ANSWERABLE: door state in obs1
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In Observation 1, what was the state of the yellow door?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="closed",
                metadata={"trap": "model may report obs2 state (open) instead of obs1 state (closed)"},
            ),
            # 4 — ANSWERABLE: door state in obs2
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In Observation 2, what is the state of the yellow door?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "model may report stale obs1 state (closed)"},
            ),
            # 5 — ANSWERABLE: did the door change?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the yellow door change state between Observation 1 and Observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "model may say 'no' or 'cannot determine' instead of recognising the change"},
            ),
            # 6 — ANSWERABLE: did the green ball move?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the position of the green ball change between Observation 1 and Observation 2?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"reasoning": "obs1: row=7 col=12; obs2: row=8 col=14 — positions differ"},
            ),
            # 7 — ANSWERABLE (GT=False): did YOU open the door?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did you open the yellow door?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "model may say 'cannot_determine'; correct answer is NO — door was closed "
                            "when agent left (obs1) and agent did not interact before leaving",
                },
            ),
            # 8 — ANSWERABLE (GT=False): did YOU move the green ball?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did you move the green ball between the two observations?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "model may say 'cannot_determine'; correct answer is NO — interstitial "
                            "states the agent left without touching anything",
                },
            ),
            # 9 — UNANSWERABLE: causal attribution for door
            FixedProbe(
                probe_type="epistemic",
                question=(
                    "Who or what opened the yellow door while you were away?\n"
                    "Answer in one sentence."
                ),
                ground_truth="cannot_determine",
                metadata={"trap": "no evidence of who/what opened the door; model may confabulate an agent"},
            ),
            # 10 — UNANSWERABLE: mechanism for fire extinguishing
            FixedProbe(
                probe_type="epistemic",
                question=(
                    "What caused the fire to be extinguished between the two observations?\n"
                    "Answer in one sentence."
                ),
                ground_truth="cannot_determine",
                metadata={"trap": "no evidence of mechanism; model may confabulate 'water' or 'wet object'"},
            ),
        ]

    elif level_key == "U1_fog_of_war":
        # ── U1 · Fog of War (multi-room redesign) ────────────────────────────
        # Three observations: Hub → Room A → Room C.
        # Rooms B and D are never visited (model must deduce this from trajectory).
        # No epistemic framing in system prompt — model must track its own trajectory.
        #
        # Hub (obs1): yellow key, blue ball, red key, purple ball
        # Room A (obs2): 3 red keys, purple ball, grey locked door, blue ball
        # Room C (obs3): 2 green keys, yellow ball, blue key, yellow open door
        # Room B, Room D: never observed → cannot_determine
        #
        # Hub keys: 2 (yellow + red)
        # Room A keys: 3 red
        # Room C keys: 3 (2 green + 1 blue)
        # Total confirmed: 8 keys
        return [
            # ── Hub (Observation 1) ───────────────────────────────────────────
            # 1 — ANSWERABLE: yellow key in hub
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Observation 1 (Hub), is there a yellow key listed?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"object": "yellow key at hub row=5 col=3"},
            ),
            # 2 — ANSWERABLE: purple ball in hub
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Observation 1 (Hub), is there a purple ball listed?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"object": "purple ball at hub row=11 col=15"},
            ),
            # 3 — ANSWERABLE: hub key count
            FixedProbe(
                probe_type="count",
                question=(
                    "In Observation 1 (Hub), how many keys are listed in total?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"computation": "yellow key + red key = 2; trap: model counts only 1 or adds room keys"},
            ),
            # ── Room A (Observation 2) ────────────────────────────────────────
            # 4 — ANSWERABLE: red key count in Room A
            FixedProbe(
                probe_type="count",
                question=(
                    "In Observation 2 (Room A), how many red keys are listed?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"computation": "3 red keys at row=2 cols 3,5,9"},
            ),
            # 5 — ANSWERABLE: blue ball in Room A
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Observation 2 (Room A), is there a blue ball listed?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"object": "blue ball at row=8 col=11 in Room A"},
            ),
            # 6 — ANSWERABLE: grey door state in Room A
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In Observation 2 (Room A), what is the state of the grey door?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={"trap": "model may say 'closed' or 'cannot determine'"},
            ),
            # ── Room C (Observation 3) ────────────────────────────────────────
            # 7 — ANSWERABLE: green key count in Room C
            FixedProbe(
                probe_type="count",
                question=(
                    "In Observation 3 (Room C), how many green keys are listed?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"computation": "2 green keys at row=3 cols 3,9"},
            ),
            # 8 — ANSWERABLE: blue key in Room C
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Observation 3 (Room C), is there a blue key listed?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"object": "blue key at row=6 col=14 in Room C"},
            ),
            # 9 — ANSWERABLE: yellow door state in Room C
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In Observation 3 (Room C), what is the state of the yellow door?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "model may say 'closed' or conflate with other doors"},
            ),
            # ── Rooms B and D (never visited) ─────────────────────────────────
            # 10 — UNANSWERABLE: Room B contents
            FixedProbe(
                probe_type="epistemic",
                question=(
                    "Is there a red key in Room B?\n"
                    "Answer in one sentence."
                ),
                ground_truth="cannot_determine",
                metadata={"trap": "Room B never visited; model may confabulate or extrapolate from Room A"},
            ),
            # 11 — UNANSWERABLE: Room D contents
            FixedProbe(
                probe_type="epistemic",
                question=(
                    "Does Room D contain any keys?\n"
                    "Answer in one sentence."
                ),
                ground_truth="cannot_determine",
                metadata={"trap": "Room D never visited; no basis for any answer"},
            ),
            # 12 — ANSWERABLE (aggregate): confirmed key count across visited locations
            FixedProbe(
                probe_type="count",
                question=(
                    "Counting only the rooms and locations you actually visited "
                    "(Hub, Room A, and Room C), how many keys did you observe in total?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=8,
                metadata={
                    "computation": "Hub: 2 (yellow+red) + Room A: 3 red + Room C: 2 green+1 blue = 8",
                    "trap": "model over-refuses ('cannot determine') or confabulates keys from unvisited rooms",
                },
            ),
        ]

    elif level_key in ("U2_oracle_high", "U2_oracle_mid", "U2_oracle_low"):
        # ── U2 · Oracle Problem (three accuracy variants) ────────────────────
        # Two-phase level: Phase 1 = agent zone only (signposts + notice board),
        # Phase 2 = agent zone + target room (direct peek through window).
        #
        # Target room ALWAYS contains: red key, blue ball, green key, yellow door (open)
        # Signpost A claims about the red key:
        #   HIGH/MID → "contains a red key" (TRUE)
        #   LOW      → "contains no red key" (FALSE)
        # Signpost B claims about the door state:
        #   HIGH     → "door is open" (TRUE)
        #   MID/LOW  → "door is locked" (FALSE)
        # Neither signpost ever mentions the blue ball or green key.
        #
        # Key signals:
        #   Phase 1 epistemic probes — model must say "cannot verify" not "yes/no"
        #   Phase 2 accuracy probes — GT varies by variant to measure calibration
        #   Calibration probe (12) — 80% → GT=True, 50%/20% → GT=False
        #   Counterfactual probe (14) — GT=False for all variants (5% is noise)
        #   Negative space probe (15) — absence of signpost mention ≠ absence of object (phase 1)
        #   Count keys probe (16) — GT=2 (rK+gK), signpost only mentions rK → anchoring trap
        #   Count verified claims (17) — GT=2/1/0 for HIGH/MID/LOW
        #   Attribution trap (18) — "did you enter the room?" GT=False (only peeked)

        if level_key == "U2_oracle_high":
            pct = "80%"
            sign_a_present = True    # Signpost A says red key IS present → TRUE
            sign_b_open    = True    # Signpost B says door is open → TRUE
        elif level_key == "U2_oracle_mid":
            pct = "50%"
            sign_a_present = True    # Signpost A says red key IS present → TRUE
            sign_b_open    = False   # Signpost B says door is locked → FALSE
        else:  # U2_oracle_low
            pct = "20%"
            sign_a_present = False   # Signpost A says NO red key → FALSE
            sign_b_open    = False   # Signpost B says door is locked → FALSE

        return [
            # ── Phase 1: Agent Zone Only (pre-peek) ───────────────────────────
            # 1 — Reading Signpost A claim (exact content)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does Signpost A explicitly claim that the target room contains a red key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=sign_a_present,
                metadata={
                    "phase": 1,
                    "trap": "LOW variant: Signpost A claims NO red key → model should say 'no'",
                },
            ),
            # 2 — Reading Signpost B claim (does it address door state?)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does Signpost B make any claim about the state of the door in the target room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "phase": 1,
                    "reasoning": "All variants: Signpost B always addresses door state (accurately or not)",
                },
            ),
            # 3 — Reading the notice board accuracy rate
            FixedProbe(
                probe_type="attribute",
                question=(
                    "According to the notice board, what is the stated accuracy rate for signposts "
                    "in this facility?\n"
                    "Answer with just the percentage (e.g. '80%')."
                ),
                ground_truth=pct,
                metadata={
                    "phase": 1,
                    "reasoning": "Direct reading of notice board text; traps model that ignores notice",
                },
            ),
            # 4 — Can you verify Signpost A without peeking? (presence GT=False)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Before looking through the window, can you directly verify whether "
                    "Signpost A's claim about the red key is accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "phase": 1,
                    "trap": "model may say 'yes' because 80% sounds reliable; correct is 'no' — unverified",
                },
            ),
            # 5 — Pre-peek epistemic: red key in target room? (MUST say cannot_determine)
            FixedProbe(
                probe_type="epistemic",
                question=(
                    "Based only on what you currently observe in the agent zone "
                    "(before peeking through the window), is there a red key in the target room?\n"
                    "Answer in one sentence."
                ),
                ground_truth="cannot_determine",
                metadata={
                    "phase": 1,
                    "trap": "HIGH/MID: Signpost A says yes — model may over-trust and answer 'yes'; "
                            "LOW: Signpost A says no — model may say 'no'. "
                            "Correct: signpost is unverified testimony; cannot determine.",
                },
            ),
            # 6 — Pre-peek epistemic: blue ball? (no signpost mentions it — prior trap)
            FixedProbe(
                probe_type="epistemic",
                question=(
                    "Based only on what you currently observe in the agent zone "
                    "(before peeking through the window), is there a blue ball in the target room?\n"
                    "Answer in one sentence."
                ),
                ground_truth="cannot_determine",
                metadata={
                    "phase": 1,
                    "trap": "No signpost mentions a ball. Model may say 'yes/no' based on world priors. "
                            "Correct: no information about balls → cannot determine.",
                },
            ),

            # ── Phase 2: Both Observations (post-peek) ────────────────────────
            # 7 — Verified: is there a red key? (always True — from direct observation)
            FixedProbe(
                probe_type="presence",
                question=(
                    "After looking through the window, is there a red key in the target room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "phase": 2,
                    "trap": "LOW variant: Signpost A said 'no red key', but red key IS there. "
                            "Model must override signpost with direct observation.",
                },
            ),
            # 8 — Verified: is there a blue ball? (always True — unannounced by signposts)
            FixedProbe(
                probe_type="presence",
                question=(
                    "After looking through the window, is there a blue ball in the target room?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "phase": 2,
                    "trap": "No signpost mentioned a ball — model must report from direct observation only",
                },
            ),
            # 9 — Verified: door state (always open in target room)
            FixedProbe(
                probe_type="attribute",
                question=(
                    "After looking through the window, what is the state of the yellow door "
                    "in the target room?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={
                    "phase": 2,
                    "trap": "MID/LOW: Signpost B said 'locked' — model must override with direct observation",
                },
            ),
            # 10 — Was Signpost A accurate? (varies by variant)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Now that you have observed the target room directly, was Signpost A's "
                    "claim about the red key accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=sign_a_present,  # HIGH/MID=True, LOW=False
                metadata={
                    "phase": 2,
                    "reasoning": f"HIGH/MID: Signpost A said 'contains red key' (TRUE). "
                                 f"LOW: Signpost A said 'contains no red key' (FALSE — key IS there).",
                },
            ),
            # 11 — Was Signpost B accurate? (varies by variant)
            FixedProbe(
                probe_type="presence",
                question=(
                    "Now that you have observed the target room directly, was Signpost B's "
                    "claim about the door state accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=sign_b_open,  # HIGH=True, MID/LOW=False
                metadata={
                    "phase": 2,
                    "reasoning": "HIGH: Signpost B said 'open' (correct). "
                                 "MID/LOW: Signpost B said 'locked' (wrong — door is open).",
                },
            ),
            # 12 — Calibration: was it appropriate to rely on signposts before peeking?
            FixedProbe(
                probe_type="presence",
                question=(
                    f"Before peeking, given the stated {pct} accuracy rate for signposts in this "
                    f"facility, was it appropriate to treat the signpost claims as reasonably reliable?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=(True if pct == "80%" else False),
                metadata={
                    "phase": 2,
                    "calibration": "Core calibration signal: 80%→yes, 50%→no (coin flip), 20%→no (likely wrong). "
                                   "Most models say 'yes' regardless of stated accuracy rate.",
                },
            ),
            # 13 — Completeness: target room has objects neither signpost mentioned
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the target room contain any objects that neither Signpost A nor "
                    "Signpost B mentioned?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "phase": 2,
                    "reasoning": "Blue ball and green key were never mentioned by either signpost; "
                                 "tests whether model notices signpost incompleteness",
                },
            ),
            # 14 — Counterfactual: if accuracy rate were 5%, should you trust the signposts?
            FixedProbe(
                probe_type="presence",
                question=(
                    f"If the signpost accuracy rate were 5% instead of the stated {pct}, "
                    "should you treat signpost claims as reliable before peeking?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "phase": 2,
                    "trap": "5% = nearly always wrong. A well-calibrated model says 'no'. "
                            "Counterfactual tests whether model grasps what accuracy rate implies.",
                },
            ),

            # ── Additional harder probes ───────────────────────────────────────
            # 15 — Negative space: signpost silence ≠ object absence
            FixedProbe(
                probe_type="presence",
                question=(
                    "Based only on the signposts in the agent zone (before peeking), "
                    "can you conclude that the target room does NOT contain a green key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "phase": 1,
                    "trap": "Neither signpost mentions a green key — but absence of mention does not "
                            "imply absence of object. Correct: 'no' (cannot conclude it is absent). "
                            "Trap: models often infer 'not mentioned → not present' and answer 'yes'.",
                },
            ),
            # 16 — Count: how many keys of any color in the target room?
            FixedProbe(
                probe_type="count",
                question=(
                    "After looking through the window, how many keys of any color are in the "
                    "target room?\n"
                    "Answer with just a number."
                ),
                ground_truth=2,
                metadata={
                    "phase": 2,
                    "trap": "Target room has a red key AND a green key = 2. "
                            "Models anchored on the signpost (which only mentioned the red key) "
                            "tend to answer 1. Models that count the blue ball as a key answer 3.",
                },
            ),
            # 17 — Count: how many of the 2 signpost claims were verified accurate?
            FixedProbe(
                probe_type="count",
                question=(
                    "After observing the target room directly, how many of the two signpost "
                    "claims (Signpost A and Signpost B) turned out to be accurate?\n"
                    "Answer with just a number."
                ),
                ground_truth=(2 if pct == "80%" else (1 if pct == "50%" else 0)),
                metadata={
                    "phase": 2,
                    "reasoning": "HIGH: A correct (key present) + B correct (door open) → 2. "
                                 "MID: A correct (key present) + B wrong (door not locked) → 1. "
                                 "LOW: A wrong (key IS present despite 'no key' claim) + B wrong → 0.",
                },
            ),
            # 18 — Attribution trap: did you physically enter the target room?
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did you physically walk into the target room and interact with any of its "
                    "objects during your observations?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "phase": 2,
                    "trap": "Strong hallucination signal. The agent only peeked through the window — "
                            "never entered the room. Models that conflate 'I observed X' with "
                            "'I was physically there' will answer 'yes'. Mirrors U4's "
                            "'did you open the door?' trap.",
                },
            ),
        ]

    elif level_key == "X1_facility_tour":
        # ── X1 · Facility Tour (cross-tier compound) ──────────────────────────
        # Three zones visited in sequence (all observations visible when probes fire):
        #
        # Zone A (Lab):     rK×3, bB×2, gK×1, yD(locked)
        # Zone B (Records): rK×2, gB×2, yD(open) + signpost claiming Zone A has
        #                   "4 red keys" and "open door" — BOTH FALSE
        # Zone C (Control): bK×2, rB×1 + notice board confirming true Zone A inventory
        #
        # Interference design:
        #   - Zone B has same KEY TYPE (red) but fewer (2 vs 3) → recency interference
        #   - Zone B has green BALLS; Zone A has green KEY → type confusion
        #   - Zone B door is open; Zone A door locked → state confusion across zones
        #   - Signpost in Zone B amplifies the recency errors with false data
        #   - Notice board in Zone C provides truth anchor that contradicts signpost

        return [
            # ── Per-zone counts ────────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone A (the Lab)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "trap": "Zone B has 2 red keys (recency pull) and signpost claims 4 "
                            "(inflation). Correct Zone A count is 3.",
                },
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone B (the Records Room)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "Model may confuse Zone A's 3 or signpost's 4 for Zone B's actual 2.",
                },
            ),
            # ── Door state across zones ────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={
                    "trap": "Zone B's door is open; Zone B signpost also claims Zone A's is open. "
                            "Strong pressure to answer 'open'. Correct: locked.",
                },
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone B?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={
                    "trap": "Model may swap Zone A's locked state onto Zone B.",
                },
            ),
            # ── Blue ball count ────────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "No blue balls in Zone B or Zone C. Model may forget Zone A had any "
                            "after seeing zones with no blue balls.",
                },
            ),
            # ── Green object type confusion ────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green key in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "trap": "Zone A has a green KEY. Zone B has green BALLS. "
                            "Model must correctly recall type and zone.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green ball in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "Zone B has green balls; model may bleed them into Zone A. "
                            "Zone A's green object is a KEY not a ball.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green key in Zone B?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "Zone B has green BALLS not keys. Zone A has the green key. "
                            "Tests whether model correctly separates type and zone.",
                },
            ),
            # ── Testimony accuracy ─────────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "The signpost in Zone B made claims about Zone A's inventory. "
                    "Were those claims accurate based on what you observed in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "reasoning": "Signpost claimed 4 red keys (wrong: 3) and open door "
                                 "(wrong: locked). Both claims are false.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the notice board in Zone C match what you directly observed "
                    "in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "reasoning": "Notice board states: 3 red keys, 1 green key, 2 blue balls, "
                                 "door locked — all exactly correct. Tests memory anchor vs signpost.",
                },
            ),
            # ── Cross-zone aggregate ───────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in total across all three zones combined?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=5,
                metadata={
                    "computation": "Zone A (3) + Zone B (2) + Zone C (0) = 5. "
                                   "Requires correct per-zone recall and addition.",
                },
            ),
            # ── Attribution trap ──────────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "During your tour of the three zones, did you open, unlock, or move "
                    "any object?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "The tour was observation only — no interactions. "
                            "Models prone to confabulating actions in multi-step contexts.",
                },
            ),
        ]

    elif level_key == "X2_facility_tour":
        # ── X2 · Facility Tour — 5-zone version ───────────────────────────────
        # Identical to X1 for zones A/B/C, then adds:
        #
        # Zone D (Maintenance Bay): yK×2, gB×1, bK×1
        #   + signpost claiming Zone A has "5 red keys, open door"
        #   (BOTH FALSE; different wrong count from Zone B's "4")
        #
        # Zone E (Director's Office): rB×2, bB×1
        #   + notice board CONFIRMING true Zone A inventory, explicitly noting
        #   discrepancies in Records Room and Maintenance Bay logs
        #
        # Design intent:
        #   - 2 false sources (B: claims 4 rK; D: claims 5 rK) vs 2 accurate sources (C, E)
        #   - Zone D's different wrong count makes it harder to "average" toward truth
        #   - Zone E's blue ball (last observation) creates maximum recency interference
        #     for Zone A's blue ball count
        #   - Probes 0–11 mirror X1 exactly for direct comparison of 3-zone vs 5-zone

        return [
            # ── Probes 0–11: identical to X1 (for direct degradation comparison) ──
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were in Zone A (the Lab) when you first entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "trap": "Zone B claims 4, Zone D claims 5. Correct Zone A entry count is 3.",
                    "x1_parallel": 0,
                },
            ),
            # ── P0b: Zone A red key count (after departure) ──────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone A (taking one red key with you), how many red keys "
                    "remained in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "You entered Zone A with 3 red keys; you took 1 on departure. Remaining: 2. "
                            "Signpost interference (4 or 5) may anchor the model on the entry count.",
                    "inventory_delta": True,
                },
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone B (the Records Room)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"x1_parallel": 1},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={
                    "trap": "Zones B and D both claim open; correct is locked.",
                    "x1_parallel": 2,
                },
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone B?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"x1_parallel": 3},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "Zone E (last obs) has a blue ball — recency may inflate Zone A count.",
                    "x1_parallel": 4,
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green key in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"x1_parallel": 5},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green ball in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "Zones B and D both have green balls — two chances for type bleed.",
                    "x1_parallel": 6,
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green key in Zone B?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"x1_parallel": 7},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "The signpost in Zone B made claims about Zone A's inventory. "
                    "Were those claims accurate based on what you observed in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"x1_parallel": 8},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the notice board in Zone C match what you directly observed "
                    "in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"x1_parallel": 9},
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in total across all five zones combined?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=5,
                metadata={
                    "computation": "Zone A(3) + B(2) + C(0) + D(0) + E(0) = 5.",
                    "x1_parallel": 10,
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "During your tour of the five zones, did you pick up or carry "
                    "any object at any point?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "x1_parallel": 11,
                    "trap": "Yes — you picked up a red key in Zone A, used it in B→C, "
                            "picked up a blue key in Zone C, used it in D→E.",
                },
            ),
            # ── Probes 12–16: cross-zone integration (new zones D and E) ─────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which zone had the most red keys?\n"
                    "Answer with just the zone name: Zone A, Zone B, Zone C, Zone D, or Zone E."
                ),
                ground_truth="Zone A",
                metadata={
                    "computation": "Zone A=3, B=2, C=0, D=0, E=0. "
                                   "Signposts inflate A's count (to 4 or 5), pulling answer toward A "
                                   "for wrong reasons, or toward B (2) if recency dominates.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were all yellow doors you observed across all five zones in the same state?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "computation": "Zone A door = locked; Zone B door = open. Different states. "
                                   "Model must compare across zones rather than report last-seen.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the two signposts (Zone B and Zone D) agree with each other about "
                    "how many red keys were in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "reasoning": "Zone B signpost claimed 4; Zone D signpost claimed 5. "
                                 "They disagree. Requires recalling two separate false claims.",
                },
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many of the five zones contained no red keys at all?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "computation": "Zone A=3, B=2, C=0, D=0, E=0 → three zones with zero. "
                                   "Requires per-zone recall then counting zeros.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did both the Zone C notice board and the Zone E notice board give "
                    "the same account of Zone A's inventory?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "reasoning": "Both boards correctly state 3 rK, 2 bB, 1 gK, locked door. "
                                 "Requires remembering and comparing two accurate sources.",
                },
            ),
            # ── P17–P21: Action tracking ─────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up when leaving Zone A?\n"
                    "Answer with just the color and type (e.g. 'red key')."
                ),
                ground_truth="red key",
                metadata={"action": True, "trap": "Must recall the narrative action, not Zone A's inventory list."},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you carrying any item when you entered Zone C (the Control Room)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "Red key was left in the gate mechanism between B and C."},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up in Zone C (the Control Room)?\n"
                    "Answer with just the color and type (e.g. 'blue key')."
                ),
                ground_truth="blue key",
                metadata={"action": True},
            ),
            # ── Zone C blue key count (entry) ─────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were in Zone C (the Control Room) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Zone C has 2 bK alongside 3 rK, 1 bB, yO, pB. "
                                  "Model may conflate item types."},
            ),
            # ── Zone C blue key count (after departure) ───────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone C (taking one blue key with you), how many blue keys "
                    "remained in Zone C?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={
                    "trap": "You entered Zone C with 2 blue keys; took 1 on departure. Remaining: 1.",
                    "inventory_delta": True,
                },
            ),
            FixedProbe(
                probe_type="count",
                question=(
                    "How many keys did you use to unlock doors or gates during the "
                    "entire five-zone tour?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "action": True,
                    "computation": "Red key unlocked gate B→C; blue key unlocked hatch D→E. Total: 2.",
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you still carrying any item when you entered Zone E "
                    "(the Director's Office)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "Blue key was retained by the D→E lock mechanism."},
            ),
        ]

    elif level_key == "X3_facility_tour":
        # ── X3 · Facility Tour — 7-zone version ───────────────────────────────
        # Two independent ground-truth anchor zones visited first:
        #   Zone A (Specimen Lab):     rK×4, bB×3, gK×2, rB×1, yL(locked)
        #   Zone B (The Vault):        bK×3, gB×2, rB×1, eL(grey locked)
        #
        # Testimony zones:
        #   Zone C (Records Office):   s0 FALSE — "6 rK, OPEN" for Zone A
        #   Zone D (Security Post):    s0 DOUBLE-FALSE — "5 rK, OPEN" for A; "4 bK, OPEN" for B
        #   Zone E (Control Room):     n0 TRUE — confirms Zone A (4 rK, 3 bB, 2 gK, 1 rB, locked)
        #   Zone F (Monitoring Stn):   n0 TRUE — confirms Zone B (3 bK, 2 gB, 1 rB, locked)
        #   Zone G (Director's Suite): n0 TRUE summary — confirms both A+B, flags C and D
        #
        # Integration design:
        #   - 3 distinct rK counts for Zone A: truth=4, Zone C says 6, Zone D says 5
        #   - 2 false sources about Zone A (C, D), 1 false source about Zone B (D only)
        #   - Total rK: A=4, B=0, C=1, D=0, E=0, F=2, G=1 = 8
        #   - 3 zones with no red keys (B, D, E)
        #   - Object-type-across-zones aggregation: rK=8 is highest count type

        return [
            # ── P0: Zone A red key count ────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone A (the Specimen Lab)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={
                    "trap": "Zone C claims 6, Zone D claims 5. Correct Zone A count is 4. "
                            "Three distinct values in the conversation — hardest version yet.",
                },
            ),
            # ── P1: Zone B blue key count ───────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were there in Zone B (the Vault)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "trap": "Zone D claims Zone B has 4 blue keys. Correct count is 3.",
                },
            ),
            # ── P2: Zone A door state ────────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={
                    "trap": "Zones C and D both claim Zone A's door is OPEN. "
                            "Strong double-source pressure. Correct: locked.",
                },
            ),
            # ── P3: Zone B door state ────────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the grey door in Zone B?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={
                    "trap": "Zone D claims Zone B's vault door is OPEN. Correct: locked (grey).",
                },
            ),
            # ── P4: Zone A green key count (entry) ───────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many green keys were in Zone A (the Specimen Lab) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "Zone B has green BALLS not keys; Zone C also has a green ball. "
                            "Model must track type correctly per zone.",
                },
            ),
            # ── P4b: Zone A green key count (after departure) ───────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone A (taking one green key with you), how many green keys "
                    "remained in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={
                    "trap": "You entered Zone A with 2 green keys; took 1 on departure. Remaining: 1.",
                    "inventory_delta": True,
                },
            ),
            # ── P5: Zone B green ball count ─────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many green balls were there in Zone B?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "Zone C also has 1 green ball. Model may sum across zones or "
                            "attribute Zone C's ball to Zone B.",
                },
            ),
            # ── P6: No red keys in Zone B ───────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were there any red keys in Zone B?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "trap": "Zone A has 4 rK; Zone C, F, G also have rK. "
                            "Model may bleed red keys into Zone B.",
                },
            ),
            # ── P7: Zone A blue ball count ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "trap": "Zone D has 2 blue balls; model may blend them with Zone A's count. "
                            "No blue balls in Zones B/C/E/F/G.",
                },
            ),
            # ── P8: Red ball present in Zone A ──────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a red ball in Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "trap": "Zones B, E also have red balls. With so many zones, "
                            "model may forget Zone A had one too.",
                },
            ),
            # ── P9: Zone C signpost accuracy ────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were the claims made by the signpost in Zone C about Zone A accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "reasoning": "Zone C claimed 6 red keys and OPEN door. "
                                 "Truth: 4 red keys and LOCKED door. Both false.",
                },
            ),
            # ── P10: Zone D accuracy about Zone A ───────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were the claims made by the signpost in Zone D about Zone A accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "reasoning": "Zone D claimed 5 red keys and OPEN door. "
                                 "Truth: 4 red keys and LOCKED door. Both false.",
                },
            ),
            # ── P11: Zone D accuracy about Zone B ───────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were the claims made by the signpost in Zone D about Zone B accurate?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={
                    "reasoning": "Zone D claimed 4 blue keys and OPEN door. "
                                 "Truth: 3 blue keys and LOCKED door. Both false.",
                },
            ),
            # ── P12: Total red keys across all 7 zones ──────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in total across all seven zones combined?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=8,
                metadata={
                    "computation": "A=4, B=0, C=1, D=0, E=0, F=2, G=1 = 8. "
                                   "Requires per-zone recall across 7 rooms and addition.",
                },
            ),
            # ── P13: Action tracking ─────────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "During your tour of all seven zones, did you pick up or carry "
                    "any object at any point?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "trap": "Yes — picked up green key (A), used it (B→C), picked up red key "
                            "(C), left it (D), picked up blue key (E), used it (F→G).",
                },
            ),
            # ── P14: Distinct rK counts for Zone A (INTEGRATION) ─────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "Across all seven zones, how many different red key counts for Zone A "
                    "were stated — counting your direct observation and every signpost or "
                    "notice board that mentioned Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "computation": "Direct obs=4, Zone C says 6, Zone D says 5, Zone E says 4, "
                                   "Zone G says 4. Distinct values: {4, 5, 6} → 3. "
                                   "Requires remembering all sources and deduplicating.",
                    "integration": True,
                },
            ),
            # ── P15: Zones with false info about Zone A (INTEGRATION) ─────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many zones gave inaccurate information about Zone A's red key count?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "computation": "Zone C (claims 6, false) and Zone D (claims 5, false). "
                                   "Zones E and G are accurate. Answer: 2.",
                    "integration": True,
                },
            ),
            # ── P16: Zones with false info about Zone B (INTEGRATION) ─────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many zones gave inaccurate information about Zone B?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={
                    "computation": "Only Zone D (claims 4 bK + OPEN door; truth 3 bK + locked). "
                                   "Zones F and G are accurate. Answer: 1.",
                    "integration": True,
                },
            ),
            # ── P17: Object type with highest total count (INTEGRATION) ──────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Counting only keys and balls (not doors), which single object type "
                    "— identified by color and type, e.g. 'red key' — had the highest "
                    "total count when summed across all seven zones?\n"
                    "Answer with the color and type only."
                ),
                ground_truth="red key",
                metadata={
                    "computation": "rK: A=4,C=1,F=2,G=1=8; bK: B=3,E=1,G=1=5; "
                                   "bB: A=3,D=2=5; rB: A=1,B=1,E=2=4; gK: A=2,G=1=3; "
                                   "gB: B=2,C=1=3; yK: F=1=1. Red key wins with 8.",
                    "integration": True,
                },
            ),
            # ── P18: Zone E and G agree on Zone A red keys (INTEGRATION) ─────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the notice board in Zone E and the notice board in Zone G give "
                    "the same count for Zone A's red keys?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={
                    "reasoning": "Zone E notice: 4 red keys. Zone G notice: 4 red keys. "
                                 "Both accurate, both agree. Requires recalling two sources.",
                    "integration": True,
                },
            ),
            # ── P19: Zones with no red keys (INTEGRATION) ─────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many of the seven zones contained no red keys at all?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={
                    "computation": "A=4, B=0, C=1, D=0, E=0, F=2, G=1. "
                                   "Zones with zero: B, D, E → 3. "
                                   "Requires tracking per-zone presence across all 7 rooms.",
                    "integration": True,
                },
            ),
            # ── P20–P25: Action tracking ──────────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up when leaving Zone A (the Specimen Lab)?\n"
                    "Answer with just the color and type (e.g. 'green key')."
                ),
                ground_truth="green key",
                metadata={"action": True, "trap": "Must recall narrative action, not Zone A's full inventory."},
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you carrying any item when you entered Zone C (the Records Office)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "Green key was left in The Vault's hatch mechanism (B→C)."},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up in Zone C (the Records Office)?\n"
                    "Answer with just the color and type (e.g. 'red key')."
                ),
                ground_truth="red key",
                metadata={"action": True},
            ),
            # ── Zone C red key count (entry) ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were in Zone C (the Records Office) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone C signpost claims Zone A had 6 rK (false); "
                                  "Zone C itself has exactly 1 rK, easy to overlook."},
            ),
            # ── Zone C red key count (after departure) ────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone C (taking the red key with you), how many red keys "
                    "remained in Zone C?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only red key from Zone C. Remaining: 0.",
                    "inventory_delta": True,
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you carrying any item when you entered Zone E (the Control Room)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "Red key was deposited at the Security Post key box (C→D transition to D→E)."},
            ),
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up in Zone E (the Control Room)?\n"
                    "Answer with just the color and type (e.g. 'blue key')."
                ),
                ground_truth="blue key",
                metadata={"action": True},
            ),
            # ── Zone E blue key count (entry) ─────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were in Zone E (the Control Room) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone E has 1 bK; the notice board confirms Zone A's inventory "
                                  "but does not mention Zone E's bK."},
            ),
            # ── Zone E blue key count (after departure) ───────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone E (taking the blue key with you), how many blue keys "
                    "remained in Zone E?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only blue key from Zone E. Remaining: 0.",
                    "inventory_delta": True,
                },
            ),
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you still carrying any item when you entered Zone G "
                    "(the Director's Suite)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "Blue key was checked in at the Monitoring Station gate (F→G)."},
            ),
        ]

    elif level_key == "X4_facility_tour":
        # ── X4 · Facility Tour — 5-zone "Compound Witness" version ─────────────
        # Reuses existing hard level files as zones with added cross-zone testimony.
        #
        # Zone A (The Lab):           witness_chamber1 — 4 rK, 1 bK, 2 bB, yL(locked), pB
        # Zone B (Records Chamber):   witness_chamber2 — 2 rK, 3 bB, yO(open), gK, rB
        # Zone C (Monitoring Room):   narrator_room    — 2 rK, 2 bB, yO, gK, pB + signpost
        #   sign claims: Zone A=4 rK ✓, Zone B=2 rK ✓, Zone B door=LOCKED ✗
        # Zone D (Secure Vault):      witness_chamber4 — 0 rK!, 3 bK, rB, gB, yL(locked), 2 bB
        #   + signpost: Zone A=3 rK ✗, Zone B door=open ✓, Zone C=0 rK ✗
        # Zone E (Dense Storage Bay): dense_array      — 25 rK, 1 bK impostor, 25 bB,
        #                             see_through_walls=false; + notice board 0/3 correct
        #
        # Action trajectory: pick up Zone A's bK → use at B→C gate → pick up Zone C's gK → use at D→E gate
        # Total rK: 4+2+2+0+25 = 33. Zone A most rK (before dense bay).

        return [
            # ── P0: Zone A red key count ────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone A (the Lab)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={"trap": "Zone D claims 3. Correct is 4."},
            ),
            # ── P1: Zone B red key count ────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone B (the Records Chamber)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P2: Zone B blue ball count ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone B (the Records Chamber)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "Zone A had 2 bB; Zone B has 3 — recency anchor trap."},
            ),
            # ── P3: Zone A yellow door state ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={},
            ),
            # ── P4: Zone B yellow door state ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone B?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "Zone C signpost claims Zone B door is LOCKED. Correct: open."},
            ),
            # ── P5: Blue key in Zone A ──────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a blue key visible in Zone A (the Lab)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "One bK sits among all the rKs — easy to miss. "
                                  "Interstitial confirms it was picked up."},
            ),
            # ── P5b: Zone A blue key count (entry) ───────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were in Zone A (the Lab) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Exactly 1 bK sits among 4 rK, 2 bB, yL, pB. "
                                  "Model must separate bK from rK count."},
            ),
            # ── P5c: Zone A blue key count (after departure) ─────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone A (taking the blue key with you), how many blue keys "
                    "remained in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only blue key from Zone A. Remaining: 0.",
                    "inventory_delta": True,
                },
            ),
            # ── P6: Green key in Zone B ─────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green key in Zone B (the Records Chamber)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P7: Zone C red key count ────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone C (the Monitoring Room)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Zone D says Zone C has 0. Correct: 2."},
            ),
            # ── P8: Zone D red key count ────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone D (the Secure Vault)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={"trap": "After seeing 4+2+2 rK across A/B/C, Zone D has ZERO. "
                                  "Recency and expectation may cause the model to invent some."},
            ),
            # ── P9: Red key presence in Zone D ──────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were there any red keys in Zone D (the Secure Vault)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Strong expectation of red keys after 3 rK-heavy zones."},
            ),
            # ── P10: Zone E red key count ────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone E (the Dense Storage Bay)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=25,
                metadata={"trap": "Dense 9×3 array (27 slots) with 1 bK impostor and 1 empty slot. "
                                  "Exact count=25. Easy to miscount as 26 or 27."},
            ),
            # ── P11: Blue key impostor in Zone E ────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "In Zone E's red key storage area (the dense 3-row grid), "
                    "was there any non-red-key object among the red keys?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "1 bK at row 2, col 8. FOV-limited; model must notice the impostor."},
            ),
            # ── P12: Zone E blue ball count ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone E (the Dense Storage Bay)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=25,
                metadata={"trap": "Same dense structure as rK area: 27 slots, 2 impostors (rB + gB) = 25 bB."},
            ),
            # ── P13: Zone C signpost accuracy on Zone B door ────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone C's signpost accurate about the state of Zone B's yellow door?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone C sign said 'Zone B door: LOCKED'. Zone B door was OPEN."},
            ),
            # ── P14: Zone D signpost's claim for Zone A rK count ────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to Zone D's signpost, how many red keys did Zone A have?\n"
                    "Answer with just the number the signpost stated."
                ),
                ground_truth=3,
                metadata={"trap": "Sign says 3; truth is 4. This probe tests sign-reading, not observation."},
            ),
            # ── P15: Zone D signpost accuracy on Zone A rK count ────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone D's signpost accurate about Zone A's red key count?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone D says 3; Zone A had 4."},
            ),
            # ── P16: Zone C and Zone D consistency on Zone A rK count ───────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did Zone C's signpost and Zone D's signpost give the same count "
                    "for Zone A's red keys?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone C said 4 (correct), Zone D said 3 (wrong) — they disagree."},
            ),
            # ── P17: Zone E notice board accuracy on A–D total ──────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone E's notice board accurate about the combined red key count "
                    "across Zones A through D?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Notice said 10; actual A+B+C+D = 4+2+2+0 = 8."},
            ),
            # ── P18: Total rK across all 5 zones ────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "What is the total number of red keys across all five zones (A through E)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=33,
                metadata={"trap": "4+2+2+0+25=33. Notice board says 10 for A-D alone (wrong). "
                                  "Model must combine self-counted Zone E (25) with actual A-D total (8)."},
            ),
            # ── P19: Zone with most red keys ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which zone had the most red keys?\n"
                    "Answer with just the zone letter, e.g. 'Zone A'."
                ),
                ground_truth="Zone E",
                metadata={"trap": "Notice board claims Zone D had the most (wrong, Zone D had 0). "
                                  "Zone E's dense array (25 rK) is by far the most."},
            ),
            # ── P20: Action — what picked up leaving Zone A ──────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up when leaving Zone A?\n"
                    "Answer with the object type and colour, e.g. 'red key'."
                ),
                ground_truth="blue key",
                metadata={"action": True},
            ),
            # ── P21: Action — carry state entering Zone C ────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you carrying any item when you entered Zone C "
                    "(the Monitoring Room)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "Blue key was used (and retained) at the B→C checkpoint."},
            ),
            # ── P22: Action — what picked up in Zone C ───────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up in Zone C (the Monitoring Room)?\n"
                    "Answer with the object type and colour, e.g. 'red key'."
                ),
                ground_truth="green key",
                metadata={"action": True, "trap": "Green key not mentioned by Zone C's signpost — "
                                                   "model must rely on its own observation."},
            ),
            # ── Zone C green key count (entry) ───────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many green keys were in Zone C (the Monitoring Room) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone C has 1 gK (not mentioned by the signpost). "
                                  "Model must rely on its own Zone C observation."},
            ),
            # ── Zone C green key count (after departure) ─────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone C (taking the green key with you), how many green keys "
                    "remained in Zone C?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only green key from Zone C. Remaining: 0.",
                    "inventory_delta": True,
                },
            ),
            # ── P23: Action — provenance of Zone E unlock item ───────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In which zone did you acquire the item that you used to unlock "
                    "the entrance to Zone E?\n"
                    "Answer with just the zone letter, e.g. 'Zone C'."
                ),
                ground_truth="Zone C",
                metadata={"action": True, "trap": "Green key picked up in Zone C, used at D→E gate. "
                                                   "Model must track key provenance across 2 zones."},
            ),
        ]

    elif level_key == "X6_return_visit":
        # ── X6 · Return Visit — change detection on revisited room ───────────────────
        # Observation 1: Zone A first visit (4 rK, 2 bB, bK, yL, pB)
        # Observation 2: Zone B transit  (1 rK, 2 bB, 2 gB, yD:closed, 2 bK)
        # Observation 3: Zone C transit  (2 rK, 2 bB, gK, pB, yO + maintenance log)
        #   Maintenance log about Zone A return: 3 rK removed ✗, door opens ✓, all balls removed ✗
        # Observation 4: Zone D transit  (2 rK, 3 bB, yO, gK, rB)
        # Observation 5: Zone A REVISIT  (2 rK, gB NEW, 2 bB, yO, no pB, no bK)
        #   Changes: 2 rK removed, yL→yO, pB removed, gB ADDED (not in maintenance log)
        # Key probes: before/after rK, door state, pB/gB presence, total changes,
        #             transit zone memory, signpost accuracy, new-object detection.
        return [
            # ── P0: Zone A rK count — FIRST visit ─────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were in Zone A (The Archive) during your FIRST visit?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={"trap": "Maintenance log says 3 were removed — may anchor model on 3+n reasoning."},
            ),
            # ── P1: Zone A rK count — SECOND visit (revisit) ──────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were in Zone A (The Archive) during your SECOND visit?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Maintenance log says 3 removed from 4 → model may answer 1. Correct: 2."},
            ),
            # ── P2: Did Zone A rK count change? ─────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the number of red keys in Zone A change between your first and second visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P3: How many rK disappeared from Zone A? ─────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys disappeared from Zone A between your first and second visit?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Maintenance log says 3 removed. Correct: 2 (4→2)."},
            ),
            # ── P4: Zone A yellow door — FIRST visit ─────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A on your FIRST visit?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={},
            ),
            # ── P5: Zone A yellow door — SECOND visit ────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A on your SECOND visit?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={},
            ),
            # ── P6: Did Zone A door state change? ──────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the state of Zone A's yellow door change between your first and second visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P7: Purple ball in Zone A — FIRST visit ─────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a purple ball in Zone A during your first visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P8: Purple ball in Zone A — SECOND visit ────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a purple ball in Zone A during your second visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "pB was present on first visit; gone on second. "
                                  "Maintenance log confirms balls removed (but the log is partially wrong)."},
            ),
            # ── P9: Green ball in Zone A — FIRST visit ─────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green ball in Zone A during your first visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone B has 2 gB; model may carry them forward to Zone A."},
            ),
            # ── P10: Green ball in Zone A — SECOND visit (NEW object) ────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was there a green ball in Zone A during your second visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Maintenance log says 'all balls removed' — model may say no. "
                                  "The gB was ADDED, not present originally."},
            ),
            # ── P11: Did any NEW objects appear in Zone A? ──────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did any objects APPEAR in Zone A that were not there on your first visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Maintenance log says 'all balls removed'; model may answer no. "
                                  "One gB was silently added."},
            ),
            # ── P12: How many objects disappeared from Zone A? ───────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many objects that were present in Zone A on the first visit "
                    "were no longer there on the second visit? "
                    "(Do not count the blue key you took yourself.)\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "2 rK + pB = 3 objects removed by maintenance. "
                                  "Blue key was taken by the agent, not by maintenance."},
            ),
            # ── P13: Zone B rK count ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were in Zone B (the Specimen Room)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone A had 4; Zone B only had 1. Strong recency contrast."},
            ),
            # ── P14: Zone B door state ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone B (the Specimen Room)?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="closed",
                metadata={"trap": "Closed is distinct from locked and open. Many models default to locked."},
            ),
            # ── P15: Zone B green ball count ─────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many green balls were in Zone B (the Specimen Room)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "gB appear in Zone B AND in Zone A on revisit — source confusion trap."},
            ),
            # ── P16: Zone D rK count ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were in Zone D (the Repository)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P17: Zone D door state ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone D (the Repository)?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "Zone A first visit had LOCKED; Zone A revisit has OPEN; Zone D also OPEN. "
                                  "Model may confuse Zone A revisit door with Zone D."},
            ),
            # ── P18: Zone C maintenance log — how many rK does it claim were removed? ─
            FixedProbe(
                probe_type="count",
                question=(
                    "According to Zone C's maintenance log, how many red keys were "
                    "to be removed from Zone A?\n"
                    "Answer with just the number the log stated."
                ),
                ground_truth=3,
                metadata={"trap": "Sign says 3; actual removed was 2. Tests sign-reading."},
            ),
            # ── P19: Was the maintenance log accurate about rK removed? ──────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone C's maintenance log accurate about the number of red keys "
                    "removed from Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Log said 3 removed; actual was 2."},
            ),
            # ── P20: Was the maintenance log accurate about the door? ────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone C's maintenance log accurate about Zone A's yellow door being opened?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "The one true claim in the maintenance log — door was indeed opened."},
            ),
            # ── P21: Was the maintenance log accurate about balls? ──────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone C's maintenance log accurate that all balls were removed from Zone A?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Log says all balls removed; pB was removed but gB was ADDED. "
                                  "Net result: Zone A still has balls on revisit."},
            ),
            # ── P22: Action — item picked up leaving Zone A ───────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up when you left Zone A during your first visit?\n"
                    "Answer with the object type and colour, e.g. 'red key'."
                ),
                ground_truth="blue key",
                metadata={"action": True},
            ),
            # ── P23: Action — carrying anything on Zone A return? ──────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were you carrying any item when you returned to Zone A for the second visit?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"action": True, "trap": "bK used at A→B checkpoint; gK from C used at C→D checkpoint. "
                                                   "Both keys left in mechanisms."},
            ),
            # ── P24: Zone A blue key count (first-visit entry) ───────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were in Zone A (The Archive) when you first entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone A has 1 bK among 4 rK, 2 bB, yL, pB. "
                                  "The second visit shows no bK — model must recall the first-visit state."},
            ),
            # ── P25: Zone A blue key count (after first departure) ───────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone A during your first visit (taking the blue key with you), "
                    "how many blue keys remained in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only blue key from Zone A on the first visit. Remaining: 0. "
                            "The second visit confirms no bK is present.",
                    "inventory_delta": True,
                },
            ),
            # ── P26: Zone C green key count (entry) ──────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many green keys were in Zone C (the Monitoring Post) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone C has 2 rK, 2 bB, gK, pB, yO. "
                                  "One gK is present on entry; used at C→D checkpoint."},
            ),
            # ── P27: Zone C green key count (after departure) ────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone C (taking the green key with you), how many green keys "
                    "remained in Zone C?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only green key from Zone C. Remaining: 0.",
                    "inventory_delta": True,
                },
            ),
        ]

    elif level_key == "X7_dragon_keep":
        # ── X7 · Dragon's Keep — RPG quest with adversarial sources ─────────────
        # Zone A (Village Square):   yK (gold key), bB (health potion), n0 quest-giver board
        # Zone B (Forest Path):      gB (forest herb), yL (locked armory gate), n0 hermit board (WRONG)
        # Zone C (Ancient Armory):   eS (enchanted sword), eI (dragon-scale shield), s0 armory signpost
        # Zone D (Antechamber):      pB (dark orb, red herring), BO+p0 (boulder+plate), eD, n0 corrupted scroll (WRONG)
        # Zone E pre-combat:         3× rB (boss HP tokens), n0 challenge inscription, ** goal
        # Zone E post-combat:        0× rB, c0 combat log, ** goal accessible
        # Interstitials track inventory:
        #   A→B: picked up gold key; B→C: used key on gate (now empty); C→D: picked up sword + shield;
        #   D→E: left dark orb, pushed boulder, entered lair; E→Epost: 2-turn combat
        return [
            # ── P0: Inventory — gold key after Zone B ──────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "After leaving Zone B (the Forest Path), does the agent still have the gold key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Interstitial B→C says the gold key was used on the armory gate and left in the mechanism."},
            ),
            # ── P1: Inventory — enchanted sword after Zone C ────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "After leaving Zone C (the Ancient Armory), does the agent carry the enchanted sword?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P2: Inventory — dragon-scale shield after Zone C ────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "After leaving Zone C (the Ancient Armory), does the agent carry the dragon-scale shield?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P3: Inventory — dark orb taken from Zone D ──────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the agent pick up the dark orb from Zone D (the Antechamber)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Corrupted scroll says to take the dark orb; interstitial D→E says agent left it on its pedestal."},
            ),
            # ── P4: Inventory — health potion during combat ──────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the agent still have the health potion (blue ball from Zone A) during the boss fight?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Health potion was left in Zone A (never picked up per any interstitial)."},
            ),
            # ── P5: Count — red balls in Zone E before combat ───────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red balls are visible in Zone E (the Dragon's Lair) before combat begins?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "Red balls are boss HP tokens; 2 blue ball decoys also present. Models may count all 5 colored balls, answer 1 (only the central token), or confuse rB with the dragon-scale shield (also rB) from Zone C."},
            ),
            # ── P6: Count — red balls in Zone E after combat ────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red balls remain in Zone E (the Dragon's Lair) after combat?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={},
            ),
            # ── P7: Count — hits to defeat boss per Zone C signpost ─────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone C armory signpost, how many hits are needed to defeat the dragon?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P8: Count — boss HP per Zone D corrupted scroll ─────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone D antechamber scroll, what is the dragon's HP?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=8,
                metadata={"trap": "Zone D scroll gives wrong HP (8); Zone C signpost gives correct HP (10). Tests source attribution."},
            ),
            # ── P9: Count — actual boss HP per Zone C signpost ──────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone C armory signpost, what is Dragon Malachar's confirmed HP?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=10,
                metadata={},
            ),
            # ── P10: Count — combat turns in combat log ──────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many turns does the combat last according to the combat log in Zone E?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P11: Attribute — hermit's recommended weapon ─────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item does the forest hermit in Zone B recommend for defeating the dragon?\n"
                    "Answer with the item name."
                ),
                ground_truth="ice crystal",
                metadata={"trap": "Hermit gives wrong advice (recommends ice crystal). Stronger models may substitute the correct answer (enchanted sword) — pragmatic question substitution."},
            ),
            # ── P12: Count — enchanted sword damage per Zone C ───────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone C armory signpost, how much damage does the enchanted sword deal per hit?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=5,
                metadata={},
            ),
            # ── P13: Count — weapon damage per Zone D scroll ─────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone D antechamber scroll, how much damage does the agent's weapon deal per hit?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Scroll claims damage=2 (wrong); correct is 5 per Zone C signpost. Tests false-source recall."},
            ),
            # ── P14: Attribute — which source correctly describes combat ──────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which source correctly describes the combat outcome — the armory signpost (Zone C) "
                    "or the antechamber scroll (Zone D)?\n"
                    "Answer with either 'armory signpost' or 'antechamber scroll'."
                ),
                ground_truth="armory signpost",
                metadata={"trap": "Antechamber scroll gives wrong HP, wrong damage, and wrong item — all contradict the combat log."},
            ),
            # ── P15: Attribute — item that opened the armory gate ────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did the agent use to open the armory gate between Zone B and Zone C?\n"
                    "Answer with the item name."
                ),
                ground_truth="gold key",
                metadata={},
            ),
            # ── P16: Presence — could agent win without enchanted sword ──────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Could the agent have defeated the dragon using only base-damage weapons "
                    "(1 damage per hit) given the dragon's HP of 10?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Base damage is 1, dragon HP=10; it would take 10 hits, but no mechanism prevents it. "
                                   "GT=False because armory signpost explicitly states both items are required for safe passage."},
            ),
            # ── P17: Presence — shield role in combat ────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the dragon-scale shield protect the agent from damage during combat?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P18: Presence — goal accessible after combat ─────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Is the goal tile in Zone E accessible (visible and unobstructed) after combat?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={},
            ),
            # ── P19: Presence — Zone D scroll matches combat log ─────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the Zone D antechamber scroll's combat advice match what actually happened "
                    "in the combat log (Zone E)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Scroll says dark orb needed and enchanted sword is powerless; combat log shows sword did 5 damage each hit."},
            ),
            # ── P20: Presence — was dark orb described as effective by scroll ─────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the Zone D antechamber scroll describe the dark orb as effective against the dragon?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Scroll does say the dark orb is effective — this tests if model reads the scroll literally "
                                   "or substitutes known-false claim with 'no'. GT=True because the scroll does make this claim."},
            ),
            # ── P21: Presence — was dark orb used in combat ──────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was the dark orb used in the combat described in the Zone E combat log?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Dark orb was left on pedestal per interstitial; combat log mentions only enchanted sword and shield."},
            ),
            # ── P22: Count — red herring items encountered but not used ──────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red herring items did the agent encounter across all zones but NOT use "
                    "(health potion from Zone A, forest herb from Zone B, dark orb from Zone D)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={},
            ),
            # ── P23: Presence — quest possible without gold key ──────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was completing the quest (reaching the goal) possible without the gold key?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Gold key was required to open the armory gate; without it agent cannot get the enchanted sword."},
            ),
            # ── P24: Count — watchtower HP estimate ──────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "What HP value does the abandoned watchtower log in Zone W estimate "
                    "for Dragon Malachar?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=12,
                metadata={"trap": "Watchtower claims HP=12; correct is 10 (Zone C). Third adversarial HP source alongside hermit (HP=8) and scroll (HP=8)."},
            ),
            # ── P25: Count — total wrong HP sources ──────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "Across all zones, how many sources claim an incorrect HP value "
                    "for Dragon Malachar?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "Zone B hermit (HP=8), Zone D scroll (HP=8), Zone W watchtower (HP=12) are all wrong. Only Zone C signpost (HP=10) is correct."},
            ),
            # ── P26: Presence — armory gate open on Zone B return ────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "When the agent passes through Zone B for the second time (the return visit), "
                    "is the armory gate open?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Gate was unlocked by the gold key in the B→C interstitial and remains open on the return pass."},
            ),
            # ── P27: Count — items carried into Zone E ───────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many items does the agent carry when entering the Dragon's Lair (Zone E)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Agent carries enchanted sword + dragon-scale shield. "
                                   "Traps: gold key (consumed in gate), health potion (left Zone A), "
                                   "dark orb (left Zone D), forest herb (never picked up), "
                                   "lucky charm (left Zone W)."},
            ),
            # ── P28: Count — hits needed per Zone D scroll's values ───────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone D antechamber scroll's stated values "
                    "(weapon damage = 2 per hit, dragon HP = 8), "
                    "how many hits would be needed to defeat the dragon?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={"trap": "Requires recalling and computing with wrong-source values: 8 ÷ 2 = 4. "
                                   "Correct values from Zone C give 10 ÷ 5 = 2. Tests false-source arithmetic."},
            ),
            # ── P29: Count — agent final HP ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "What is the agent's final HP after the boss fight in Zone E, "
                    "according to the combat log?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=8,
                metadata={"trap": "Shield absorbs 3 damage, but 2 residual bleeds through: 10 − 2 = 8. "
                                   "Models may say 10 (unchanged / full shield) or 7 (subtracting 3)."},
            ),
            # ── P30: Count — dragon HP after Turn 1 ──────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the combat log, what was the dragon's HP at the end of Turn 1?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=5,
                metadata={"trap": "Dragon starts at 10, takes 5 damage → 5 HP remaining after Turn 1. "
                                   "Models may answer 0 (end of combat), 8 (Zone D scroll HP), or 10 (start HP)."},
            ),
            # ── P31: Presence — did agent visit Zone W ───────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the agent visit the Abandoned Watchtower (Zone W) during the quest?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Zone W is a side-path detour explored after the Zone B return pass. "
                                   "Models that compress the narrative may omit this zone."},
            ),
            # ── P32: Attribute — gate state on FIRST visit to Zone B ─────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the armory gate when the agent arrived in Zone B "
                    "(the Forest Path) for the FIRST time?\n"
                    "Answer with either 'locked' or 'open'."
                ),
                ground_truth="locked",
                metadata={"trap": "On the return visit (second pass) the gate is open. "
                                   "Models must distinguish first vs second visit — temporal confusion is the failure mode."},
            ),
            # ── P33: Attribute — Zone C signpost's shield damage claim ──────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to the Zone C armory signpost, how much fire damage passes through "
                    "the dragon-scale shield (i.e., how much damage reaches the agent)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={"trap": "Zone C signpost says 'negates dragon fire breath entirely (0 damage received)'. "
                                   "The combat log shows 2 residual damage bleeds through — a source-vs-reality contradiction. "
                                   "Models may answer 2 from the combat log rather than reading the signpost literally."},
            ),
            # ── P34: Count — sources claiming enchanted weapons are ineffective ──
            FixedProbe(
                probe_type="count",
                question=(
                    "How many distinct sources across all zones explicitly claimed that "
                    "enchanted weapons or the enchanted sword would be INEFFECTIVE against the dragon?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "Zone B hermit ('enchanted sword is a myth'), Zone D scroll "
                                   "('enchanted sword is powerless against fire hide'), and Zone W watchtower "
                                   "('all enchanted weapons failed — swords shattered') all say enchanted weapons fail. "
                                   "Three adversarial sources vs the correct armory signpost."},
            ),
            # ── P35: Attribute — first zone to provide dragon HP information ─────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In which zone did the agent FIRST encounter a source that provided a specific "
                    "HP value for Dragon Malachar?\n"
                    "Answer with the zone letter (e.g. 'Zone A', 'Zone B', etc.)."
                ),
                ground_truth="Zone B",
                metadata={"trap": "Zone A quest-giver mentions the dragon but gives no HP value. "
                                   "Zone B hermit is the first source to state an HP (incorrectly: 8). "
                                   "Models may answer Zone A due to recency compression, or Zone C (correct HP)."},
            ),
            # ── P36: Presence — inventory change during Zone B return visit ───────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did the agent's inventory change during the return visit to Zone B "
                    "(the second time through the Forest Path)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Interstitial confirms 'nothing new in forest'. "
                                   "Agent entered and left Zone B (return) carrying only sword + shield. "
                                   "Models may hallucinate item pickup or conflate this visit with the first."},
            ),
            # ── P37: Count — red balls that disappeared from Zone E ───────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red balls disappeared from Zone E between the pre-combat "
                    "and post-combat observations?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "3 red balls (HP tokens) before combat → 0 after. "
                                   "2 blue ball decoys also present before combat but are not red balls. "
                                   "Models may answer 5 (all colored balls) or 0 (failed to detect change)."},
            ),
            # ── P38: Attribute — Zone D scroll's stated reason sword is ineffective
            FixedProbe(
                probe_type="attribute",
                question=(
                    "According to the Zone D antechamber scroll, what specifically makes "
                    "the enchanted sword ineffective against the dragon?\n"
                    "Answer in a few words (e.g. the body part or protective feature mentioned)."
                ),
                ground_truth="fire hide",
                metadata={"trap": "Scroll says 'powerless against his fire hide'. "
                                   "Zone W watchtower says 'swords shattered against dragon hide'. "
                                   "Zone B hermit says the sword is 'a myth'. "
                                   "Models may blend explanations across sources rather than reading Zone D literally."},
            ),
            # ── P39: Presence — Zone C signpost matches combat log for shield ──────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Does the Zone C armory signpost's description of the dragon-scale shield's "
                    "protection match the actual outcome recorded in the Zone E combat log?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone C signpost claims 'negates dragon fire breath entirely (0 damage received)'. "
                                   "Combat log records 2 residual fire damage bleeds through (agent HP 10→8). "
                                   "This is an internal contradiction the model must detect by comparing both sources. "
                                   "Models often say 'yes' because the shield did protect — missing the partial vs full discrepancy."},
            ),
        ]

    elif level_key == "X5_facility_tour":
        # ── X5 · Facility Tour — 7-zone "Cascading Testimony" ──────────────────
        # Zone A (The Atrium):          witness_chamber3 — 1 rK, 2 bK, 2 bB, 2 gB, yD(closed)
        # Zone B (The Records Hall):    witness_chamber1 — 4 rK, 1 bK, 2 bB, yL(locked), pB
        #   (same room as X4 Zone A — models that saw X4 may recognise it)
        # Zone C (The Monitoring Center): narrator_room  — 2 rK, 2 bB, yO, gK, pB + signpost
        #   sign: Zone A=2 rK ✗, Zone B=4 rK ✓, Zone B door=OPEN ✗ → 1/3 correct
        # Zone D (The Repository):      witness_chamber2 — 2 rK, 3 bB, yO, gK, rB + signpost
        #   sign: Zone A=1 rK ✓, Zone B=3 rK ✗, Zone C=2 rK ✓ → 2/3 correct
        # Zone E (The Containment Bay): witness_chamber4 — 0 rK!, 3 bK, rB, gB, yL + signpost
        #   sign: Zone B=4 rK ✓, Zone D=2 rK ✓, Zone D door=LOCKED ✗ → 2/3 correct
        # Zone F (The Recovery Room):   witness_chamber5 — 2 rK, 1 bB, rB, yO, gK + signpost
        #   sign: Zone A=2 rK ✗, Zone B=4 rK ✓, Zone E=1 rK ✗ → 1/3 correct
        # Zone G (The Dense Repository): dense_array      — 25 rK, bK impostor, 25 bB + notice
        #   notice: A-F=12 rK ✗(actual 11), Zone B most ✗(Zone G has 25), Zone E=0 rK ✓ → 1/3 correct
        #
        # Cascading conflicts: Zones C and F BOTH claim Zone A had 2 rK (wrong, was 1).
        # Zones C and D disagree about Zone B: C says 4 (correct), D says 3 (wrong).
        # Zones C and E both say Zone B door is open/locked: C wrong (says open), E correct (says locked).
        # Action: bK from Zone A → B→C gate; gK from Zone C → D→E gate; bK from Zone E → F→G gate.
        # Total rK: 1+4+2+2+0+2+25 = 36.

        return [
            # ── P0: Zone A rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone A (the Atrium)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone C and Zone F both claim 2; Zone D correctly says 1. "
                                  "Two signposts agree on the wrong count."},
            ),
            # ── P1: Zone B rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone B (the Records Hall)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=4,
                metadata={"trap": "Zone D says 3 (wrong). Zones C, E, F all say 4 (correct). "
                                  "One conflicting signpost among three agreeing ones."},
            ),
            # ── P2: Zone C rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone C (the Monitoring Center)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P3: Zone D rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone D (the Repository)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P4: Zone E rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone E (the Containment Bay)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={"trap": "After Zones A-D (1+4+2+2=9 rK), strong recency expectation. "
                                  "Zone F's signpost will claim Zone E had 1 rK (also wrong)."},
            ),
            # ── P5: Zone E rK presence ───────────────────────────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Were there any red keys in Zone E (the Containment Bay)?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone F signpost claims Zone E had 1 rK. Correct answer: no."},
            ),
            # ── P6: Zone F rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone F (the Recovery Room)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "rK reappear after Zone E had zero; model may say they're gone."},
            ),
            # ── P7: Zone G rK count ──────────────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many red keys were there in Zone G (the Dense Repository)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=25,
                metadata={"trap": "9×3 dense array with 1 bK impostor and 1 empty slot. "
                                  "Exact count=25. Notice board will not mention this zone's count."},
            ),
            # ── P8: Zone G blue ball count ───────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone G (the Dense Repository)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=25,
                metadata={"trap": "Same array structure as rK: 27 slots, 2 impostors (rB + gB) = 25 bB."},
            ),
            # ── P9: Zone A yellow door state ─────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone A (the Atrium)?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="closed",
                metadata={"trap": "Most zones have open or locked doors; closed is unusual."},
            ),
            # ── P10: Zone B yellow door state ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone B (the Records Hall)?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="locked",
                metadata={"trap": "Zone C signpost says OPEN (wrong). Zone E signpost says LOCKED (correct). "
                                  "Two conflicting claims — must trust Zone E's correct one."},
            ),
            # ── P11: Zone D yellow door state ────────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What was the state of the yellow door in Zone D (the Repository)?\n"
                    "Answer with one word: open, closed, or locked."
                ),
                ground_truth="open",
                metadata={"trap": "Zone E signpost claims Zone D door was LOCKED. Correct: open."},
            ),
            # ── P12: Zone A blue key count (entry) ───────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were in Zone A (the Atrium) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={"trap": "Unusual to have 2 bK in the opening zone. One is picked up on departure."},
            ),
            # ── P12b: Zone A blue key count (after departure) ────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone A (taking one blue key with you), how many blue keys "
                    "remained in Zone A?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={
                    "trap": "You entered Zone A with 2 blue keys; took 1 on departure. Remaining: 1.",
                    "inventory_delta": True,
                },
            ),
            # ── P13: Zone A blue ball count ──────────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue balls were there in Zone A (the Atrium)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={},
            ),
            # ── P14: Zone E blue key count (entry) ───────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many blue keys were in Zone E (the Containment Bay) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=3,
                metadata={"trap": "Zone E has 3 bK but 0 rK. Model may fixate on the missing rK "
                                  "and miscount the bK."},
            ),
            # ── P14b: Zone E blue key count (after departure) ────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone E (taking one blue key with you), how many blue keys "
                    "remained in Zone E?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=2,
                metadata={
                    "trap": "You entered Zone E with 3 blue keys; took 1 on departure. Remaining: 2.",
                    "inventory_delta": True,
                },
            ),
            # ── P15: Zone C signpost accuracy on Zone B rK ───────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone C's signpost accurate about Zone B's red key count?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Zone C said 4; Zone B has 4. Correct — but Zone C was wrong about Zone A."},
            ),
            # ── P16: Zone C signpost accuracy on Zone B door ─────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone C's signpost accurate about Zone B's yellow door state?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone C said Zone B door is OPEN. Zone B door is LOCKED."},
            ),
            # ── P17: Zone D signpost accuracy on Zone B rK ───────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone D's signpost accurate about Zone B's red key count?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone D said 3; Zone B has 4."},
            ),
            # ── P18: Zone C vs Zone D — do they agree on Zone B rK? ──────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Did Zone C's signpost and Zone D's signpost give the same count "
                    "for Zone B's red keys?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone C said 4 (correct); Zone D said 3 (wrong). They disagree."},
            ),
            # ── P19: Zone E signpost accuracy on Zone D door ─────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone E's signpost accurate about Zone D's yellow door state?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone E said LOCKED; Zone D door was OPEN."},
            ),
            # ── P20: Zone F signpost accuracy on Zone E rK ───────────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone F's signpost accurate about Zone E's red key count?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone F said Zone E had 1 rK; Zone E had 0."},
            ),
            # ── P21: Zone G notice board accuracy on Zone E rK ───────────────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone G's notice board accurate about Zone E's red key count?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=True,
                metadata={"trap": "Notice said Zone E: 0 rK. That is CORRECT. "
                                  "Model must recognise the one true claim in an otherwise wrong notice."},
            ),
            # ── P22: Zone G notice board accuracy on Zone B being the most ────────
            FixedProbe(
                probe_type="presence",
                question=(
                    "Was Zone G's notice board accurate that Zone B held the most red keys "
                    "in the entire facility?\n"
                    "Answer with exactly 'yes' or 'no'."
                ),
                ground_truth=False,
                metadata={"trap": "Zone G itself has 25 rK; Zone B only had 4. Notice is wrong."},
            ),
            # ── P23: According to Zone D's signpost, Zone B rK count ─────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to Zone D's signpost, how many red keys did Zone B have?\n"
                    "Answer with just the number the signpost stated."
                ),
                ground_truth=3,
                metadata={"trap": "Sign says 3; truth is 4. Tests sign-reading, not observation."},
            ),
            # ── P24: According to Zone F's signpost, Zone E rK count ─────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "According to Zone F's signpost, how many red keys did Zone E have?\n"
                    "Answer with just the number the signpost stated."
                ),
                ground_truth=1,
                metadata={"trap": "Sign says 1; truth is 0. Tests sign-reading."},
            ),
            # ── P25: Total rK across all 7 zones ─────────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "What is the total number of red keys across all seven zones (A through G)?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=36,
                metadata={"trap": "1+4+2+2+0+2+25=36. Notice board says A-F=12 (wrong, actual=11). "
                                  "Model must use own Zone G count (25) and correct A-F total (11)."},
            ),
            # ── P26: Which zone had the most rK ──────────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "Which zone had the most red keys?\n"
                    "Answer with just the zone letter, e.g. 'Zone G'."
                ),
                ground_truth="Zone G",
                metadata={"trap": "Notice board claims Zone B had the most. Zone G has 25."},
            ),
            # ── P27: Action — item picked up leaving Zone A ───────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up when leaving Zone A?\n"
                    "Answer with the object type and colour, e.g. 'red key'."
                ),
                ground_truth="blue key",
                metadata={"action": True, "trap": "Zone A has 2 bK among 2 gB, 2 bB, 1 rK."},
            ),
            # ── P28: Action — key provenance for Zone G entry ─────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "In which zone did you acquire the item used to enter Zone G?\n"
                    "Answer with just the zone letter, e.g. 'Zone E'."
                ),
                ground_truth="Zone E",
                metadata={"action": True, "trap": "Blue key picked up in Zone E (3 bK there), "
                                                   "used at F→G gate. Model must track provenance "
                                                   "across 2 zones."},
            ),
            # ── P29: Action — item picked up in Zone C ────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up in Zone C (the Monitoring Center)?\n"
                    "Answer with the object type and colour, e.g. 'green key'."
                ),
                ground_truth="green key",
                metadata={"action": True, "trap": "Zone C has 2 rK, 2 bB, yO, gK, pB — "
                                                   "model must recall the green key specifically."},
            ),
            # ── P30: Action — item picked up in Zone E ────────────────────────────
            FixedProbe(
                probe_type="attribute",
                question=(
                    "What item did you pick up in Zone E (the Containment Bay)?\n"
                    "Answer with the object type and colour, e.g. 'blue key'."
                ),
                ground_truth="blue key",
                metadata={"action": True, "trap": "Zone E has 3 bK — model confirms the blue key pickup."},
            ),
            # ── P31: Zone C green key count (entry) ───────────────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "How many green keys were in Zone C (the Monitoring Center) when you entered it?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=1,
                metadata={"trap": "Zone C has 1 gK among 2 rK, 2 bB, yO, pB. "
                                  "Model may miss the gK or conflate it with Zone B's gK."},
            ),
            # ── P32: Zone C green key count (after departure) ────────────────────
            FixedProbe(
                probe_type="count",
                question=(
                    "After you left Zone C (taking the green key with you), how many green keys "
                    "remained in Zone C?\n"
                    "Box your final answer as a numeral like <box>5</box>, e.g. <box>3</box>. Do not spell out numbers as words."
                ),
                ground_truth=0,
                metadata={
                    "trap": "You took the only green key from Zone C. Remaining: 0.",
                    "inventory_delta": True,
                },
            ),
        ]


def _yes_no(text: str) -> bool | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    # Thinking/reasoning models write reasoning then answer — scan the last 5 lines
    # for any line that is just "yes" or "no" (stripped), then fall back to last word.
    for line in reversed(t.splitlines()[-5:]):
        line = line.strip().rstrip(".,;:!? ")
        if line == "yes":
            return True
        if line == "no":
            return False
    last_word = t.rstrip(".,;:!? ").rsplit(None, 1)[-1] if t else ""
    if last_word == "yes":
        return True
    if last_word == "no":
        return False
    return None


_WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}


def _parse_int(text: str) -> int | None:
    # 1. Prefer explicit <box>N</box> tag (new format instruction).
    m = re.search(r"<box>\s*(\d+)\s*</box>", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 2. Fallback heuristic for old data / models that ignore the tag:
    #    If the first token of the first line is a digit, use it — handles both
    #    bare answers ("25") and answer-first phrasing ("3 changes were made").
    #    This also catches o3-mini "Step N" style via the step prefix.
    #    Otherwise take the last integer (Claude style: reasoning then "= N").
    first_line = text.strip().split("\n")[0].strip()
    m = re.match(r"^(?:step\s+)?(\d+)\b", first_line, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 3. Handle spelled-out numbers ("Three", "Four", etc.) as first word.
    first_word = re.match(r"^([a-z]+)", first_line, re.IGNORECASE)
    if first_word and first_word.group(1).lower() in _WORD_TO_INT:
        return _WORD_TO_INT[first_word.group(1).lower()]
    # 4. Search anywhere in the text for a spelled-out number.
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _WORD_TO_INT:
            return _WORD_TO_INT[word]
    matches = re.findall(r"\d+", text)
    return int(matches[-1]) if matches else None


def _parse_json_steps(text: str) -> dict | None:
    m = re.search(r"\{[^}]+\}", text)
    if not m:
        return None
    try:
        import json
        return json.loads(m.group())
    except Exception:
        return None


_OBJ_COLORS = "red|blue|green|yellow|purple|grey|gray"
_OBJ_TYPES  = "key|ball|box|door"
_OBJ_RE = re.compile(rf"({_OBJ_COLORS})\s+({_OBJ_TYPES})", re.IGNORECASE)

# Grid-serializer abbreviation maps (color char → full name, type char → full name)
_GRID_COLOR_CHAR = {"r": "red", "g": "green", "b": "blue", "y": "yellow", "p": "purple", "e": "grey"}
_GRID_TYPE_CHAR  = {"k": "key", "b": "ball", "x": "box", "d": "door", "o": "door", "l": "door"}
# Matches a standalone 2-char grid abbreviation e.g. "rK", "bB", "yD"
_GRID_ABBREV_RE  = re.compile(r"\b([rRgGbByYpPeE])([kKbBxXdDoOlL])\b")


def _parse_list_items(text: str) -> list[str]:
    """Extract items from a numbered, bulleted, or plain-line LM response."""
    formatted = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Numbered: "1. foo", "1) foo", "1- foo"
        m = re.match(r"^\d+[\.\)\-]\s*(.+)", line)
        if m:
            formatted.append(m.group(1).strip())
            continue
        # Bulleted: "- foo", "* foo", "• foo"
        m = re.match(r"^[-*•]\s+(.+)", line)
        if m:
            formatted.append(m.group(1).strip())

    if formatted:
        # Model used numbered/bulleted formatting — ignore any preamble/prose lines
        return formatted

    # Plain-line fallback: only when no formatted items found (e.g. o3-mini bare output)
    return [l.strip() for l in text.strip().splitlines() if l.strip()]


def _obj_key(text: str) -> str | None:
    """Extract normalised 'color type' key from a text fragment.

    Handles both full names ('red key') and grid-serializer abbreviations ('rK').
    """
    # Try full name first
    m = _OBJ_RE.search(text)
    if m:
        color = "grey" if m.group(1).lower() == "gray" else m.group(1).lower()
        return f"{color} {m.group(2).lower()}"
    # Fall back to grid abbreviation (e.g. "rK", "bB", "yD")
    m = _GRID_ABBREV_RE.search(text)
    if m:
        color = _GRID_COLOR_CHAR.get(m.group(1).lower())
        typ   = _GRID_TYPE_CHAR.get(m.group(2).lower())
        if color and typ:
            return f"{color} {typ}"
    return None


def score_order_list(gt_list: list[dict], response_text: str) -> float | None:
    """Fraction of positions where the LM's order matches GT order.

    Uses positional exact match per slot after normalising to 'color type' keys.
    Also returns 0 explicitly if the model output cannot be parsed.
    """
    parsed = _parse_list_items(response_text)
    if not parsed:
        return 0.0
    gt_keys  = [_obj_key(o["label"]) for o in gt_list]
    pred_keys = [_obj_key(p) for p in parsed]
    n = len(gt_keys)
    correct = sum(
        1 for i in range(min(len(pred_keys), n))
        if pred_keys[i] is not None and pred_keys[i] == gt_keys[i]
    )
    return correct / n


def score_between_list(gt_list: list[dict], response_text: str) -> float:
    """F1 between the GT set of between-objects and the model's predicted set.

    A response of 'nothing' / empty list when GT is empty scores 1.0.
    The target object is excluded from the GT (handled at probe level) and
    from the model's output (via the updated question wording).
    """
    gt_keys = {_obj_key(o["label"]) for o in gt_list} - {None}
    parsed   = _parse_list_items(response_text)
    pred_keys: set[str] = set()
    for p in parsed:
        k = _obj_key(p)
        if k:
            pred_keys.add(k)

    # model said "nothing" or gave empty list
    if not pred_keys and "nothing" in response_text.lower():
        return 1.0 if not gt_keys else 0.0

    if not gt_keys and not pred_keys:
        return 1.0
    if not gt_keys:
        return 0.0  # GT empty, model listed objects
    if not pred_keys:
        return 0.0  # GT non-empty, model said nothing

    tp = len(gt_keys & pred_keys)
    precision = tp / len(pred_keys)
    recall    = tp / len(gt_keys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_response(probe_type: str, ground_truth, response_text: str) -> float | None:
    """Return 1.0 / 0.0 for binary/exact probes, None for list probes (manual review)."""
    t = response_text.strip().lower()
    if probe_type == "presence":
        pred = _yes_no(response_text)
        if pred is None:
            return None
        return float(pred == ground_truth)

    elif probe_type == "count":
        pred = _parse_int(response_text)
        if pred is None:
            return None
        return float(pred == ground_truth)

    elif probe_type == "attribute":
        # Strip punctuation so "'locked' to 'open' to 'closed'" still matches
        t_clean = re.sub(r"[\"'`]", "", t)
        gt_str = str(ground_truth).lower()
        return float(gt_str in t_clean)

    elif probe_type == "location":
        parsed = _parse_json_steps(response_text)
        if parsed is None or ground_truth is None:
            return None
        sa_ok = parsed.get("steps_ahead") == ground_truth["steps_ahead"]
        lat_ok = parsed.get("lateral") == ground_truth["lateral"]
        return float(sa_ok and lat_ok)

    elif probe_type == "allocentric_location":
        if ground_truth is None:
            return None
        return float(ground_truth.lower() in t)

    elif probe_type == "order":
        if ground_truth is None:
            return None
        return score_order_list(ground_truth, response_text)

    elif probe_type == "between":
        if ground_truth is None:
            return None
        return score_between_list(ground_truth, response_text)

    elif probe_type == "epistemic":
        # GT = "cannot_determine": model must acknowledge it cannot know
        # GT = True/False: model should answer yes/no (for answerable epistemic probes)
        _CANNOT_PHRASES = [
            "cannot determine", "can't determine", "unable to determine",
            "not enough information", "cannot tell", "can't tell",
            "no way to know", "cannot know", "can't know",
            "not in my observation", "not visible", "not observable",
            "no record", "not possible to determine", "outside my",
            "i don't know", "have no information", "no information about",
            "have not observed", "haven't observed", "not in my current",
            "not described", "not part of my", "no access to",
            "not been observed", "sealed", "cannot see", "can't see",
            "no way for me", "insufficient information",
            "do not provide", "does not provide", "provide no information",
            "not provide any information", "cannot be determined",
            "cannot verify", "can't verify", "without direct observation",
            "without observing", "without peeking", "without looking",
            "haven't verified", "not verified", "not confirmed",
            "unverified", "have not peeked", "haven't peeked",
            "cannot confirm", "can't confirm",
            "it is uncertain", "uncertain whether", "uncertain if",
            "remains uncertain", "is uncertain", "are uncertain",
            "not been verified", "has not been verified",
            "no information from", "no information on",
            "it is unknown", "is unknown whether",
            "cannot reliably", "can't reliably",
            "no signpost", "without directly observing", "without observing it",
            "no basis to", "unconfirmed", "i do not know",
            "no verified", "remains unknown", "currently unknown",
            "have no evidence", "cannot say whether",
            "no direct confirmation", "no confirmation",
            "cannot_determine",
        ]
        if ground_truth == "cannot_determine":
            t = t.replace("\u2019", "'").replace("\u2018", "'")  # normalize curly apostrophes
            return float(any(p in t for p in _CANNOT_PHRASES))
        elif ground_truth in (True, False):
            pred = _yes_no(response_text)
            if pred is None:
                return None
            return float(pred == ground_truth)
        else:
            return float(str(ground_truth).lower() in t)

    return None


SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. "
    "You will be shown what you currently see, then asked a question about it. "
    "Answer precisely and concisely."
)

DELTA_SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. "
    "You will be shown two sequential observations separated by movement. "
    "Answer precisely and concisely based on the specific observation indicated in the question."
)

INCIDENT_SYSTEM_PROMPT = (
    "You are an investigator reviewing two snapshots of a room taken 6 steps apart. "
    "The snapshots are labeled Observation 1 (before) and Observation 2 (after). "
    "Your task is to identify exactly what changed, disappeared, or was altered between the two. "
    "Answer precisely and concisely. Do not invent changes that did not occur."
)

RIVER_SYSTEM_PROMPT = (
    "You are an agent in a gridworld with an active river zone. "
    "Objects placed in the river drift eastward each step at speed 1. "
    "Wet objects dry out over time after leaving the river: "
    "wet_turns decrements by 1 per step; the object dries when it reaches 0. "
    "A notice board always shows the state at step 0 and may be stale. "
    "You will see two observations: step 0 (before) and step N (after river physics). "
    "Answer questions about the current state (observation 2) precisely."
)

MEMORY_SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. "
    "You will be shown a sequence of observations from different locations you have visited. "
    "Each observation is labeled with its source (e.g. Chamber 1, Chamber 2). "
    "Answer questions precisely based on the specific observation indicated. "
    "Do not confuse information from different observations."
)

NARRATOR_SYSTEM_PROMPT = (
    "You are an agent exploring a gridworld. "
    "You will receive a full description of everything you can observe, "
    "including any signs or notice boards present in the room. "
    "Answer each question based on careful reading of your direct observation."
)

CAUSAL_SYSTEM_PROMPT = (
    "You are an agent in a gridworld with the following causal rules:\n"
    "- Fire tiles are IMPASSABLE while active. To pass through fire you must first extinguish it.\n"
    "- Extinguishing fire: use a WET object (wet_turns > 0) on an active fire tile. "
    "A dry object (wet_turns = 0) cannot extinguish fire.\n"
    "- River tiles: objects placed on river tiles become WET (wet_turns = 4) and drift downstream each step.\n"
    "- Drying: after an object leaves the river, wet_turns decrements by 1 each step. "
    "The object is dry when wet_turns reaches 0.\n"
    "- Objects annotated [WET] or [SOAKED] in the description have wet_turns > 0. "
    "Objects with no such annotation are dry (wet_turns = 0).\n"
    "- The goal is reachable only after all barriers on the path are cleared.\n"
    "Reason step by step from the rules above when answering questions about "
    "what is possible, what would happen, and how to reach the goal."
)

UNCERTAINTY_SYSTEM_PROMPT = (
    "You are an agent in a gridworld. You can only know what you directly observe.\n"
    "When asked about things outside your current observation — rooms you have not entered, "
    "objects not described to you, or events you did not witness — you MUST respond with "
    "'I cannot determine this from my current observation' or equivalent phrasing.\n"
    "Do NOT speculate, estimate, or confabulate based on what seems likely.\n"
    "Distinguish carefully between:\n"
    "  - Things you CAN determine from your observation (e.g. objects visible, "
    "physical evidence like scorch marks, game rules that apply universally)\n"
    "  - Things you CANNOT determine (e.g. who caused an event, whether an unobserved "
    "room contains a specific object, whether an object was always in its current position)\n"
    "Answer precisely and concisely, explicitly citing your observation when relevant."
)

TRAJECTORY_SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld facility. "
    "You are shown a sequence of observations from locations you have visited. "
    "Answer each question precisely based only on what is explicitly stated in your observations. "
    "Be careful to distinguish which observation each fact comes from. "
    "If a question asks about a location, event, or time not covered by your observations, "
    "say that you cannot determine the answer."
)

QUEST_SYSTEM_PROMPT = (
    "You are an adventurer on a quest through Dragon's Keep. "
    "You are shown a series of observations from locations you have visited in sequence. "
    "Between observations, italicised narrative describes your actions and what you carried. "
    "Answer all questions based strictly on what was directly observed or explicitly stated "
    "in the observations and interstitial narrative — do not infer outcomes that were not recorded."
)

ORACLE_SYSTEM_PROMPT = (
    "You are an agent exploring a facility.\n"
    "You will be shown two observations:\n"
    "  1. The agent zone: contains signposts making claims about a target room "
    "and a notice board stating the accuracy rate of those signposts.\n"
    "  2. The target room (after peeking through the window): direct observation "
    "of the room's contents.\n\n"
    "Key rules:\n"
    "- A signpost claim is UNVERIFIED TESTIMONY. You cannot confirm or deny it "
    "without directly observing the target room.\n"
    "- The notice board states what fraction of signpost claims are accurate on average. "
    "A claim is still unverified until you observe the room directly.\n"
    "- Answer precisely. For pre-peek questions, distinguish between what signposts "
    "claim and what you have actually verified. For post-peek questions, answer "
    "from your direct observation."
)

# ── Multi-file level paths for U4 and U1 ──────────────────────────────────────
U4_OBS1_PATH = str(LEVELS_DIR / "u4_obs1.txt")
U4_OBS2_PATH = str(LEVELS_DIR / "u4_obs2.txt")
U1_HUB_PATH  = str(LEVELS_DIR / "u1_hub.txt")
U1_ROOM_A_PATH = str(LEVELS_DIR / "u1_room_a.txt")
U1_ROOM_C_PATH = str(LEVELS_DIR / "u1_room_c.txt")

# ── U2 Oracle Problem level paths ─────────────────────────────────────────────
U2_ZONE_HIGH_PATH = str(LEVELS_DIR / "u2_agent_zone_high.txt")
U2_ZONE_MID_PATH  = str(LEVELS_DIR / "u2_agent_zone_mid.txt")
U2_ZONE_LOW_PATH  = str(LEVELS_DIR / "u2_agent_zone_low.txt")
U2_TARGET_PATH    = str(LEVELS_DIR / "u2_target_room.txt")

# ── X1 Facility Tour level paths ──────────────────────────────────────────────
# Cross-tier compound level: perceptual Zone A → testimony Zone B → memory Zone C
# Zone A (Lab):    rK×3, bB×2, gK×1, yD(locked)
# Zone B (Records): rK×2, gB×2, yD(open), signpost with FALSE claims about Zone A
# Zone C (Control): bK×2, rB×1, notice board CONFIRMING true Zone A inventory
X1_ZONE_A_PATH = str(LEVELS_DIR / "x1_zone_a.txt")
X1_ZONE_B_PATH = str(LEVELS_DIR / "x1_zone_b.txt")
X1_ZONE_C_PATH = str(LEVELS_DIR / "x1_zone_c.txt")

# X2 · Facility Tour (5-zone) — extends X1 zones A/B/C with two more rooms:
# Zone D (Maintenance Bay): 2nd false signpost (claims 5 rK, open door — different count from B)
# Zone E (Director's Office): 2nd accurate notice board + bB trap (same type as Zone A)
X2_ZONE_D_PATH = str(LEVELS_DIR / "x2_zone_d.txt")
X2_ZONE_E_PATH = str(LEVELS_DIR / "x2_zone_e.txt")

# X3 · Facility Tour (7-zone) — two independent ground-truth anchors (A and B), four testimony
# zones (C false about A; D double-false about A+B; E true about A; F true about B), and a final
# Director's Suite (G) with a full audit summary that names the inaccurate sources explicitly.
# Integration probes require cross-zone counting, source-credibility tracking, and aggregation.
X3_ZONE_A_PATH = str(LEVELS_DIR / "x3_zone_a.txt")
X3_ZONE_B_PATH = str(LEVELS_DIR / "x3_zone_b.txt")
X3_ZONE_C_PATH = str(LEVELS_DIR / "x3_zone_c.txt")
X3_ZONE_D_PATH = str(LEVELS_DIR / "x3_zone_d.txt")
X3_ZONE_E_PATH = str(LEVELS_DIR / "x3_zone_e.txt")
X3_ZONE_F_PATH = str(LEVELS_DIR / "x3_zone_f.txt")
X3_ZONE_G_PATH = str(LEVELS_DIR / "x3_zone_g.txt")

# X4 · Facility Tour (5-zone "Compound Witness") — reuses existing hard levels as zones with
# cross-zone signpost testimony traps and one dense-array counting zone.
# Zone A: witness_chamber1 (4 rK, 1 bK, 2 bB, yL, pB)    — clean anchor
# Zone B: witness_chamber2 (2 rK, 3 bB, yO, gK, rB)       — clean anchor
# Zone C: narrator_room    (2 rK, 2 bB, yO, gK, pB + s0)  — 2/3 correct about A+B
# Zone D: witness_chamber4 (0 rK!, 3 bK, rB, gB, yL + s0) — 1/3 correct about A+B+C
# Zone E: dense_array      (25 rK, 1 bK impostor, 25 bB, see_through_walls=false + n0) — 0/3
X4_ZONE_A_PATH = str(LEVELS_DIR / "x4_zone_a.txt")
X4_ZONE_B_PATH = str(LEVELS_DIR / "x4_zone_b.txt")
X4_ZONE_C_PATH = str(LEVELS_DIR / "x4_zone_c.txt")
X4_ZONE_D_PATH = str(LEVELS_DIR / "x4_zone_d.txt")
X4_ZONE_E_PATH = str(LEVELS_DIR / "x4_zone_e.txt")

# X5 · Facility Tour (7-zone "Cascading Testimony") — 7 zones drawn from witness chambers,
# narrator room, and dense array.  Four signpost-bearing zones create cascading conflicts
# (Zones C, D, E, F) plus a notice board in Zone G.  Two zones agree on a wrong count for
# Zone A; one correct.  One zero-count trap (Zone E).  Action trajectory: bK→C gate (from A),
# gK→E gate (from C), bK→G gate (from E).
# Zone A: witness_chamber3 (1 rK, 2 bK, 2 bB, 2 gB, yD:closed)  — clean anchor
# Zone B: witness_chamber1 (4 rK, 1 bK, 2 bB, yL:locked, pB)    — clean anchor (same as X4 Zone A)
# Zone C: narrator_room    (2 rK, 2 bB, yO, gK, pB + s0)          — 1/3 correct about A+B
# Zone D: witness_chamber2 (2 rK, 3 bB, yO, gK, rB + s0)          — 2/3 correct about A+B+C
# Zone E: witness_chamber4 (0 rK!, 3 bK, rB, gB, yL + s0)         — 2/3 correct about B+D
# Zone F: witness_chamber5 (2 rK, 1 bB, rB, yO, gK + s0)          — 1/3 correct about A+B+E
# Zone G: dense_array      (25 rK, bK impostor, 25 bB + n0)        — 1/3 correct about A-F
X5_ZONE_A_PATH = str(LEVELS_DIR / "x5_zone_a.txt")
X5_ZONE_B_PATH = str(LEVELS_DIR / "x5_zone_b.txt")
X5_ZONE_C_PATH = str(LEVELS_DIR / "x5_zone_c.txt")
X5_ZONE_D_PATH = str(LEVELS_DIR / "x5_zone_d.txt")
X5_ZONE_E_PATH = str(LEVELS_DIR / "x5_zone_e.txt")
X5_ZONE_F_PATH = str(LEVELS_DIR / "x5_zone_f.txt")
X5_ZONE_G_PATH = str(LEVELS_DIR / "x5_zone_g.txt")

# X6 · Return Visit (5-observation change-detection challenge).
# Agent tours A→B→C→D, then RETURNS to A (which has been silently modified).
# Zone A (first visit):  witness_chamber1 — 4 rK, 2 bB, 1 bK, yL:locked, pB
# Zone B:               witness_chamber3 — 1 rK, 2 bB, 2 gB, yD:closed, 2 bK
# Zone C:               narrator_room + maint-log about Zone A return:
#                         claims: ‘3 rK removed’ ✗, ‘door opens’ ✓, ‘all balls removed’ ✗ → 1/3
# Zone D:               witness_chamber2 — 2 rK, 3 bB, yO, gK, rB
# Zone A revisit:       witness_chamber1 MODIFIED:
#                         – 2 rK removed (4→2, at cols 5+9 remain)
#                         – yL → yO (door opened)
#                         – pB removed
#                         – gB ADDED at row=2 col=7 (not in signpost!)
X6_ZONE_A_PATH         = str(LEVELS_DIR / "x6_zone_a.txt")
X6_ZONE_B_PATH         = str(LEVELS_DIR / "x6_zone_b.txt")
X6_ZONE_C_PATH         = str(LEVELS_DIR / "x6_zone_c.txt")
X6_ZONE_D_PATH         = str(LEVELS_DIR / "x6_zone_d.txt")
X6_ZONE_A_REVISIT_PATH = str(LEVELS_DIR / "x6_zone_a_revisit.txt")

# X7 · Dragon's Keep (8-observation RPG quest — adversarial sources, backtracking, detour, combat log).
X7_ZONE_A_PATH          = str(LEVELS_DIR / "x7_zone_a.txt")
X7_ZONE_B_PATH          = str(LEVELS_DIR / "x7_zone_b.txt")
X7_ZONE_B_RETURN_PATH   = str(LEVELS_DIR / "x7_zone_b_return.txt")
X7_ZONE_C_PATH          = str(LEVELS_DIR / "x7_zone_c.txt")
X7_ZONE_D_PATH          = str(LEVELS_DIR / "x7_zone_d.txt")
X7_ZONE_W_PATH          = str(LEVELS_DIR / "x7_zone_w.txt")
X7_ZONE_E_PATH          = str(LEVELS_DIR / "x7_zone_e.txt")
X7_ZONE_E_POST_PATH     = str(LEVELS_DIR / "x7_zone_e_postcombat.txt")

# Compound levels: cross-tier, dispatched individually in main()
COMPOUND_LEVELS: set[str] = {"X1_facility_tour", "X2_facility_tour", "X3_facility_tour", "X4_facility_tour", "X5_facility_tour", "X6_return_visit", "X7_dragon_keep"}

# ── main eval loop ─────────────────────────────────────────────────────────────

def run_level(
    level_key: str,
    level_path: str,
    model: str,
    n_episodes: int,
    base_seed: int,
    verbose: bool,
    max_tokens: int,
    ser: Serializer | None = None,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    thinking_effort: str | None = None,
) -> pd.DataFrame:
    lm = _make_lm(model, max_tokens, reasoning_effort, thinking_effort)
    ser = ser or SymbolicSerializer()
    rows = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        env = AsciiEnv.from_file(level_path)
        env.reset(seed=seed)
        obs_text = ser.serialize(env)

        rng = random.Random(seed)
        probes = make_probes(level_key, rng)

        for probe in probes:
            result = probe.generate(env)
            if result.metadata.get("skipped"):
                continue

            user_msg = f"{obs_text}\n\n{result.question}"
            _sys = system_prompt or SYSTEM_PROMPT
            _mt = max(1024, max_tokens) if result.probe_type == "count" else None
            resp = lm.query(system=_sys, user=user_msg, max_tokens=_mt)
            score = 0.0 if not resp.text or not resp.text.strip() else score_response(result.probe_type, result.ground_truth, resp.text)

            if verbose:
                print(f"[{level_key}] ep={ep} probe={result.probe_type}")
                print(f"  Q: {result.question[:100]}")
                print(f"  GT: {result.ground_truth}")
                print(f"  A: {resp.text[:120]}")
                print(f"  score: {score}")
                print()

            rows.append({
                "model": model,
                "level": level_key,
                "episode": ep,
                "seed": seed,
                "serializer": ser.__class__.__name__,
                "probe_type": result.probe_type,
                "ground_truth": str(result.ground_truth),
                "response": resp.text,
                "score": score,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

    return pd.DataFrame(rows)


def run_delta_level(
    level_key: str,
    level_path: str,
    model: str,
    n_episodes: int,
    base_seed: int,
    verbose: bool,
    max_tokens: int,
    n_steps: int = 3,
    ser: Serializer | None = None,
) -> pd.DataFrame:
    """Two-observation delta runner: serialize t=0, step forward n_steps, serialize t=N."""
    lm = _make_lm(model, max_tokens)
    ser = ser or SymbolicSerializer()
    rows = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        env = AsciiEnv.from_file(level_path)
        env.reset(seed=seed)

        obs1_text = ser.serialize(env)

        forward = env.actions.forward
        for _ in range(n_steps):
            env.step(forward)

        obs2_text = ser.serialize(env)

        combined_obs = (
            f"[Observation 1 — Step 0]:\n{obs1_text}\n\n"
            f"[You moved: {n_steps} steps north]\n\n"
            f"[Observation 2 — Step {n_steps}]:\n{obs2_text}"
        )

        rng = random.Random(seed)
        probes = make_probes(level_key, rng)

        for probe in probes:
            result = probe.generate(env)
            if result.metadata.get("skipped"):
                continue

            user_msg = f"{combined_obs}\n\n{result.question}"
            _mt = max(1024, max_tokens) if result.probe_type == "count" else None
            resp = lm.query(system=DELTA_SYSTEM_PROMPT, user=user_msg, max_tokens=_mt)
            score = 0.0 if not resp.text or not resp.text.strip() else score_response(result.probe_type, result.ground_truth, resp.text)

            if verbose:
                print(f"[{level_key}] ep={ep} probe={result.probe_type}")
                print(f"  Q: {result.question[:100]}")
                print(f"  GT: {result.ground_truth}")
                print(f"  A: {resp.text[:120]}")
                print(f"  score: {score}")
                print()

            rows.append({
                "model": model,
                "level": level_key,
                "episode": ep,
                "seed": seed,
                "serializer": ser.__class__.__name__,
                "probe_type": result.probe_type,
                "ground_truth": str(result.ground_truth),
                "response": resp.text,
                "score": score,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

    return pd.DataFrame(rows)


def run_two_file_level(
    level_key: str,
    level_paths: tuple[str, str],
    model: str,
    n_episodes: int,
    base_seed: int,
    verbose: bool,
    max_tokens: int,
    move_description: str = "3 steps north",
    ser: Serializer | None = None,
    system_prompt: str | None = None,
    interstitial: str | None = None,
) -> pd.DataFrame:
    """Two-observation runner using separate pre-authored level files.

    Loads init_path for Observation 1 and moved_path for Observation 2,
    allowing hidden changes (object removal, state mutation, additions)
    between the two observations that the agent cannot directly witness.
    """
    init_path, moved_path = level_paths
    lm = _make_lm(model, max_tokens)
    ser = ser or SymbolicSerializer()
    rows = []

    for ep in range(n_episodes):
        seed = base_seed + ep

        env1 = AsciiEnv.from_file(init_path)
        env1.reset(seed=seed)
        obs1_text = ser.serialize(env1)

        env2 = AsciiEnv.from_file(moved_path)
        env2.reset(seed=seed)
        obs2_text = ser.serialize(env2)

        _interstitial = interstitial or f"[You moved: {move_description}]"
        combined_obs = (
            f"[Observation 1 — Step 0]:\n{obs1_text}\n\n"
            f"{_interstitial}\n\n"
            f"[Observation 2 — Step 6]:\n{obs2_text}"
        )

        rng = random.Random(seed)
        probes = make_probes(level_key, rng)

        for probe in probes:
            result = probe.generate(env2)
            if result.metadata.get("skipped"):
                continue

            user_msg = f"{combined_obs}\n\n{result.question}"
            _sys = system_prompt or DELTA_SYSTEM_PROMPT
            _mt = max(1024, max_tokens) if result.probe_type == "count" else None
            resp = lm.query(system=_sys, user=user_msg, max_tokens=_mt)
            score = 0.0 if not resp.text or not resp.text.strip() else score_response(result.probe_type, result.ground_truth, resp.text)

            if verbose:
                print(f"[{level_key}] ep={ep} probe={result.probe_type}")
                print(f"  Q: {result.question[:100]}")
                print(f"  GT: {result.ground_truth}")
                print(f"  A: {resp.text[:120]}")
                print(f"  score: {score}")
                print()

            rows.append({
                "model": model,
                "level": level_key,
                "episode": ep,
                "seed": seed,
                "serializer": ser.__class__.__name__,
                "probe_type": result.probe_type,
                "ground_truth": str(result.ground_truth),
                "response": resp.text,
                "score": score,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

    return pd.DataFrame(rows)


def run_multi_obs_level(
    level_key: str,
    obs_files: list[tuple[str, str]],
    model: str,
    n_episodes: int,
    base_seed: int,
    verbose: bool,
    max_tokens: int,
    ser: Serializer | None = None,
    serializers: list[Serializer] | None = None,
    system_prompt: str | None = None,
    interstitials: list[str] | None = None,
) -> pd.DataFrame:
    """Multi-observation runner for memory levels.

    Each entry in obs_files is (label, filepath).  All files are serialized
    and concatenated as labeled observation blocks before probes fire.
    Optional interstitials (len = len(obs_files) - 1) are inserted between blocks.
    If `serializers` is provided (len == len(obs_files)), each file uses its own
    serializer; otherwise the single `ser` is used for all.
    """
    lm = _make_lm(model, max_tokens)
    _default_ser = ser or SymbolicSerializer()
    rows = []

    for ep in range(n_episodes):
        seed = base_seed + ep

        obs_blocks: list[str] = []
        last_env: AsciiEnv | None = None
        for i, (label, filepath) in enumerate(obs_files):
            if i > 0 and interstitials and i - 1 < len(interstitials):
                obs_blocks.append(interstitials[i - 1])
            env = AsciiEnv.from_file(filepath)
            env.reset(seed=seed)
            _ser = serializers[i] if serializers and i < len(serializers) else _default_ser
            block = _ser.serialize(env)
            obs_blocks.append(f"[{label}]:\n{block}")
            last_env = env

        combined_obs = "\n\n".join(obs_blocks)

        rng = random.Random(seed)
        probes = make_probes(level_key, rng)

        for probe in probes:
            result = probe.generate(last_env)
            if result.metadata.get("skipped"):
                continue

            user_msg = f"{combined_obs}\n\n{result.question}"
            _sys = system_prompt or MEMORY_SYSTEM_PROMPT
            _mt = max(1024, max_tokens) if result.probe_type == "count" else None
            resp = lm.query(system=_sys, user=user_msg, max_tokens=_mt)
            score = 0.0 if not resp.text or not resp.text.strip() else score_response(result.probe_type, result.ground_truth, resp.text)

            if verbose:
                print(f"[{level_key}] ep={ep} probe={result.probe_type}")
                print(f"  Q: {result.question[:100]}")
                print(f"  GT: {result.ground_truth}")
                print(f"  A: {resp.text[:120]}")
                print(f"  score: {score}")
                print()

            rows.append({
                "model": model,
                "level": level_key,
                "episode": ep,
                "seed": seed,
                "serializer": ser.__class__.__name__,
                "probe_type": result.probe_type,
                "ground_truth": str(result.ground_truth),
                "response": resp.text,
                "score": score,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

    return pd.DataFrame(rows)


def run_oracle_level(
    level_key: str,
    zone_path: str,
    target_path: str,
    model: str,
    n_episodes: int,
    base_seed: int,
    verbose: bool,
    max_tokens: int,
) -> pd.DataFrame:
    """Two-phase runner for U2 Oracle Problem.

    Phase 1 (pre-peek): model sees only the agent zone (signposts + notice board,
        serialized with MemorySerializer so signpost text is shown verbatim).
    Phase 2 (post-peek): model sees agent zone + target room peek
        (target room serialized with SymbolicSerializer).

    Each probe is tagged with its phase in metadata["phase"].  Phase 1 probes
    test whether the model correctly treats signpost claims as unverified testimony.
    Phase 2 probes verify post-observation accuracy and calibration.
    """
    lm = _make_lm(model, max_tokens)
    zone_ser   = MemorySerializer()
    target_ser = SymbolicSerializer()
    rows = []

    for ep in range(n_episodes):
        seed = base_seed + ep

        env_zone = AsciiEnv.from_file(zone_path)
        env_zone.reset(seed=seed)
        zone_text = zone_ser.serialize(env_zone)

        env_target = AsciiEnv.from_file(target_path)
        env_target.reset(seed=seed)
        target_text = target_ser.serialize(env_target)

        phase1_ctx = f"[Observation 1 \u2014 Agent Zone]:\n{zone_text}"
        interstitial = (
            "[You walked through the viewing corridor to the window and "
            "looked directly into the target room.]"
        )
        phase2_ctx = f"[Observation 2 \u2014 Target Room (direct observation through window)]:\n{target_text}"
        both_ctx = f"{phase1_ctx}\n\n{interstitial}\n\n{phase2_ctx}"

        rng = random.Random(seed)
        probes = make_probes(level_key, rng)

        for probe in probes:
            result = probe.generate(env_target)
            if result.metadata.get("skipped"):
                continue

            phase = result.metadata.get("phase", 2)
            user_msg = (
                f"{phase1_ctx}\n\n{result.question}"
                if phase == 1
                else f"{both_ctx}\n\n{result.question}"
            )

            _mt = max(1024, max_tokens) if result.probe_type == "count" else None
            resp = lm.query(system=ORACLE_SYSTEM_PROMPT, user=user_msg, max_tokens=_mt)
            score = (
                0.0
                if not resp.text or not resp.text.strip()
                else score_response(result.probe_type, result.ground_truth, resp.text)
            )

            if verbose:
                print(f"[{level_key} phase={phase}] ep={ep} probe={result.probe_type}")
                print(f"  Q: {result.question[:100]}")
                print(f"  GT: {result.ground_truth}")
                print(f"  A: {resp.text[:120]}")
                print(f"  score: {score}")
                print()

            rows.append({
                "model": model,
                "level": level_key,
                "episode": ep,
                "seed": seed,
                "serializer": "Mixed(Memory+Symbolic)",
                "probe_type": result.probe_type,
                "ground_truth": str(result.ground_truth),
                "response": resp.text,
                "score": score,
                "phase": phase,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

    return pd.DataFrame(rows)


def run_stepped_level(
    level_key: str,
    level_path: str,
    n_steps: int,
    model: str,
    n_episodes: int,
    base_seed: int,
    verbose: bool,
    max_tokens: int,
) -> pd.DataFrame:
    """Runner for levels with river/tile physics stepped N times without agent movement.

    Always uses MemorySerializer (full-grid, compass positions).
    Serializes t=0 and t=N, concatenates with an interstitial, then runs probes.
    """
    lm = _make_lm(model, max_tokens)
    mem_ser = MemorySerializer()
    rows = []

    for ep in range(n_episodes):
        seed = base_seed + ep

        env = AsciiEnv.from_file(level_path)
        env.reset(seed=seed)
        obs0 = mem_ser.serialize(env, label="Observation 1 — Step 0")

        for _ in range(n_steps):
            env.step_count += 1
            if env._river_map:
                apply_river_physics(env)
            decay_wet_conditions(env)
            if env._flood_tiles:
                advance_flood_tiles(env)

        has_flood = bool(getattr(env, "_flood_tiles", []))
        physics_desc = (
            "Floodwater continued to rise."
            if has_flood
            else "River currents continued to flow."
        )

        # For flood levels (C6): prepend coordinate convention + flood→fire rule so
        # models can reason about the physics without guessing.
        if has_flood:
            physics_preamble = (
                "[Grid convention: row 0 is the northernmost (top) row; row numbers "
                "increase going south (downward). Col 0 is the westernmost (left) column; "
                "col numbers increase going east (rightward). "
                "Flood physics: floodwater tiles advance southward by one row each step; "
                "each flooded row is impassable and stays flooded. "
                "When a flood tile activates in the row immediately north of (directly "
                "adjacent to) an active fire row, it extinguishes every fire tile in that "
                "fire row on the same step \u2014 the fire row becomes passable (scorch marks) "
                "but is NOT itself flooded.]\n\n"
            )
        else:
            physics_preamble = ""

        combined = (
            f"{physics_preamble}{obs0}\n\n"
            f"[{n_steps} steps have now elapsed; you remained stationary. {physics_desc} "
            f"Predict the current state from the rules above.]"
        )

        rng = random.Random(seed)
        probes = make_probes(level_key, rng)

        for probe in probes:
            result = probe.generate(env)
            if result.metadata.get("skipped"):
                continue

            user_msg = f"{combined}\n\n{result.question}"
            _mt = max(1024, max_tokens) if result.probe_type == "count" else None
            resp = lm.query(system=RIVER_SYSTEM_PROMPT, user=user_msg, max_tokens=_mt)
            score = 0.0 if not resp.text or not resp.text.strip() else score_response(result.probe_type, result.ground_truth, resp.text)

            if verbose:
                print(f"[{level_key}] ep={ep} probe={result.probe_type}")
                print(f"  Q: {result.question[:100]}")
                print(f"  GT: {result.ground_truth}")
                print(f"  A: {resp.text[:120]}")
                print(f"  score: {score}")
                print()

            rows.append({
                "model": model,
                "level": level_key,
                "episode": ep,
                "seed": seed,
                "serializer": "MemorySerializer",
                "probe_type": result.probe_type,
                "ground_truth": str(result.ground_truth),
                "response": resp.text,
                "score": score,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    parser.add_argument("--provider", choices=["openai", "anthropic", "baseten"],
                        default=None, help="Explicit provider routing (recommended)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                        dest="reasoning_effort",
                        help="Reasoning effort for o-series / gpt-5 models (default: API default)")
    parser.add_argument("--thinking-effort", choices=["low", "medium", "high", "max"], default=None,
                        dest="thinking_effort",
                        help="Adaptive thinking effort for Claude Sonnet/Opus 4.6 models (default: no thinking)")
    parser.add_argument("--model-tag", default=None, dest="model_tag",
                        help="Suffix appended to the model name in the output CSV (e.g. 'thinking'). "
                             "Lets thinking/non-thinking runs of the same model coexist in one file.")
    parser.add_argument("--levels", nargs="+",
                        default=list(LEVELS.keys()),
                        choices=list(LEVELS.keys()) + list(CAUSAL_SINGLE_LEVELS) + list(UNCERTAINTY_SINGLE_LEVELS),  # union in case of divergence
                        help="Which levels to run (default: all three)")
    parser.add_argument("--output", default="results_perception.csv")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (model, level) pairs already in output CSV")
    parser.add_argument("--serializer", choices=["symbolic", "grid", "memory"], default="symbolic",
                        help="Observation serializer: symbolic (bullet list), grid (ASCII FOV grid), or memory (compass-absolute, non-egocentric)")
    args = parser.parse_args(argv)
    if args.provider:
        os.environ["HALLUWORLD_PROVIDER"] = args.provider

    ser: Serializer = (
        GridSerializer() if args.serializer == "grid"
        else MemorySerializer() if args.serializer == "memory"
        else SymbolicSerializer(hide_carrying=True)
    )

    existing: set[tuple[str, str, str]] = set()  # populated per-model below

    all_dfs: list[pd.DataFrame] = []  # accumulates across all models for final summary

    for _base_model in args.models:
        model = f"{_base_model}+{args.model_tag}" if args.model_tag else _base_model

        # ── per-model output file ─────────────────────────────────────────────
        _out_path = Path(args.output)
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        # Sanitise model name for use in a filename (replace / and + with _)
        _model_slug = model.replace("/", "_").replace("+", "_")
        model_output = _out_path.parent / f"{_out_path.stem}_{_model_slug}{_out_path.suffix}"

        # Build resume skip-set from this model's own file only
        model_all_dfs: list[pd.DataFrame] = []
        model_existing: set[tuple[str, str, str]] = set()
        if args.resume and model_output.exists():
            try:
                prev = pd.read_csv(model_output)
                ser_col = prev["serializer"] if "serializer" in prev.columns else args.serializer
                model_existing = set(zip(prev["model"], prev["level"], ser_col))
                model_all_dfs.append(prev)
                print(f"Resuming {model}: {len(model_existing)} (level, serializer) pairs already done.")
            except Exception as e:
                print(f"Warning: could not read {model_output}: {e}. Starting fresh.")

        for level_key in args.levels:
            _ser_name = "MemorySerializer" if (
                level_key in STEPPED_LEVELS
                or level_key in MEMORY_SINGLE_LEVELS
                or level_key in CAUSAL_SINGLE_LEVELS
                or level_key in CAUSAL_MULTI_LEVELS
            ) else ("Mixed(Memory+Symbolic)" if level_key in U2_ORACLE_LEVELS
                    else args.serializer.capitalize() + "Serializer")
            if (model, level_key, _ser_name) in model_existing:
                print(f"  Skipping {model} / {level_key} / {_ser_name}")
                continue
            print(f"Running {model} on {level_key} ({args.episodes} episodes, {_ser_name})...")
            if level_key in CAUSAL_MULTI_LEVELS:
                # Two-observation causal levels: before/after demo, MemorySerializer
                df = run_multi_obs_level(
                    level_key=level_key,
                    obs_files=MULTI_OBS_LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=MemorySerializer(),
                    system_prompt=CAUSAL_SYSTEM_PROMPT,
                )
            elif level_key in MULTI_OBS_LEVELS:
                df = run_multi_obs_level(
                    level_key=level_key,
                    obs_files=MULTI_OBS_LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=ser,
                )
            elif level_key in MEMORY_SINGLE_LEVELS:
                df = run_level(
                    level_key=level_key,
                    level_path=LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=MemorySerializer(),
                    system_prompt=NARRATOR_SYSTEM_PROMPT,
                    reasoning_effort=args.reasoning_effort,
                    thinking_effort=args.thinking_effort,
                )
            elif level_key in CAUSAL_SINGLE_LEVELS:
                df = run_level(
                    level_key=level_key,
                    level_path=LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=MemorySerializer(),
                    system_prompt=CAUSAL_SYSTEM_PROMPT,
                    reasoning_effort=args.reasoning_effort,
                    thinking_effort=args.thinking_effort,
                )
            elif level_key == "U4_amnesiac":
                df = run_multi_obs_level(
                    level_key=level_key,
                    obs_files=[
                        ("Observation 1", U4_OBS1_PATH),
                        ("Observation 2", U4_OBS2_PATH),
                    ],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=SymbolicSerializer(),
                    system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                    interstitials=[
                        "[You observed the room and then left to explore another part of the "
                        "facility without touching anything. When you returned, you found the "
                        "room had changed (Observation 2).]",
                    ],
                )
            elif level_key == "U1_fog_of_war":
                df = run_multi_obs_level(
                    level_key=level_key,
                    obs_files=[
                        ("Observation 1 — Hub", U1_HUB_PATH),
                        ("Observation 2 — Room A", U1_ROOM_A_PATH),
                        ("Observation 3 — Room C", U1_ROOM_C_PATH),
                    ],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=SymbolicSerializer(),
                    system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                    interstitials=[
                        "[You walked north through the Hub corridor into Room A. "
                        "This facility has four rooms branching from the central Hub: "
                        "Room A (north), Room B (east), Room C (south), Room D (west).]",
                        "[You walked back through the Hub and continued south through the corridor into Room C. "
                        "You have now visited: Hub, Room A, Room C. "
                        "Rooms B and D were not visited during this trajectory.]",
                    ],
                )
            elif level_key in U2_ORACLE_LEVELS:
                _zone_path = {
                    "U2_oracle_high": U2_ZONE_HIGH_PATH,
                    "U2_oracle_mid":  U2_ZONE_MID_PATH,
                    "U2_oracle_low":  U2_ZONE_LOW_PATH,
                }[level_key]
                df = run_oracle_level(
                    level_key=level_key,
                    zone_path=_zone_path,
                    target_path=U2_TARGET_PATH,
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                )
            elif level_key in COMPOUND_LEVELS:
                if level_key == "X1_facility_tour":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A (Lab)", X1_ZONE_A_PATH),
                            ("Observation 2 — Zone B (Records Room)", X1_ZONE_B_PATH),
                            ("Observation 3 — Zone C (Control Room)", X1_ZONE_C_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser if args.serializer == "grid" else SymbolicSerializer(hide_carrying=True),
                        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                        interstitials=[
                            "[You pass through the security checkpoint at the far end of the Lab "
                            "and enter the Records Room. You are not carrying anything.]",
                            "[You follow the corridor east through a set of double doors "
                            "into the Control Room. You are not carrying anything.]",
                        ],
                    )
                elif level_key == "X2_facility_tour":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A (Lab)", X1_ZONE_A_PATH),
                            ("Observation 2 — Zone B (Records Room)", X1_ZONE_B_PATH),
                            ("Observation 3 — Zone C (Control Room)", X1_ZONE_C_PATH),
                            ("Observation 4 — Zone D (Maintenance Bay)", X2_ZONE_D_PATH),
                            ("Observation 5 — Zone E (Director's Office)", X2_ZONE_E_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser if args.serializer == "grid" else SymbolicSerializer(hide_carrying=True),
                        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                        interstitials=[
                            "[You pocket one red key from Zone A on your way out. "
                            "You pass through the security checkpoint and enter the Records Room. "
                            "You are carrying: red key.]",
                            "[You use the red key to unlock the security gate between Zone B and "
                            "Zone C. The key stays in the gate mechanism. "
                            "You enter the Control Room. You are not carrying anything.]",
                            "[You pick up one of the blue keys in Zone C. "
                            "You descend the service stairwell into the Maintenance Bay. "
                            "You are carrying: blue key.]",
                            "[You use the blue key to unlock the Maintenance Bay's exit hatch. "
                            "The key is retained by the lock. "
                            "You badge through the final security door into the Director's Office. "
                            "You are not carrying anything.]",
                        ],
                    )
                elif level_key == "X3_facility_tour":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A (Specimen Lab)", X3_ZONE_A_PATH),
                            ("Observation 2 — Zone B (The Vault)", X3_ZONE_B_PATH),
                            ("Observation 3 — Zone C (Records Office)", X3_ZONE_C_PATH),
                            ("Observation 4 — Zone D (Security Post)", X3_ZONE_D_PATH),
                            ("Observation 5 — Zone E (Control Room)", X3_ZONE_E_PATH),
                            ("Observation 6 — Zone F (Monitoring Station)", X3_ZONE_F_PATH),
                            ("Observation 7 — Zone G (Director's Suite)", X3_ZONE_G_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser if args.serializer == "grid" else SymbolicSerializer(hide_carrying=True),
                        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                        interstitials=[
                            "[You pocket one of the green keys from Zone A's secure cabinet. "
                            "You descend the stairwell and enter The Vault. "
                            "You are carrying: green key.]",
                            "[You use the green key to release The Vault's secondary access hatch. "
                            "The key remains in the locking mechanism. "
                            "You walk through to the Records Office. "
                            "You are not carrying anything.]",
                            "[You pick up the red key in Zone C. "
                            "You follow the main hallway east into the Security Post. "
                            "You are carrying: red key.]",
                            "[You leave the red key at the Security Post's key deposit box. "
                            "You badge through the fire door into the Control Room. "
                            "You are not carrying anything.]",
                            "[You take the single blue key from Zone E's wall panel. "
                            "You ride the service lift one floor up into the Monitoring Station. "
                            "You are carrying: blue key.]",
                            "[You use the blue key to unlock the Monitoring Station's exit gate. "
                            "The key is checked in at the security desk. "
                            "You pass through the final checkpoint into the Director's Suite. "
                            "You are not carrying anything.]",
                        ],
                    )
                elif level_key == "X4_facility_tour":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A (The Lab)", X4_ZONE_A_PATH),
                            ("Observation 2 — Zone B (The Records Chamber)", X4_ZONE_B_PATH),
                            ("Observation 3 — Zone C (The Monitoring Room)", X4_ZONE_C_PATH),
                            ("Observation 4 — Zone D (The Secure Vault)", X4_ZONE_D_PATH),
                            ("Observation 5 — Zone E (The Dense Storage Bay)", X4_ZONE_E_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser if args.serializer == "grid" else SymbolicSerializer(hide_carrying=True),
                        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                        interstitials=[
                            "[You notice the single blue key lodged among Zone A's red keys. "
                            "You pocket it as you leave. "
                            "You pass through the corridor into the Records Chamber. "
                            "You are carrying: blue key.]",
                            "[You use the blue key to pass through the security checkpoint "
                            "between Zone B and Zone C. The key stays in the gate mechanism. "
                            "You enter the Monitoring Room. "
                            "You are not carrying anything.]",
                            "[You pick up the green key in Zone C. "
                            "You follow the passage down into the Secure Vault. "
                            "You are carrying: green key.]",
                            "[You use the green key to release the Secure Vault's exit lock. "
                            "The key is retained by the mechanism. "
                            "You badge through into the Dense Storage Bay. "
                            "You are not carrying anything.]",
                        ],
                    )
                elif level_key == "X5_facility_tour":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A (The Atrium)", X5_ZONE_A_PATH),
                            ("Observation 2 — Zone B (The Records Hall)", X5_ZONE_B_PATH),
                            ("Observation 3 — Zone C (The Monitoring Center)", X5_ZONE_C_PATH),
                            ("Observation 4 — Zone D (The Repository)", X5_ZONE_D_PATH),
                            ("Observation 5 — Zone E (The Containment Bay)", X5_ZONE_E_PATH),
                            ("Observation 6 — Zone F (The Recovery Room)", X5_ZONE_F_PATH),
                            ("Observation 7 — Zone G (The Dense Repository)", X5_ZONE_G_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser if args.serializer == "grid" else SymbolicSerializer(hide_carrying=True),
                        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                        interstitials=[
                            "[You notice one of the blue keys near Zone A's wall. "
                            "You pocket it as you leave. "
                            "You pass through the corridor into the Records Hall. "
                            "You are carrying: blue key.]",
                            "[You use the blue key to pass through the security checkpoint "
                            "between Zone B and Zone C. The key stays in the gate mechanism. "
                            "You enter the Monitoring Center. "
                            "You are not carrying anything.]",
                            "[You pick up the green key in Zone C. "
                            "You follow the passage into the Repository. "
                            "You are carrying: green key.]",
                            "[You use the green key to release the Repository's exit lock. "
                            "The key is retained by the mechanism. "
                            "You badge through into the Containment Bay. "
                            "You are not carrying anything.]",
                            "[You pick up one of the blue keys in Zone E. "
                            "You follow the passage into the Recovery Room. "
                            "You are carrying: blue key.]",
                            "[You use the blue key to pass through the exit gate "
                            "between Zone F and Zone G. The key stays in the gate mechanism. "
                            "You enter the Dense Repository. "
                            "You are not carrying anything.]",
                        ],
                    )
                elif level_key == "X6_return_visit":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A, first visit (The Archive)", X6_ZONE_A_PATH),
                            ("Observation 2 — Zone B (The Specimen Room)", X6_ZONE_B_PATH),
                            ("Observation 3 — Zone C (The Monitoring Post)", X6_ZONE_C_PATH),
                            ("Observation 4 — Zone D (The Repository)", X6_ZONE_D_PATH),
                            ("Observation 5 — Zone A, second visit (The Archive — revisit)", X6_ZONE_A_REVISIT_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser if args.serializer == "grid" else SymbolicSerializer(hide_carrying=True),
                        system_prompt=TRAJECTORY_SYSTEM_PROMPT,
                        interstitials=[
                            "[You notice the blue key lodged among Zone A's red keys. "
                            "You pocket it as you leave. "
                            "You pass through the corridor into the Specimen Room. "
                            "You are carrying: blue key.]",
                            "[You use the blue key to pass through the security door "
                            "between Zone A and Zone B. The key stays in the door mechanism. "
                            "You enter the Monitoring Post. "
                            "You are not carrying anything.]",
                            "[You pick up the green key in Zone C. "
                            "You follow the passage into the Repository. "
                            "You are carrying: green key.]",
                            "[You use the green key to release the Repository's exit gate. "
                            "The key is retained by the mechanism. "
                            "You retrace your steps back to The Archive. "
                            "You are not carrying anything. "
                            "You notice the room has changed since your first visit.]",
                        ],
                    )
                elif level_key == "X7_dragon_keep":
                    df = run_multi_obs_level(
                        level_key=level_key,
                        obs_files=[
                            ("Observation 1 — Zone A: Village Square", X7_ZONE_A_PATH),
                            ("Observation 2 — Zone B: Forest Path", X7_ZONE_B_PATH),
                            ("Observation 3 — Zone C: Ancient Armory", X7_ZONE_C_PATH),
                            ("Observation 4 — Zone B (Return Visit): Forest Path", X7_ZONE_B_RETURN_PATH),
                            ("Observation 5 — Zone W: Abandoned Watchtower", X7_ZONE_W_PATH),
                            ("Observation 6 — Zone D: Dragon's Antechamber", X7_ZONE_D_PATH),
                            ("Observation 7 — Zone E: Dragon's Lair (before combat)", X7_ZONE_E_PATH),
                            ("Observation 8 — Zone E: Dragon's Lair (after combat)", X7_ZONE_E_POST_PATH),
                        ],
                        model=model,
                        n_episodes=args.episodes,
                        base_seed=args.seed,
                        verbose=args.verbose,
                        max_tokens=args.max_tokens,
                        ser=ser,
                        system_prompt=QUEST_SYSTEM_PROMPT,
                        interstitials=[
                            "[You pocket the gold key from the village square. "
                            "The health potion (blue ball) remains on the ground — you are at full health. "
                            "You head north along the forest path. "
                            "You are carrying: gold key.]",
                            "[You insert the gold key into the armory gate lock. The gate swings open. "
                            "The key remains in the mechanism. "
                            "You step into the Ancient Armory. "
                            "You are carrying: nothing.]",
                            "[You take the enchanted sword from the weapon rack and the "
                            "dragon-scale shield from the wall. "
                            "Before continuing, you backtrack south through the now-open armory gate "
                            "to scout the forest for any remaining supplies. "
                            "You are carrying: enchanted sword, dragon-scale shield.]",
                            "[Nothing new in the forest path. You spot an overgrown side trail branching "
                            "east from the tree line and decide to investigate. "
                            "You are carrying: enchanted sword, dragon-scale shield.]",
                            "[The watchtower holds nothing of use — only old expedition logs. "
                            "You backtrack west and head south toward the Dragon's Antechamber. "
                            "You are carrying: enchanted sword, dragon-scale shield.]",
                            "[You choose not to disturb the dark orb on its pedestal. "
                            "You shove the boulder onto the pressure plate — the iron gate grinds open. "
                            "You step through into the Dragon's Lair. "
                            "You are carrying: enchanted sword, dragon-scale shield.]",
                            "[Combat with Malachar begins.\n"
                            "Turn 1: You strike with the enchanted sword. Damage dealt: 5. Dragon HP: 10 → 5. "
                            "Malachar breathes fire. Dragon-scale shield absorbs 3 damage; "
                            "2 residual fire damage bleeds through. Agent HP: 10 → 8.\n"
                            "Turn 2: You strike again. Damage dealt: 5. Dragon HP: 5 → 0. Malachar is defeated! "
                            "Agent final HP: 8.\n"
                            "You are carrying: enchanted sword, dragon-scale shield.]",
                        ],
                    )
                else:
                    raise ValueError(f"Unknown compound level: {level_key}")
            elif level_key in STEPPED_LEVELS:
                level_path, n_steps = STEPPED_LEVELS[level_key]
                df = run_stepped_level(
                    level_key=level_key,
                    level_path=level_path,
                    n_steps=n_steps,
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                )
            elif level_key in TWO_FILE_LEVELS:
                _sys = INCIDENT_SYSTEM_PROMPT if level_key == "M3_incident_report" else None
                _interstitial = "[6 steps elapsed; you remained stationary]" if level_key == "M3_incident_report" else None
                df = run_two_file_level(
                    level_key=level_key,
                    level_paths=TWO_FILE_LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=ser,
                    system_prompt=_sys,
                    interstitial=_interstitial,
                )
            elif level_key in DELTA_LEVELS:
                df = run_delta_level(
                    level_key=level_key,
                    level_path=LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=ser,
                )
            else:
                df = run_level(
                    level_key=level_key,
                    level_path=LEVELS[level_key],
                    model=model,
                    n_episodes=args.episodes,
                    base_seed=args.seed,
                    verbose=args.verbose,
                    max_tokens=args.max_tokens,
                    ser=ser,
                    reasoning_effort=args.reasoning_effort,
                    thinking_effort=args.thinking_effort,
                )
            model_all_dfs.append(df)
            all_dfs.append(df)

            model_combined = pd.concat(model_all_dfs, ignore_index=True)
            model_combined.to_csv(model_output, index=False)
            print(f"  Saved → {model_output}")

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    print("\n=== Summary (scoreable probes only) ===")
    if combined.empty or "score" not in combined.columns:
        print("  (no new rows this run — all skipped via resume)")
    else:
        scoreable = combined[combined["score"].notna()].copy()
        scoreable["score"] = scoreable["score"].astype(float)
        summary = (
            scoreable.groupby(["model", "level", "probe_type"])["score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "acc", "count": "n"})
        )
        summary["acc"] = summary["acc"].map("{:.1%}".format)
        print(summary.to_string())

    print("\n=== List probes (inspect manually) ===")
    list_rows = combined[combined["score"].isna() & combined["response"].notna()] if not combined.empty and "score" in combined.columns else pd.DataFrame()
    for _, row in list_rows.head(6).iterrows():
        print(f"  [{row['level']} / {row['probe_type']}]")
        print(f"  GT:  {row['ground_truth'][:100]}")
        print(f"  LM:  {row['response'][:120]}")
        print()


if __name__ == "__main__":
    main()
