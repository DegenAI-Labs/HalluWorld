#!/usr/bin/env python3
"""Test save-then-load workflow for innav traces.

This demonstrates:
1. First run: Navigate fresh, save trace
2. Second run: Load saved trace, skip navigation
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import shutil

load_dotenv()
sys.path.insert(0, '.')

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.lm.openai_lm import OpenAILM
from halluworld.tracks.grid.probes.visibility import (
    PresenceProbe,
    CountProbe,
    AttributeProbe,
    AllocentricLocationProbe,
)
from halluworld.tracks.innav.engine import run_innav_episodes
from halluworld.data import LEVELS_DIR

def test_workflow():
    """Test: Save trace, then load it"""

    print("="*70)
    print("TESTING SAVE-THEN-LOAD WORKFLOW")
    print("="*70)
    print()

    # Setup
    level_path = str(LEVELS_DIR / "corridor_gauntlet.txt")
    seed = 999  # Use unique seed to avoid conflicts

    # Clean up any existing traces for this seed to start fresh
    trace_pattern = f"trace_*_seed{seed}_*.json"
    for old_trace in Path("navigation_traces").glob(trace_pattern):
        print(f"Removing old trace: {old_trace}")
        old_trace.unlink()

    # Models
    lm = OpenAILM(model="gpt-5.4-mini", max_tokens=256, reasoning_effort="medium")
    navigation_lm = OpenAILM(model="gpt-5.4", max_tokens=256, reasoning_effort="medium")

    # Environment setup
    env = AsciiEnv.from_file(level_path)
    serializer = SymbolicSerializer()

    # Probes
    probes = [
        PresenceProbe(),
        CountProbe(),
        AttributeProbe(),
        AllocentricLocationProbe(),
    ]

    # ===== RUN 1: Navigate fresh and save =====
    print("RUN 1: Navigate fresh (should see 🚀 FRESH NAVIGATION)")
    print("-"*70)

    df1 = run_innav_episodes(
        env=env,
        lm=lm,
        serializer=serializer,
        probes=probes,
        n_episodes=1,
        base_seed=seed,
        probe_timesteps=3,
        max_steps=50,
        navigation_lm=navigation_lm,
        save_traces=True,  # Save trace
        trace_dir=None,  # No pre-existing traces
    )

    print(f"✅ Run 1 complete:")
    print(f"   Episodes: {len(df1)//4}")  # 4 probes per episode
    print(f"   Steps: {df1['steps_taken'].iloc[0]}")
    print(f"   Reached goal: {df1['reached_goal'].iloc[0]}")
    print(f"   Ego accuracy: {df1['innav_accuracy'].iloc[0]:.1%}")
    print()

    # Check that trace was saved
    expected_trace = Path(f"navigation_traces/trace_SymbolicSerializer_seed{seed}_OpenAILM.json")
    if expected_trace.exists():
        print(f"✅ Trace saved: {expected_trace}")
    else:
        print(f"❌ Trace NOT saved to expected location: {expected_trace}")
        return False
    print()

    # ===== RUN 2: Load saved trace =====
    print("RUN 2: Load trace (should see ✅ REUSING navigation trace)")
    print("-"*70)

    # Fresh environment
    env2 = AsciiEnv.from_file(level_path)

    df2 = run_innav_episodes(
        env=env2,
        lm=lm,
        serializer=serializer,
        probes=probes,
        n_episodes=1,
        base_seed=seed,
        probe_timesteps=3,
        max_steps=50,
        navigation_lm=navigation_lm,
        save_traces=False,  # Don't overwrite
        trace_dir="navigation_traces",  # Load from here!
    )

    print(f"✅ Run 2 complete:")
    print(f"   Episodes: {len(df2)//4}")
    print(f"   Steps: {df2['steps_taken'].iloc[0]}")
    print(f"   Reached goal: {df2['reached_goal'].iloc[0]}")
    print(f"   Ego accuracy: {df2['innav_accuracy'].iloc[0]:.1%}")
    print()

    # Verify both runs match
    print("="*70)
    print("VERIFICATION")
    print("="*70)
    steps1 = df1['steps_taken'].iloc[0]
    steps2 = df2['steps_taken'].iloc[0]
    goal1 = df1['reached_goal'].iloc[0]
    goal2 = df2['reached_goal'].iloc[0]

    print(f"Steps match: {steps1 == steps2} ({steps1} vs {steps2})")
    print(f"Goal match: {goal1 == goal2}")
    print()

    if steps1 == steps2 and goal1 == goal2:
        print("✅ SUCCESS: Trace was loaded and reused!")
        print("   (Same navigation trajectory in both runs)")
        return True
    else:
        print("❌ FAILED: Results differ (trace may not have been reused)")
        return False


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
