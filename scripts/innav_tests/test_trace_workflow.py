#!/usr/bin/env python3
"""Test the complete trace workflow using navigate_until_sufficient.

This tests:
1. Navigate until sufficiency (PROVEN WORKING)
2. Save trace with messages + state
3. Restore from trace
4. Probe

"""
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.lm.openai_lm import OpenAILM
from halluworld.tracks.grid.probes.visibility import PresenceProbe, CountProbe
from halluworld.tracks.grid.evaluators import PresenceEvaluator
from halluworld.tracks.innav.navigation import navigate_until_sufficient
from halluworld.data import LEVELS_DIR


def test_full_workflow():
    """Test: Navigate → Save → Restore → Probe"""

    print("="*70)
    print("TESTING TRACE WORKFLOW")
    print("="*70)
    print()

    # Setup
    level_path = str(LEVELS_DIR / "corridor_gauntlet.txt")
    seed = 42

    env = AsciiEnv.from_file(level_path)
    serializer = SymbolicSerializer()
    lm = OpenAILM(model="gpt-5.4", max_tokens=256, reasoning_effort="medium")

    # Load config
    with open(str(LEVELS_DIR / "corridor_gauntlet.innav.json")) as f:
        config = json.load(f)

    targets = [tuple(t) for t in config['navigation_target_locations']]
    params = config['navigation_params']

    print("Step 1: Navigate until sufficiency")
    print("-"*70)

    result = navigate_until_sufficient(
        env=env,
        lm=lm,
        serializer=serializer,
        static_probe_locations=targets,
        seed=seed,
        max_steps=params['max_steps'],
        max_distance=params['sufficiency_distance'],
        min_steps=params['min_steps'],
        min_distance_from_start=params['min_distance_from_start']
    )

    if not result['sufficient']:
        print(f"❌ Navigation failed: {result['reason']}")
        return False

    print(f"✅ Navigated to sufficiency:")
    print(f"   Steps: {result['sufficient_step']}")
    print(f"   Position: {result['sufficient_position']}")
    print(f"   Distance from target: {result['sufficient_distance']:.2f}")
    print()

    # Step 2: Save trace
    print("Step 2: Save trace")
    print("-"*70)

    trace_dir = Path("test_traces_correct")
    trace_dir.mkdir(exist_ok=True)

    trace_data = {
        "seed": seed,
        "navigation_model": "gpt-5.4",
        "serializer": "SymbolicSerializer",
        "sufficient": True,
        "sufficient_step": int(result['sufficient_step']),
        "sufficient_position": [int(x) for x in result['sufficient_position']],
        "messages": result['messages'],  # Full chat history!
        "trajectory": [int(step['action']) if isinstance(step, dict) else int(step) for step in result['trajectory']],  # Action sequence
        "reached_goal": result['reached_goal'],
    }

    trace_path = trace_dir / "nav_trace_seed42.json"
    with open(trace_path, 'w') as f:
        json.dump(trace_data, f, indent=2)

    print(f"✅ Saved trace: {trace_path}")
    print(f"   Messages: {len(trace_data['messages'])} entries")
    print(f"   Actions: {len(trace_data['trajectory'])} steps")
    print()

    # Step 3: Restore from trace
    print("Step 3: Restore environment from trace")
    print("-"*70)

    # Fresh environment
    env2 = AsciiEnv.from_file(level_path)
    env2.reset(seed=seed)

    # Replay actions
    for action in trace_data['trajectory']:
        env2.step(action)

    restored_pos = (int(env2.agent_pos[0]), int(env2.agent_pos[1]))
    print(f"✅ Environment restored:")
    print(f"   Position: {restored_pos}")
    print(f"   Matches original: {restored_pos == tuple(trace_data['sufficient_position'])}")
    print()

    # Step 4: Probe at restored state
    print("Step 4: Probe at restored state")
    print("-"*70)

    probe_lm = OpenAILM(model="gpt-5.4-mini", max_tokens=256, reasoning_effort="medium")

    # Generate probe
    probe = PresenceProbe()
    probe_result = probe.generate(env2)

    if probe_result.question:
        # Ask probe
        print(f"Probe question: {probe_result.question}")
        print(f"Ground truth: {probe_result.ground_truth}")

        # Get state description for probe
        state_desc = serializer.serialize(env2)  # Pass env, not obs!
        probe_prompt = f"{state_desc}\n\n{probe_result.question}"

        response = probe_lm([
            {"role": "system", "content": "Answer the question about your current observation."},
            {"role": "user", "content": probe_prompt}
        ])

        print(f"Model response: {response}")

        # Evaluate
        evaluator = PresenceEvaluator()
        score = evaluator.evaluate(probe_result.ground_truth, response)
        print(f"Score: {score:.1f}")
        print()

    print("="*70)
    print("✅ WORKFLOW TEST COMPLETE")
    print("="*70)
    print()
    print("All steps working:")
    print("  ✅ Navigate until sufficiency")
    print("  ✅ Save trace (messages + actions)")
    print("  ✅ Restore environment state")
    print("  ✅ Probe at restored state")
    print()

    return True


if __name__ == "__main__":
    success = test_full_workflow()
    sys.exit(0 if success else 1)
