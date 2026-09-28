#!/usr/bin/env python3
"""
Comprehensive probing launcher - runs each world×model independently.

Strategy:
- Each world×model runs as separate task (failures isolated)
- Batched by provider to respect rate limits
- Progress tracked in real-time
- Can resume if interrupted
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime

# Comprehensive world coverage (16 worlds total)
# HalluWorld-Hard subset (9/12) + X-tier expansion (6) + P4b (1)
WORLDS = [
    # HalluWorld-Hard (9 worlds)
    ("P1_dense_array", "grid"),
    ("P2_corridor_gauntlet", "grid"),
    ("P4_harder_array", "grid"),
    ("C1a_persistent_chain", "memory"),
    ("C1a_noboard", "memory"),
    ("C1b_noboard", "memory"),
    ("C5a_adversarial_board", "memory"),
    ("C6_flood_fire_escape", "memory"),
    ("X5_facility_tour", "grid"),

    # X-tier expansion (matching Emmy's coverage)
    ("X1_facility_3zone", "grid"),
    ("X2_facility_5zone", "grid"),
    ("X3_facility_7zone", "grid"),
    ("X4_compound_witness", "grid"),
    ("X6_return_visit", "grid"),
    ("X7_dragon_keep", "grid"),

    # P-tier expansion
    ("P4b_delta_perception", "grid"),
]

# Models by provider (14 total)
MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-5.4-mini",
        "gpt-5.5",
    ],
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-sonnet-4-6_thinking",
        "claude-opus-4-6",
        "claude-opus-4-6_thinking",
    ],
    "baseten": [
        "zai-org/GLM-5",
        "moonshotai/Kimi-K2.6",
        "deepseek-ai/DeepSeek-V3-0324",
    ],
}

# Provider-specific settings
PROVIDER_DELAYS = {
    "openai": 2,      # 2s between OpenAI calls
    "anthropic": 3,   # 3s between Anthropic calls
    "baseten": 1,     # 1s between Baseten calls
}


def run_probe(world, serializer, model, provider):
    """Run single probe task independently."""

    # Sanitize model name for filename
    model_safe = model.replace("/", "-").replace("_", "-")
    world_safe = world.replace("_", "-")

    output_csv = f"probe_{world_safe}_{serializer}_{model_safe}.csv"
    log_file = f"probe_{world_safe}_{serializer}_{model_safe}.log"

    # Check if already completed
    if Path(output_csv).exists():
        size_kb = Path(output_csv).stat().st_size / 1024
        if size_kb > 1:  # More than 1KB suggests completion
            print(f"  ⏭️  SKIP {world} + {model} (already complete, {size_kb:.1f}KB)")
            return "skipped", 0

    # Build command
    cmd = [
        "python", "run_innav_eval.py",
        "--models", model,
        "--levels", world,
        "--serializer", serializer,
        "--episodes", "5",
        "--seed", "999",
        "--trace-dir", "navigation_traces",
        "--navigation-model", "gpt-5.4-mini",  # Reuse gpt-5.4-mini navigation traces
        "--output", output_csv,
    ]

    print(f"  🚀 LAUNCH {world} + {model}")

    start_time = time.time()

    try:
        # Run with timeout (15 min per world×model)
        with open(log_file, "w") as log:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,  # 15 minutes
                check=False
            )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  ✅ SUCCESS {world} + {model} ({elapsed:.1f}s)")
            return "success", elapsed
        else:
            print(f"  ❌ FAILED {world} + {model} (exit code {result.returncode})")
            return "failed", elapsed

    except subprocess.TimeoutExpired:
        print(f"  ⏱️  TIMEOUT {world} + {model} (>15min)")
        return "timeout", 900
    except Exception as e:
        print(f"  💥 ERROR {world} + {model}: {str(e)[:100]}")
        return "error", 0


def main():
    print("=" * 80)
    print("COMPREHENSIVE PROBING - HalluWorld-Hard Subset")
    print("=" * 80)
    print()
    print(f"Worlds: {len(WORLDS)}")
    print(f"Models: {sum(len(m) for m in MODELS.values())}")
    print(f"Total combinations: {len(WORLDS) * sum(len(m) for m in MODELS.values())}")
    print()

    # Stats tracking
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "timeout": 0,
        "error": 0,
    }

    start_time = datetime.now()

    # Process by provider (sequential within provider, can parallelize across)
    for provider, models in MODELS.items():
        print()
        print(f"{'=' * 80}")
        print(f"PROVIDER: {provider.upper()} ({len(models)} models)")
        print(f"{'=' * 80}")
        print()

        delay = PROVIDER_DELAYS[provider]

        for model in models:
            print(f"\n--- MODEL: {model} ({len(WORLDS)} worlds) ---")

            for world, serializer in WORLDS:
                status, elapsed = run_probe(world, serializer, model, provider)
                stats[status] += 1

                # Brief delay between calls to same provider
                if status != "skipped":
                    time.sleep(delay)

            print()

    # Final summary
    elapsed_total = datetime.now() - start_time
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total time: {elapsed_total}")
    print()
    print(f"✅ Success:  {stats['success']}")
    print(f"⏭️  Skipped:  {stats['skipped']}")
    print(f"❌ Failed:   {stats['failed']}")
    print(f"⏱️  Timeout:  {stats['timeout']}")
    print(f"💥 Error:    {stats['error']}")
    print()
    print(f"Total: {sum(stats.values())}")
    print()

    # List output files
    probe_files = sorted(Path(".").glob("probe_*.csv"))
    total_size = sum(f.stat().st_size for f in probe_files) / (1024 * 1024)
    print(f"Probe CSV files: {len(probe_files)} ({total_size:.1f} MB)")
    print()


if __name__ == "__main__":
    main()
