#!/usr/bin/env python3
"""
Parallel experiment launcher for Varun's perception experiments.
Launches multiple experiments concurrently to save time.
"""
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Experiment configurations
EXPERIMENTS = {
    "exp1_baseline": [
        # Format: (model, seed, output_file)
        ("gpt-4o-mini", 42, "results_varun_exp1/gpt4omini_seed42.csv"),
        ("gpt-4o-mini", 123, "results_varun_exp1/gpt4omini_seed123.csv"),
        ("gpt-4o-mini", 999, "results_varun_exp1/gpt4omini_seed999.csv"),
        ("gpt-4o", 42, "results_varun_exp1/gpt4o_seed42.csv"),
        ("gpt-4o", 123, "results_varun_exp1/gpt4o_seed123.csv"),
        ("gpt-4o", 999, "results_varun_exp1/gpt4o_seed999.csv"),
        ("gpt-5-mini", 42, "results_varun_exp1/gpt5mini_seed42.csv"),
        ("gpt-5-mini", 123, "results_varun_exp1/gpt5mini_seed123.csv"),
        ("gpt-5-mini", 999, "results_varun_exp1/gpt5mini_seed999.csv"),
        ("GLM-5", 42, "results_varun_exp1/glm5_seed42.csv"),
        ("GLM-5", 123, "results_varun_exp1/glm5_seed123.csv"),
        ("GLM-5", 999, "results_varun_exp1/glm5_seed999.csv"),
    ],
    "exp2_reasoning": [
        # GPT-5-mini with different reasoning efforts
        ("gpt-5-mini", 42, "minimal", "results_varun_exp2/gpt5mini_minimal_seed42.csv"),
        ("gpt-5-mini", 123, "minimal", "results_varun_exp2/gpt5mini_minimal_seed123.csv"),
        ("gpt-5-mini", 999, "minimal", "results_varun_exp2/gpt5mini_minimal_seed999.csv"),
        ("gpt-5-mini", 42, "low", "results_varun_exp2/gpt5mini_low_seed42.csv"),
        ("gpt-5-mini", 123, "low", "results_varun_exp2/gpt5mini_low_seed123.csv"),
        ("gpt-5-mini", 999, "low", "results_varun_exp2/gpt5mini_low_seed999.csv"),
        ("gpt-5-mini", 42, "medium", "results_varun_exp2/gpt5mini_medium_seed42.csv"),
        ("gpt-5-mini", 123, "medium", "results_varun_exp2/gpt5mini_medium_seed123.csv"),
        ("gpt-5-mini", 999, "medium", "results_varun_exp2/gpt5mini_medium_seed999.csv"),
        ("gpt-5-mini", 42, "high", "results_varun_exp2/gpt5mini_high_seed42.csv"),
        ("gpt-5-mini", 123, "high", "results_varun_exp2/gpt5mini_high_seed123.csv"),
        ("gpt-5-mini", 999, "high", "results_varun_exp2/gpt5mini_high_seed999.csv"),
    ],
}

def run_experiment(model, seed, output, reasoning_effort=None, episodes=20):
    """Run a single experiment."""
    cmd = [
        "python", "run_perception_eval.py",
        "--models", model,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", output,
    ]

    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])

    try:
        print(f"▶️  Starting: {model} (seed={seed}, effort={reasoning_effort})")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ Completed: {model} (seed={seed}) → {output}")
        return True, model, seed, reasoning_effort
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {model} (seed={seed})")
        print(f"   Error: {e.stderr[:200]}")
        return False, model, seed, reasoning_effort

def main():
    # Create output directories
    Path("results_varun_exp1").mkdir(exist_ok=True)
    Path("results_varun_exp2").mkdir(exist_ok=True)

    print("🚀 Varun's Parallel Experiment Launcher")
    print("=" * 60)
    print()

    # Parse command line args
    if len(sys.argv) > 1 and sys.argv[1] in EXPERIMENTS:
        experiment = sys.argv[1]
        experiments_to_run = {experiment: EXPERIMENTS[experiment]}
    else:
        experiments_to_run = EXPERIMENTS

    # Determine parallelism
    max_workers = 4  # Run 4 experiments in parallel (adjust based on API limits)

    for exp_name, configs in experiments_to_run.items():
        print(f"\n📊 Running {exp_name}")
        print(f"   Total experiments: {len(configs)}")
        print(f"   Parallelism: {max_workers}")
        print()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if exp_name == "exp1_baseline":
                futures = {
                    executor.submit(run_experiment, model, seed, output): (model, seed)
                    for model, seed, output in configs
                }
            else:  # exp2_reasoning
                futures = {
                    executor.submit(run_experiment, model, seed, output, effort): (model, seed, effort)
                    for model, seed, effort, output in configs
                }

            completed = 0
            failed = 0
            for future in as_completed(futures):
                success, model, seed, effort = future.result()
                completed += 1
                if not success:
                    failed += 1
                print(f"   Progress: {completed}/{len(configs)} ({failed} failed)")

    print()
    print("=" * 60)
    print("✅ All experiments complete!")
    print()
    print("📂 Results saved to:")
    print("   - results_varun_exp1/ (baseline)")
    print("   - results_varun_exp2/ (reasoning effort)")
    print()
    print("📝 Next: Update VARUN_EXPERIMENTS.md with results")

if __name__ == "__main__":
    main()
