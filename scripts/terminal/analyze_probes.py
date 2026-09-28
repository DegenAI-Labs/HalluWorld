#!/usr/bin/env python3
"""Analyze Derek's 530 curated probes for distribution statistics."""

import json
from pathlib import Path
from collections import Counter, defaultdict

PROBE_DIR = Path("extracted_llm_probes_20260504")

def analyze_probes():
    """Generate comprehensive statistics on the 530 probes."""

    probe_files = sorted(PROBE_DIR.glob("*.json"))
    # Exclude summary.json
    probe_files = [f for f in probe_files if f.name != 'summary.json']
    print(f"Total probe files: {len(probe_files)}\n")

    # Collectors
    probe_types = Counter()
    tasks = Counter()
    difficulty_scores = Counter()
    answerability_scores = Counter()
    answer_schemas = Counter()
    failure_modes = Counter()
    injectors = Counter()

    # Per-task breakdown
    task_probe_types = defaultdict(Counter)
    task_difficulties = defaultdict(list)

    # Difficulty score distributions per probe type
    type_difficulties = defaultdict(list)

    for probe_file in probe_files:
        # Skip summary.json which has different structure
        if probe_file.name == 'summary.json':
            continue

        with open(probe_file) as f:
            probe = json.load(f)

            # Skip if no metadata
            if 'metadata' not in probe:
                continue

            meta = probe['metadata']

            # Basic counts
            probe_type = meta.get('probe_type', 'unknown')
            task_name = meta.get('task_name', 'unknown')
            difficulty = meta.get('difficulty_score', 0)
            answerability = meta.get('answerability_score', 0)
            answer_schema = meta.get('answer_schema', 'unknown')
            failure_mode = meta.get('failure_mode_target', 'unknown')
            injector = meta.get('injector', 'unknown')

            probe_types[probe_type] += 1
            tasks[task_name] += 1
            difficulty_scores[difficulty] += 1
            answerability_scores[answerability] += 1
            answer_schemas[answer_schema] += 1
            failure_modes[failure_mode] += 1
            injectors[injector] += 1

            # Per-task breakdown
            task_probe_types[task_name][probe_type] += 1
            task_difficulties[task_name].append(difficulty)

            # Type-difficulty correlation
            type_difficulties[probe_type].append(difficulty)

    # Print results
    print("=" * 80)
    print("PROBE TYPE DISTRIBUTION")
    print("=" * 80)
    for probe_type, count in probe_types.most_common():
        pct = 100 * count / len(probe_files)
        print(f"  {probe_type:25s} {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("DIFFICULTY SCORE DISTRIBUTION")
    print("=" * 80)
    for score in sorted(difficulty_scores.keys()):
        count = difficulty_scores[score]
        pct = 100 * count / len(probe_files)
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("ANSWERABILITY SCORE DISTRIBUTION")
    print("=" * 80)
    for score in sorted(answerability_scores.keys()):
        count = answerability_scores[score]
        pct = 100 * count / len(probe_files)
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("FAILURE MODE TARGETS")
    print("=" * 80)
    for failure_mode, count in failure_modes.most_common():
        pct = 100 * count / len(probe_files)
        print(f"  {failure_mode:30s} {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("ANSWER SCHEMA PATTERNS (Top 15)")
    print("=" * 80)
    for schema, count in answer_schemas.most_common(15):
        pct = 100 * count / len(probe_files)
        # Truncate long schemas
        display_schema = schema if len(schema) <= 50 else schema[:47] + "..."
        print(f"  {count:4d} ({pct:5.1f}%) {display_schema}")

    print("\n" + "=" * 80)
    print("TASK COVERAGE (Top 20 tasks by probe count)")
    print("=" * 80)
    for task_name, count in tasks.most_common(20):
        pct = 100 * count / len(probe_files)
        avg_difficulty = sum(task_difficulties[task_name]) / len(task_difficulties[task_name])
        print(f"  {count:3d} ({pct:4.1f}%) {task_name:40s} avg_difficulty={avg_difficulty:.1f}")

    print("\n" + "=" * 80)
    print("AVERAGE DIFFICULTY BY PROBE TYPE")
    print("=" * 80)
    for probe_type in sorted(type_difficulties.keys()):
        scores = type_difficulties[probe_type]
        avg = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(f"  {probe_type:25s} avg={avg:.2f} min={min_score} max={max_score} n={len(scores)}")

    # Summary statistics
    all_difficulties = [meta.get('difficulty_score', 0) for f in probe_files
                        for meta in [json.load(open(f))['metadata']]]
    all_answerabilities = [meta.get('answerability_score', 0) for f in probe_files
                           for meta in [json.load(open(f))['metadata']]]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"  Total probes:                  {len(probe_files)}")
    print(f"  Unique tasks:                  {len(tasks)}")
    print(f"  Unique probe types:            {len(probe_types)}")
    print(f"  Unique answer schemas:         {len(answer_schemas)}")
    print(f"  Avg difficulty (all):          {sum(all_difficulties) / len(all_difficulties):.2f}")
    print(f"  Avg answerability (all):       {sum(all_answerabilities) / len(all_answerabilities):.2f}")
    print(f"  Difficulty >= 4:               {sum(1 for d in all_difficulties if d >= 4)} ({100*sum(1 for d in all_difficulties if d >= 4)/len(all_difficulties):.1f}%)")

    # Output per-task breakdown to file for reference
    with open('probe_task_breakdown.txt', 'w') as out:
        out.write("DETAILED TASK BREAKDOWN\n")
        out.write("=" * 100 + "\n\n")

        for task_name in sorted(tasks.keys()):
            out.write(f"\nTask: {task_name}\n")
            out.write(f"  Total probes: {tasks[task_name]}\n")
            out.write(f"  Avg difficulty: {sum(task_difficulties[task_name]) / len(task_difficulties[task_name]):.2f}\n")
            out.write(f"  Probe types:\n")
            for ptype, count in task_probe_types[task_name].most_common():
                out.write(f"    {ptype:25s} {count:3d}\n")

    print(f"\n✓ Detailed task breakdown written to: probe_task_breakdown.txt")

if __name__ == "__main__":
    analyze_probes()
