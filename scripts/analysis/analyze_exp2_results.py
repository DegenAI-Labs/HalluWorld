#!/usr/bin/env python3
"""
Analyze Experiment 2 results: Reasoning effort ablation study
"""

import pandas as pd
import os
from pathlib import Path

def analyze_exp2(results_dir="results_varun_exp2"):
    """Analyze reasoning effort ablation results"""

    # Load all CSV files
    csv_files = sorted(Path(results_dir).glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    # Combine all results
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Extract reasoning_effort from model name (stored as "gpt-5-mini")
    # We need to get it from filename instead
    file_metadata = []
    for csv_file in csv_files:
        # Parse filename: gpt5mini_{effort}_seed{seed}.csv
        filename = csv_file.stem
        parts = filename.split('_')

        if len(parts) >= 3:
            effort = parts[1]  # minimal, low, medium, high
            seed = parts[2].replace('seed', '')

            # Load this file's data
            df = pd.read_csv(csv_file)
            df['reasoning_effort'] = effort
            df['seed'] = int(seed)
            file_metadata.append(df)

    if not file_metadata:
        print("Could not parse filenames")
        return

    all_data = pd.concat(file_metadata, ignore_index=True)

    # Filter to rows with valid scores (exclude NaN)
    scoreable = all_data[all_data['score'].notna()].copy()

    print("=== Experiment 2: Reasoning Effort Ablation ===\n")

    # Group by reasoning_effort, level, probe_type
    grouped = scoreable.groupby(['reasoning_effort', 'level', 'probe_type'])['score'].agg(['mean', 'count']).reset_index()
    grouped.columns = ['reasoning_effort', 'level', 'probe_type', 'mean_score', 'n']
    grouped['acc'] = grouped['mean_score'].apply(lambda x: f"{x*100:.1f}%")

    # Pivot table for easy comparison
    pivot = grouped.pivot_table(
        index=['level', 'probe_type'],
        columns='reasoning_effort',
        values='mean_score',
        aggfunc='first'
    )

    # Order columns by effort level
    effort_order = ['minimal', 'low', 'medium', 'high']
    available_efforts = [e for e in effort_order if e in pivot.columns]
    pivot = pivot[available_efforts]

    # Format as percentages
    pivot_pct = pivot * 100

    print("=== Performance by Reasoning Effort ===")
    print(pivot_pct.to_string())
    print()

    # Highlight allocentric performance
    print("=== Allocentric Location Probe (Key Finding) ===")
    allocentric = grouped[grouped['probe_type'] == 'allocentric_location']
    allocentric_pivot = allocentric.pivot_table(
        index='level',
        columns='reasoning_effort',
        values='mean_score',
        aggfunc='first'
    )
    allocentric_pivot = allocentric_pivot[available_efforts] * 100
    print(allocentric_pivot.to_string())
    print()

    # Calculate seed variance for each effort level
    print("=== Seed Variance Analysis ===")
    seed_variance = scoreable.groupby(['reasoning_effort', 'level', 'probe_type', 'seed'])['score'].mean().reset_index()
    variance_summary = seed_variance.groupby(['reasoning_effort', 'level', 'probe_type'])['score'].agg(['min', 'max', 'std']).reset_index()
    variance_summary['range'] = (variance_summary['max'] - variance_summary['min']) * 100
    variance_summary['std_pct'] = variance_summary['std'] * 100

    # Show probes with high variance
    high_variance = variance_summary[variance_summary['range'] > 10].sort_values('range', ascending=False)
    if not high_variance.empty:
        print("Probes with >10% seed variance:")
        print(high_variance[['reasoning_effort', 'level', 'probe_type', 'range', 'std_pct']].to_string(index=False))
    else:
        print("No probes with >10% seed variance")
    print()

    # Count probe analysis (known to have high variance)
    print("=== Count Probe Deep Dive ===")
    count_data = scoreable[scoreable['probe_type'] == 'count']
    if not count_data.empty:
        count_by_effort = count_data.groupby(['reasoning_effort', 'seed'])['score'].mean() * 100
        count_pivot = count_by_effort.unstack()
        count_pivot = count_pivot.reindex(available_efforts)
        print(count_pivot.to_string())
        print()

    # Summary statistics
    print("=== Overall Summary ===")
    overall = scoreable.groupby('reasoning_effort')['score'].agg(['mean', 'std', 'count']).reset_index()
    overall['mean_pct'] = overall['mean'] * 100
    overall['std_pct'] = overall['std'] * 100
    overall = overall.reindex(overall['reasoning_effort'].map({e: i for i, e in enumerate(effort_order)}).sort_values().index)
    print(overall[['reasoning_effort', 'mean_pct', 'std_pct', 'count']].to_string(index=False))
    print()

    # Files analyzed
    print(f"=== Files Analyzed ({len(csv_files)}) ===")
    for csv_file in sorted(csv_files):
        size_kb = csv_file.stat().st_size / 1024
        print(f"  {csv_file.name:40s} {size_kb:6.1f} KB")

if __name__ == "__main__":
    analyze_exp2()
