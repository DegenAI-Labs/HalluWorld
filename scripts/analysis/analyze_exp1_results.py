#!/usr/bin/env python3
"""Analyze Experiment 1 results and generate summary tables."""
import pandas as pd
from pathlib import Path
import sys

def analyze_results(results_dir="results_varun_exp1"):
    """Load and analyze all experiment 1 results."""
    results_path = Path(results_dir)

    # Load all CSV files
    all_data = []
    for csv_file in sorted(results_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
            print(f"✅ Loaded: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"❌ Error loading {csv_file.name}: {e}")

    if not all_data:
        print("No data files found!")
        return

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total rows: {len(combined)}")
    print(f"📊 Models: {combined['model'].unique()}")
    print(f"📊 Levels: {combined['level'].unique()}")

    # Filter scoreable probes
    scoreable = combined[combined['score'].notna()].copy()
    scoreable['score'] = scoreable['score'].astype(float)

    # Overall summary by model × level × probe
    summary = (
        scoreable.groupby(['model', 'level', 'probe_type'])['score']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary['mean'] = summary['mean'] * 100  # Convert to percentage
    summary['std'] = summary['std'] * 100
    summary = summary.round(1)

    print("\n" + "="*80)
    print("EXPERIMENT 1: COMPREHENSIVE RESULTS")
    print("="*80)
    print("\nBy Model × Level × Probe Type:")
    print(summary.to_string(index=False))

    # Pivot table: Models as rows, probes as columns
    print("\n" + "="*80)
    print("PIVOT: Model Performance Across All Probes")
    print("="*80)

    pivot = summary.pivot_table(
        index='model',
        columns=['level', 'probe_type'],
        values='mean',
        aggfunc='first'
    )
    print(pivot.to_string())

    # Key comparisons
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # P1 Count performance
    p1_count = summary[
        (summary['level'] == 'P1_dense_array') &
        (summary['probe_type'] == 'count')
    ][['model', 'mean', 'std']].sort_values('mean', ascending=False)
    print("\n🔢 P1 Dense Array - Count Probe (pattern violation detection):")
    print(p1_count.to_string(index=False))

    # P2 Order performance
    p2_order = summary[
        (summary['level'] == 'P2_corridor_gauntlet') &
        (summary['probe_type'] == 'order')
    ][['model', 'mean', 'std']].sort_values('mean', ascending=False)
    print("\n📏 P2 Corridor Gauntlet - Order Probe (spatial ordering):")
    print(p2_order.to_string(index=False))

    # P3 Allocentric performance
    p3_allo = summary[
        (summary['level'] == 'P3_rotation_challenge') &
        (summary['probe_type'] == 'allocentric_location')
    ][['model', 'mean', 'std']].sort_values('mean', ascending=False)
    print("\n🧭 P3 Rotation Challenge - Allocentric Location (compass directions):")
    print(p3_allo.to_string(index=False))

    # Seed variance analysis
    print("\n" + "="*80)
    print("SEED VARIANCE ANALYSIS")
    print("="*80)

    # Extract seed from file names via original data
    seed_data = []
    for csv_file in sorted(results_path.glob("*.csv")):
        df = pd.read_csv(csv_file)
        # Extract seed from filename (e.g., gpt4o_seed42.csv -> 42)
        seed = csv_file.stem.split('_seed')[-1]
        df['seed'] = seed
        seed_data.append(df)

    seed_combined = pd.concat(seed_data, ignore_index=True)
    seed_scoreable = seed_combined[seed_combined['score'].notna()].copy()
    seed_scoreable['score'] = seed_scoreable['score'].astype(float) * 100

    # Variance by probe
    variance_summary = (
        seed_scoreable.groupby(['model', 'probe_type', 'seed'])['score']
        .mean()
        .reset_index()
        .pivot_table(
            index=['model', 'probe_type'],
            columns='seed',
            values='score',
            aggfunc='first'
        )
    )
    variance_summary['range'] = variance_summary.max(axis=1) - variance_summary.min(axis=1)
    variance_summary = variance_summary.round(1)

    print("\nVariance Across Seeds (columns = seeds, range = max-min):")
    print(variance_summary.sort_values('range', ascending=False).head(15).to_string())

    return combined, summary

if __name__ == "__main__":
    analyze_results()
