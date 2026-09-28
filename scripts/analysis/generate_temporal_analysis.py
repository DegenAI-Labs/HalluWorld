#!/usr/bin/env python3
"""Temporal analysis of probe results during navigation.

Analyzes how hallucination rates change as navigation progresses.

Usage:
    python generate_temporal_analysis.py probe_perception_batch1.csv
    python generate_temporal_analysis.py probe_*.csv --output temporal_analysis.txt
"""

import argparse
import pandas as pd
from pathlib import Path


def temporal_analysis(csv_path: str, output_path: str | None = None):
    """Generate temporal analysis report."""
    df = pd.read_csv(csv_path)

    # Calculate probe position in trajectory (0.0 = start, 1.0 = end)
    df['probe_position'] = df['timestep'] / df['steps_taken']

    # Create 5 temporal buckets
    df['temporal_bucket'] = pd.cut(
        df['probe_position'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['1/5 Early', '2/5', '3/5 Mid', '4/5', '5/5 Late']
    )

    # Overall summary
    report = []
    report.append("=" * 80)
    report.append(f"TEMPORAL HALLUCINATION ANALYSIS")
    report.append(f"Input: {csv_path}")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("=== OVERALL SUMMARY ===")
    report.append(f"Total probe tests: {len(df):,}")
    report.append(f"Models: {', '.join(df['model'].unique())}")
    report.append(f"Levels: {', '.join(df['level'].unique())}")
    report.append(f"Episodes per level: {df.groupby('level')['episode'].nunique().mean():.1f}")
    report.append("")

    # Temporal distribution
    report.append("=== TEMPORAL DISTRIBUTION ===")
    temporal_stats = df.groupby('temporal_bucket').agg({
        'innav_score': ['mean', 'std', 'count'],
        'controlled_static_score': ['mean', 'std']
    }).round(3)

    temporal_stats.columns = ['_'.join(col).strip() for col in temporal_stats.columns.values]
    temporal_stats['gap'] = (
        temporal_stats['controlled_static_score_mean'] -
        temporal_stats['innav_score_mean']
    ).round(3)

    report.append(temporal_stats.to_string())
    report.append("")

    # Per-model temporal analysis
    report.append("=== PER-MODEL TEMPORAL PATTERNS ===")
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        model_stats = model_df.groupby('temporal_bucket').agg({
            'innav_score': 'mean',
            'controlled_static_score': 'mean'
        }).round(3)
        model_stats['gap'] = (
            model_stats['controlled_static_score'] -
            model_stats['innav_score']
        ).round(3)

        report.append(f"\n{model}:")
        report.append(model_stats.to_string())

    report.append("")

    # Per-level temporal analysis
    report.append("=== PER-LEVEL TEMPORAL PATTERNS ===")
    for level in sorted(df['level'].unique()):
        level_df = df[df['level'] == level]
        level_stats = level_df.groupby('temporal_bucket').agg({
            'innav_score': 'mean',
            'controlled_static_score': 'mean'
        }).round(3)
        level_stats['gap'] = (
            level_stats['controlled_static_score'] -
            level_stats['innav_score']
        ).round(3)

        report.append(f"\n{level}:")
        report.append(level_stats.to_string())

    report.append("")

    # Probe type analysis
    report.append("=== PER-PROBE-TYPE TEMPORAL PATTERNS ===")
    for probe_type in sorted(df['probe_type'].unique()):
        probe_df = df[df['probe_type'] == probe_type]
        if len(probe_df) < 10:  # Skip rare probe types
            continue

        probe_stats = probe_df.groupby('temporal_bucket').agg({
            'innav_score': 'mean',
            'controlled_static_score': 'mean'
        }).round(3)
        probe_stats['gap'] = (
            probe_stats['controlled_static_score'] -
            probe_stats['innav_score']
        ).round(3)

        report.append(f"\n{probe_type} (n={len(probe_df)}):")
        report.append(probe_stats.to_string())

    report.append("")
    report.append("=" * 80)
    report.append("KEY INSIGHTS:")
    report.append("")
    report.append("1. Cognitive Load Over Time: Does the gap increase as navigation progresses?")
    report.append("2. Model Differences: Do reasoning models show different temporal patterns?")
    report.append("3. Level Complexity: Do complex levels show stronger temporal effects?")
    report.append("4. Probe Types: Which probe types are most affected by navigation depth?")
    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Temporal analysis written to: {output_path}")
    else:
        print(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate temporal analysis of innav probe results"
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Probe result CSV file(s)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: print to stdout)"
    )

    args = parser.parse_args()

    # Combine multiple CSV files if provided
    if len(args.csv_files) > 1:
        dfs = [pd.read_csv(f) for f in args.csv_files]
        combined = pd.concat(dfs, ignore_index=True)
        temp_csv = "temp_combined_probes.csv"
        combined.to_csv(temp_csv, index=False)
        temporal_analysis(temp_csv, args.output)
        Path(temp_csv).unlink()  # Clean up
    else:
        temporal_analysis(args.csv_files[0], args.output)


if __name__ == "__main__":
    main()
