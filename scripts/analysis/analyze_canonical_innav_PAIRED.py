#!/usr/bin/env python3
"""
PAIRED comparison of innav vs controlled_static.

Each innav probe is compared with its EXACT corresponding controlled_static probe:
- Same world
- Same seed/episode
- Same timestep
- Same probe instance

This avoids conflating trace-level or probe-point-level difficulty differences.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path


def categorize_world(level):
    """Categorize world by type."""
    if level.startswith('P'):
        return 'Perception'
    elif level.startswith('C'):
        return 'Causal'
    elif level.startswith('M'):
        return 'Memory'
    elif level.startswith('U'):
        return 'Uncertainty'
    elif level.startswith('X'):
        return 'X-Levels'
    return 'Other'


def load_canonical_probes():
    """Load all canonical serialization probe data."""

    CANONICAL = {
        'C1a_noboard': 'memory', 'C1a_persistent_chain': 'memory',
        'C1b_continuous_chain': 'memory', 'C1b_noboard': 'memory',
        'C2_fire_crossing': 'memory', 'C3_flood_room': 'memory',
        'C4_forking_paths': 'memory', 'C5a_adversarial_board': 'memory',
        'C6_flood_fire_escape': 'memory', 'M1_river_field': 'memory',
        'M2_witness_stand': 'symbolic', 'M3_incident_report': 'symbolic',
        'M4_narrator': 'memory', 'P1_dense_array': 'grid',
        'P2_corridor_gauntlet': 'grid', 'P3_rotation_challenge': 'grid',
        'P4_harder_array': 'grid', 'P4b_delta_perception': 'symbolic',
        'P5_object_permanence': 'grid', 'U1_fog_of_war': 'memory',
        'U2_oracle_high': 'memory', 'U2_oracle_low': 'memory',
        'U2_oracle_mid': 'memory', 'U4_amnesiac': 'memory',
        'X1_facility_3zone': 'symbolic', 'X2_facility_5zone': 'symbolic',
        'X3_facility_7zone': 'symbolic', 'X4_compound_witness': 'symbolic',
        'X5_facility_tour': 'symbolic', 'X6_return_visit': 'symbolic',
        'X7_dragon_keep': 'symbolic',
    }

    probe_files = glob.glob('probe_*.csv')
    exclude = ['batch', 'stuck', 'clean', 'model_', 'retry']
    probe_files = [f for f in probe_files if not any(x in f for x in exclude)]

    all_data = []

    for f in probe_files:
        try:
            df = pd.read_csv(f)

            # Parse filename
            parts = Path(f).stem.replace('probe_', '').split('_')
            if len(parts) >= 3:
                model = parts[-1]
                serializer = parts[-2]
                world = '_'.join(parts[:-2])
                world = world.replace('-', '_')

                # Check if canonical
                if world in CANONICAL and CANONICAL[world] == serializer:
                    df['world'] = world
                    df['serializer'] = serializer
                    df['model_name'] = model
                    df['world_category'] = categorize_world(world)
                    all_data.append(df)
        except:
            continue

    combined = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(combined)} canonical probes")
    print(f"   Models: {combined['model'].nunique()}")
    print(f"   Worlds: {combined['level'].nunique()}")
    print()

    return combined


def paired_comparison(df):
    """
    PAIRED comparison with hierarchical aggregation:
    Probe point → Trace → World × Serialization → Model

    This gives equal weight to each trace regardless of probe count,
    and equal weight to each world regardless of trace count.
    """

    # For each probe, compute paired difference
    df['ego_hallucinated'] = (df['innav_score'] == 0).astype(int)
    df['static_hallucinated'] = (df['controlled_static_score'] == 0).astype(int)
    df['paired_difference'] = df['ego_hallucinated'] - df['static_hallucinated']

    # Step 1: Aggregate to TRACE level
    by_trace = df.groupby(['model', 'world', 'serializer', 'seed']).agg({
        'paired_difference': 'mean',
        'ego_hallucinated': 'mean',
        'static_hallucinated': 'mean'
    }).reset_index()

    # Step 2: Aggregate to WORLD × SERIALIZATION level
    by_world = by_trace.groupby(['model', 'world', 'serializer']).agg({
        'paired_difference': 'mean',
        'ego_hallucinated': 'mean',
        'static_hallucinated': 'mean',
        'seed': 'count'  # Number of traces
    }).reset_index()
    by_world.rename(columns={'seed': 'n_traces'}, inplace=True)

    # Step 3: Aggregate to MODEL level (across worlds)
    by_model = by_world.groupby('model').agg({
        'paired_difference': ['mean', 'std'],
        'ego_hallucinated': 'mean',
        'static_hallucinated': 'mean',
        'world': 'count'
    }).reset_index()

    by_model.columns = ['model', 'paired_diff', 'paired_diff_std', 'ego_halluc', 'static_halluc', 'n_worlds']
    by_model['cognitive_load_effect'] = by_model['paired_diff'] * 100
    by_model['std_error'] = by_model['paired_diff_std'] * 100 / np.sqrt(by_model['n_worlds'])
    by_model['ego_halluc'] *= 100
    by_model['static_halluc'] *= 100

    # Compute 95% confidence intervals
    by_model['ci_lower'] = by_model['cognitive_load_effect'] - 1.96 * by_model['std_error']
    by_model['ci_upper'] = by_model['cognitive_load_effect'] + 1.96 * by_model['std_error']

    by_model = by_model.sort_values('cognitive_load_effect')

    print("=" * 80)
    print("PAIRED COMPARISON: InNav vs Controlled_Static")
    print("Hierarchical aggregation: Probe → Trace → World → Model")
    print("=" * 80)
    print()
    print(by_model[['model', 'ego_halluc', 'static_halluc', 'cognitive_load_effect', 'std_error', 'n_worlds']].to_string(index=False))
    print()

    return by_model, df, by_world


def paired_comparison_by_category(by_world):
    """
    Paired comparison by world category.
    Uses world-level aggregates (stops at world level, doesn't aggregate across worlds).
    """

    # Add category to world-level data
    by_world['world_category'] = by_world['world'].apply(categorize_world)

    # Aggregate within category (across worlds in that category)
    by_category = by_world.groupby(['world_category', 'model']).agg({
        'paired_difference': ['mean', 'std'],
        'ego_hallucinated': 'mean',
        'static_hallucinated': 'mean',
        'world': 'count'
    }).reset_index()

    by_category.columns = ['category', 'model', 'paired_diff', 'paired_diff_std', 'ego_halluc', 'static_halluc', 'n_worlds']
    by_category['cognitive_load_effect'] = by_category['paired_diff'] * 100
    by_category['std_error'] = by_category['paired_diff_std'] * 100 / np.sqrt(by_category['n_worlds'])
    by_category['ego_halluc'] *= 100
    by_category['static_halluc'] *= 100

    print("=" * 80)
    print("PAIRED COMPARISON BY WORLD CATEGORY")
    print("Aggregation: Probe → Trace → World → Category")
    print("=" * 80)

    for category in sorted(by_category['category'].unique()):
        cat_data = by_category[by_category['category'] == category].sort_values('cognitive_load_effect')
        print(f"\n{category} ({cat_data['n_worlds'].iloc[0]} worlds):")
        print(cat_data[['model', 'ego_halluc', 'static_halluc', 'cognitive_load_effect', 'std_error']].to_string(index=False))

    print()

    return by_category


def plot_cognitive_load_effect(by_model, output_file='cognitive_load_effect_canonical.png'):
    """Plot paired cognitive load effect with error bars."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine significance threshold (can use CI or simple threshold)
    colors = ['#2ecc71' if x < -1 else '#e74c3c' if x > 1 else '#95a5a6'
              for x in by_model['cognitive_load_effect']]

    y_pos = np.arange(len(by_model))
    bars = ax.barh(y_pos, by_model['cognitive_load_effect'], color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5,
                   xerr=by_model['std_error'], capsize=5, error_kw={'linewidth': 2})

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(by_model['model'])
    ax.set_xlabel('Cognitive Load Effect (%)\nPositive = Navigation Hurts | Negative = Navigation Helps',
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    ax.set_title('Cognitive Load Effect: InNav vs Controlled Static\n(Paired Comparison with 95% Confidence Intervals)',
                 fontsize=13, fontweight='bold', pad=20)

    # Value labels with significance
    for i, row in by_model.iterrows():
        val = row['cognitive_load_effect']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']

        # Check if CI excludes zero (statistically significant)
        sig = '*' if (ci_lower > 0 or ci_upper < 0) else ''

        label_x = val + (1.5 if val > 0 else -1.5)
        ax.text(label_x, i, f'{val:+.1f}%{sig}', va='center',
                ha='left' if val > 0 else 'right', fontweight='bold', fontsize=9)

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    return fig


def main():
    print("=" * 80)
    print("CANONICAL SERIALIZATION: PAIRED INNAV ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    df = load_canonical_probes()

    # Paired comparison
    by_model, df_with_diff, by_world = paired_comparison(df)

    # By category
    by_category = paired_comparison_by_category(by_world)

    # Plot
    plot_cognitive_load_effect(by_model)

    # Save
    by_model.to_csv('innav_vs_static_canonical.csv', index=False)
    by_category.to_csv('innav_vs_static_by_category_canonical.csv', index=False)

    print()
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE (95% CI excludes zero)")
    print("=" * 80)

    # Statistically significant effects (CI excludes zero)
    sig_helps = by_model[(by_model['ci_upper'] < 0)]
    sig_hurts = by_model[(by_model['ci_lower'] > 0)]

    print(f"\n✅ Navigation significantly HELPS ({len(sig_helps)} models):")
    for _, row in sig_helps.iterrows():
        print(f"   {row['model']:35s} {row['cognitive_load_effect']:+.1f}% ± {row['std_error']:.1f}% (95% CI: [{row['ci_lower']:+.1f}%, {row['ci_upper']:+.1f}%])")

    print(f"\n❌ Navigation significantly HURTS ({len(sig_hurts)} models):")
    for _, row in sig_hurts.iterrows():
        print(f"   {row['model']:35s} {row['cognitive_load_effect']:+.1f}% ± {row['std_error']:.1f}% (95% CI: [{row['ci_lower']:+.1f}%, {row['ci_upper']:+.1f}%])")

    neutral = by_model[(by_model['ci_lower'] <= 0) & (by_model['ci_upper'] >= 0)]
    print(f"\n⏸️  Not significant ({len(neutral)} models):")
    for _, row in neutral.iterrows():
        print(f"   {row['model']:35s} {row['cognitive_load_effect']:+.1f}% ± {row['std_error']:.1f}% (95% CI: [{row['ci_lower']:+.1f}%, {row['ci_upper']:+.1f}%])")

    print()
    print("=" * 80)
    print(f"Total probes analyzed: {len(df_with_diff):,}")
    print(f"Worlds: {by_model['n_worlds'].iloc[0]}")
    print(f"Hierarchical aggregation: Probe → Trace → World → Model")
    print("=" * 80)
    print()
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
