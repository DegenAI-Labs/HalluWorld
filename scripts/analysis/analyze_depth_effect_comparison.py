#!/usr/bin/env python3
"""
Compare trajectory depth effects in INNAV vs CONTROLLED_STATIC.

Key question: Does hallucination increase with depth in BOTH modes?
- If YES → States get harder, not cognitive load accumulation
- If ego slope > static slope → True cognitive load accumulation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_canonical_probes():
    """Load all canonical probe data."""
    import glob
    from pathlib import Path

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
    all_data = []

    for f in probe_files:
        try:
            df = pd.read_csv(f)
            parts = Path(f).stem.replace('probe_', '').split('_')
            if len(parts) >= 3:
                model = parts[-1]
                serializer = parts[-2]
                world = '_'.join(parts[:-2]).replace('-', '_')

                if world in CANONICAL and CANONICAL[world] == serializer:
                    df['world'] = world
                    df['serializer'] = serializer
                    df['model_name'] = model
                    all_data.append(df)
        except:
            continue

    return pd.concat(all_data, ignore_index=True)


def compute_depth_slopes(df):
    """
    Compute trajectory depth slope for BOTH innav and controlled_static.

    Returns: DataFrame with model, ego_slope, static_slope, diff_slope
    """

    # Calculate trajectory position
    df['trajectory_position'] = df['timestep'] / df['steps_taken']

    # Quintiles
    df['quintile'] = pd.cut(df['trajectory_position'],
                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            labels=[0.1, 0.3, 0.5, 0.7, 0.9],
                            include_lowest=True)
    df['quintile'] = df['quintile'].astype(float)

    # Hallucination indicators
    df['ego_hallucinated'] = (df['innav_score'] == 0).astype(int)
    df['static_hallucinated'] = (df['controlled_static_score'] == 0).astype(int)

    # Compute slopes per model
    slopes = []

    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()

        # Aggregate by quintile
        by_quintile = model_data.groupby('quintile').agg({
            'ego_hallucinated': 'mean',
            'static_hallucinated': 'mean'
        }).reset_index()

        # Linear regression for innav
        if len(by_quintile) >= 3:
            # Fit line: halluc = slope * quintile + intercept
            ego_coeffs = np.polyfit(by_quintile['quintile'], by_quintile['ego_hallucinated'], 1)
            ego_slope = ego_coeffs[0]

            static_coeffs = np.polyfit(by_quintile['quintile'], by_quintile['static_hallucinated'], 1)
            static_slope = static_coeffs[0]

            # Compute R^2
            ego_yhat = np.polyval(ego_coeffs, by_quintile['quintile'])
            ego_ss_res = np.sum((by_quintile['ego_hallucinated'] - ego_yhat)**2)
            ego_ss_tot = np.sum((by_quintile['ego_hallucinated'] - by_quintile['ego_hallucinated'].mean())**2)
            ego_r2 = 1 - (ego_ss_res / ego_ss_tot) if ego_ss_tot > 0 else 0

            static_yhat = np.polyval(static_coeffs, by_quintile['quintile'])
            static_ss_res = np.sum((by_quintile['static_hallucinated'] - static_yhat)**2)
            static_ss_tot = np.sum((by_quintile['static_hallucinated'] - by_quintile['static_hallucinated'].mean())**2)
            static_r2 = 1 - (static_ss_res / static_ss_tot) if static_ss_tot > 0 else 0

            slopes.append({
                'model': model,
                'ego_slope': ego_slope * 100,  # % per quintile
                'static_slope': static_slope * 100,
                'diff_slope': (ego_slope - static_slope) * 100,
                'ego_r2': ego_r2,
                'static_r2': static_r2
            })

    return pd.DataFrame(slopes)


def plot_comparison(slopes):
    """Plot ego vs static depth slopes."""

    slopes = slopes.sort_values('diff_slope')

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(slopes))
    width = 0.35

    bars1 = ax.bar(x - width/2, slopes['ego_slope'], width,
                   label='InNav', alpha=0.8, color='#e74c3c')
    bars2 = ax.bar(x + width/2, slopes['static_slope'], width,
                   label='Controlled Static', alpha=0.8, color='#3498db')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trajectory Depth Effect (% hallucination increase per quintile)',
                  fontsize=11, fontweight='bold')
    ax.set_title('Does Hallucination Increase with Trajectory Depth?\nInNav vs Controlled Static Comparison',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(slopes['model'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_depth_ego_vs_static.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: trajectory_depth_ego_vs_static.png")

    return fig


def main():
    print("=" * 80)
    print("TRAJECTORY DEPTH EFFECT: INNAV VS CONTROLLED_STATIC")
    print("=" * 80)
    print()

    df = load_canonical_probes()
    print(f"Loaded {len(df):,} canonical probes")
    print()

    slopes = compute_depth_slopes(df)

    print("=" * 80)
    print("TRAJECTORY DEPTH SLOPES (% hallucination increase per quintile)")
    print("=" * 80)
    print()
    print(slopes[['model', 'ego_slope', 'static_slope', 'diff_slope']].to_string(index=False))
    print()

    # Save results
    slopes.to_csv('trajectory_depth_slopes_comparison.csv', index=False)

    # Plot
    plot_comparison(slopes)

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    both_increase = slopes[(slopes['ego_slope'] > 0) & (slopes['static_slope'] > 0)]
    ego_steeper = slopes[slopes['diff_slope'] > 1]
    static_steeper = slopes[slopes['diff_slope'] < -1]

    print(f"\n📈 Both modes show depth effect (both > 0): {len(both_increase)}/{len(slopes)} models")
    print("   → Suggests states get intrinsically harder with depth\n")

    print(f"📊 InNav slope > Static slope (+1% threshold): {len(ego_steeper)}/{len(slopes)} models")
    print("   → True cognitive load accumulation (dual-task gets worse over time)\n")

    print(f"📊 Static slope > InNav slope (-1% threshold): {len(static_steeper)}/{len(slopes)} models")
    print("   → Navigation actually mitigates depth effect\n")

    print("\nModels with TRUE cognitive load accumulation (ego much steeper than static):")
    for _, row in ego_steeper.iterrows():
        print(f"  {row['model']:30s}  Ego: +{row['ego_slope']:.2f}%  Static: +{row['static_slope']:.2f}%  Diff: +{row['diff_slope']:.2f}%")

    print("\nModels where navigation HELPS with depth (static steeper than ego):")
    for _, row in static_steeper.iterrows():
        print(f"  {row['model']:30s}  Ego: +{row['ego_slope']:.2f}%  Static: +{row['static_slope']:.2f}%  Diff: {row['diff_slope']:.2f}%")

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
