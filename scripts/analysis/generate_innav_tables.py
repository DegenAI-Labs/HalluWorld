#!/usr/bin/env python3
"""
Generate publication-quality tables for innav vs controlled_static comparison.
Creates both absolute rates and cognitive load effect tables.
"""

import pandas as pd
import numpy as np

def load_results():
    """Load the analysis results."""
    by_model = pd.read_csv('innav_vs_static_canonical.csv')
    by_category = pd.read_csv('innav_vs_static_by_category_canonical.csv')
    return by_model, by_category


def create_main_table(by_model):
    """
    Create main table: Overall results with absolute rates and cognitive load.

    Format:
    Model | InNav (%) | Controlled Static (%) | Cognitive Load Effect (%) | Sig
    """

    # Determine significance
    by_model['sig'] = by_model.apply(
        lambda row: '**' if (row['ci_lower'] > 0 or row['ci_upper'] < 0) else '',
        axis=1
    )

    # Format for display
    table = by_model[['model', 'ego_halluc', 'static_halluc', 'cognitive_load_effect', 'std_error', 'sig']].copy()
    table = table.sort_values('cognitive_load_effect')

    # Create formatted strings
    table['InNav'] = table['ego_halluc'].apply(lambda x: f"{x:.1f}")
    table['Controlled Static'] = table['static_halluc'].apply(lambda x: f"{x:.1f}")
    table['Cognitive Load'] = table.apply(
        lambda row: f"{row['cognitive_load_effect']:+.1f} ± {row['std_error']:.1f}{row['sig']}",
        axis=1
    )

    output = table[['model', 'InNav', 'Controlled Static', 'Cognitive Load']].copy()
    output.columns = ['Model', 'InNav (%)', 'Controlled Static (%)', 'Cognitive Load Effect (%)']

    return output


def create_category_table(by_category):
    """
    Create category breakdown table.

    Pivot format:
    Model | Perception | Causal | Memory | Uncertainty | X-Levels
    """

    categories = ['Perception', 'Causal', 'Memory', 'Uncertainty', 'X-Levels']

    # Create pivot for cognitive load effect
    pivot_load = by_category.pivot(index='model', columns='category', values='cognitive_load_effect')
    pivot_stderr = by_category.pivot(index='model', columns='category', values='std_error')

    # Compute significance per category
    by_category['ci_lower'] = by_category['cognitive_load_effect'] - 1.96 * by_category['std_error']
    by_category['ci_upper'] = by_category['cognitive_load_effect'] + 1.96 * by_category['std_error']
    by_category['sig'] = by_category.apply(
        lambda row: '**' if (row['ci_lower'] > 0 or row['ci_upper'] < 0) else '',
        axis=1
    )
    pivot_sig = by_category.pivot(index='model', columns='category', values='sig')

    # Format combined strings
    result = pd.DataFrame(index=pivot_load.index)

    for cat in categories:
        if cat in pivot_load.columns:
            result[cat] = pivot_load[cat].combine(
                pivot_stderr[cat],
                lambda load, err: f"{load:+.1f} ± {err:.1f}" if pd.notna(load) else "—"
            )
            # Add significance markers
            if cat in pivot_sig.columns:
                result[cat] = result[cat] + pivot_sig[cat].fillna('')

    # Add overall cognitive load
    overall = pd.read_csv('innav_vs_static_canonical.csv')
    overall = overall.set_index('model')
    result['Overall'] = overall['cognitive_load_effect'].apply(lambda x: f"{x:+.1f}")

    # Reorder columns
    cols = ['Overall'] + [c for c in categories if c in result.columns]
    result = result[cols]

    # Sort by overall effect
    result = result.sort_values('Overall', key=lambda x: x.str.replace('+', '').astype(float))

    result.index.name = 'Model'
    return result


def create_absolute_rates_table(by_category):
    """
    Create table showing absolute hallucination rates by category.

    Format: Model | Category | InNav (%) | Controlled Static (%)
    """

    table = by_category[['model', 'category', 'ego_halluc', 'static_halluc']].copy()
    table['InNav'] = table['ego_halluc'].apply(lambda x: f"{x:.1f}")
    table['Controlled Static'] = table['static_halluc'].apply(lambda x: f"{x:.1f}")

    output = table[['model', 'category', 'InNav', 'Controlled Static']].copy()
    output.columns = ['Model', 'Category', 'InNav (%)', 'Controlled Static (%)']

    return output


def generate_latex_table(df, caption, label):
    """Convert dataframe to LaTeX table."""

    latex = df.to_latex(
        index=True,
        escape=False,
        column_format='l' + 'r' * len(df.columns),
        caption=caption,
        label=label
    )

    # Add booktabs styling
    latex = latex.replace('\\toprule', '\\toprule\n\\rowcolor{gray!20}')
    latex = latex.replace('**', '\\textbf{**}')

    return latex


def main():
    print("=" * 80)
    print("GENERATING PUBLICATION TABLES")
    print("=" * 80)
    print()

    by_model, by_category = load_results()

    # Table 1: Main results (overall)
    print("\nTable 1: Overall Cognitive Load Effect")
    print("=" * 80)
    table1 = create_main_table(by_model)
    print(table1.to_string(index=False))
    table1.to_csv('table1_overall_cognitive_load.csv', index=False)

    # Table 2: Category breakdown
    print("\n\nTable 2: Cognitive Load Effect by World Category")
    print("=" * 80)
    table2 = create_category_table(by_category)
    print(table2.to_string())
    table2.to_csv('table2_category_breakdown.csv')

    # Table 3: Absolute rates by category
    print("\n\nTable 3: Absolute Hallucination Rates by Category")
    print("=" * 80)
    table3 = create_absolute_rates_table(by_category)
    print(table3.head(20).to_string(index=False))
    print("...")
    table3.to_csv('table3_absolute_rates.csv', index=False)

    # Generate LaTeX versions
    latex1 = generate_latex_table(
        table1.set_index('Model'),
        "Overall cognitive load effect: innav vs controlled static. ** indicates 95\% CI excludes zero.",
        "tab:cognitive_load_overall"
    )

    latex2 = generate_latex_table(
        table2,
        "Cognitive load effect by world category. ** indicates 95\% CI excludes zero.",
        "tab:cognitive_load_by_category"
    )

    with open('tables_latex.tex', 'w') as f:
        f.write("% Table 1: Overall Results\n")
        f.write(latex1)
        f.write("\n\n")
        f.write("% Table 2: Category Breakdown\n")
        f.write(latex2)

    print("\n\n" + "=" * 80)
    print("✅ Tables generated:")
    print("  - table1_overall_cognitive_load.csv")
    print("  - table2_category_breakdown.csv")
    print("  - table3_absolute_rates.csv")
    print("  - tables_latex.tex")
    print("=" * 80)

    # Summary statistics
    print("\n\nKEY FINDINGS FOR PAPER:")
    print("=" * 80)

    helps = by_model[by_model['ci_upper'] < 0]
    hurts = by_model[by_model['ci_lower'] > 0]

    print(f"\n✅ Navigation significantly HELPS: {len(helps)}/{len(by_model)} models")
    print(f"   Mean effect: {helps['cognitive_load_effect'].mean():.1f}% reduction")
    print(f"   Range: {helps['cognitive_load_effect'].min():.1f}% to {helps['cognitive_load_effect'].max():.1f}%")

    print(f"\n❌ Navigation significantly HURTS: {len(hurts)}/{len(by_model)} models")

    print(f"\n⏸️  Not significant: {len(by_model) - len(helps) - len(hurts)}/{len(by_model)} models")

    # Category insights
    print("\n\nCATEGORY-SPECIFIC INSIGHTS:")
    print("=" * 80)

    for cat in sorted(by_category['category'].unique()):
        cat_data = by_category[by_category['category'] == cat]
        cat_mean = cat_data['cognitive_load_effect'].mean()
        cat_sig = cat_data[(cat_data['ci_lower'] > 0) | (cat_data['ci_upper'] < 0)]

        print(f"\n{cat}:")
        print(f"  Mean effect: {cat_mean:+.1f}%")
        print(f"  Significant effects: {len(cat_sig)}/{len(cat_data)} models")

        # Best and worst
        best = cat_data.loc[cat_data['cognitive_load_effect'].idxmin()]
        worst = cat_data.loc[cat_data['cognitive_load_effect'].idxmax()]
        print(f"  Best (most help): {best['model']} ({best['cognitive_load_effect']:+.1f}%)")
        print(f"  Worst: {worst['model']} ({worst['cognitive_load_effect']:+.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
