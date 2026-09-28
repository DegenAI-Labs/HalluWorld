"""Generate hallucination table matching Emmy's format from innav results.

InNav results have TWO accuracy columns:
- innav_accuracy: Accuracy when agent was navigating (cognitive load)
- controlled_static_accuracy: Accuracy for pure observation (no navigation context)

We'll create TWO tables:
1. InNav hallucination rates (1 - innav_accuracy)
2. Static hallucination rates (1 - controlled_static_accuracy)
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List


def load_results(csv_files: List[str]) -> pd.DataFrame:
    """Load and combine multiple result CSVs."""
    dfs = []
    for csv_file in csv_files:
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            dfs.append(df)
        else:
            print(f"Warning: {csv_file} not found", file=sys.stderr)

    if not dfs:
        raise ValueError("No valid CSV files found")

    return pd.concat(dfs, ignore_index=True)


def compute_hallucination_rates(df: pd.DataFrame) -> Dict:
    """Compute hallucination rates per model per level.

    Returns dict with structure:
    {
        'innav': {(model, level): hallucination_rate, ...},
        'static': {(model, level): hallucination_rate, ...}
    }
    """
    # Group by model and level, take first occurrence of accuracy per episode
    # (all probe rows for an episode have the same accuracy value)
    episode_accuracies = df.groupby(['model', 'level', 'episode']).first().reset_index()

    # Compute mean accuracy per model-level
    summary = episode_accuracies.groupby(['model', 'level']).agg({
        'innav_accuracy': 'mean',
        'controlled_static_accuracy': 'mean',
    }).reset_index()

    # Convert to hallucination rates (1 - accuracy)
    summary['innav_hallucination'] = 1.0 - summary['innav_accuracy']
    summary['static_hallucination'] = 1.0 - summary['controlled_static_accuracy']

    # Convert to dict
    ego_rates = {}
    static_rates = {}
    for _, row in summary.iterrows():
        key = (row['model'], row['level'])
        ego_rates[key] = row['innav_hallucination']
        static_rates[key] = row['static_hallucination']

    return {'innav': ego_rates, 'static': static_rates}


def format_table(rates: Dict, models: List[str], levels: List[tuple]) -> str:
    """Format hallucination rates as Emmy's table.

    Args:
        rates: Dict from (model, level) to hallucination rate
        models: List of model names (column headers)
        levels: List of (level_code, level_name) tuples

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    header = f"{'Level':<35}"
    for model in models:
        header += f"{model:>15}"
    lines.append(header)
    lines.append("")

    # Group levels by category
    categories = {
        'P': "── Perception (P) " + "─" * 80,
        'M': "── Memory / Testimony (M) " + "─" * 74,
        'C': "── Causal Dynamics (C) " + "─" * 77,
        'U': "── Uncertainty (U) " + "─" * 80,
        'X': "── Multi-Zone Compound (X) " + "─" * 73,
    }

    current_category = None
    for level_code, level_name in levels:
        # Extract category (first letter)
        category = level_code[0]

        # Print category header if changed
        if category != current_category:
            if current_category is not None:
                lines.append("")  # Blank line between categories
            lines.append(categories.get(category, ""))
            current_category = category

        # Format level name
        level_str = f"{level_code:<4}{level_name}"
        line = f"{level_str:<35}"

        # Add model results
        for model in models:
            key = (model, level_code)
            if key in rates:
                rate = rates[key]
                line += f"{rate:>14.0%}"
            else:
                line += f"{'—':>15}"

        lines.append(line)

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate hallucination table from innav results")
    parser.add_argument("csv_files", nargs="+", help="CSV result files to process")
    parser.add_argument("--output", default="hallucination_table_innav.txt", help="Output file")
    args = parser.parse_args()

    # Load results
    print(f"Loading {len(args.csv_files)} CSV files...")
    df = load_results(args.csv_files)
    print(f"Loaded {len(df)} rows, {df.groupby(['model', 'level', 'episode']).ngroups} episodes")

    # Compute rates
    rates_dict = compute_hallucination_rates(df)

    # Get unique models and levels
    models = sorted(df['model'].unique())
    levels_raw = sorted(df['level'].unique())

    # Map level codes to names (simplified - extract from level strings)
    level_map = {
        'P2_corridor_gauntlet': ('P2', 'Corridor Gauntlet'),
        'M1_river_field': ('M1', 'River'),
        'C1a_persistent_chain': ('C1a', 'Persistent Chain'),
        # Add more as needed
    }

    # Build levels list and create reverse mapping (code -> full_name)
    levels = []
    code_to_full = {}  # Maps abbreviated code to full level name
    for level_raw in levels_raw:
        if level_raw in level_map:
            code, name = level_map[level_raw]
            levels.append((code, name))
            code_to_full[code] = level_raw
        else:
            # Fallback: use raw name
            code = level_raw.split('_')[0].upper()
            name = ' '.join(level_raw.split('_')[1:]).title()
            levels.append((code, name))
            code_to_full[code] = level_raw

    # Remap rates dict to use abbreviated codes instead of full names
    for rate_type in ['innav', 'static']:
        remapped = {}
        for (model, level_full), rate in rates_dict[rate_type].items():
            # Find the code for this full level name
            for code, full_name in code_to_full.items():
                if full_name == level_full:
                    remapped[(model, code)] = rate
                    break
        rates_dict[rate_type] = remapped

    # Generate tables
    with open(args.output, 'w') as f:
        f.write("HalluWorld Benchmark — InNav vs Static Hallucination Rates\n")
        f.write("Generated from innav evaluation results\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write("'—' = not yet evaluated.\n\n")

        f.write("=" * 100 + "\n")
        f.write("TABLE 1: INNAV HALLUCINATION RATES (agent navigating while being probed)\n")
        f.write("=" * 100 + "\n\n")
        f.write(format_table(rates_dict['innav'], models, levels))

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("TABLE 2: CONTROLLED STATIC HALLUCINATION RATES (retrospective probing, no navigation context)\n")
        f.write("=" * 100 + "\n\n")
        f.write(format_table(rates_dict['static'], models, levels))

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("DIFFERENCE (InNav - Controlled Static)\n")
        f.write("Positive values = cognitive load increases hallucinations (innav worse than controlled static)\n")
        f.write("=" * 100 + "\n\n")

        # Compute differences
        diff_rates = {}
        for key in rates_dict['innav']:
            if key in rates_dict['static']:
                diff = rates_dict['innav'][key] - rates_dict['static'][key]
                diff_rates[key] = diff

        # Format difference table (show as percentage points)
        header = f"{'Level':<35}"
        for model in models:
            header += f"{model:>15}"
        f.write(header + "\n\n")

        current_category = None
        categories = {
            'P': "── Perception (P) " + "─" * 80,
            'M': "── Memory / Testimony (M) " + "─" * 74,
            'C': "── Causal Dynamics (C) " + "─" * 77,
        }

        for level_code, level_name in levels:
            category = level_code[0]
            if category != current_category:
                if current_category is not None:
                    f.write("\n")
                f.write(categories.get(category, "") + "\n")
                current_category = category

            level_str = f"{level_code:<4}{level_name}"
            line = f"{level_str:<35}"

            for model in models:
                key = (model, level_code)
                if key in diff_rates:
                    diff = diff_rates[key]
                    # Show as percentage points with sign
                    line += f"{diff:>+14.0%}"
                else:
                    line += f"{'—':>15}"

            f.write(line + "\n")

    print(f"\nSaved hallucination table to: {args.output}")
    print("\nSummary:")
    print(format_table(rates_dict['innav'], models, levels))


if __name__ == "__main__":
    main()
