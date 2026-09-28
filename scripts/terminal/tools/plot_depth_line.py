#!/usr/bin/env python3
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("runs/probe_eval_sonnet")
OUT_DIR = Path("analysis_plots")

PROBE_TYPE_COLORS = {
    "perceptual":          "#4C72B0",
    "memory":              "#DD8452",
    "causal":              "#55A868",
    "uncertainty":         "#C44E52",
    "cross-tier compound": "#8172B2",
}

# Load results
rows = []
for path in sorted(RESULTS_DIR.glob("*.result.json")):
    d = json.loads(path.read_text())
    if d.get("status") != "ok":
        continue
    m = d.get("metadata") or {}
    depth = m.get("step_command_index") or m.get("trigger_step_index")
    if depth is None:
        continue
    rows.append({
        "depth": int(depth),
        "correct": bool(d.get("correct")),
        "probe_type": m.get("probe_type", "unknown"),
    })

# Quantile bins (equal-count)
depths = np.array([r["depth"] for r in rows])
bin_edges = np.quantile(depths, [0, 0.25, 0.5, 0.75, 1.0])
bin_edges = np.unique(np.round(bin_edges).astype(int))
bin_edges[0] -= 1
bin_edges[-1] += 1

bin_labels = [f"{bin_edges[i]+1}–{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
n_bins = len(bin_labels)

def assign_bin(d):
    for i in range(n_bins):
        if bin_edges[i] < d <= bin_edges[i+1]:
            return i
    return n_bins - 1

probe_types = ["perceptual", "memory", "causal", "uncertainty", "cross-tier compound"]
bin_pt = {b: {pt: [] for pt in probe_types} for b in range(n_bins)}
bin_all = {b: [] for b in range(n_bins)}

for r in rows:
    b = assign_bin(r["depth"])
    val = 1 - int(r["correct"])
    bin_all[b].append(val)
    if r["probe_type"] in probe_types:
        bin_pt[b][r["probe_type"]].append(val)

x = np.arange(n_bins)

fig, ax = plt.subplots(figsize=(9, 5))

# Per-probe-type lines
for pt in probe_types:
    means = [np.mean(bin_pt[b][pt]) if bin_pt[b][pt] else np.nan for b in range(n_bins)]
    ses   = [np.std(bin_pt[b][pt]) / np.sqrt(len(bin_pt[b][pt])) if bin_pt[b][pt] else np.nan for b in range(n_bins)]
    ax.errorbar(x, means, yerr=ses, marker="o", lw=2, capsize=4,
                label=pt, color=PROBE_TYPE_COLORS[pt])

# Overall line
overall_means = [np.mean(bin_all[b]) if bin_all[b] else np.nan for b in range(n_bins)]
overall_ses   = [np.std(bin_all[b]) / np.sqrt(len(bin_all[b])) if bin_all[b] else np.nan for b in range(n_bins)]
ax.errorbar(x, overall_means, yerr=overall_ses, marker="D", lw=2.5, capsize=4,
            linestyle="--", color="black", label="overall", zorder=5)

ns = [len(bin_all[b]) for b in range(n_bins)]
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={ns[i]})" for i, l in enumerate(bin_labels)], fontsize=10)
ax.set_xlabel("Trajectory Depth Bin (command index at injection)", fontsize=11)
ax.set_ylabel("Hallucination Rate", fontsize=11)
ax.set_title("Hallucination Rate vs. Trajectory Depth\nClaude Sonnet 4.6 · 529 LLM-generated probes · error bars = ±1 SE", fontsize=12)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.axhline(np.mean([1-r["correct"] for r in rows]), color="gray", ls=":", lw=1.2, alpha=0.6, label="_nolegend_")

plt.tight_layout()
out = OUT_DIR / "depth_vs_hallucination_line.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
