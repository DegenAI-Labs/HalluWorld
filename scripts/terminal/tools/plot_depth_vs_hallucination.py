#!/usr/bin/env python3
"""
Plot hallucination rate vs. trajectory depth for the 530 curated probes.

Usage:
  python3 plot_depth_vs_hallucination.py
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

RESULTS_DIR = Path("runs/probe_eval_sonnet")
OUT_DIR = Path("analysis_plots")
OUT_DIR.mkdir(exist_ok=True)

PROBE_TYPE_COLORS = {
    "perceptual":          "#4C72B0",
    "memory":              "#DD8452",
    "causal":              "#55A868",
    "uncertainty":         "#C44E52",
    "cross-tier compound": "#8172B2",
}

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

rows = []
for path in sorted(RESULTS_DIR.glob("*.result.json")):
    d = json.loads(path.read_text())
    if d.get("status") not in ("ok",):
        continue
    m = d.get("metadata") or {}
    depth = m.get("step_command_index") or m.get("trigger_step_index")
    if depth is None:
        continue
    rows.append({
        "depth": int(depth),
        "correct": bool(d.get("correct")),
        "probe_type": m.get("probe_type", "unknown"),
        "task": m.get("task_name", "unknown"),
        "prompt_chars": d.get("prompt_chars", 0),
    })

print(f"Loaded {len(rows)} result rows")

# ---------------------------------------------------------------------------
# Per-task aggregation (all 5 probes share the same depth)
# ---------------------------------------------------------------------------

task_data = defaultdict(lambda: {"depth": None, "correct": [], "probe_types": []})
for r in rows:
    t = task_data[r["task"]]
    t["depth"] = r["depth"]
    t["correct"].append(r["correct"])
    t["probe_types"].append(r["probe_type"])

tasks = []
for name, td in task_data.items():
    n = len(td["correct"])
    n_correct = sum(td["correct"])
    tasks.append({
        "task": name,
        "depth": td["depth"],
        "n_probes": n,
        "n_correct": n_correct,
        "hallucination_rate": 1 - n_correct / n,
        "accuracy": n_correct / n,
    })

tasks.sort(key=lambda x: x["depth"])
depths_task = np.array([t["depth"] for t in tasks])
hall_task   = np.array([t["hallucination_rate"] for t in tasks])
print(f"Tasks: {len(tasks)}, depth range: {depths_task.min()}–{depths_task.max()}")

# Per-probe arrays
depths_probe = np.array([r["depth"] for r in rows])
correct_probe = np.array([r["correct"] for r in rows], dtype=float)
hall_probe    = 1 - correct_probe

# ---------------------------------------------------------------------------
# Figure 1 — Scatter: hallucination rate per task vs depth
#            + LOWESS trend on per-probe binary outcomes
# ---------------------------------------------------------------------------

from scipy.ndimage import uniform_filter1d

def lowess_manual(x, y, frac=0.4):
    """Simple LOWESS using a Gaussian-weighted local regression."""
    from scipy.stats import norm
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bw = np.std(xs) * frac * len(xs) ** 0.2
    smoothed = np.empty_like(xs, dtype=float)
    for i, xi in enumerate(xs):
        w = np.exp(-0.5 * ((xs - xi) / bw) ** 2)
        smoothed[i] = np.average(ys, weights=w)
    return xs, smoothed


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hallucination Rate vs. Trajectory Depth\n(Claude Sonnet 4.6 on 529 LLM-generated probes)", fontsize=13)

# --- left: per-task scatter ---
ax = axes[0]
ax.scatter(depths_task, hall_task, alpha=0.7, s=60, color="#4C72B0", zorder=3, label="task (5 probes avg)")

# LOWESS on per-probe binary outcomes
xs_lo, ys_lo = lowess_manual(depths_probe, hall_probe, frac=0.5)
ax.plot(xs_lo, ys_lo, color="crimson", lw=2, label="LOWESS trend (per probe)")

# Pearson on per-task
r, p = pearsonr(depths_task, hall_task)
ax.set_xlabel("Trajectory Depth (command index at injection)", fontsize=11)
ax.set_ylabel("Hallucination Rate", fontsize=11)
ax.set_title(f"Per-task (n={len(tasks)})\nPearson r={r:.3f}, p={p:.3f}", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- right: log-scale x ---
ax2 = axes[1]
ax2.scatter(depths_task, hall_task, alpha=0.7, s=60, color="#4C72B0", zorder=3)
ax2.plot(xs_lo, ys_lo, color="crimson", lw=2, label="LOWESS trend")
ax2.set_xscale("log")
ax2.set_xlabel("Trajectory Depth (log scale)", fontsize=11)
ax2.set_ylabel("Hallucination Rate", fontsize=11)
ax2.set_title("Same data — log x-axis\n(depths span 2 → 1643)", fontsize=10)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out1 = OUT_DIR / "depth_vs_hallucination_scatter.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Figure 2 — Binned bar chart with error bars + per-probe-type breakdown
# ---------------------------------------------------------------------------

# Quantile bins so each bin has ~equal number of tasks
n_bins = 4
quantiles = np.quantile(depths_task, np.linspace(0, 1, n_bins + 1))
quantiles = np.unique(np.round(quantiles).astype(int))
# Build bin edges ensuring coverage
bin_edges = [depths_task.min() - 1] + list(quantiles[1:-1]) + [depths_task.max() + 1]

bin_labels = []
for i in range(len(bin_edges) - 1):
    lo = bin_edges[i] + 1
    hi = bin_edges[i + 1]
    bin_labels.append(f"{lo}–{hi}")

def assign_bin(d):
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] < d <= bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2

# Per-probe-type breakdown within each bin
probe_types = ["perceptual", "memory", "causal", "uncertainty", "cross-tier compound"]
bin_pt_hall = {b: {pt: [] for pt in probe_types} for b in range(len(bin_labels))}
bin_overall  = {b: [] for b in range(len(bin_labels))}

for r in rows:
    b = assign_bin(r["depth"])
    bin_overall[b].append(1 - r["correct"])
    pt = r["probe_type"]
    if pt in probe_types:
        bin_pt_hall[b][pt].append(1 - r["correct"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hallucination Rate by Trajectory Depth Bin\n(Claude Sonnet 4.6)", fontsize=13)

# --- left: overall per bin ---
ax = axes[0]
means, errs, ns = [], [], []
for b in range(len(bin_labels)):
    vals = bin_overall[b]
    if vals:
        means.append(np.mean(vals))
        errs.append(np.std(vals) / np.sqrt(len(vals)))
        ns.append(len(vals))
    else:
        means.append(0); errs.append(0); ns.append(0)

x = np.arange(len(bin_labels))
bars = ax.bar(x, means, yerr=errs, capsize=5, color="#4C72B0", alpha=0.8, zorder=3)
for xi, m, n in zip(x, means, ns):
    ax.text(xi, m + 0.02, f"n={n}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=9)
ax.set_xlabel("Trajectory Depth Bin (command index)", fontsize=11)
ax.set_ylabel("Hallucination Rate", fontsize=11)
ax.set_title("Overall hallucination rate per depth bin\n(error bars = ±1 SE)", fontsize=10)
ax.set_ylim(0, 0.6)
ax.axhline(1 - 0.7913, color="gray", ls="--", lw=1.2, label=f"Overall mean ({1-0.7913:.3f})")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

# --- right: per-probe-type stacked lines ---
ax2 = axes[1]
for pt in probe_types:
    pt_means = []
    for b in range(len(bin_labels)):
        vals = bin_pt_hall[b][pt]
        pt_means.append(np.mean(vals) if vals else np.nan)
    ax2.plot(x, pt_means, marker="o", label=pt, color=PROBE_TYPE_COLORS[pt], lw=2)

ax2.set_xticks(x)
ax2.set_xticklabels(bin_labels, fontsize=9)
ax2.set_xlabel("Trajectory Depth Bin (command index)", fontsize=11)
ax2.set_ylabel("Hallucination Rate", fontsize=11)
ax2.set_title("By probe type across depth bins", fontsize=10)
ax2.set_ylim(0, 0.8)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out2 = OUT_DIR / "depth_vs_hallucination_binned.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\nDepth bin summary:")
print(f"{'Bin':<12} {'n_probes':>8} {'hall_rate':>10} {'±SE':>8}")
for b, label in enumerate(bin_labels):
    vals = bin_overall[b]
    if vals:
        m = np.mean(vals)
        se = np.std(vals) / np.sqrt(len(vals))
        print(f"{label:<12} {len(vals):>8} {m:>10.3f} {se:>8.3f}")

r_log, p_log = pearsonr(np.log1p(depths_task), hall_task)
print(f"\nPearson r (log depth vs hall rate per task): r={r_log:.3f}, p={p_log:.3f}")
