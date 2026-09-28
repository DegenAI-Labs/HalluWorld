#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("analysis_plots")

PROBE_TYPE_COLORS = {
    "perceptual":          "#4C72B0",
    "memory":              "#DD8452",
    "causal":              "#55A868",
    "uncertainty":         "#C44E52",
    "cross-tier compound": "#8172B2",
}
PROBE_TYPES = list(PROBE_TYPE_COLORS.keys())

EVAL_MODELS = {
    "claude-sonnet-4-6": {
        "dir": Path("runs/probe_eval_sonnet"),
        "color": "#2196F3",
        "ls": "-",
    },
    "gpt-4o-mini": {
        "dir": Path("runs/probe_eval_4o-mini"),
        "color": "#FF9800",
        "ls": "--",
    },
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_results(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*.result.json")):
        d = json.loads(path.read_text())
        if d.get("status") != "ok":
            continue
        m = d.get("metadata") or {}
        depth = m.get("step_command_index") or m.get("trigger_step_index")
        if depth is None:
            continue
        rows.append({
            "depth": int(depth),
            "hallucinated": 1 - int(bool(d.get("correct"))),
            "probe_type": m.get("probe_type", "unknown"),
        })
    return rows

# ---------------------------------------------------------------------------
# Gaussian kernel smoother in log-depth space
# ---------------------------------------------------------------------------
def smooth(x_data, y_data, x_query, bw=0.8):
    log_x = np.log(x_data)
    log_q = np.log(x_query)
    y_smooth = np.empty_like(x_query, dtype=float)
    ci_lo    = np.empty_like(x_query, dtype=float)
    ci_hi    = np.empty_like(x_query, dtype=float)
    for i, lq in enumerate(log_q):
        w = np.exp(-0.5 * ((log_x - lq) / bw) ** 2)
        w_sum = w.sum()
        if w_sum < 1e-12:
            y_smooth[i] = ci_lo[i] = ci_hi[i] = np.nan
            continue
        mu = np.dot(w, y_data) / w_sum
        n_eff = w_sum ** 2 / (w ** 2).sum()
        se = np.sqrt(mu * (1 - mu) / max(n_eff, 1))
        y_smooth[i] = mu
        ci_lo[i]    = max(0, mu - 1.96 * se)
        ci_hi[i]    = min(1, mu + 1.96 * se)
    return y_smooth, ci_lo, ci_hi

# ---------------------------------------------------------------------------
# Load all models
# ---------------------------------------------------------------------------
all_data = {}
for model_name, cfg in EVAL_MODELS.items():
    rows = load_results(cfg["dir"])
    all_data[model_name] = rows
    hall = np.mean([r["hallucinated"] for r in rows])
    print(f"{model_name}: n={len(rows)}, hallucination_rate={hall:.3f}, accuracy={1-hall:.3f}")

# Shared depth range and query points
all_depths = np.concatenate([[r["depth"] for r in rows] for rows in all_data.values()])
x_query = np.exp(np.linspace(np.log(all_depths.min()), np.log(all_depths.max()), 300))

# ---------------------------------------------------------------------------
# Figure 1 — Overall smooth per evaluator model
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))

for model_name, rows in all_data.items():
    cfg = EVAL_MODELS[model_name]
    depths = np.array([r["depth"] for r in rows])
    halls  = np.array([r["hallucinated"] for r in rows], dtype=float)
    overall_mean = halls.mean()

    y, lo, hi = smooth(depths, halls, x_query, bw=0.8)
    ax.fill_between(x_query, lo, hi, alpha=0.12, color=cfg["color"])
    ax.plot(x_query, y, color=cfg["color"], lw=2.5, ls=cfg["ls"],
            label=f"{model_name}  (hall. {overall_mean:.1%}  acc. {1-overall_mean:.1%})")

# Rug
unique_depths = np.unique(all_depths)
ax.plot(unique_depths, np.full_like(unique_depths, -0.025, dtype=float),
        "|", color="gray", alpha=0.5, ms=8, markeredgewidth=1.2, clip_on=False)

ax.set_xscale("log")
ax.set_xlim(all_depths.min() * 0.9, all_depths.max() * 1.1)
ax.set_ylim(-0.04, 0.95)
ax.set_xlabel("Trajectory Depth — command index at injection (log scale)", fontsize=11)
ax.set_ylabel("Hallucination Rate", fontsize=11)
ax.set_title(
    "Hallucination Rate vs. Trajectory Depth\n"
    "Gaussian kernel smoother (bw=0.8 in log space) · shaded = 95% CI · rug = observed depths",
    fontsize=11,
)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.25, which="both")
xticks = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
ax.set_xticks(xticks)
ax.set_xticklabels([str(t) for t in xticks])

plt.tight_layout()
out = OUT_DIR / "depth_vs_hallucination_smooth.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 2 — Per-probe-type, one subplot per evaluator model
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.suptitle(
    "Hallucination Rate vs. Trajectory Depth — by Probe Type\n"
    "Gaussian kernel smoother (bw=0.8 in log space) · shaded = 95% CI",
    fontsize=12,
)

for ax, (model_name, rows) in zip(axes, all_data.items()):
    cfg = EVAL_MODELS[model_name]
    depths = np.array([r["depth"] for r in rows])
    halls  = np.array([r["hallucinated"] for r in rows], dtype=float)
    types  = np.array([r["probe_type"] for r in rows])

    # Overall
    y, lo, hi = smooth(depths, halls, x_query, bw=0.8)
    ax.fill_between(x_query, lo, hi, alpha=0.10, color="black")
    ax.plot(x_query, y, color="black", lw=2, ls="--", label="overall")

    for pt in PROBE_TYPES:
        mask = types == pt
        if mask.sum() < 5:
            continue
        y_pt, lo_pt, hi_pt = smooth(depths[mask], halls[mask], x_query, bw=0.8)
        color = PROBE_TYPE_COLORS[pt]
        ax.fill_between(x_query, lo_pt, hi_pt, alpha=0.07, color=color)
        ax.plot(x_query, y_pt, color=color, lw=1.8, label=pt)

    ax.set_xscale("log")
    ax.set_xlim(all_depths.min() * 0.9, all_depths.max() * 1.1)
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlabel("Trajectory Depth (log scale)", fontsize=10)
    ax.set_ylabel("Hallucination Rate", fontsize=10)
    overall_hall = halls.mean()
    ax.set_title(f"{model_name}\nhall. {overall_hall:.1%}  ·  acc. {1-overall_hall:.1%}", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.axhline(overall_hall, color="gray", ls=":", lw=1, alpha=0.5)

plt.tight_layout()
out2 = OUT_DIR / "depth_vs_hallucination_smooth_by_type.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
