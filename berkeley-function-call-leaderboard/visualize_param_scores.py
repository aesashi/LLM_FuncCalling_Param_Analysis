"""
Visualize param_scores_aggregated.csv.

Produces two figures saved to score/:
  1. error_heatmap.png  — model × error type mean (averaged across categories)
  2. accuracy_by_category.png — accuracy per model per test category
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

SCORE_DIR = Path(__file__).parent / "score"
CSV = SCORE_DIR / "param_scores_aggregated.csv"

# Ratio metrics: score = correct/total (0–1, higher=better) → error = 1 - mean
RATIO_METRICS = [
    "missing_information",
    "hallucinated_params",
    "specification_mismatch",
    "task_deviation",
]

# Count metric: mean = avg extra params per sample (higher = more errors, no upper bound)
COUNT_METRIC = "redundant_information"

METRICS = RATIO_METRICS + [COUNT_METRIC]

METRIC_LABELS = {
    "missing_information":    "Missing\nInfo",
    "hallucinated_params":    "Hallucinated\nParams",
    "specification_mismatch": "Spec\nMismatch",
    "task_deviation":         "Task\nDeviation",
    "redundant_information":  "Redundant\nInfo\n(count)",
}

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

# Ratio metrics → error rate (1 - mean), higher = more errors, range 0–1
for m in RATIO_METRICS:
    col = f"{m}_mean"
    if col in df.columns:
        df[f"{m}_error"] = 1.0 - df[col].fillna(1.0)

# Count metric → use mean directly (avg extra params per sample), already "higher = worse"
df["redundant_information_error"] = df["redundant_information_mean"].fillna(0.0)


# ── Figure 1: Error heatmap (model × error type) ─────────────────────────────
mean_errors = (
    df.groupby("model")[[f"{m}_error" for m in METRICS]]
    .mean()
    .rename(columns={f"{m}_error": METRIC_LABELS[m] for m in METRICS})
)
mean_errors.index.name = "Model"

# Split into two sub-tables: ratio errors (0–1) and count (unbounded)
ratio_labels  = [METRIC_LABELS[m] for m in RATIO_METRICS]
count_label   = METRIC_LABELS[COUNT_METRIC]
ratio_data    = mean_errors[ratio_labels]
count_data    = mean_errors[[count_label]]

# Two side-by-side heatmaps with independent color scales
fig1, (ax_r, ax_c) = plt.subplots(
    1, 2,
    figsize=(11, max(3, 0.7 * len(mean_errors) + 1.5)),
    gridspec_kw={"width_ratios": [len(RATIO_METRICS), 1.2]},
)

sns.heatmap(
    ratio_data, ax=ax_r,
    annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    vmin=0, vmax=ratio_data.values.max() * 1.1 or 0.1,
    cbar_kws={"label": "Error rate  (1 − mean score)"},
)
ax_r.set_title("Ratio-based errors\n(higher = worse)", pad=10)
ax_r.set_ylabel("Model")
ax_r.tick_params(axis="x", labelsize=9)
ax_r.tick_params(axis="y", labelsize=9, rotation=0)

sns.heatmap(
    count_data, ax=ax_c,
    annot=True, fmt=".3f", cmap="Blues",
    linewidths=0.5, linecolor="white",
    vmin=0,
    cbar_kws={"label": "Avg extra params / sample"},
)
ax_c.set_title("Count-based error\n(avg extra params)", pad=10)
ax_c.set_ylabel("")
ax_c.tick_params(axis="x", labelsize=9)
ax_c.tick_params(axis="y", labelleft=False)

fig1.suptitle("Error rates by model (averaged across all test categories)", y=1.02)
fig1.tight_layout()
out1 = SCORE_DIR / "error_heatmap.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close(fig1)


# ── Figure 2: Accuracy by category ───────────────────────────────────────────
# Sort categories by mean accuracy across models for readability
cat_order = (
    df.groupby("test_category")["accuracy"].mean()
    .sort_values(ascending=True)
    .index.tolist()
)

models = sorted(df["model"].unique())
n_models = len(models)
n_cats = len(cat_order)
x = np.arange(n_cats)
width = 0.8 / max(n_models, 1)

fig2, ax2 = plt.subplots(figsize=(14, 5))
colors = sns.color_palette("Set2", n_models)

for i, model in enumerate(models):
    sub = df[df["model"] == model].set_index("test_category")
    vals = [sub.loc[c, "accuracy"] if c in sub.index else np.nan for c in cat_order]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax2.bar(x + offset, vals, width=width * 0.9, label=model, color=colors[i], zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(cat_order, rotation=40, ha="right", fontsize=8)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax2.set_ylim(0, 1.05)
ax2.axhline(1.0, color="gray", lw=0.5, ls="--", zorder=2)
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy by test category and model")
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(axis="y", alpha=0.3, zorder=0)
fig2.tight_layout()
out2 = SCORE_DIR / "accuracy_by_category.png"
fig2.savefig(out2, dpi=150)
print(f"Saved → {out2}")
plt.close(fig2)


# ── Figure 3: Error breakdown per category ────────────────────────────────────
# Top row: stacked bar of ratio errors (0–1) per model
# Bottom row: bar of redundant_information count per model (separate scale)
fig3, axes = plt.subplots(
    2, n_models,
    figsize=(7 * n_models, 9),
    gridspec_kw={"height_ratios": [3, 1.5]},
    squeeze=False,
)
palette_ratio = sns.color_palette("Set1", len(RATIO_METRICS))

for col_i, model in enumerate(models):
    sub = df[df["model"] == model].set_index("test_category")

    # ── top: stacked ratio errors ──────────────────────────────────────────
    ax_top = axes[0][col_i]
    ratio_error_cols = [f"{m}_error" for m in RATIO_METRICS]
    plot_ratio = sub.reindex(cat_order)[ratio_error_cols].fillna(0)
    plot_ratio.columns = [METRIC_LABELS[m] for m in RATIO_METRICS]

    bottom = np.zeros(len(cat_order))
    for j, metric_col in enumerate(plot_ratio.columns):
        ax_top.bar(range(len(cat_order)), plot_ratio[metric_col],
                   bottom=bottom, label=metric_col,
                   color=palette_ratio[j], zorder=3)
        bottom += plot_ratio[metric_col].values

    ax_top.set_title(model, fontsize=9)
    ax_top.set_xticks(range(len(cat_order)))
    ax_top.set_xticklabels(cat_order, rotation=40, ha="right", fontsize=7)
    ax_top.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_top.set_ylim(0, 1.05)
    ax_top.grid(axis="y", alpha=0.3, zorder=0)
    ax_top.set_ylabel("Error rate (1 − score)" if col_i == 0 else "")

    # ── bottom: redundant count ────────────────────────────────────────────
    ax_bot = axes[1][col_i]
    redund_vals = [
        sub.loc[c, "redundant_information_error"] if c in sub.index else 0.0
        for c in cat_order
    ]
    ax_bot.bar(range(len(cat_order)), redund_vals, color="#4C72B0", zorder=3)
    ax_bot.set_xticks(range(len(cat_order)))
    ax_bot.set_xticklabels(cat_order, rotation=40, ha="right", fontsize=7)
    ax_bot.grid(axis="y", alpha=0.3, zorder=0)
    ax_bot.set_ylabel("Avg extra params / sample" if col_i == 0 else "")

handles, labels = axes[0][0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="upper right", fontsize=8, title="Error type (ratio)")
fig3.suptitle(
    "Error breakdown by category and model\n"
    "Top: ratio errors (stacked) · Bottom: redundant info count",
    y=1.01,
)
fig3.tight_layout()
out3 = SCORE_DIR / "error_breakdown_by_category.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")
plt.close(fig3)
