"""Figures for Chapter Four."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"LR": ("#9e9e9e", "o", ":"), "RF": ("#6d4c41", "s", ":"), "XGB": ("#1565c0", "D", "-"),
         "XGB+graph": ("#00897b", "D", "--"), "GCN": ("#ef6c00", "^", "-"), "GraphSAGE": ("#c62828", "v", "-"),
         "GAT": ("#6a1b9a", "P", "-")}
NICE = {"gender": "Gender", "age": "Age 35+", "disability": "Disability", "imd": "Deprivation (IMD)"}


def _save(fig, cfg, name):
    d = Path(cfg["output_dir"], "figures")
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(out, cfg):
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    s = out["t03_performance_summary"]

    # 1. PR-AUC and recall at budget across weeks, Split B
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "pr_auc", "Test PR-AUC"), (axes[1], "recall_at_budget", "Recall at 20% alert budget")):
        for m, (c, mk, ls) in STYLE.items():
            d = s[(s["split"] == "B") & (s["model"] == m)].sort_values("week")
            if len(d):
                ax.errorbar(d["week"], d[f"{metric}_mean"], yerr=d[f"{metric}_std"], color=c, marker=mk,
                            linestyle=ls, capsize=3, label=m, markersize=5)
        ax.set_xlabel("Week of presentation")
        ax.set_title(f"{title} (Split B: 2013 → 2014J)")
        ax.set_xticks(sorted(s["week"].unique()))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    _save(fig, cfg, "fig_prauc_by_week")

    # 2. Split A vs Split B for key models
    keys = ["LR", "XGB", "XGB+graph", "GCN", "GraphSAGE", "GAT"]
    weeks = sorted(s["week"].unique())
    fig, axes = plt.subplots(1, len(weeks), figsize=(2.6 * len(weeks), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, weeks):
        x = np.arange(len(keys))
        for off, split, col in ((-0.2, "A", "#bdbdbd"), (0.2, "B", "#455a64")):
            vals = [s[(s["split"] == split) & (s["week"] == w) & (s["model"] == m)]["pr_auc_mean"].mean() for m in keys]
            ax.bar(x + off, vals, width=0.4, color=col, label=f"Split {split}")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=60, fontsize=8)
        ax.set_title(f"Week {w}")
        lo = s["pr_auc_mean"].min()
        ax.set_ylim(max(0, lo - 0.08), min(1, s["pr_auc_mean"].max() + 0.03))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Test PR-AUC")
    axes[0].legend(fontsize=8)
    _save(fig, cfg, "fig_split_a_vs_b")

    # 3. EOD best GNN vs best gradient boosting
    h3 = out["t11_h3"]
    attrs = list(NICE)
    fig, axes = plt.subplots(1, len(attrs), figsize=(12, 3.6), sharey=True)
    for ax, a in zip(axes, attrs):
        d = h3[h3["attribute"] == a].sort_values("week")
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["eod_gb"], 0.4, color="#1565c0", label="Best gradient boosting")
        ax.bar(x + 0.2, d["eod_gnn"], 0.4, color="#c62828", label="Best GNN")
        ax.set_xticks(x)
        ax.set_xticklabels([f"W{w}" for w in d["week"]])
        ax.set_title(NICE[a])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Equalized odds difference")
    axes[0].legend(fontsize=8)
    _save(fig, cfg, "fig_fairness_eod")

    # 4. SHAP and GNNExplainer importance (week 4 if available)
    shap_t, gx = out["t15_shap"], out["t16_gnnexplainer"]
    wk = 4 if 4 in set(shap_t["week"]) else shap_t["week"].min()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.75)
    d = shap_t[shap_t["week"] == wk].head(12).iloc[::-1]
    axes[0].barh(d["feature"], d["mean_abs_shap"], color="#1565c0")
    axes[0].set_title(f"SHAP: {d['model'].iloc[0]} (week {wk})")
    axes[0].set_xlabel("Mean |SHAP value|")
    perm = out.get("t19_gnn_permutation")
    if perm is not None and len(perm):
        d = perm[perm["week"] == wk].head(12).iloc[::-1]
        axes[1].barh(d["feature"], d["pr_auc_drop"], color="#c62828")
        axes[1].set_title(f"Permutation importance: {d['model'].iloc[0]} (week {wk})")
        axes[1].set_xlabel("Drop in test PR-AUC when shuffled")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    _save(fig, cfg, "fig_explanations")

    # 5. mitigation
    mit = out["t12_mitigation"]
    agg = mit.groupby(["role"])[["eod_before", "eod_after", "recall_before", "recall_after"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(agg))
    ax.bar(x - 0.3, agg["eod_before"], 0.2, label="EOD before", color="#ef9a9a")
    ax.bar(x - 0.1, agg["eod_after"], 0.2, label="EOD after", color="#c62828")
    ax.bar(x + 0.1, agg["recall_before"], 0.2, label="Recall before", color="#90caf9")
    ax.bar(x + 0.3, agg["recall_after"], 0.2, label="Recall after", color="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels(["Best gradient boosting" if r == "best_gb" else "Best GNN" for r in agg.index])
    ax.set_title("Group-specific thresholds (mean over weeks and attributes)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, cfg, "fig_mitigation")
