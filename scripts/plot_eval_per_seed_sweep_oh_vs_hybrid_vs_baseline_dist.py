"""
Per-protein BoN (N=10) metric distributions, model-by-model comparison.

Reads the BoN per-protein values produced by
plot_eval_per_seed_sweep_oh_vs_hybrid_vs_baseline_by_length.py
(per_protein.csv: ca_rmsd=min across seeds, tm_score=max across seeds, with
the failed-fold sentinel 7W2P pre-filtered).

For each (dataset, metric) we plot:
  - ECDF grid:                  one ECDF subplot per iter (2x3)
  - Per-iter standalone:        histogram + ECDF in a single figure
  - Violin summary:             per-iter violins for all three models

Threshold count annotations (rmsd ≤ 5/3/2/1, tm-score ≥ 0.5/0.7/0.8) overlay
the ECDF as horizontal reference lines.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SRC_CSV = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/"
               f"oh_vs_hybrid_vs_baseline_by_length/per_protein.csv")
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/"
               f"oh_vs_hybrid_vs_baseline_dist")

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]
METRICS = ["ca_rmsd", "tm_score"]

OH_LABEL = "oh-xhigh (recommended)"
HYBRID_LABEL = "hybrid plain (recommended)"
BASELINE_LABEL = "baseline (max-T anneal)"

SERIES_ORDER = [OH_LABEL, HYBRID_LABEL, BASELINE_LABEL]
SERIES_COLOR = {OH_LABEL: "#9467bd", HYBRID_LABEL: "#d62728",
                BASELINE_LABEL: "#1f77b4"}
SHORT = {OH_LABEL: "oh-xhigh", HYBRID_LABEL: "hybrid",
         BASELINE_LABEL: "baseline"}

# Recommended decoding per (model, iter), used only for plot annotations.
DEC_BY_ITER = {
    OH_LABEL: {1: "annealing0.5_0.1", 3: "annealing1.0_0.1",
               5: "annealing0.7_0.1", 10: "annealing0.3_0.1",
               30: "argmax", 100: "annealing4.0_0.1"},
    HYBRID_LABEL: {1: "annealing0.5_0.1", 3: "annealing0.5_0.1",
                   5: "annealing0.1_0.01", 10: "argmax",
                   30: "argmax", 100: "annealing4.0_0.1"},
    BASELINE_LABEL: {1: "annealing0.5_0.1", 3: "annealing1.0_0.1",
                     5: "annealing1.0_0.1", 10: "annealing2.0_0.1",
                     30: "annealing2.0_0.1", 100: "annealing4.0_0.1"},
}


def thresholds(metric: str):
    if metric == "tm_score":
        return [("≥0.5", 0.5), ("≥0.7", 0.7), ("≥0.8", 0.8)]
    return [("≤5", 5), ("≤3", 3), ("≤2", 2), ("≤1", 1)]


def _passes(metric: str, y, t):
    return (y >= t).sum() if metric == "tm_score" else (y <= t).sum()


def _counts_table(sub: pd.DataFrame, metric: str) -> str:
    ths = thresholds(metric)
    col_w = 6
    lbl_w = max(len(SHORT[l]) for l in SERIES_ORDER)
    header = (" " * (lbl_w + 2)) + "".join(f"{n:>{col_w}}" for n, _ in ths) + "   n"
    lines = [header]
    for lbl in SERIES_ORDER:
        s = sub[sub["label"] == lbl][metric]
        cs = "".join(f"{int(_passes(metric, s, t)):>{col_w}}" for _, t in ths)
        lines.append(f"{SHORT[lbl]:<{lbl_w}}  {cs}  {len(s):>3}")
    return "\n".join(lines)


def _ecdf(values: np.ndarray):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _plot_ecdf(ax, sub_iter: pd.DataFrame, metric: str,
               show_legend: bool = True):
    for lbl in SERIES_ORDER:
        s = sub_iter[sub_iter["label"] == lbl][metric].to_numpy(dtype=float)
        if len(s) == 0:
            continue
        x, y = _ecdf(s)
        ax.step(x, y, where="post", color=SERIES_COLOR[lbl],
                linewidth=2.0, label=lbl, zorder=3)
    # threshold reference lines (vertical)
    for name, t in thresholds(metric):
        ax.axvline(t, color="grey", linestyle=":", linewidth=0.8,
                   alpha=0.7, zorder=1)
        # label at top inside axes
        ax.annotate(name, xy=(t, 1.0), xytext=(2, -2),
                    textcoords="offset points", fontsize=7,
                    color="grey", ha="left", va="top")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _plot_hist(ax, sub_iter: pd.DataFrame, metric: str, bins=30):
    # shared range across models for fair comparison
    vals_all = sub_iter[metric].to_numpy(dtype=float)
    if len(vals_all) == 0:
        return
    lo, hi = float(np.min(vals_all)), float(np.max(vals_all))
    if hi == lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    for lbl in SERIES_ORDER:
        s = sub_iter[sub_iter["label"] == lbl][metric].to_numpy(dtype=float)
        if len(s) == 0:
            continue
        ax.hist(s, bins=edges, color=SERIES_COLOR[lbl], alpha=0.40,
                label=lbl, zorder=2)
    for _, t in thresholds(metric):
        ax.axvline(t, color="grey", linestyle=":", linewidth=0.8,
                   alpha=0.7, zorder=1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _counts_textbox(ax, sub: pd.DataFrame, metric: str,
                    loc=(0.99, 0.02), ha="right", va="bottom"):
    ax.text(loc[0], loc[1], _counts_table(sub, metric),
            transform=ax.transAxes, ha=ha, va=va,
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))


def _iter_dec_str(it: int) -> str:
    return ", ".join(f"{SHORT[l]}={DEC_BY_ITER[l][it]}"
                     for l in SERIES_ORDER)


def plot_ecdf_grid(records: pd.DataFrame, dataset: str, metric: str,
                   out_path: Path):
    sub = records[records["dataset"] == dataset]
    if sub.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, it in zip(axes, ITERS):
        sub_it = sub[sub["iter"] == it]
        _plot_ecdf(ax, sub_it, metric, show_legend=False)
        ax.set_title(f"iter={it}  ({_iter_dec_str(it)})", fontsize=9)
        # counts box: rmsd → upper-left (low-x is good); tm → lower-right
        if metric == "tm_score":
            _counts_textbox(ax, sub_it, metric,
                            loc=(0.02, 0.98), ha="left", va="top")
        else:
            _counts_textbox(ax, sub_it, metric,
                            loc=(0.98, 0.02), ha="right", va="bottom")
    for ax in axes[-3:]:
        ax.set_xlabel(metric)
    for ax in axes[::3]:
        ax.set_ylabel("ECDF  P(X ≤ x)")
    handles = [plt.Line2D([0], [0], color=SERIES_COLOR[l], linewidth=2.4,
                          label=l) for l in SERIES_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.99), fontsize=10, frameon=False)
    fig.suptitle(f"{dataset} | {metric} | per-protein BoN (N=10) ECDF",
                 y=1.005, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_iter(records: pd.DataFrame, dataset: str, metric: str,
                  iter_n: int, out_path: Path):
    sub = records[(records["dataset"] == dataset) & (records["iter"] == iter_n)]
    if sub.empty:
        return
    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(15, 5.5))
    _plot_hist(ax_hist, sub, metric)
    ax_hist.set_xlabel(metric)
    ax_hist.set_ylabel("count of proteins")
    ax_hist.set_title("histogram (overlapping, alpha=0.4)", fontsize=10)

    _plot_ecdf(ax_ecdf, sub, metric, show_legend=False)
    ax_ecdf.set_xlabel(metric)
    ax_ecdf.set_ylabel("ECDF  P(X ≤ x)")
    ax_ecdf.set_title("ECDF", fontsize=10)

    if metric == "tm_score":
        _counts_textbox(ax_ecdf, sub, metric,
                        loc=(0.02, 0.98), ha="left", va="top")
    else:
        _counts_textbox(ax_ecdf, sub, metric,
                        loc=(0.98, 0.02), ha="right", va="bottom")

    handles = [plt.Line2D([0], [0], color=SERIES_COLOR[l], linewidth=2.4,
                          label=l) for l in SERIES_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=10, frameon=False)
    fig.suptitle(
        f"{dataset} | {metric} | iter={iter_n} | BoN N=10\n"
        f"({_iter_dec_str(iter_n)})", y=1.10, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_violin(records: pd.DataFrame, dataset: str, metric: str,
                out_path: Path):
    sub = records[records["dataset"] == dataset]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    n_models = len(SERIES_ORDER)
    width = 0.24
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width
    iter_to_pos = {it: i for i, it in enumerate(ITERS)}

    for j, lbl in enumerate(SERIES_ORDER):
        positions, data = [], []
        for it in ITERS:
            s = sub[(sub["iter"] == it) & (sub["label"] == lbl)][metric]
            s = s.to_numpy(dtype=float)
            if len(s) == 0:
                continue
            positions.append(iter_to_pos[it] + offsets[j])
            data.append(s)
        if not data:
            continue
        parts = ax.violinplot(data, positions=positions, widths=width * 0.95,
                              showmeans=False, showmedians=True,
                              showextrema=False)
        color = SERIES_COLOR[lbl]
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.2)

    ax.set_xticks(list(iter_to_pos.values()))
    ax.set_xticklabels([str(it) for it in ITERS])
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} (per-protein BoN, N=10)")
    ax.set_title(f"{dataset} | {metric} | distribution per iter (violin)",
                 fontsize=11)
    for _, t in thresholds(metric):
        ax.axhline(t, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.3, axis="y")

    handles = [plt.Line2D([0], [0], color=SERIES_COLOR[l], linewidth=8,
                          alpha=0.55, label=l) for l in SERIES_ORDER]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_CSV.exists():
        raise SystemExit(f"missing {SRC_CSV}; run plot_..._by_length.py first.")
    records = pd.read_csv(SRC_CSV)
    print(f"loaded {len(records)} rows from {SRC_CSV}")

    for ds in DATASETS:
        for metric in METRICS:
            grid_out = OUT_DIR / f"{ds}__{metric}__ecdf_grid.png"
            plot_ecdf_grid(records, ds, metric, grid_out)
            print(f"wrote {grid_out}")

            violin_out = OUT_DIR / f"{ds}__{metric}__violin.png"
            plot_violin(records, ds, metric, violin_out)
            print(f"wrote {violin_out}")

            for it in ITERS:
                out_it = OUT_DIR / f"{ds}__{metric}__iter{it}.png"
                plot_per_iter(records, ds, metric, it, out_it)
                print(f"wrote {out_it}")


if __name__ == "__main__":
    main()
