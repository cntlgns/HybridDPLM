"""
Per-protein length-vs-metric scatter for the three models in
oh_vs_hybrid_vs_baseline:
  - oh-xhigh (recommended)
  - hybrid plain (recommended)
  - baseline (max-T anneal)

Each iter uses its model-specific recommended decoding (mirrors the picks in
plot_eval_per_seed_sweep_oh_vs_hybrid_vs_baseline.py and recommended/summary.csv).

For each (model, iter, decoding, dataset, protein) we collapse seeds with the
BoN oracle (min for ca_rmsd, max for tm_score) — the best of N=10 seeds — and
plot length on the x-axis, the metric on the y-axis.

Outputs:
  - Grid figure per (dataset, metric):  2x3 subplots (one per iter)
  - Standalone per-iter figure:         <dataset>__<metric>__iter<N>.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = Path(f"{PROJECT_DIR}/generation-results")
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/"
               f"oh_vs_hybrid_vs_baseline_by_length")

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]

METRICS = {
    "ca_rmsd":  ("ca_rmsd",   "min"),
    "tm_score": ("bb_tmscore", "max"),
}

OH_LABEL = "oh-xhigh (recommended)"
HYBRID_LABEL = "hybrid plain (recommended)"
BASELINE_LABEL = "baseline (max-T anneal)"

OH_COLOR = "#9467bd"
HYBRID_COLOR = "#d62728"
BASELINE_COLOR = "#1f77b4"

SERIES_ORDER = [OH_LABEL, HYBRID_LABEL, BASELINE_LABEL]
SERIES_COLOR = {OH_LABEL: OH_COLOR, HYBRID_LABEL: HYBRID_COLOR,
                BASELINE_LABEL: BASELINE_COLOR}

# (model_dir, ckpt_dir, recommended decoding by iter)
SERIES = {
    OH_LABEL: (
        "oh-xhigh_noise-full-3e5_lr-fs0-normemb",
        "step_10256.0-loss_0.70",
        {1: "annealing0.5_0.1",
         3: "annealing1.0_0.1",
         5: "annealing0.7_0.1",
         10: "annealing0.3_0.1",
         30: "argmax",
         100: "annealing4.0_0.1"},
    ),
    HYBRID_LABEL: (
        "emb-high_noise-full-1e4_lr",
        "step_9843.0-loss_0.68",
        {1: "annealing0.5_0.1",
         3: "annealing0.5_0.1",
         5: "annealing0.1_0.01",
         10: "argmax",
         30: "argmax",
         100: "annealing4.0_0.1"},
    ),
    BASELINE_LABEL: (
        "baseline-full-1e4_lr-fs0_uncond",
        "step_6971.0-loss_0.22",
        {1: "annealing0.5_0.1",
         3: "annealing1.0_0.1",
         5: "annealing1.0_0.1",
         10: "annealing2.0_0.1",
         30: "annealing2.0_0.1",
         100: "annealing4.0_0.1"},
    ),
}


def protein_id(sample_path: str) -> str:
    m = re.search(r"eval/(length_\d+/[^/]+)/sample\.pdb", sample_path)
    return m.group(1) if m else sample_path


def collect():
    """Return long-form DataFrame with per-(label, iter, dataset, protein) rows.

    Columns: label, iter, dataset, decoding, protein_id, length, ca_rmsd,
             tm_score (mean across seeds).
    """
    rows = []
    for label in SERIES_ORDER:
        model_dir, ckpt_dir, dec_by_iter = SERIES[label]
        for it in ITERS:
            dec = dec_by_iter[it]
            for ds in DATASETS:
                csv = (GEN_ROOT / f"eval_per_seed_sweep_{it}iter" / model_dir /
                       ds / ckpt_dir / dec / "per_sample_by_seed.csv")
                if not csv.exists():
                    print(f"[warn] missing {csv}")
                    continue
                df = pd.read_csv(csv)
                if df.empty:
                    continue
                df["protein_id"] = df["sample_path"].apply(protein_id)
                # Drop folding-failure sentinels (ca_rmsd=100, tm_score=0).
                # PDB_date/length_64/7W2P consistently fails across all
                # (model, iter, decoding) and would dominate the trend line.
                df = df[(df["ca_rmsd"] < 100) & (df["bb_tmscore"] > 0)]
                if df.empty:
                    continue
                # one row per protein: BoN oracle across seeds (min for
                # ca_rmsd, max for tm_score); length is constant per protein.
                grp = df.groupby("protein_id", as_index=False).agg(
                    length=("length", "first"),
                    ca_rmsd=("ca_rmsd", "min"),
                    tm_score=("bb_tmscore", "max"),
                )
                grp["label"] = label
                grp["iter"] = it
                grp["dataset"] = ds
                grp["decoding"] = dec
                rows.append(grp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


SHORT_LABEL = {
    OH_LABEL: "oh-xhigh",
    HYBRID_LABEL: "hybrid",
    BASELINE_LABEL: "baseline",
}


def _series_legend_handles():
    return [plt.Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor=SERIES_COLOR[lbl],
                       markeredgecolor="none", markersize=8, label=lbl)
            for lbl in SERIES_ORDER]


def _compute_counts(sub: pd.DataFrame, metric: str):
    """label -> (n, [(threshold_name, count), ...])."""
    if metric == "tm_score":
        ths = [("≥0.5", 0.5), ("≥0.7", 0.7), ("≥0.8", 0.8)]
        passes = lambda y, t: (y >= t).sum()
    else:
        ths = [("≤5", 5), ("≤3", 3), ("≤2", 2), ("≤1", 1)]
        passes = lambda y, t: (y <= t).sum()
    out = {}
    for label in SERIES_ORDER:
        s = sub[sub["label"] == label]
        out[label] = (
            len(s),
            [(name, int(passes(s[metric], t))) for name, t in ths],
        )
    return out


def _legend_handles_with_counts(sub: pd.DataFrame, metric: str):
    """Legend handles: model entries + a header row + per-model count rows."""
    handles = _series_legend_handles()
    counts = _compute_counts(sub, metric)
    if metric == "tm_score":
        ths = ["≥0.5", "≥0.7", "≥0.8"]
        col_w = 6
    else:
        ths = ["≤5", "≤3", "≤2", "≤1"]
        col_w = 6
    lbl_w = max(len(SHORT_LABEL[l]) for l in SERIES_ORDER)
    header = (" " * (lbl_w + 2)) + "".join(f"{t:>{col_w}}" for t in ths) + "   n"
    handles.append(plt.Line2D([0], [0], color="none", marker="None",
                              label=" "))  # separator
    handles.append(plt.Line2D([0], [0], color="none", marker="None",
                              label=header))
    for lbl in SERIES_ORDER:
        n, items = counts[lbl]
        cs = "".join(f"{c:>{col_w}}" for _, c in items)
        text = f"{SHORT_LABEL[lbl]:<{lbl_w}}  {cs}  {n:>3}"
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="None",
                                  markerfacecolor=SERIES_COLOR[lbl],
                                  markeredgecolor="none", markersize=8,
                                  label=text))
    return handles


def _counts_textbox(ax, sub: pd.DataFrame, metric: str, loc=(0.99, 0.99),
                    ha="right", va="top"):
    counts = _compute_counts(sub, metric)
    if metric == "tm_score":
        ths = ["≥0.5", "≥0.7", "≥0.8"]
        col_w = 6
    else:
        ths = ["≤5", "≤3", "≤2", "≤1"]
        col_w = 6
    lbl_w = max(len(SHORT_LABEL[l]) for l in SERIES_ORDER)
    lines = [(" " * (lbl_w + 2)) + "".join(f"{t:>{col_w}}" for t in ths) + "   n"]
    for lbl in SERIES_ORDER:
        n, items = counts[lbl]
        cs = "".join(f"{c:>{col_w}}" for _, c in items)
        lines.append(f"{SHORT_LABEL[lbl]:<{lbl_w}}  {cs}  {n:>3}")
    ax.text(loc[0], loc[1], "\n".join(lines), transform=ax.transAxes,
            ha=ha, va=va, fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))


def _gaussian_trend(x: np.ndarray, y: np.ndarray,
                    bandwidth_frac: float = 0.15, n_grid: int = 120):
    """Gaussian-kernel smoother of y(x). Bandwidth is `bandwidth_frac` of the
    x-range. Returns (xg, yg) on a regular grid spanning [xmin, xmax]."""
    if len(x) < 3:
        return x, y
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax == xmin:
        return np.array([xmin]), np.array([float(np.mean(y))])
    h = max((xmax - xmin) * bandwidth_frac, 1e-6)
    xg = np.linspace(xmin, xmax, n_grid)
    # (n_grid, n_pts) weight matrix
    diff = (xg[:, None] - x[None, :]) / h
    w = np.exp(-0.5 * diff * diff)
    wsum = w.sum(axis=1)
    safe = wsum > 1e-12
    yg = np.full_like(xg, np.nan)
    yg[safe] = (w[safe] @ y) / wsum[safe]
    return xg, yg


def _scatter_iter(ax, sub_iter: pd.DataFrame, metric: str):
    for label in SERIES_ORDER:
        s = sub_iter[sub_iter["label"] == label]
        if s.empty:
            continue
        color = SERIES_COLOR[label]
        ax.scatter(s["length"], s[metric], s=18,
                   color=color, alpha=0.30,
                   edgecolors="none", label=label, zorder=2)
        x = s["length"].to_numpy(dtype=float)
        y = s[metric].to_numpy(dtype=float)
        order = np.argsort(x)
        xg, yg = _gaussian_trend(x[order], y[order])
        ax.plot(xg, yg, color=color, linewidth=2.4, zorder=4)
    ax.grid(True, alpha=0.3)


def _iter_dec_str(it: int) -> str:
    return ", ".join(f"{lbl.split()[0]}={SERIES[lbl][2][it]}"
                     for lbl in SERIES_ORDER)


def plot_grid(records: pd.DataFrame, dataset: str, metric: str, out_path: Path):
    sub = records[records["dataset"] == dataset]
    if sub.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    # ca_rmsd: low is good (data low) -> text upper-right
    # tm_score: high is good (data high) -> text lower-right
    if metric == "tm_score":
        text_loc, ha, va = (0.99, 0.02), "right", "bottom"
    else:
        text_loc, ha, va = (0.99, 0.98), "right", "top"
    for ax, it in zip(axes, ITERS):
        sub_it = sub[sub["iter"] == it]
        _scatter_iter(ax, sub_it, metric)
        ax.set_title(f"iter={it}  ({_iter_dec_str(it)})", fontsize=9)
        _counts_textbox(ax, sub_it, metric, loc=text_loc, ha=ha, va=va)
    for ax in axes[-3:]:
        ax.set_xlabel("protein length")
    for ax in axes[::3]:
        ax.set_ylabel(metric)
    fig.legend(handles=_series_legend_handles(), loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=10, frameon=False)
    fig.suptitle(f"{dataset} | {metric} vs protein length "
                 f"(per-protein BoN, N=10 seeds)",
                 y=1.005, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_iter(records: pd.DataFrame, dataset: str, metric: str,
                  iter_n: int, out_path: Path):
    sub = records[(records["dataset"] == dataset) & (records["iter"] == iter_n)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_iter(ax, sub, metric)
    ax.set_xlabel("protein length")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset} | {metric} vs protein length | iter={iter_n} | "
                 f"BoN N=10\n({_iter_dec_str(iter_n)})", fontsize=10)
    if metric == "tm_score":
        legend_loc = "lower right"
    else:
        legend_loc = "upper right"
    ax.legend(handles=_legend_handles_with_counts(sub, metric), loc=legend_loc,
              fontsize=8, framealpha=0.9, prop={"family": "monospace",
                                                "size": 8})
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect()
    csv_path = OUT_DIR / "per_protein.csv"
    records.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(records)} rows)")

    for ds in DATASETS:
        for metric in METRICS:
            out = OUT_DIR / f"{ds}__{metric}.png"
            plot_grid(records, ds, metric, out)
            print(f"wrote {out}")
            for it in ITERS:
                out_it = OUT_DIR / f"{ds}__{metric}__iter{it}.png"
                plot_per_iter(records, ds, metric, it, out_it)
                print(f"wrote {out_it}")


if __name__ == "__main__":
    main()
