"""
Single-model per-seed sweep plot: emb-1high_noise-full-1e4_lr-linear (FT10).

Layout mirrors plot_eval_per_seed_sweep_freeze_vs_redraw_linear.py for a single
model: each iter slot shows argmax + every available annealing decoding, with
filled = typical (median/mean across proteins) and hollow = oracle best per
protein.

8 plots: 2 datasets x {ca_rmsd, tm_score} x {median, mean}.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = Path(f"{PROJECT_DIR}/generation-results")
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/emb1high")

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]

MODEL_DIR = "emb-1high_noise-full-1e4_lr-linear"
CKPT = "step_15995.0-loss_0.21"
TITLE_LABEL = "emb-1high (lr-linear)"

METRICS = {
    "ca_rmsd":  ("ca_rmsd",   "min"),
    "tm_score": ("bb_tmscore", "max"),
}

ANNEAL_RE = re.compile(r"annealing([0-9.]+)_([0-9.]+)")


def _anneal_temp(name):
    m = ANNEAL_RE.match(name)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def protein_id(sample_path: str) -> str:
    m = re.search(r"eval/(length_\d+/[^/]+)/sample\.pdb", sample_path)
    return m.group(1) if m else sample_path


def collect_records():
    rows = []
    for it in ITERS:
        root = GEN_ROOT / f"eval_per_seed_sweep_{it}iter"
        if not root.exists():
            continue
        for ds in DATASETS:
            ckpt_dir = root / MODEL_DIR / ds / CKPT
            if not ckpt_dir.exists():
                continue
            for dec_dir in sorted(ckpt_dir.iterdir()):
                if not dec_dir.is_dir():
                    continue
                csv_path = dec_dir / "per_sample_by_seed.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                df["protein_id"] = df["sample_path"].apply(protein_id)
                for mname, (col, direction) in METRICS.items():
                    if col not in df.columns:
                        continue
                    per_protein_mean = df.groupby("protein_id")[col].mean()
                    if direction == "min":
                        per_protein_best = df.groupby("protein_id")[col].min()
                    else:
                        per_protein_best = df.groupby("protein_id")[col].max()
                    rows.append({
                        "iter": it,
                        "dataset": ds,
                        "decoding": dec_dir.name,
                        "metric": mname,
                        "median":      per_protein_mean.median(),
                        "mean":        per_protein_mean.mean(),
                        "best_median": per_protein_best.median(),
                        "best_mean":   per_protein_best.mean(),
                        "n_proteins":  per_protein_mean.shape[0],
                        "n_seeds":     int(df["seed"].nunique()),
                    })
    return pd.DataFrame(rows)


def _decoding_sort_key(d):
    if d == "argmax":
        return (0, 0.0, 0.0)
    t = _anneal_temp(d)
    if t is None:
        return (2, 0.0, 0.0)
    return (1, t[0], t[1])


def _decoding_color(d, anneal_count, anneal_idx):
    if d == "argmax":
        return "#000000"
    if anneal_count <= 1:
        return plt.cm.viridis(0.5)
    norm = anneal_idx / (anneal_count - 1)
    return plt.cm.viridis(0.15 + 0.70 * norm)


def _decoding_label(d):
    if d == "argmax":
        return "argmax"
    t = _anneal_temp(d)
    return f"{t[0]:g}_{t[1]:g}"


SLOT_WIDTH = 0.80


def plot_one(records: pd.DataFrame, dataset: str, metric: str, agg: str,
             out_path: Path):
    sub = records[(records["dataset"] == dataset) & (records["metric"] == metric)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6.5))

    typical_col = agg
    best_col = f"best_{agg}"
    iter_to_pos = {it: i for i, it in enumerate(ITERS)}

    for it, isub in sub.groupby("iter"):
        decs = sorted(isub["decoding"].unique(), key=_decoding_sort_key)
        n = len(decs)
        if n == 0:
            continue
        if n == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-SLOT_WIDTH / 2, SLOT_WIDTH / 2, n)

        anneal_decs = [d for d in decs if d != "argmax"]
        anneal_n = len(anneal_decs)
        base_x = iter_to_pos[it]

        for i, d in enumerate(decs):
            x = base_x + offsets[i]
            row = isub[isub["decoding"] == d].iloc[0]
            if d == "argmax":
                color = _decoding_color(d, anneal_n, 0)
            else:
                a_idx = anneal_decs.index(d)
                color = _decoding_color(d, anneal_n, a_idx)

            ax.scatter([x], [row[typical_col]], color=color, marker="o", s=70,
                       facecolors=color, edgecolors="black", linewidths=0.5,
                       zorder=3)
            ax.scatter([x], [row[best_col]], color=color, marker="o", s=70,
                       facecolors="none", edgecolors=color, linewidths=1.6,
                       zorder=3)
            ax.plot([x, x], [row[typical_col], row[best_col]],
                    color=color, alpha=0.5, linewidth=1, zorder=1)

            y_low = min(row[typical_col], row[best_col])
            ax.annotate(_decoding_label(d), xy=(x, y_low),
                        xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=7,
                        color=color, rotation=90)

    for i in range(len(ITERS) - 1):
        ax.axvline(i + 0.5, color="0.80", linewidth=0.9, zorder=0)

    ax.set_xticks(list(iter_to_pos.values()))
    ax.set_xticklabels([str(it) for it in ITERS])
    ax.set_xlim(-0.6, len(ITERS) - 0.4)
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} ({agg} across proteins)")
    ax.set_title(f"{TITLE_LABEL} | {dataset} | {metric} | "
                 f"{agg} (filled) & best_{agg} (hollow)")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markeredgecolor="black", markersize=9, label="argmax"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=plt.cm.viridis(0.15),
               markeredgecolor=plt.cm.viridis(0.15),
               markersize=9, label="annealing low T"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=plt.cm.viridis(0.85),
               markeredgecolor=plt.cm.viridis(0.85),
               markersize=9, label="annealing high T"),
        Line2D([0], [0], marker="o", color="k", markerfacecolor="k",
               markersize=9, linestyle="None", label=f"{agg} (typical)"),
        Line2D([0], [0], marker="o", color="k", markerfacecolor="none",
               markersize=9, linestyle="None", label=f"best_{agg} (oracle)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    csv_path = OUT_DIR / "summary.csv"
    records.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(records)} rows)")

    for ds in DATASETS:
        for metric in METRICS:
            for agg in ("median", "mean"):
                out = OUT_DIR / f"{ds}__{metric}__{agg}.png"
                plot_one(records, ds, metric, agg, out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
