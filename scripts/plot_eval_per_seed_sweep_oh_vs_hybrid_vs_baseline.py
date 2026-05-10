"""
Compare emb-hybrid (hybrid plain, recommended) vs oh-xhigh (recommended) vs
baseline (max-T anneal) — three series, one plot per (dataset, metric, agg).

Data sources:
  - emb-hybrid + baseline: analysis/eval_per_seed_sweep/recommended/summary.csv
  - oh-xhigh: analysis/eval_per_seed_sweep/oh/summary.csv (filtered to OH_PICKS)

oh-xhigh per-iter pick (combined-rank across {ca_rmsd, tm_score} x
{cameo2022, PDB_date} x {median, mean, best_median, best_mean}; iter 3 is a
manual override — combined-rank winner there was 0.5_0.1):
    iter 1   -> annealing0.5_0.1
    iter 3   -> annealing1.0_0.1
    iter 5   -> annealing0.7_0.1
    iter 10  -> annealing0.3_0.1
    iter 30  -> argmax
    iter 100 -> annealing4.0_0.1

8 plots: 2 datasets x {ca_rmsd, tm_score} x {median, mean}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
OH_CSV = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/oh/summary.csv")
REC_CSV = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/recommended/summary.csv")
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/oh_vs_hybrid_vs_baseline")

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]
METRICS = ["ca_rmsd", "tm_score"]

OH_PICKS = {
    1:   "annealing0.5_0.1",
    3:   "annealing1.0_0.1",
    5:   "annealing0.7_0.1",
    10:  "annealing0.3_0.1",
    30:  "argmax",
    100: "annealing4.0_0.1",
}

OH_LABEL = "oh-xhigh (recommended)"
OH_COLOR = "#9467bd"   # purple
HYBRID_LABEL = "hybrid plain (recommended)"
HYBRID_COLOR = "#d62728"  # red
BASELINE_LABEL = "baseline (max-T anneal)"
BASELINE_COLOR = "#1f77b4"  # blue

SERIES_ORDER = [OH_LABEL, HYBRID_LABEL, BASELINE_LABEL]
SERIES_COLOR = {OH_LABEL: OH_COLOR, HYBRID_LABEL: HYBRID_COLOR,
                BASELINE_LABEL: BASELINE_COLOR}


def load_oh_series():
    df = pd.read_csv(OH_CSV)
    rows = []
    for it, dec in OH_PICKS.items():
        sub = df[(df["iter"] == it) & (df["decoding"] == dec)]
        for _, r in sub.iterrows():
            rows.append({
                "label": OH_LABEL, "color": OH_COLOR,
                "decoding": dec, "iter": it,
                "dataset": r["dataset"], "metric": r["metric"],
                "median": r["median"], "mean": r["mean"],
                "best_median": r["best_median"], "best_mean": r["best_mean"],
            })
    return pd.DataFrame(rows)


def load_recommended_series(label, color):
    df = pd.read_csv(REC_CSV)
    df = df[(df["label"] == label) & (df["iter"].isin(ITERS))].copy()
    df["color"] = color
    return df[["label", "color", "decoding", "iter", "dataset", "metric",
               "median", "mean", "best_median", "best_mean"]]


def _short_dec(d):
    if d == "argmax":
        return "argm"
    return d.replace("annealing", "a").replace("_", "/")


def plot_one(records, dataset, metric, agg, out_path):
    sub = records[(records["dataset"] == dataset) & (records["metric"] == metric)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))

    typ_col = agg
    ora_col = f"best_{agg}"
    iter_to_pos = {it: i for i, it in enumerate(ITERS)}

    for label in SERIES_ORDER:
        s = sub[sub["label"] == label].sort_values("iter")
        if s.empty:
            continue
        color = SERIES_COLOR[label]
        xs = [iter_to_pos[it] for it in s["iter"]]
        ys_typ = s[typ_col].tolist()
        ys_ora = s[ora_col].tolist()
        decs = s["decoding"].tolist()

        ax.plot(xs, ys_typ, color=color, linewidth=1.5, zorder=2)
        ax.plot(xs, ys_ora, color=color, linewidth=1.2, linestyle="--", zorder=2)
        ax.scatter(xs, ys_typ, color=color, s=70, marker="o",
                   facecolors=color, edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(xs, ys_ora, color=color, s=70, marker="o",
                   facecolors="none", edgecolors=color, linewidths=1.6, zorder=3)

        for x, y, d in zip(xs, ys_ora, decs):
            ax.annotate(_short_dec(d), xy=(x, y),
                        xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color=color)

    ax.set_xticks(list(iter_to_pos.values()))
    ax.set_xticklabels([str(it) for it in ITERS])
    ax.set_xlim(-0.4, len(ITERS) - 0.6)
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} ({agg} across proteins)")
    ax.set_title(f"oh-xhigh vs emb-hybrid vs baseline (recommended) | {dataset} | "
                 f"{metric} | {agg} (filled) & best_{agg} (hollow)")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.lines import Line2D
    handles = []
    for label in SERIES_ORDER:
        c = SERIES_COLOR[label]
        handles.append(Line2D([0], [0], marker="o", color=c, markerfacecolor=c,
                              markeredgecolor="black", markersize=9, label=label))
    handles += [
        Line2D([0], [0], marker="o", color="k", markerfacecolor="k",
               markersize=9, linestyle="None", label=f"{agg} (typical)"),
        Line2D([0], [0], marker="o", color="k", markerfacecolor="none",
               markersize=9, linestyle="None", label=f"best_{agg} (BoN N=10 oracle)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = pd.concat([
        load_oh_series(),
        load_recommended_series(HYBRID_LABEL, HYBRID_COLOR),
        load_recommended_series(BASELINE_LABEL, BASELINE_COLOR),
    ], ignore_index=True)
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
