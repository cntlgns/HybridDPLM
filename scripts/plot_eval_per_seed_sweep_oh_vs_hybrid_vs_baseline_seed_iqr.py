"""
Per-seed mean/IQR comparison vs BoN(N=10) for the recommended decoding picks.

Two pairwise comparisons are produced (each shares the baseline series):
  - oh_vs_baseline:  oh-xhigh (recommended)  vs  baseline (max-T anneal)
  - emb_vs_baseline: hybrid plain (recommended)  vs  baseline (max-T anneal)

For each (label, iter, dataset) we load the strategy's per_sample_by_seed.csv
and:
  * compute one value per seed = aggregate (mean or median) of the metric
    across proteins
  * across seeds: mean (bold line) and IQR Q1-Q3 (shaded band)
  * BoN N=10 dotted line = aggregate across proteins of the per-protein oracle
    (min for ca_rmsd, max for tm_score across the 10 seeds)

8 plots per comparison: 2 datasets x {ca_rmsd, tm_score} x {median, mean}.

Per-iter recommended decoding picks mirror
plot_eval_per_seed_sweep_oh_vs_hybrid_vs_baseline_by_length.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


_PROTEIN_RE = re.compile(r"eval/(length_\d+/[^/]+)/sample\.pdb")


def _protein_id(sample_path: str) -> str:
    m = _PROTEIN_RE.search(sample_path)
    return m.group(1) if m else sample_path


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = Path(f"{PROJECT_DIR}/generation-results")
OUT_ROOT = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/"
                f"oh_vs_hybrid_vs_baseline_seed_iqr")

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]
METRICS = ["ca_rmsd", "tm_score"]  # name in summary; column in csv differs for tm
METRIC_COL = {"ca_rmsd": "ca_rmsd", "tm_score": "bb_tmscore"}
BON_FN = {"ca_rmsd": "min", "tm_score": "max"}

OH_LABEL = "hybrid(one-hot)"
HYBRID_LABEL = "hybrid(embed)"
BASELINE_LABEL = "baseline(disc. FT)"

OH_COLOR = "#d62728"
HYBRID_COLOR = "#d62728"
BASELINE_COLOR = "#404040"

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

COMPARISONS = {
    "oh_vs_baseline":  [OH_LABEL, BASELINE_LABEL],
    "emb_vs_baseline": [HYBRID_LABEL, BASELINE_LABEL],
}


def _load_strategy_df(label: str, it: int, ds: str) -> pd.DataFrame | None:
    model_dir, ckpt_dir, dec_by_iter = SERIES[label]
    dec = dec_by_iter[it]
    csv = (GEN_ROOT / f"eval_per_seed_sweep_{it}iter" / model_dir /
           ds / ckpt_dir / dec / "per_sample_by_seed.csv")
    if not csv.exists():
        print(f"[warn] missing {csv}")
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    # Drop folding-failure sentinels (ca_rmsd=100, tm_score=0) so a single
    # bad protein does not dominate per-seed means.
    df = df[(df["ca_rmsd"] < 100) & (df["bb_tmscore"] > 0)]
    if df.empty:
        return None
    df = df.copy()
    df["protein_id"] = df["sample_path"].apply(_protein_id)
    return df


def _agg_per_seed(df: pd.DataFrame, metric: str, agg: str) -> np.ndarray:
    """Per-seed scalar = agg(metric_col across proteins). One row per seed."""
    col = METRIC_COL[metric]
    grp = df.groupby("seed")[col]
    return (grp.mean() if agg == "mean" else grp.median()).to_numpy(dtype=float)


def _bon(df: pd.DataFrame, metric: str, agg: str) -> float:
    """BoN N=10: per-protein oracle across seeds, then agg across proteins."""
    col = METRIC_COL[metric]
    fn = BON_FN[metric]
    # protein_id grouping; sample_path is unique per protein
    bon = df.groupby("protein_id")[col].agg(fn)
    return float(bon.mean() if agg == "mean" else bon.median())


def collect():
    """Return dict[(label, it, ds, metric, agg)] -> dict with seed_values, bon, dec."""
    out = {}
    for label in SERIES:
        for it in ITERS:
            for ds in DATASETS:
                df = _load_strategy_df(label, it, ds)
                if df is None:
                    continue
                dec = SERIES[label][2][it]
                for metric in METRICS:
                    for agg in ("median", "mean"):
                        seed_vals = _agg_per_seed(df, metric, agg)
                        bon = _bon(df, metric, agg)
                        out[(label, it, ds, metric, agg)] = {
                            "seed_values": seed_vals,
                            "bon": bon,
                            "decoding": dec,
                        }
    return out


def plot_one(records, comparison_labels, dataset, metric, agg, out_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    iter_to_pos = {it: i for i, it in enumerate(ITERS)}

    for label in comparison_labels:
        color = SERIES_COLOR[label]
        xs, means, q1s, q3s, bons = [], [], [], [], []
        for it in ITERS:
            key = (label, it, dataset, metric, agg)
            if key not in records:
                continue
            sv = records[key]["seed_values"]
            if len(sv) == 0:
                continue
            xs.append(iter_to_pos[it])
            means.append(float(np.mean(sv)))
            q1s.append(float(np.quantile(sv, 0.25)))
            q3s.append(float(np.quantile(sv, 0.75)))
            bons.append(records[key]["bon"])

        if not xs:
            continue

        ax.fill_between(xs, q1s, q3s, facecolor=color, alpha=0.28,
                        edgecolor=color, linewidth=0.8, zorder=1)
        ax.plot(xs, means, color=color, linewidth=2.6, zorder=3,
                solid_capstyle="round")
        ax.scatter(xs, means, color=color, s=40, marker="o", zorder=4)
        ax.plot(xs, bons, color=color, linewidth=1.5, linestyle=":",
                zorder=2)
        ax.scatter(xs, bons, color=color, s=40, marker="o", zorder=4)

    ax.set_xticks(list(iter_to_pos.values()))
    ax.set_xticklabels([str(it) for it in ITERS])
    ax.set_xlim(-0.4, len(ITERS) - 0.6)
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} ({agg} across proteins)")

    pretty = " vs ".join(comparison_labels)
    ax.set_title(
        f"{pretty} | {dataset} | {metric} ({agg} across proteins)\n"
        f"bold = mean across 10 seeds; shaded = IQR (Q1-Q3); "
        f"dotted = BoN N=10 oracle",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis="y")

    handles = []
    for label in comparison_labels:
        c = SERIES_COLOR[label]
        handles.append(Line2D([0], [0], marker="o", color=c, markerfacecolor=c,
                              markeredgecolor=c, markersize=8, label=label))
        handles.append(Line2D([0], [0], color=c, linewidth=8, alpha=0.28,
                              label=f"  └ IQR (Q1-Q3) across seeds"))
    handles += [
        Line2D([0], [0], color="k", linewidth=2.6, label="mean across seeds"),
        Line2D([0], [0], color="k", linewidth=1.5, linestyle=":",
               label="BoN N=10 (oracle best across seeds)"),
    ]
    legend_loc = "upper right" if metric == "ca_rmsd" else "lower right"
    ax.legend(handles=handles, loc=legend_loc, fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def dump_summary(records, out_csv: Path):
    rows = []
    for (label, it, ds, metric, agg), v in records.items():
        sv = v["seed_values"]
        rows.append({
            "label": label, "iter": it, "dataset": ds,
            "metric": metric, "agg": agg,
            "decoding": v["decoding"], "n_seeds": int(len(sv)),
            "seed_mean": float(np.mean(sv)),
            "seed_median": float(np.median(sv)),
            "seed_q1": float(np.quantile(sv, 0.25)),
            "seed_q3": float(np.quantile(sv, 0.75)),
            "seed_min": float(np.min(sv)),
            "seed_max": float(np.max(sv)),
            "bon": v["bon"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = collect()
    dump_summary(records, OUT_ROOT / "summary.csv")
    print(f"wrote {OUT_ROOT / 'summary.csv'} ({len(records)} entries)")

    for cmp_name, labels in COMPARISONS.items():
        sub_dir = OUT_ROOT / cmp_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        for ds in DATASETS:
            for metric in METRICS:
                for agg in ("median", "mean"):
                    out = sub_dir / f"{ds}__{metric}__{agg}.png"
                    plot_one(records, labels, ds, metric, agg, out)
                    print(f"wrote {out}")


if __name__ == "__main__":
    main()
