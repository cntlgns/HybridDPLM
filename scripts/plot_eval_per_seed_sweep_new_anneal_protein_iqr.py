"""
Compare the three method ckpts (oh-xhigh hybrid, FT9 hybrid, cutL0 noise)
against the baseline FT ckpt, using the *new* high-min-T annealing schedules
(annealing@X:1.0). Layout follows
plot_eval_per_seed_sweep_oh_vs_hybrid_vs_baseline_protein_iqr.py:

  * Per protein, compute typical (mean or median across 10 seeds) and Q1/Q3.
  * Across proteins, plot agg(typical) bold, shaded agg(Q1)–agg(Q3), and a
    dotted BoN N=10 oracle curve.

Per-iter recommended decoding was picked automatically from each per-model
summary CSV by combined rank across {cameo2022, PDB_date, cath_4.2_all,
cath_4.3_all} x {ca_rmsd, tm_score} (typical=1.0 weight, BoN=0.5 weight). See
the SERIES dict for the resulting picks. Iters 1 and 3 are dropped since none
of the new annealing schedules were run there.

Datasets: cameo2022, PDB_date, cath_4.2_all, cath_4.3_all
Iters:    5, 10, 30, 100, 500 (only noise has 500)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = Path(f"{PROJECT_DIR}/generation-results")
OUT_ROOT = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/"
                f"new_anneal_protein_iqr")

ITERS = [5, 10, 30, 100, 500]
DATASETS = ["cameo2022", "PDB_date", "cath_4.2_all", "cath_4.3_all"]
METRICS = ["ca_rmsd", "tm_score"]
METRIC_COL = {"ca_rmsd": "ca_rmsd", "tm_score": "bb_tmscore"}
BON_FN = {"ca_rmsd": "min", "tm_score": "max"}

OH_LABEL = "hybrid(one-hot)"
HYBRID_LABEL = "hybrid(embed)"
NOISE_LABEL = "noise"
BASELINE_LABEL = "baseline(disc. FT)"

SERIES_ORDER = [OH_LABEL, HYBRID_LABEL, NOISE_LABEL, BASELINE_LABEL]
SERIES_COLOR = {
    OH_LABEL:       "#9467bd",  # purple
    HYBRID_LABEL:   "#d62728",  # red
    NOISE_LABEL:    "#2ca02c",  # green
    BASELINE_LABEL: "#404040",  # dark gray
}

# (model_dir, ckpt_dir, recommended decoding by iter) — picks chosen by
# combined-rank across (dataset x metric) on the new annealing schedules in
# each per-model summary.csv.
SERIES = {
    OH_LABEL: (
        "oh-xhigh_noise-full-3e5_lr-fs0-normemb",
        "step_10256.0-loss_0.70",
        {5:   "annealing1.0_1.0",
         10:  "annealing2.0_1.0",
         30:  "annealing2.0_1.0",
         100: "annealing5.0_1.0"},
    ),
    HYBRID_LABEL: (
        "emb-high_noise-full-1e4_lr",
        "step_9843.0-loss_0.68",
        {5:   "annealing1.0_1.0",
         10:  "annealing2.0_1.0",
         30:  "annealing3.0_1.0",
         100: "annealing2.0_1.0"},
    ),
    NOISE_LABEL: (
        "noise-full-1e4_lr-fs0-cutL0_freeze",
        "step_8202.0-loss_0.70",
        {5:   "annealing1.0_1.0",
         10:  "annealing2.0_1.0",
         30:  "annealing1.0_1.0",
         100: "annealing2.0_1.0",
         500: "annealing5.0_1.0"},
    ),
    BASELINE_LABEL: (
        "baseline-full-1e4_lr-fs0_uncond",
        "step_6971.0-loss_0.22",
        {5:   "annealing1.0_1.0",
         10:  "annealing2.0_1.0",
         30:  "annealing2.0_1.0",
         100: "annealing2.0_1.0"},
    ),
}


_PROTEIN_RE = re.compile(r"eval/(length_\d+/[^/]+)/sample\.pdb")


def _protein_id(sample_path: str) -> str:
    m = _PROTEIN_RE.search(sample_path)
    return m.group(1) if m else sample_path


def _load_strategy_df(label: str, it: int, ds: str):
    model_dir, ckpt_dir, dec_by_iter = SERIES[label]
    if it not in dec_by_iter:
        return None, None
    dec = dec_by_iter[it]
    csv = (GEN_ROOT / f"eval_per_seed_sweep_{it}iter" / model_dir /
           ds / ckpt_dir / dec / "per_sample_by_seed.csv")
    if not csv.exists():
        print(f"[warn] missing {csv}")
        return None, dec
    df = pd.read_csv(csv)
    if df.empty:
        return None, dec
    df = df[(df["ca_rmsd"] < 100) & (df["bb_tmscore"] > 0)]
    if df.empty:
        return None, dec
    df = df.copy()
    df["protein_id"] = df["sample_path"].apply(_protein_id)
    return df, dec


def _per_protein_stats(df: pd.DataFrame, metric: str, agg: str):
    col = METRIC_COL[metric]
    fn_bon = BON_FN[metric]
    grp = df.groupby("protein_id")[col]
    typ = grp.mean() if agg == "mean" else grp.median()
    q1 = grp.quantile(0.25)
    q3 = grp.quantile(0.75)
    bon = grp.agg(fn_bon)
    return pd.DataFrame({"typ": typ, "q1": q1, "q3": q3, "bon": bon})


def _agg_across_proteins(stats: pd.DataFrame, agg: str) -> dict:
    fn = (lambda s: float(s.mean())) if agg == "mean" else (lambda s: float(s.median()))
    return {
        "typ":  fn(stats["typ"]),
        "q1":   fn(stats["q1"]),
        "q3":   fn(stats["q3"]),
        "bon":  fn(stats["bon"]),
        "n_proteins": int(len(stats)),
    }


def collect():
    out = {}
    for label in SERIES:
        for it in ITERS:
            for ds in DATASETS:
                df, dec = _load_strategy_df(label, it, ds)
                if df is None:
                    continue
                for metric in METRICS:
                    for agg in ("median", "mean"):
                        stats = _per_protein_stats(df, metric, agg)
                        rec = _agg_across_proteins(stats, agg)
                        rec["decoding"] = dec
                        out[(label, it, ds, metric, agg)] = rec
    return out


def plot_one(records, dataset, metric, agg, out_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    iter_to_pos = {it: i for i, it in enumerate(ITERS)}

    for label in SERIES_ORDER:
        color = SERIES_COLOR[label]
        xs, typs, q1s, q3s, bons, decs = [], [], [], [], [], []
        for it in ITERS:
            key = (label, it, dataset, metric, agg)
            if key not in records:
                continue
            r = records[key]
            xs.append(iter_to_pos[it])
            typs.append(r["typ"])
            q1s.append(r["q1"])
            q3s.append(r["q3"])
            bons.append(r["bon"])
            decs.append(r["decoding"])

        if not xs:
            continue

        ax.fill_between(xs, q1s, q3s, facecolor=color, alpha=0.22,
                        edgecolor=color, linewidth=0.8, zorder=1)
        ax.plot(xs, typs, color=color, linewidth=2.4, zorder=3,
                solid_capstyle="round")
        ax.scatter(xs, typs, color=color, s=40, marker="o", zorder=4)
        ax.plot(xs, bons, color=color, linewidth=1.4, linestyle=":", zorder=2)
        ax.scatter(xs, bons, color=color, s=40, marker="o", zorder=4)

        # Annotate the recommended decoding tag next to each point.
        for x, y, d in zip(xs, typs, decs):
            short = d.replace("annealing", "a").replace("_", "/")
            ax.annotate(short, xy=(x, y), xytext=(4, 6),
                        textcoords="offset points",
                        fontsize=7, color=color, alpha=0.9)

    ax.set_xticks(list(iter_to_pos.values()))
    ax.set_xticklabels([str(it) for it in ITERS])
    ax.set_xlim(-0.4, len(ITERS) - 0.6)
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} ({agg} across proteins)")

    ax.set_title(
        f"new annealing (per-iter best) | {dataset} | {metric} "
        f"({agg} across proteins)\n"
        f"bold = {agg}(per-protein {agg} across seeds); "
        f"shaded = {agg}(per-protein Q1)–{agg}(per-protein Q3); "
        f"dotted = BoN N=10",
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.3, axis="y")

    handles = []
    for label in SERIES_ORDER:
        c = SERIES_COLOR[label]
        handles.append(Line2D([0], [0], marker="o", color=c, markerfacecolor=c,
                              markeredgecolor=c, markersize=8, label=label))
        handles.append(Line2D([0], [0], color=c, linewidth=8, alpha=0.22,
                              label=f"  └ {agg}(per-protein Q1)–{agg}(per-protein Q3)"))
    handles += [
        Line2D([0], [0], color="k", linewidth=2.4,
               label=f"{agg}(per-protein {agg} across seeds)"),
        Line2D([0], [0], color="k", linewidth=1.4, linestyle=":",
               label="BoN N=10 (oracle best across seeds)"),
    ]
    legend_loc = "upper right" if metric == "ca_rmsd" else "lower right"
    ax.legend(handles=handles, loc=legend_loc, fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def dump_summary(records, out_csv: Path):
    rows = []
    for (label, it, ds, metric, agg), v in records.items():
        rows.append({
            "label": label, "iter": it, "dataset": ds,
            "metric": metric, "agg": agg,
            "decoding": v["decoding"], "n_proteins": v["n_proteins"],
            "typ": v["typ"], "q1": v["q1"], "q3": v["q3"], "bon": v["bon"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = collect()
    dump_summary(records, OUT_ROOT / "summary.csv")
    print(f"wrote {OUT_ROOT / 'summary.csv'} ({len(records)} entries)")

    for ds in DATASETS:
        for metric in METRICS:
            for agg in ("median", "mean"):
                out = OUT_ROOT / f"{ds}__{metric}__{agg}.png"
                plot_one(records, ds, metric, agg, out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
