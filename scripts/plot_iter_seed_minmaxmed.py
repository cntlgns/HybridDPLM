"""
Plot iter-vs-metric (ca_rmsd, tm_score) comparing 4 methods. For each method
and iteration, 10 seeds were generated/evaluated against the same dataset
samples, so each (method, iter, sample) has a min/max/median across seeds.
We aggregate per-sample stats by averaging across the dataset's samples and
plot the mean per-sample-min, per-sample-median, per-sample-max as a function
of iter.

Layout (rooted at PROJECT_DIR/generation-results):
  - hybrid (emb-high, emb-xhigh):
      iter=1   -> hybrid_invfold_FT7final_refine_1iter_seedsweep
      iter=30  -> hybrid_invfold_FT7final_noODEstep_30iter_seedsweep
      iter=100 -> hybrid_invfold_FT7final_noODEstep_100iter_seedsweep
      <root>/<method_dir>/<dataset>/<ckpt>/per_sample_by_seed.csv
  - dplm2_650m (uncond, no_remask):
      <PROJECT>/dplm2_650m_invfold_seedsweep/<dataset>/annealing@2.0:0.1_<variant>_iter<N>/per_sample_by_seed.csv

Outputs to PROJECT_DIR/analysis/iter_seed_minmaxmed/:
  - per_sample_stats.csv           (one row per method,iter,dataset,sample)
  - dataset_aggregates.csv         (one row per method,iter,dataset)
  - plots: ca_rmsd__<dataset>.png, tm_score__<dataset>.png
"""
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = f"{PROJECT_DIR}/generation-results"
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/iter_seed_minmaxmed")

ITERS = [1, 10, 30, 100]
DATASETS = ["PDB_date", "cameo2022"]
METRICS = {
    "ca_rmsd": "ca_rmsd",       # display -> column in per_sample_by_seed.csv
    "tm_score": "bb_tmscore",
}


def hybrid_root(it: int) -> str:
    # noODEstep variants only exist for 30/100; iter=1/10 fall back to refine
    # (the strategy distinction matters less at very low iter counts).
    if it in (1, 10):
        sub = f"hybrid_invfold_FT7final_refine_{it}iter_seedsweep"
    else:
        sub = f"hybrid_invfold_FT7final_noODEstep_{it}iter_seedsweep"
    return f"{GEN_ROOT}/{sub}"


# (label, builder(dataset, iter) -> path-to-per_sample_by_seed.csv)
METHODS = [
    (
        "hybrid_dplm2_650m",
        lambda ds, it: (
            f"{hybrid_root(it)}/emb-high_noise-full-ema_lr_745-fs0-normemb/"
            f"{ds}/step_9005.0-loss_0.67/per_sample_by_seed.csv"
        ),
    ),
    (
        "hybrid: emb-xhigh",
        lambda ds, it: (
            f"{hybrid_root(it)}/emb-xhigh_noise-full-low_lr-fs0-normemb/"
            f"{ds}/step_26017.0-loss_0.72/per_sample_by_seed.csv"
        ),
    ),
    (
        "dplm2_650m",
        lambda ds, it: (
            f"{GEN_ROOT}/dplm2_650m_invfold_seedsweep/{ds}/"
            f"annealing@2.0:0.1_uncond_iter{it}/per_sample_by_seed.csv"
        ),
    ),
    (
        "dplm2_650m no_remask",
        lambda ds, it: (
            f"{GEN_ROOT}/dplm2_650m_invfold_seedsweep/{ds}/"
            f"annealing@2.0:0.1_no_remask_iter{it}/per_sample_by_seed.csv"
        ),
    ),
]
# methods to skip in plots (kept in CSVs for reference)
PLOT_EXCLUDE = {"dplm2_650m no_remask", "hybrid: emb-xhigh"}
# explicit color overrides for plotted methods
METHOD_COLOR_OVERRIDE = {
    "hybrid_dplm2_650m": "tab:red",
    "dplm2_650m": "tab:blue",
}
# metrics where lower is better (legends should sit in the upper region)
LOWER_IS_BETTER = {"ca_rmsd"}

# expected number of seeds; samples with fewer are still kept but logged
EXPECTED_NUM_SEEDS = 10

SAMPLE_ID_RE = re.compile(r"length_(\d+)/([^/]+)/sample\.pdb")


def extract_sample_id(path: str) -> str:
    """Pull a stable per-sample id (e.g. '243/7YK9') from a sample.pdb path."""
    m = SAMPLE_ID_RE.search(path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # fallback: parent dir + parent-of-parent
    parts = path.rsplit("/", 3)
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else path


def per_sample_minmaxmed(df: pd.DataFrame) -> pd.DataFrame:
    """Group by sample_id and compute min/max/median for each metric across seeds."""
    df = df.copy()
    df["sample_id"] = df["sample_path"].astype(str).map(extract_sample_id)
    rows = []
    grouped = df.groupby("sample_id", sort=False)
    for sid, g in grouped:
        row = {"sample_id": sid, "n_seeds": len(g)}
        for disp, col in METRICS.items():
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
            if len(vals):
                row[f"{disp}_min"] = float(np.min(vals))
                row[f"{disp}_max"] = float(np.max(vals))
                row[f"{disp}_median"] = float(np.median(vals))
            else:
                row[f"{disp}_min"] = np.nan
                row[f"{disp}_max"] = np.nan
                row[f"{disp}_median"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_sample_records = []
    aggregate_records = []

    for method_label, path_fn in METHODS:
        for ds in DATASETS:
            for it in ITERS:
                csv_path = path_fn(ds, it)
                if not os.path.exists(csv_path):
                    print(f"[skip] missing: {csv_path}")
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"[error] could not read {csv_path}: {e}")
                    continue
                if "sample_path" not in df.columns:
                    print(f"[error] no sample_path column in {csv_path}")
                    continue

                stats = per_sample_minmaxmed(df)
                low_seed = stats[stats["n_seeds"] < EXPECTED_NUM_SEEDS]
                if len(low_seed):
                    print(
                        f"[warn] {method_label} | {ds} | iter={it}: "
                        f"{len(low_seed)}/{len(stats)} samples have <{EXPECTED_NUM_SEEDS} seeds"
                    )

                stats.insert(0, "iter", it)
                stats.insert(0, "dataset", ds)
                stats.insert(0, "method", method_label)
                per_sample_records.append(stats)

                agg_row = {"method": method_label, "dataset": ds, "iter": it,
                           "n_samples": len(stats)}
                for disp in METRICS:
                    for stat in ("min", "max", "median"):
                        col = f"{disp}_{stat}"
                        agg_row[f"{col}__mean_over_samples"] = float(stats[col].mean())
                        agg_row[f"{col}__median_over_samples"] = float(stats[col].median())
                aggregate_records.append(agg_row)

    if not per_sample_records:
        print("No data found; aborting.")
        return

    per_sample_df = pd.concat(per_sample_records, ignore_index=True)
    agg_df = pd.DataFrame(aggregate_records)

    per_sample_df.to_csv(OUT_DIR / "per_sample_stats.csv", index=False)
    agg_df.to_csv(OUT_DIR / "dataset_aggregates.csv", index=False)
    print(f"Wrote {OUT_DIR/'per_sample_stats.csv'} ({len(per_sample_df)} rows)")
    print(f"Wrote {OUT_DIR/'dataset_aggregates.csv'} ({len(agg_df)} rows)")

    # ---------------- plots ----------------
    method_labels = [m[0] for m in METHODS if m[0] not in PLOT_EXCLUDE]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_color = {}
    fallback_idx = 0
    for m in method_labels:
        if m in METHOD_COLOR_OVERRIDE:
            method_color[m] = METHOD_COLOR_OVERRIDE[m]
        else:
            method_color[m] = color_cycle[fallback_idx % len(color_cycle)]
            fallback_idx += 1

    # 8 plots total: 2 datasets x 2 metrics x 2 sample-aggregations (mean/median)
    # Each plot: x = iter; one color per method; per-method 3 lines:
    #   solid  = per-sample MEDIAN across seeds (aggregated over samples)
    #   dotted = per-sample MIN  across seeds (aggregated over samples)
    #   dotted = per-sample MAX  across seeds (aggregated over samples)
    # The "aggregated over samples" step is mean (..._mean plot) or median
    # (..._median plot).
    from matplotlib.lines import Line2D

    SAMPLE_AGGS = ("mean", "median")  # how per-sample stats are reduced over samples

    for ds in DATASETS:
        for disp in METRICS:
            for sample_agg in SAMPLE_AGGS:
                col_suffix = f"{sample_agg}_over_samples"
                fig, ax = plt.subplots(figsize=(7.5, 5.0))
                for method in method_labels:
                    sub = (agg_df[(agg_df["method"] == method) & (agg_df["dataset"] == ds)]
                           .sort_values("iter"))
                    if sub.empty:
                        continue
                    xs = sub["iter"].to_numpy()
                    med = sub[f"{disp}_median__{col_suffix}"].to_numpy()
                    lo = sub[f"{disp}_min__{col_suffix}"].to_numpy()
                    hi = sub[f"{disp}_max__{col_suffix}"].to_numpy()
                    color = method_color[method]
                    ax.plot(xs, med, color=color, linewidth=2.0,
                            linestyle="-", label=method)
                    ax.plot(xs, lo, color=color, linewidth=1.0,
                            linestyle=":", alpha=0.9)
                    ax.plot(xs, hi, color=color, linewidth=1.0,
                            linestyle=":", alpha=0.9)

                style_handles = [
                    Line2D([0], [0], color="black", linestyle="-", linewidth=2.0,
                           label="seed-median"),
                    Line2D([0], [0], color="black", linestyle=":", linewidth=1.0,
                           label="seed-min / seed-max"),
                ]
                # For lower-is-better metrics the meaningful values cluster near
                # the bottom — keep both legends in the upper region. Otherwise
                # (higher-is-better) place them at the bottom.
                if disp in LOWER_IS_BETTER:
                    method_loc, style_loc = "upper left", "upper right"
                else:
                    method_loc, style_loc = "lower left", "lower right"
                method_legend = ax.legend(fontsize=8, loc=method_loc, title="method")
                ax.add_artist(method_legend)
                ax.legend(handles=style_handles, fontsize=8, loc=style_loc,
                          title="line style")

                ax.set_xscale("log")
                ax.set_xticks(ITERS)
                ax.set_xticklabels([str(i) for i in ITERS])
                ax.set_xlabel("iterations")
                ax.set_ylabel(f"{disp} ({sample_agg} over samples)")
                ax.set_title(f"{ds} — {disp} ({sample_agg})")
                ax.grid(True, which="both", alpha=0.3)
                fig.tight_layout()
                out_path = OUT_DIR / f"{disp}_{sample_agg}__{ds}.png"
                fig.savefig(out_path, dpi=160)
                plt.close(fig)
                print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
