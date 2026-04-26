"""
Aggregate hybrid_invfold seedsweep per-seed results into a single CSV.

Expected layout under <root>:
    <root>/<ft_method>/<dataset>/<checkpoint>/per_sample_by_seed.csv
            (the per_sample_by_seed.csv is produced by
             scripts/aggregate_seed_metrics.py)

One row per (ft_method, checkpoint, seed-or-aggregate) with each dataset's
metrics side by side. For each dataset x metric, reports median and mean of
per-sample values. Aggregate rows (mean_across_seeds, std_across_seeds per
checkpoint) are placed at the bottom.

Usage:
    python scripts/aggregate_seedsweep.py <root> [--out PATH]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


METRIC_SRC = {
    "ca_rmsd": "ca_rmsd",
    "bb_rmsd": "bb_rmsd",
    "tm_score": "bb_tmscore",
    "plddt": "mean_plddt",
    "seq_recovery": "inv_fold_seq_recovery",
}
METRIC_ORDER = ["ca_rmsd", "bb_rmsd", "tm_score", "plddt", "seq_recovery"]
# Preferred ordering for well-known dataset names; unknown datasets are appended
# in alphabetical order after these.
DATASET_PRIORITY = ["PDB_date", "cameo2022"]


def per_seed_stats(df_seed, prefix):
    out = {}
    for name, src in METRIC_SRC.items():
        vals = pd.to_numeric(df_seed[src], errors="coerce").dropna().values
        out[f"{prefix}_{name}_median"] = float(np.median(vals)) if len(vals) else np.nan
        out[f"{prefix}_{name}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
    return out


def discover_datasets(root):
    names = set()
    for ft_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(ft_dir):
            continue
        for ds in os.listdir(ft_dir):
            if os.path.isdir(os.path.join(ft_dir, ds)):
                names.add(ds)
    ordered = [d for d in DATASET_PRIORITY if d in names]
    ordered += sorted(n for n in names if n not in DATASET_PRIORITY)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Sweep root, e.g. generation-results/hybrid_invfold_FT4_seedsweep")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <root>/seedsweep_summary.csv)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = args.out or os.path.join(root, "seedsweep_summary.csv")

    datasets = discover_datasets(root)
    if not datasets:
        raise SystemExit(f"No dataset subdirs found under {root}")

    stat_cols = [
        f"{ds}_{m}_{s}"
        for ds in datasets
        for m in METRIC_ORDER
        for s in ("median", "mean")
    ]
    n_sample_cols = [f"{ds}_n_samples" for ds in datasets]

    per_seed_out = []
    agg_out = []

    ft_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
    )
    for ft_dir in ft_dirs:
        ft = os.path.basename(ft_dir)

        # Collect checkpoints across datasets; key = checkpoint name,
        # value = {dataset: checkpoint_dir}
        ck_to_dsdir = {}
        for ds in datasets:
            ds_dir = os.path.join(ft_dir, ds)
            if not os.path.isdir(ds_dir):
                continue
            for ck in sorted(os.listdir(ds_dir)):
                ck_dir = os.path.join(ds_dir, ck)
                if os.path.isdir(ck_dir):
                    ck_to_dsdir.setdefault(ck, {})[ds] = ck_dir

        for ck in sorted(ck_to_dsdir):
            ds_paths = ck_to_dsdir[ck]

            per_sample = {}
            seed_set = None
            for ds, ck_dir in ds_paths.items():
                psv = os.path.join(ck_dir, "per_sample_by_seed.csv")
                if not os.path.exists(psv):
                    print(f"[warn] missing {psv}")
                    continue
                df = pd.read_csv(psv)
                per_sample[ds] = df
                seeds = set(df["seed"].dropna().astype(int).unique().tolist())
                seed_set = seeds if seed_set is None else (seed_set | seeds)
            if not per_sample:
                continue

            per_seed_rows = []
            for s in sorted(seed_set):
                row = {"ft_method": ft, "checkpoint": ck, "seed": str(s)}
                for ds in datasets:
                    if ds not in per_sample:
                        continue
                    sub = per_sample[ds][per_sample[ds]["seed"] == s]
                    row.update(per_seed_stats(sub, ds))
                    row[f"{ds}_n_samples"] = len(sub)
                per_seed_rows.append(row)
            per_seed_out.extend(per_seed_rows)

            seed_df = pd.DataFrame(per_seed_rows)
            for stat_name, fn in [
                ("mean_across_seeds", lambda x: np.mean(x)),
                ("std_across_seeds",
                 lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
            ]:
                agg = {"ft_method": ft, "checkpoint": ck, "seed": stat_name}
                for c in stat_cols:
                    if c in seed_df.columns:
                        agg[c] = fn(seed_df[c].values)
                for c in n_sample_cols:
                    if c in seed_df.columns:
                        agg[c] = int(seed_df[c].mean())
                agg_out.append(agg)

    rows = per_seed_out + agg_out
    cols = ["ft_method", "checkpoint", "seed"] + n_sample_cols + stat_cols
    out_df = pd.DataFrame(rows)
    cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[cols]
    out_df.to_csv(out, index=False)
    print(f"[write] {out}  rows={len(out_df)}  cols={len(cols)}  datasets={datasets}")


if __name__ == "__main__":
    main()
