"""
Aggregate per-seed hybrid-eval metrics into a summary CSV.

Expected layout under --root:
    <root>/seed_<s>/inverse_folding/aatype/eval/inverse_fold_metrics.csv
    <root>/seed_<s>/inverse_folding/aatype/eval/all_top_samples.csv

Writes:
    <root>/seed_summary.csv         (one row per seed + aggregate rows)
    <root>/per_sample_by_seed.csv   (concatenation of all_top_samples.csv,
                                     adds a 'seed' column; optional)

Usage:
    python scripts/aggregate_seed_metrics.py --root <dir> [--task inverse_folding]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def _seed_dirs(root):
    """Return sorted list of (seed:int, path:str) under root."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "seed_*"))):
        if not os.path.isdir(p):
            continue
        m = re.match(r"seed_(-?\d+)$", os.path.basename(p))
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    return out


def _metrics_csv(seed_dir, task):
    return os.path.join(seed_dir, task, "aatype", "eval", "inverse_fold_metrics.csv")


def _samples_csv(seed_dir, task):
    return os.path.join(seed_dir, task, "aatype", "eval", "all_top_samples.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Directory containing seed_<s>/ subdirectories.")
    parser.add_argument("--task", default="inverse_folding")
    parser.add_argument("--out_summary", default=None,
                        help="Default: <root>/seed_summary.csv")
    parser.add_argument("--out_per_sample", default=None,
                        help="Default: <root>/per_sample_by_seed.csv "
                             "(set to empty '' to skip).")
    args = parser.parse_args()

    seeds = _seed_dirs(args.root)
    if not seeds:
        raise SystemExit(f"No seed_*/ dirs found under {args.root}")

    rows = []
    per_sample_frames = []
    missing = []
    for seed, sdir in seeds:
        mp = _metrics_csv(sdir, args.task)
        if not os.path.exists(mp):
            missing.append((seed, mp))
            continue
        df = pd.read_csv(mp)
        assert len(df) == 1, f"expected 1-row metrics CSV, got {len(df)} in {mp}"
        row = df.iloc[0].to_dict()
        row["seed"] = seed
        rows.append(row)

        sp = _samples_csv(sdir, args.task)
        if os.path.exists(sp):
            sdf = pd.read_csv(sp)
            sdf["seed"] = seed
            per_sample_frames.append(sdf)

    if missing:
        print(f"[warn] {len(missing)} seed(s) missing metrics:")
        for seed, p in missing:
            print(f"  - seed={seed}: {p}")

    if not rows:
        raise SystemExit("No metrics found; nothing to aggregate.")

    summary = pd.DataFrame(rows)

    # numeric columns = anything convertible to float & not the seed column
    numeric_cols = [
        c for c in summary.columns
        if c != "seed" and pd.api.types.is_numeric_dtype(summary[c])
    ]

    agg_rows = []
    for stat, fn in [
        ("mean", np.mean),
        ("std",  lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
        ("min",  np.min),
        ("max",  np.max),
    ]:
        ar = {c: fn(summary[c].values) for c in numeric_cols}
        ar["seed"] = f"_{stat}"
        agg_rows.append(ar)
    agg_rows.append({"seed": "_n", **{c: len(summary) for c in numeric_cols}})

    summary_out = pd.concat(
        [summary, pd.DataFrame(agg_rows)],
        ignore_index=True,
    )
    # put seed first
    cols = ["seed"] + [c for c in summary_out.columns if c != "seed"]
    summary_out = summary_out[cols]

    out_summary = args.out_summary or os.path.join(args.root, "seed_summary.csv")
    summary_out.to_csv(out_summary, index=False)
    print(f"[write] {out_summary} ({len(summary)} seeds, {len(numeric_cols)} metrics)")

    out_per_sample = (
        args.out_per_sample
        if args.out_per_sample is not None
        else os.path.join(args.root, "per_sample_by_seed.csv")
    )
    if out_per_sample and per_sample_frames:
        cat = pd.concat(per_sample_frames, ignore_index=True)
        cat.to_csv(out_per_sample, index=False)
        print(f"[write] {out_per_sample} ({len(cat)} rows)")


if __name__ == "__main__":
    main()
