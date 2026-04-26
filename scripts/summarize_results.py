"""Summarize all_top_samples.csv and append one row to a shared summary CSV.

Usage:
    python summarize_results.py \
        --exp_name reproduction \
        --dataset cameo2022 \
        --model_name dplm2_650m \
        --sampling_strategy argmax \
        --remasking_strategy uncond
"""

import argparse
import os
import fcntl

import pandas as pd
import numpy as np


METRIC_COLS = [
    "ca_rmsd",
    "bb_rmsd",
    "bb_tmscore",
    "mean_plddt",
    "inv_fold_seq_recovery",
    "pmpnn_seq_recovery",
    "pmpnn_bb_rmsd",
    "length",
    "helix_percent",
    "strand_percent",
]

STAT_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
}


PROJECT_DIR = '/data_fast/home/sihun/diffprotein/dplm'
# os.path.dirname(os.path.abspath(__file__))
# import ipdb; ipdb.set_trace()  # --- IGNORE ---


def summarize(exp_name, dataset, model_name, sampling_strategy, remasking_strategy="uncond", max_iter=1):
    base_dir = os.path.join(PROJECT_DIR, "generation-results", exp_name)
    run_dir = os.path.join(base_dir, dataset, model_name, sampling_strategy, remasking_strategy)
    if int(max_iter) != 1:
        run_dir = os.path.join(run_dir, f"iter{max_iter}")
    input_csv = os.path.join(
        run_dir, "inverse_folding", "aatype", "eval", "all_top_samples.csv",
    )
    output_csv = os.path.join(base_dir, "summary.csv")

    df = pd.read_csv(input_csv)

    row = {
        "exp_name": exp_name,
        "dataset": dataset,
        "model_name": model_name,
        "sampling_strategy": sampling_strategy,
        "remasking_strategy": remasking_strategy,
        "max_iter": int(max_iter),
        "num_samples": len(df),
    }

    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        for stat_name, stat_fn in STAT_FUNCS.items():
            row[f"{col}_{stat_name}"] = stat_fn(values)

    row_df = pd.DataFrame([row])

    # File-lock to avoid concurrent write corruption
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    with open(output_csv, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        row_df.to_csv(f, header=write_header, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)

    print(f"Summary appended to {output_csv}")
    print(row_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--sampling_strategy", required=True)
    parser.add_argument("--remasking_strategy", default="uncond")
    parser.add_argument("--max_iter", type=int, default=1)
    args = parser.parse_args()

    summarize(
        args.exp_name,
        args.dataset,
        args.model_name,
        args.sampling_strategy,
        args.remasking_strategy,
        args.max_iter,
    )
