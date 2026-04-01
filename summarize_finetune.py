"""Summarize all checkpoints under a single finetuning experiment directory.

Given a path like:
    generation-results/candi_invfold_test/candi_const_weight_ft2

It discovers all `all_top_samples.csv` files under {dataset}/{checkpoint}/
and produces a summary CSV sorted by dataset and checkpoint step.

Usage:
    python summarize_finetune.py \
        generation-results/candi_invfold_test/candi_const_weight_ft2
"""

import argparse
import os
import re
import glob

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


def parse_step(checkpoint_name: str) -> float:
    """Extract step number from checkpoint name for sorting. Non-step names sort to -1."""
    m = re.match(r"step_([\d.]+)", checkpoint_name)
    return float(m.group(1)) if m else -1


def summarize_csv(csv_path: str, dataset: str, checkpoint: str) -> dict:
    df = pd.read_csv(csv_path)

    row = {
        "dataset": dataset,
        "checkpoint": checkpoint,
        "num_samples": len(df),
    }

    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        for stat_name, stat_fn in STAT_FUNCS.items():
            row[f"{col}_{stat_name}"] = stat_fn(values)

    return row


def summarize_finetune(ft_dir: str):
    ft_dir = os.path.abspath(ft_dir)
    ft_name = os.path.basename(ft_dir)

    csv_pattern = os.path.join(ft_dir, "*", "*", "inverse_folding", "aatype", "eval", "all_top_samples.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"No all_top_samples.csv found under {ft_dir}")
        return

    rows = []
    for csv_path in csv_files:
        # Extract dataset and checkpoint from path
        # .../ft_dir/{dataset}/{checkpoint}/inverse_folding/aatype/eval/all_top_samples.csv
        rel = os.path.relpath(csv_path, ft_dir)
        parts = rel.split(os.sep)
        dataset, checkpoint = parts[0], parts[1]
        rows.append(summarize_csv(csv_path, dataset, checkpoint))

    summary_df = pd.DataFrame(rows)

    # Sort: dataset ascending, then step number (non-step checkpoints first)
    summary_df["_step"] = summary_df["checkpoint"].apply(parse_step)
    summary_df = summary_df.sort_values(["dataset", "_step"]).drop(columns=["_step"]).reset_index(drop=True)

    output_csv = os.path.join(ft_dir, "summary.csv")
    summary_df.to_csv(output_csv, index=False)

    print(f"Finetuning: {ft_name}")
    print(f"Summary saved to {output_csv}")
    print(f"Found {len(rows)} checkpoint(s) across {summary_df['dataset'].nunique()} dataset(s)\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize all checkpoints under a finetuning experiment directory."
    )
    parser.add_argument("ft_dir", help="Path to the finetuning directory (e.g. generation-results/.../candi_const_weight_ft2)")
    args = parser.parse_args()

    summarize_finetune(args.ft_dir)
