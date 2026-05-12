"""Pick the best MCTS output per pdb_name and emit a per-pdb metric CSV.

Reads the enriched ``mcts_rewards.csv`` written by ``generate_dplm2_mcts.py``
(which records, per output, the same metrics the eval pipeline produces:
``bb_tmscore``, ``bb_rmsd``, ``ca_rmsd``, ``mean_plddt``,
``inv_fold_seq_recovery``, ``length``), groups by ``pdb_name``, and
keeps the single row with the best ranking metric (default: max
``bb_tmscore``). The output CSV has one row per pdb, ready for plotting
or aggregation.

Usage:
    python scripts/mcts_best_per_pdb.py \
        --csv .../seed_42/inverse_folding/mcts_rewards.csv \
        --out .../seed_42/inverse_folding/mcts_best_per_pdb.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True,
                    help="Path to the per-output mcts_rewards.csv.")
    ap.add_argument("--out", default=None,
                    help="Output path for the per-pdb best CSV. Defaults "
                         "to <csv_dir>/mcts_best_per_pdb.csv.")
    ap.add_argument("--rank-by", default="bb_tmscore",
                    help="Column used to pick the best row per pdb_name.")
    ap.add_argument("--rank-order", default="max",
                    choices=("max", "min"),
                    help="'max' → keep row with largest --rank-by; "
                         "'min' → smallest. Default: max.")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"Input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "pdb_name" not in df.columns:
        sys.exit("CSV missing 'pdb_name' column; not a mcts_rewards.csv?")
    if args.rank_by not in df.columns:
        sys.exit(f"Ranking column {args.rank_by!r} not in CSV columns: "
                 f"{list(df.columns)}. Re-run generate_dplm2_mcts.py to "
                 f"populate metric columns.")

    if args.rank_order == "max":
        idx = df.groupby("pdb_name")[args.rank_by].idxmax()
    else:
        idx = df.groupby("pdb_name")[args.rank_by].idxmin()
    best = df.loc[idx].sort_values("pdb_name").reset_index(drop=True)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)), "mcts_best_per_pdb.csv"
    )
    best.to_csv(out_path, index=False)
    print(f"[mcts_best_per_pdb] wrote {len(best)} rows -> {out_path}")
    print(f"  ranking: {args.rank_order}({args.rank_by})")
    metric_cols = [c for c in
                   ("bb_tmscore", "bb_rmsd", "ca_rmsd", "mean_plddt",
                    "inv_fold_seq_recovery", "length")
                   if c in best.columns]
    if metric_cols:
        print("  per-pdb means:")
        for c in metric_cols:
            print(f"    {c:<24s} {best[c].mean():.4f}")


if __name__ == "__main__":
    main()
