import pandas as pd
import os
import sys


def format_val(mean, std, decimals=4):
    return f"{mean:.{decimals}f}({std:.{decimals}f})"


def format_summary(input_path: str):
    df = pd.read_csv(input_path)

    # Get sorted unique datasets
    datasets = sorted(df["dataset"].unique())

    # Group rows by (model_name, sampling_strategy, remasking_strategy)
    group_keys = ["model_name", "sampling_strategy", "remasking_strategy"]
    metrics = [
        ("inv_fold_seq_recovery_mean", "inv_fold_seq_recovery_std", "inv_fold_seq_recovery_median"),
        ("bb_tmscore_mean", "bb_tmscore_std", "bb_tmscore_median"),
        ("mean_plddt_mean", "mean_plddt_std", "mean_plddt_median"),
        ("ca_rmsd_mean", "ca_rmsd_std", "ca_rmsd_median"),
        ("bb_rmsd_mean", "bb_rmsd_std", "bb_rmsd_median"),
    ]

    # Build column headers
    col_headers = group_keys[:]
    for ds in datasets:
        for mean_col, std_col, median_col in metrics:
            base = mean_col.replace("_mean", "")
            col_headers.append(f"{ds}_{base}_mean(std)")
            col_headers.append(f"{ds}_{base}_median")

    rows = []
    for group_vals, group_df in df.groupby(group_keys, sort=False):
        row = list(group_vals)
        for ds in datasets:
            ds_df = group_df[group_df["dataset"] == ds]
            for mean_col, std_col, median_col in metrics:
                if len(ds_df) == 0:
                    row.append("")
                    row.append("")
                else:
                    r = ds_df.iloc[0]
                    row.append(format_val(r[mean_col], r[std_col]))
                    row.append(f"{r[median_col]:.4f}")
        rows.append(row)

    result_df = pd.DataFrame(rows, columns=col_headers)

    # Sort rows by group keys for consistency
    result_df = result_df.sort_values(group_keys).reset_index(drop=True)

    # Save output
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(dir_name, f"formatted_{base_name}")
    result_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_summary.py <path_to_csv>")
        sys.exit(1)
    format_summary(sys.argv[1])
