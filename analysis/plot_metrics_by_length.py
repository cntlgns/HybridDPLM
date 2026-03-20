import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


METRICS = ['ca_rmsd', 'bb_rmsd', 'bb_tmscore', 'mean_plddt', 'inv_fold_seq_recovery']
METRIC_LABELS = {
    'ca_rmsd': 'CA RMSD',
    'bb_rmsd': 'BB RMSD',
    'bb_tmscore': 'BB TM-Score',
    'mean_plddt': 'pLDDT',
    'inv_fold_seq_recovery': 'Inv. Fold Seq Recovery',
}

DEFAULT_OUTPUT_DIR = os.path.expanduser('~/diffprotein/dplm/length_analyze')


def assign_length_clusters(lengths: pd.Series, n_clusters: int) -> pd.Series:
    """Assign each length to a cluster using quantile-based binning for balanced group sizes."""
    bins = np.quantile(lengths, np.linspace(0, 1, n_clusters + 1))
    # Remove duplicate bin edges
    bins = np.unique(bins)
    labels = []
    for i in range(len(bins) - 1):
        lo = int(bins[i])
        hi = int(bins[i + 1])
        labels.append(f"{lo}-{hi}")
    return pd.cut(lengths, bins=bins, labels=labels, include_lowest=True)


def extract_exp_info(csv_path: str) -> dict:
    """Extract DATASET, MODEL_NAME, SAMPLING_STRATEGY, REMASKING_STRATEGY from CSV path.

    Expected path structure:
      .../generation-results/{EXP_NAME}/{DATASET}/{MODEL_NAME}/{SAMPLING_STRATEGY}/{REMASKING_STRATEGY}/...
    """
    parts = os.path.normpath(csv_path).split(os.sep)
    try:
        idx = parts.index('generation-results')
    except ValueError:
        return None
    # generation-results / EXP_NAME / DATASET / MODEL_NAME / SAMPLING_STRATEGY / REMASKING_STRATEGY / ...
    if len(parts) < idx + 6:
        return None
    return {
        'dataset': parts[idx + 2],
        'model_name': parts[idx + 3],
        'sampling_strategy': parts[idx + 4],
        'remasking_strategy': parts[idx + 5],
    }


def plot_metric(df, metric, cluster_order, prefix=''):
    """Create a single box-and-whisker plot for one metric."""
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [df.loc[df['length_cluster'] == c, metric].dropna().values for c in cluster_order]
    bp = ax.boxplot(data, tick_labels=cluster_order, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cluster_order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    title = f'{prefix}-{metric}' if prefix else METRIC_LABELS[metric]
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Protein Length', fontsize=12)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    for i, c in enumerate(cluster_order):
        n = len(df[df['length_cluster'] == c])
        ax.text(i + 1, ax.get_ylim()[0], f'n={n}', ha='center', va='bottom', fontsize=9, color='gray')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Box-and-whisker plots of metrics by protein length clusters')
    parser.add_argument('csv_path', type=str, help='Path to all_top_samples.csv')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of length clusters (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ~/diffprotein/dplm/length_analyze)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Validate columns
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    if 'length' not in df.columns:
        raise ValueError("CSV must contain a 'length' column")

    cluster_col = assign_length_clusters(df['length'], args.n_clusters)
    df['length_cluster'] = cluster_col
    df = df.dropna(subset=['length_cluster'])
    cluster_order = sorted(df['length_cluster'].unique(), key=lambda x: int(x.split('-')[0]))

    # Determine output directory and filename prefix
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(out_dir, exist_ok=True)

    # Build filename prefix from path info
    exp_info = extract_exp_info(os.path.abspath(args.csv_path))
    if exp_info:
        prefix = '-'.join([
            exp_info['dataset'],
            exp_info['model_name'],
            exp_info['sampling_strategy'],
            exp_info['remasking_strategy'],
        ])
    else:
        prefix = 'unknown'

    # Save one plot per metric
    for metric in METRICS:
        fig = plot_metric(df, metric, cluster_order, prefix=prefix)
        fname = f"{prefix}-{metric}_by_length.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
