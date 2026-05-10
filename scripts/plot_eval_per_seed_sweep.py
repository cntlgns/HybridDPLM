"""
Plot iter-vs-metric for 4 models across 2 datasets and decoding methods,
using per_sample_by_seed.csv produced by eval_per_seed_sweep_<N>iter runs.

For each (model, iter, dataset, decoding) tuple we compute, per protein:
  - mean across seeds  -> typical-sample value
  - best across seeds  -> oracle best-of-N value
                          (min for ca_rmsd, max for tm_score)
Then we aggregate across proteins with median and mean, giving:
  - median, mean              (representative of the typical sample)
  - best_median, best_mean    (representative of per-protein best-of-seeds)

8 plots (2 datasets x {ca_rmsd, tm_score} x {median, mean}). The x-axis is the
iteration count; decoding methods differ per iteration so points are NOT
connected. Models are encoded by color, decoding methods by marker shape;
"best" values are drawn with hollow markers and the "typical" values with
filled markers.
"""
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = Path(f"{PROJECT_DIR}/generation-results")
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep")

ITERS = [1, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]

# model dir name -> checkpoint subdir name (single subdir under <model>/<dataset>/)
MODEL_CKPT = {
    "baseline-full-1e4_lr-fs0":                       "step_6971.0-loss_0.22",
    "emb-high_noise-full-ema_lr_745-fs0-normemb":     "step_9005.0-loss_0.67",
    "noise-full-1e4_lr-fs0-cutL0":                    "step_8202.0-loss_0.70",
    "pretrained_hf":                                  "dplm2_650m",
}

METRICS = {
    "ca_rmsd":  ("ca_rmsd",   "min"),  # lower is better
    "tm_score": ("bb_tmscore", "max"),  # higher is better
}


def protein_id(sample_path: str) -> str:
    """Extract a stable protein identifier from sample_path."""
    m = re.search(r"eval/(length_\d+/[^/]+)/sample\.pdb", sample_path)
    return m.group(1) if m else sample_path


def collect_records():
    rows = []
    for it in ITERS:
        root = GEN_ROOT / f"eval_per_seed_sweep_{it}iter"
        if not root.exists():
            continue
        for model, ckpt in MODEL_CKPT.items():
            for ds in DATASETS:
                ckpt_dir = root / model / ds / ckpt
                if not ckpt_dir.exists():
                    continue
                for dec_dir in sorted(ckpt_dir.iterdir()):
                    if not dec_dir.is_dir():
                        continue
                    csv_path = dec_dir / "per_sample_by_seed.csv"
                    if not csv_path.exists():
                        continue
                    df = pd.read_csv(csv_path)
                    if df.empty:
                        continue
                    df["protein_id"] = df["sample_path"].apply(protein_id)
                    for mname, (col, direction) in METRICS.items():
                        if col not in df.columns:
                            continue
                        # per-protein mean across seeds
                        per_protein_mean = df.groupby("protein_id")[col].mean()
                        # per-protein best across seeds (oracle)
                        if direction == "min":
                            per_protein_best = df.groupby("protein_id")[col].min()
                        else:
                            per_protein_best = df.groupby("protein_id")[col].max()

                        rows.append({
                            "iter": it,
                            "model": model,
                            "dataset": ds,
                            "decoding": dec_dir.name,
                            "metric": mname,
                            "median":      per_protein_mean.median(),
                            "mean":        per_protein_mean.mean(),
                            "best_median": per_protein_best.median(),
                            "best_mean":   per_protein_best.mean(),
                            "n_proteins":  per_protein_mean.shape[0],
                            "n_seeds":     int(df["seed"].nunique()),
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- plot
MODEL_COLORS = {
    "baseline-full-1e4_lr-fs0":                   "#1f77b4",
    "emb-high_noise-full-ema_lr_745-fs0-normemb": "#d62728",
    "noise-full-1e4_lr-fs0-cutL0":                "#2ca02c",
    "pretrained_hf":                              "#7f7f7f",
}
MODEL_SHORT = {
    "baseline-full-1e4_lr-fs0":                   "baseline",
    "emb-high_noise-full-ema_lr_745-fs0-normemb": "emb-high (normemb)",
    "noise-full-1e4_lr-fs0-cutL0":                "noise (cutL0)",
    "pretrained_hf":                              "pretrained_hf",
}
# Marker semantics (per iter):
#   argmax                  -> 'o' (circle)
#   higher annealing temp   -> '^' (triangle up)
#   lower annealing temp    -> 'v' (triangle down)
ANNEAL_RE = re.compile(r"annealing([0-9.]+)_")


def _anneal_temp(name: str):
    m = ANNEAL_RE.match(name)
    return float(m.group(1)) if m else None


def build_marker_map(records: pd.DataFrame) -> dict:
    """Marker key is (iter, decoding) since "high/low annealing" depends on iter."""
    out = {}
    for it, sub in records.groupby("iter"):
        decs = sorted(sub["decoding"].unique())
        anneal_temps = sorted(
            [(d, _anneal_temp(d)) for d in decs if _anneal_temp(d) is not None],
            key=lambda kv: kv[1],
        )
        for d in decs:
            if d == "argmax":
                out[(it, d)] = "o"
            elif anneal_temps and d == anneal_temps[-1][0]:
                out[(it, d)] = "^"
            elif anneal_temps and d == anneal_temps[0][0]:
                out[(it, d)] = "v"
            else:
                out[(it, d)] = "s"  # fallback
    return out


def plot_one(records: pd.DataFrame, dataset: str, metric: str, agg: str,
             marker_map: dict, out_path: Path):
    """agg in {'median','mean'} chooses median/mean (typical) and the
    matching best_<agg> column."""
    sub = records[(records["dataset"] == dataset) & (records["metric"] == metric)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))

    typical_col = agg
    best_col = f"best_{agg}"
    # tiny x-jitter so identical-iter points from different decodings don't overlap
    decs = sorted(sub["decoding"].unique())
    n_dec = len(decs)
    dec_to_jit = {d: (i - (n_dec - 1) / 2) * 0.06 for i, d in enumerate(decs)}

    for model, mdf in sub.groupby("model"):
        color = MODEL_COLORS.get(model, "k")
        for _, row in mdf.iterrows():
            x = row["iter"]
            mk = marker_map[(row["iter"], row["decoding"])]
            ax.scatter(x, row[typical_col], color=color,
                       marker=mk, s=80,
                       facecolors=color, edgecolors=color, linewidths=1.2,
                       zorder=3)
            ax.scatter(x, row[best_col], color=color,
                       marker=mk, s=80,
                       facecolors="none", edgecolors=color, linewidths=1.6,
                       zorder=3)
            # thin vertical line connecting typical -> best for the same point
            ax.plot([x, x], [row[typical_col], row[best_col]],
                    color=color, alpha=0.25, linewidth=1, zorder=1)

    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([str(i) for i in ITERS])
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} ({agg} across proteins)")
    ax.set_title(f"{dataset} | {metric} | {agg} (filled) & best_{agg} (hollow)")
    ax.grid(True, alpha=0.3)

    # legend: models (color) + decodings (marker) + filled/hollow meaning
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS[m],
               markeredgecolor=MODEL_COLORS[m], markersize=9,
               label=MODEL_SHORT.get(m, m))
        for m in MODEL_COLORS
        if m in sub["model"].unique()
    ]
    # build per-iter description of which annealing temperature maps to high/low
    iter_to_temps = {}
    for it, isub in sub.groupby("iter"):
        decs_it = sorted(isub["decoding"].unique())
        anneal = sorted(
            [(d, _anneal_temp(d)) for d in decs_it if _anneal_temp(d) is not None],
            key=lambda kv: kv[1],
        )
        iter_to_temps[it] = {
            "low":  anneal[0][1] if anneal else None,
            "high": anneal[-1][1] if len(anneal) > 1 else None,
        }
    sorted_iters = sorted(iter_to_temps)
    def _fmt_per_iter(key):
        chunks = [
            f"it{it}={iter_to_temps[it][key]:g}"
            for it in sorted_iters if iter_to_temps[it][key] is not None
        ]
        # wrap to two per line for readability
        lines = [", ".join(chunks[i:i+2]) for i in range(0, len(chunks), 2)]
        return "\n  ".join(lines)
    dec_handles = [
        Line2D([0], [0], marker="o", color="k", linestyle="None",
               markersize=9, label="argmax"),
        Line2D([0], [0], marker="^", color="k", linestyle="None",
               markersize=9, label=f"annealing higher T:\n  {_fmt_per_iter('high')}"),
        Line2D([0], [0], marker="v", color="k", linestyle="None",
               markersize=9, label=f"annealing lower T:\n  {_fmt_per_iter('low')}"),
    ]
    style_handles = [
        Line2D([0], [0], marker="o", color="k", markerfacecolor="k",
               markersize=9, linestyle="None", label=f"{agg} (typical)"),
        Line2D([0], [0], marker="o", color="k", markerfacecolor="none",
               markersize=9, linestyle="None", label=f"best_{agg} (oracle)"),
    ]
    leg1 = ax.legend(handles=model_handles, title="model", loc="upper left",
                     bbox_to_anchor=(1.02, 1.00), borderaxespad=0,
                     frameon=True, fontsize=9)
    leg2 = ax.legend(handles=dec_handles, title="decoding", loc="upper left",
                     bbox_to_anchor=(1.02, 0.65), borderaxespad=0,
                     frameon=True, fontsize=8)
    leg3 = ax.legend(handles=style_handles, title="style", loc="upper left",
                     bbox_to_anchor=(1.02, 0.18), borderaxespad=0,
                     frameon=True, fontsize=9)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    csv_path = OUT_DIR / "summary.csv"
    records.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(records)} rows)")

    marker_map = build_marker_map(records)
    for ds in DATASETS:
        for metric in METRICS:
            for agg in ("median", "mean"):
                out = OUT_DIR / f"{ds}__{metric}__{agg}.png"
                plot_one(records, ds, metric, agg, marker_map, out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
