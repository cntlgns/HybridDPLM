"""
baseline (uncond) vs noise (lr-linear) comparison.

Selection rule per series:
  - baseline (no argmax):
      iter 1   -> annealing0.5_0.1
      other it -> highest annealing T (largest first param)
  - noise lr-linear (FREEZE and REDRAW):
      argmax at every iter
      iter 1   -> annealing0.5_0.1
      other it -> highest annealing T

Marker semantics:
  - argmax       -> 'o'
  - annealing    -> '^'
Filled = typical (median/mean across proteins);
hollow = oracle best per protein.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
GEN_ROOT = Path(f"{PROJECT_DIR}/generation-results")
OUT_DIR = Path(f"{PROJECT_DIR}/analysis/eval_per_seed_sweep/baseline_vs_noise_linear")

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]

# (model_dir, ckpt_stem, short_label, allow_argmax)
MODELS = [
    ("baseline-full-1e4_lr-fs0_uncond",     "step_6971.0-loss_0.22", "baseline (uncond)",       False),
    ("noise-full-1e4_lr-linear_freeze",     "step_9025.0-loss_0.22", "noise lr-linear FREEZE",  True),
    ("noise-full-1e4_lr-linear_redraw",     "step_9025.0-loss_0.22", "noise lr-linear REDRAW",  True),
]

MODEL_COLORS = {
    "baseline (uncond)":      "#1f77b4",  # blue
    "noise lr-linear FREEZE": "#ff7f0e",  # orange
    "noise lr-linear REDRAW": "#2ca02c",  # green
}

METRICS = {
    "ca_rmsd":  ("ca_rmsd",   "min"),
    "tm_score": ("bb_tmscore", "max"),
}

ANNEAL_RE = re.compile(r"annealing([0-9.]+)_([0-9.]+)")


def _anneal_temp(name):
    m = ANNEAL_RE.match(name)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def protein_id(sample_path: str) -> str:
    m = re.search(r"eval/(length_\d+/[^/]+)/sample\.pdb", sample_path)
    return m.group(1) if m else sample_path


def collect_records():
    rows = []
    for it in ITERS:
        root = GEN_ROOT / f"eval_per_seed_sweep_{it}iter"
        if not root.exists():
            continue
        for model_dir, ckpt, short, _ in MODELS:
            for ds in DATASETS:
                ckpt_dir = root / model_dir / ds / ckpt
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
                        per_protein_mean = df.groupby("protein_id")[col].mean()
                        if direction == "min":
                            per_protein_best = df.groupby("protein_id")[col].min()
                        else:
                            per_protein_best = df.groupby("protein_id")[col].max()
                        rows.append({
                            "iter": it,
                            "model": short,
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


def _selected_anneal_for_iter(decs, it):
    """iter 1 -> 'annealing0.5_0.1' if present; otherwise highest first-param."""
    if it == 1 and "annealing0.5_0.1" in decs:
        return "annealing0.5_0.1"
    anneal = sorted(
        [(d, _anneal_temp(d)) for d in decs if _anneal_temp(d) is not None],
        key=lambda kv: kv[1],
    )
    return anneal[-1][0] if anneal else None


def build_keep_map(records: pd.DataFrame) -> dict:
    """(iter, model, decoding) -> marker, restricted by per-model rules."""
    out = {}
    allow_argmax = {short: aa for _, _, short, aa in MODELS}
    for (it, model), sub in records.groupby(["iter", "model"]):
        decs = list(sub["decoding"].unique())
        chosen_anneal = _selected_anneal_for_iter(decs, it)
        if chosen_anneal is not None:
            out[(it, model, chosen_anneal)] = "^"
        if allow_argmax.get(model, False) and "argmax" in decs:
            out[(it, model, "argmax")] = "o"
    return out


def plot_one(records: pd.DataFrame, dataset: str, metric: str, agg: str,
             keep_map: dict, out_path: Path):
    sub = records[(records["dataset"] == dataset) & (records["metric"] == metric)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))

    typical_col = agg
    best_col = f"best_{agg}"

    for _, row in sub.iterrows():
        key = (row["iter"], row["model"], row["decoding"])
        mk = keep_map.get(key)
        if mk is None:
            continue
        color = MODEL_COLORS.get(row["model"], "k")
        x = row["iter"]
        ax.scatter(x, row[typical_col], color=color,
                   marker=mk, s=80,
                   facecolors=color, edgecolors=color, linewidths=1.2,
                   zorder=3)
        ax.scatter(x, row[best_col], color=color,
                   marker=mk, s=80,
                   facecolors="none", edgecolors=color, linewidths=1.6,
                   zorder=3)
        ax.plot([x, x], [row[typical_col], row[best_col]],
                color=color, alpha=0.25, linewidth=1, zorder=1)

    ax.set_xscale("log")
    ax.set_xticks(ITERS)
    ax.set_xticklabels([str(i) for i in ITERS])
    ax.set_xlabel("decoding iterations")
    ax.set_ylabel(f"{metric} ({agg} across proteins)")
    ax.set_title(f"baseline vs noise lr-linear | {dataset} | {metric} | "
                 f"{agg} (filled) & best_{agg} (hollow)")
    ax.grid(True, alpha=0.3)

    # Per-iter annealing T choice (taken from any model that has it; identical
    # across models since they share decoding sets).
    iter_to_T = {}
    for it, isub in sub.groupby("iter"):
        decs = list(isub["decoding"].unique())
        chosen = _selected_anneal_for_iter(decs, it)
        if chosen is not None:
            iter_to_T[it] = _anneal_temp(chosen)

    def _fmt_iter_temps():
        chunks = [
            f"it{it}={iter_to_T[it][0]:g}_{iter_to_T[it][1]:g}"
            for it in sorted(iter_to_T)
        ]
        lines = [", ".join(chunks[i:i + 3]) for i in range(0, len(chunks), 3)]
        return "\n  ".join(lines)

    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODEL_COLORS[m],
               markeredgecolor=MODEL_COLORS[m], markersize=9, label=m)
        for m in MODEL_COLORS
        if m in sub["model"].unique()
    ]
    dec_handles = [
        Line2D([0], [0], marker="o", color="k", linestyle="None",
               markersize=9, label="argmax (noise only)"),
        Line2D([0], [0], marker="^", color="k", linestyle="None",
               markersize=9, label=f"annealing:\n  {_fmt_iter_temps()}"),
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
                     bbox_to_anchor=(1.02, 0.25), borderaxespad=0,
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

    keep_map = build_keep_map(records)
    for ds in DATASETS:
        for metric in METRICS:
            for agg in ("median", "mean"):
                out = OUT_DIR / f"{ds}__{metric}__{agg}.png"
                plot_one(records, ds, metric, agg, keep_map, out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
