"""
Per-iteration sweep of sequence-level diversity and novelty for the
inverse-folding (struct -> seq) generation pipeline.

Methods (5) compared at each (dataset, max_iter):
  hybrid(one-hot)   = oh-xhigh_noise-full-3e5_lr-fs0-normemb / step_10256.0-loss_0.70
  hybrid(embed)     = emb-high_noise-full-1e4_lr           / step_9843.0-loss_0.68
  noise(freeze)     = noise-full-1e4_lr-fs0-cutL0_freeze   / step_8202.0-loss_0.70
                       (sample_noise_every_step = False)
  noise(redraw)     = noise-full-1e4_lr-fs0-cutL0_redraw   / step_8202.0-loss_0.70
                       (sample_noise_every_step = True; same checkpoint as freeze)
  baseline(disc.FT) = baseline-full-1e4_lr-fs0_uncond      / step_6971.0-loss_0.22

Decoding strategies per iter:
  - oh / embed / baseline: same per-iter decoding as
    `plot_eval_per_seed_sweep_oh_vs_hybrid_vs_baseline_protein_iqr.py`
  - noise(freeze): same per-iter decoding as `analysis/eval_per_seed_sweep/recommended2`
  - noise(redraw): identical decoding-per-iter to noise(freeze), so the only
    difference is the inference-time `sample_noise_every_step` flag.

For each (label, iter, dataset):
  * Load 10 seeds (42..51) of generated aatype.fasta.
  * For every protein_id present in all 10 seeds:
      diversity = 1 - mean(pairwise sequence identity across the 10 seeds)
      novelty   = 1 - mean(per-seed identity vs the native sequence)
  * Aggregate per-protein diversity / novelty across proteins with both
    median and mean.

Outputs in analysis/eval_per_seed_sweep/seq_diversity_novelty/:
  - summary.csv    (label, iter, dataset, metric, agg, value, n_proteins, decoding)
  - per_protein.csv (label, iter, dataset, protein_id, diversity, novelty)
  - {dataset}__{metric}__{agg}.png  (line plot, x = iter)
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/data_fast/home/sihun/diffprotein/dplm")
GEN_ROOT = PROJECT_DIR / "generation-results"
NATIVE_ROOT = PROJECT_DIR / "data-bin"
OUT_ROOT = PROJECT_DIR / "analysis/eval_per_seed_sweep/seq_diversity_novelty"

ITERS = [1, 3, 5, 10, 30, 100]
DATASETS = ["cameo2022", "PDB_date"]
SEEDS = list(range(42, 52))

OH_LABEL = "hybrid(one-hot)"
EMB_LABEL = "hybrid(embed)"
NOISE_FREEZE_LABEL = "noise(freeze)"
NOISE_REDRAW_LABEL = "noise(redraw)"
BASELINE_LABEL = "baseline(disc. FT)"

# Per-iter decoding shared by both noise variants (from recommended2).
_NOISE_DEC_BY_ITER = {1: "annealing0.7_0.1",
                      3: "argmax",
                      5: "argmax",
                      10: "argmax",
                      30: "argmax",
                      100: "argmax"}

# (model_dir, ckpt_dir, decoding-by-iter)
SERIES = {
    OH_LABEL: (
        "oh-xhigh_noise-full-3e5_lr-fs0-normemb",
        "step_10256.0-loss_0.70",
        {1: "annealing0.5_0.1",
         3: "annealing1.0_0.1",
         5: "annealing0.7_0.1",
         10: "annealing0.3_0.1",
         30: "argmax",
         100: "annealing4.0_0.1"},
    ),
    EMB_LABEL: (
        "emb-high_noise-full-1e4_lr",
        "step_9843.0-loss_0.68",
        {1: "annealing0.5_0.1",
         3: "annealing0.5_0.1",
         5: "annealing0.1_0.01",
         10: "argmax",
         30: "argmax",
         100: "annealing4.0_0.1"},
    ),
    NOISE_FREEZE_LABEL: (
        "noise-full-1e4_lr-fs0-cutL0_freeze",
        "step_8202.0-loss_0.70",
        _NOISE_DEC_BY_ITER,
    ),
    NOISE_REDRAW_LABEL: (
        "noise-full-1e4_lr-fs0-cutL0_redraw",
        "step_8202.0-loss_0.70",
        _NOISE_DEC_BY_ITER,
    ),
    BASELINE_LABEL: (
        "baseline-full-1e4_lr-fs0_uncond",
        "step_6971.0-loss_0.22",
        {1: "annealing0.5_0.1",
         3: "annealing1.0_0.1",
         5: "annealing1.0_0.1",
         10: "annealing2.0_0.1",
         30: "annealing2.0_0.1",
         100: "annealing4.0_0.1"},
    ),
}

LABEL_COLOR = {
    OH_LABEL:           "#d62728",
    EMB_LABEL:          "#ff7f0e",
    NOISE_FREEZE_LABEL: "#1f77b4",
    NOISE_REDRAW_LABEL: "#2ca02c",
    BASELINE_LABEL:     "#404040",
}


def parse_fasta(path: Path) -> dict[str, str]:
    """Return {name -> sequence}. Tolerates wrapped lines and blank lines."""
    out: dict[str, list[str]] = {}
    name: str | None = None
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                out[name] = []
            elif name is not None:
                out[name].append(line)
    return {k: "".join(v) for k, v in out.items()}


def load_native(dataset: str) -> dict[str, str]:
    return parse_fasta(NATIVE_ROOT / dataset / "aatype.fasta")


def fasta_for_seed(label: str, it: int, dataset: str, seed: int) -> Path | None:
    model_dir, ckpt_dir, dec_by_iter = SERIES[label]
    dec = dec_by_iter[it]
    path = (GEN_ROOT / f"eval_per_seed_sweep_{it}iter" / model_dir /
            dataset / ckpt_dir / dec / f"seed_{seed}" /
            "inverse_folding" / "aatype.fasta")
    return path if path.exists() else None


def load_all_seeds(label: str, it: int, dataset: str
                   ) -> tuple[dict[str, dict[int, str]], str]:
    """Return ({protein_id -> {seed -> seq}}, decoding_used)."""
    _, _, dec_by_iter = SERIES[label]
    dec = dec_by_iter[it]
    by_protein: dict[str, dict[int, str]] = {}
    for seed in SEEDS:
        p = fasta_for_seed(label, it, dataset, seed)
        if p is None:
            continue
        seqs = parse_fasta(p)
        for pid, s in seqs.items():
            by_protein.setdefault(pid, {})[seed] = s
    return by_protein, dec


def seq_identity(a: str, b: str) -> float:
    """Position-wise identity. Compares only over min(len(a), len(b)).
    Returns NaN if the comparison length is 0."""
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    aa = np.frombuffer(a[:n].encode(), dtype=np.uint8)
    bb = np.frombuffer(b[:n].encode(), dtype=np.uint8)
    return float((aa == bb).mean())


def per_protein_metrics(by_protein: dict[str, dict[int, str]],
                        natives: dict[str, str]) -> pd.DataFrame:
    rows = []
    for pid, seed_to_seq in by_protein.items():
        seqs = list(seed_to_seq.values())
        if len(seqs) < 2:
            continue
        # diversity = 1 - mean pairwise identity across seeds
        ids = [seq_identity(a, b) for a, b in combinations(seqs, 2)]
        ids = [x for x in ids if not np.isnan(x)]
        if not ids:
            continue
        diversity = 1.0 - float(np.mean(ids))

        # novelty = 1 - mean(per-seed identity vs native), if native present
        novelty = float("nan")
        native = natives.get(pid)
        if native is not None:
            ids_n = [seq_identity(s, native) for s in seqs]
            ids_n = [x for x in ids_n if not np.isnan(x)]
            if ids_n:
                novelty = 1.0 - float(np.mean(ids_n))

        rows.append({"protein_id": pid,
                     "diversity": diversity,
                     "novelty": novelty,
                     "n_seeds": len(seqs)})
    return pd.DataFrame(rows)


def collect():
    per_protein_rows = []
    summary_rows = []
    for label in SERIES:
        for ds in DATASETS:
            natives = load_native(ds)
            for it in ITERS:
                by_protein, dec = load_all_seeds(label, it, ds)
                if not by_protein:
                    print(f"[warn] no data: {label} | iter={it} | {ds} ({dec})")
                    continue
                pp = per_protein_metrics(by_protein, natives)
                if pp.empty:
                    print(f"[warn] empty per-protein: {label}|iter={it}|{ds}")
                    continue
                pp_out = pp.copy()
                pp_out.insert(0, "label", label)
                pp_out.insert(1, "iter", it)
                pp_out.insert(2, "dataset", ds)
                pp_out.insert(3, "decoding", dec)
                per_protein_rows.append(pp_out)
                for metric in ("diversity", "novelty"):
                    vals = pp[metric].dropna()
                    if vals.empty:
                        continue
                    for agg in ("median", "mean"):
                        v = float(vals.median()) if agg == "median" else float(vals.mean())
                        summary_rows.append({
                            "label": label, "iter": it, "dataset": ds,
                            "metric": metric, "agg": agg,
                            "value": v,
                            "n_proteins": int(len(vals)),
                            "decoding": dec,
                        })
                print(f"[ok] {label} | iter={it:>3} | {ds:>10} | "
                      f"dec={dec:<20} | n_proteins={len(pp)}")
    per_protein_df = (pd.concat(per_protein_rows, ignore_index=True)
                      if per_protein_rows else pd.DataFrame())
    summary_df = pd.DataFrame(summary_rows)
    return per_protein_df, summary_df


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    iter_to_pos = {it: i for i, it in enumerate(ITERS)}
    for ds in DATASETS:
        for metric in ("diversity", "novelty"):
            for agg in ("median", "mean"):
                fig, ax = plt.subplots(figsize=(8.5, 5.0))
                sub = summary[(summary["dataset"] == ds) &
                              (summary["metric"] == metric) &
                              (summary["agg"] == agg)]
                if sub.empty:
                    plt.close(fig)
                    continue
                for label in SERIES:
                    sl = (sub[sub["label"] == label]
                          .sort_values("iter"))
                    if sl.empty:
                        continue
                    xs = [iter_to_pos[it] for it in sl["iter"]]
                    ys = sl["value"].tolist()
                    color = LABEL_COLOR[label]
                    ax.plot(xs, ys, color=color, marker="o", linewidth=2.2,
                            markersize=6.5, label=label)
                ax.set_xticks(list(iter_to_pos.values()))
                ax.set_xticklabels([str(it) for it in ITERS])
                ax.set_xlabel("decoding iterations (max_iter)")
                ax.set_ylabel(f"{metric} ({agg} across proteins)")
                ax.set_title(f"{ds} | sequence {metric} ({agg} across proteins)\n"
                             "diversity = 1 - mean pairwise seq identity (10 seeds); "
                             "novelty = 1 - mean identity vs native",
                             fontsize=9.5)
                ax.grid(True, alpha=0.3, axis="y")
                ax.legend(fontsize=9, loc="best", framealpha=0.9)
                fig.tight_layout()
                out = out_dir / f"{ds}__{metric}__{agg}.png"
                fig.savefig(out, dpi=140, bbox_inches="tight")
                plt.close(fig)
                print(f"wrote {out}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_protein_df, summary_df = collect()
    if per_protein_df.empty or summary_df.empty:
        print("no records produced; aborting plot/save.")
        return
    per_protein_df.to_csv(OUT_ROOT / "per_protein.csv", index=False)
    summary_df.to_csv(OUT_ROOT / "summary.csv", index=False)
    print(f"wrote {OUT_ROOT/'per_protein.csv'} ({len(per_protein_df)} rows)")
    print(f"wrote {OUT_ROOT/'summary.csv'} ({len(summary_df)} rows)")
    plot_summary(summary_df, OUT_ROOT)


if __name__ == "__main__":
    main()
