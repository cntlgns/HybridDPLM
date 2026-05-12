"""SLURM launcher: per-seed fan-out for MCTS inverse-folding evaluation.

Mirrors scripts/sweep_eval_per_seed.py but invokes
scripts/run_eval_single_seed_mcts.sh, which drives
generate_dplm2_mcts.py instead of the per-variant generators.

Adds two extra sweep axes specific to MCTS:
    * MCTS_HP_SWEEP — list of (M, K, c_uct, τ, num_outputs) tuples
    * one job per (variant, ckpt, dataset, sampling, max_iter, MCTS hp, seed)

Result layout under generation-results/<RESULT_SUBDIR>_{max_iter}iter/:
    <EXP>/<DATASET>/<CKPT>/<strategy_tag>__mcts_M{M}_K{K}/seed_<s>/...

Skip rule: a (combo, seed) is skipped if its inverse_fold_metrics.csv
already exists on NFS (matches sweep_eval_per_seed.py).
"""
import os
import sys

from slurm_launcher.sbatch_launcher import launch_tasks

PYTHON_BIN = "/data_fast/home/sihun/diffprotein/dplm/.venv/bin/python"
PART_TO_PY = {"rtx3090": PYTHON_BIN, "ada": PYTHON_BIN}

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/run_eval_single_seed_mcts.sh"

RESULT_SUBDIR = "mcts_eval_protinvtree_seed_sweep"


def result_subdir(max_iter: int) -> str:
    return f"{RESULT_SUBDIR}_{max_iter}iter"


def result_base(max_iter: int) -> str:
    return f"{PROJECT_DIR}/generation-results/{result_subdir(max_iter)}"

# ─── Checkpoints by variant ───────────────────────────────────────────────────
NOISE_CKPTS: list[str] = [
    f"{PROJECT_DIR}/train_logs/noiseFT/noise-full-1e4_lr-fs0-cutL0/checkpoints/step_8202.0-loss_0.70.ckpt",
]

BASELINE_CKPTS: list[str] = [
    f"{PROJECT_DIR}/train_logs/noiseFT/baseline-full-1e4_lr-fs0/checkpoints/step_6971.0-loss_0.22.ckpt",
]

HYBRID_CKPTS: list[str] = [
    f"{PROJECT_DIR}/train_logs/FT9/emb-high_noise-full-1e4_lr/checkpoints/step_9843.0-loss_0.68.ckpt",
    f"{PROJECT_DIR}/train_logs/FTO3/oh-xhigh_noise-full-3e5_lr-fs0-normemb/checkpoints/step_10256.0-loss_0.70.ckpt",
]

DATASETS = ["cath_4.2_all", "cath_4.3_all"] # "cameo2022", "PDB_date", "cath_4.2_all", "cath_4.3_all"

BATCH_SIZE_BY_DATASET = {
    "cameo2022": 1,
    "PDB_date": 1,
    "cath_4.2_all": 1,
    "cath_4.3_all": 1,
}

# ─── Sweep axes ───────────────────────────────────────────────────────────────
# Outer: max_iter (denoising depth). Inner: annealing configs.
ANNEALING_SWEEP_BY_MAX_ITER: dict[int, list[tuple[str, str]]] = {
    3: [
        ("annealing@1.0:1.0", "annealing1.0_1.0"),
    ],
}
MAX_ITER_SWEEP: list[int] = sorted(ANNEALING_SWEEP_BY_MAX_ITER.keys())

# MCTS hyperparameter combinations to sweep.
# Each tuple is (mcts_iterations M, mcts_expansions K, c_uct, τ, num_outputs N).
MCTS_HP_SWEEP: list[tuple[int, int, float, float, int]] = [
    # Paper defaults (Sec. 5.1).
    (3, 3, 0.01, 0.99, 1),
    # A budget-matched comparison with N=10 BoN (10 fold calls per design).
    # (10, 2, 0.01, 0.99, 10),
]


def mcts_tag(M: int, K: int, c_uct: float, tau: float, N: int) -> str:
    return f"mcts_M{M}_K{K}_c{c_uct}_t{tau}_N{N}"


SEED_BASE = 42
NUM_SEEDS = 1   # MCTS is more expensive than vanilla generate; default fewer seeds.


def _seeds(n: int) -> list[int]:
    return [SEED_BASE + i for i in range(n)]


SEEDS_BY_VARIANT: dict[str, list[int]] = {
    "noise":    _seeds(NUM_SEEDS),
    "baseline": _seeds(NUM_SEEDS),
    "hybrid":   _seeds(NUM_SEEDS),
}

REMASKINGS_BY_VARIANT: dict[str, list[str]] = {
    "noise":    ["no_remask"],          # ignored; placeholder
    "baseline": ["uncond"],
    "hybrid":   ["no_remask"],          # ignored; placeholder
}

NOISE_SAMPLE_MODES_BY_VARIANT: dict[str, list[str]] = {
    "noise":    ["false"],
    "baseline": [""],
    "hybrid":   [""],
}


def build_plan(variant: str) -> list[tuple[str, str, list[int], int, tuple]]:
    """Return list of (sampling, base_tag, seeds, max_iter, mcts_hp)."""
    plan: list[tuple[str, str, list[int], int, tuple]] = []
    for max_iter in MAX_ITER_SWEEP:
        for sampling, base_tag in ANNEALING_SWEEP_BY_MAX_ITER[max_iter]:
            for hp in MCTS_HP_SWEEP:
                plan.append((sampling, base_tag,
                             SEEDS_BY_VARIANT[variant], max_iter, hp))
    return plan


GROUPS: list[tuple[str, list[str]]] = [
    ("noise",    NOISE_CKPTS),
    ("baseline", BASELINE_CKPTS),
    ("hybrid",   HYBRID_CKPTS),
]


HF_PREFIX = "hf:"


def is_hf(ckpt_path: str) -> bool:
    return ckpt_path.startswith(HF_PREFIX)


_NOISE_MODE_SUFFIX = {"true": "_redraw", "false": "_freeze", "": ""}


def exp_and_base(
    ckpt_path: str, variant: str, remasking: str, noise_mode: str,
) -> tuple[str, str]:
    if is_hf(ckpt_path):
        exp = "pretrained_hf"
        base = ckpt_path[len(HF_PREFIX):].split("/")[-1]
    else:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        exp = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    if variant == "baseline":
        exp = f"{exp}_{remasking}"
    if variant == "noise":
        exp = f"{exp}{_NOISE_MODE_SUFFIX[noise_mode]}"
    return exp, base


def metric_csv_path(ckpt_path: str, ds: str, strategy_tag: str, seed: int,
                    max_iter: int, variant: str, remasking: str,
                    noise_mode: str) -> str:
    exp, base = exp_and_base(ckpt_path, variant, remasking, noise_mode)
    return os.path.join(
        result_base(max_iter), exp, ds, base, strategy_tag,
        f"seed_{seed}", "inverse_folding", "aatype", "eval",
        "inverse_fold_metrics.csv",
    )


def run_slurm():
    all_ckpts = [(k, p) for k, lst in GROUPS for p in lst]
    if not all_ckpts:
        print("No checkpoints — edit NOISE_CKPTS / BASELINE_CKPTS / HYBRID_CKPTS "
              "in scripts/sweep_mcts_eval_per_seed.py.")
        return

    missing = [(k, p) for k, p in all_ckpts if not is_hf(p) and not os.path.isfile(p)]
    if missing:
        print("The following checkpoints do not exist:")
        for k, p in missing:
            print(f"  - [{k}] {p}")
        sys.exit(1)

    print(f"Sweeping max_iter: {MAX_ITER_SWEEP}")
    for mi in MAX_ITER_SWEEP:
        tags = [t for _, t in ANNEALING_SWEEP_BY_MAX_ITER[mi]]
        print(f"  max_iter={mi} -> {result_base(mi)}")
        print(f"    annealing={tags}")
    print(f"MCTS hp combos: {MCTS_HP_SWEEP}")
    for variant, ckpts in GROUPS:
        if not ckpts:
            continue
        plan = build_plan(variant)
        plan_summary = ", ".join(
            f"{tag}/{mcts_tag(*hp)}x{len(seeds)}"
            for _, tag, seeds, _, hp in plan
        )
        print(f"  [{variant}] {len(ckpts)} ckpts × {len(DATASETS)} datasets × ({plan_summary})")

    combos: list[str] = []
    skipped = 0
    for variant, ckpts in GROUPS:
        plan = build_plan(variant)
        remaskings = REMASKINGS_BY_VARIANT[variant]
        noise_modes = NOISE_SAMPLE_MODES_BY_VARIANT[variant]
        for ckpt_path in ckpts:
            for ds in DATASETS:
                bs = BATCH_SIZE_BY_DATASET.get(ds, 1)
                for sampling, base_tag, seeds, max_iter, hp in plan:
                    M, K, c_uct, tau, N = hp
                    full_tag = f"{base_tag}__{mcts_tag(M, K, c_uct, tau, N)}"
                    subdir = result_subdir(max_iter)
                    for remasking in remaskings:
                        for noise_mode in noise_modes:
                            for seed in seeds:
                                if os.path.isfile(metric_csv_path(
                                    ckpt_path, ds, full_tag, seed, max_iter,
                                    variant, remasking, noise_mode,
                                )):
                                    skipped += 1
                                    continue
                                tail_extra = (
                                    f" {M} {K} {c_uct} {tau} {N}"
                                )
                                tail_noise = f" {noise_mode}" if noise_mode else " _"
                                cmd = (
                                    f"{SCRIPT_PATH} {variant} {ckpt_path} {ds} {seed} "
                                    f"{sampling} {max_iter} {bs} {subdir} {full_tag} "
                                    f"{remasking}{tail_noise}{tail_extra}"
                                )
                                combos.append(cmd)

    print(f"Skipped (already done): {skipped}")
    print(f"Submitting {len(combos)} jobs")
    if not combos:
        print("Nothing to submit.")
        return

    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": combos},
        partition="a100",
        exclude="radish",
        qos="normal",
        timeout="5-0",
        job_name="mcts_per_seed",
        max_job_num=92,
        part_to_py=PART_TO_PY,
    )


if __name__ == "__main__":
    run_slurm()
