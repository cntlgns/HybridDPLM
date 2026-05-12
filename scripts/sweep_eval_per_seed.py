"""
SLURM launcher: per-seed fan-out for inverse-folding evaluation.

Compared to sweep_hybrid_eval_seeds.py (one job per (ckpt, dataset)), this
launcher submits one SLURM job per (model_kind, ckpt, dataset, sampling, seed)
so each seed claims its own GPU and runs in parallel.

Result layout under generation-results/<RESULT_SUBDIR>/:
    <EXP>/<DATASET>/<CKPT>/<strategy_tag>/seed_<s>/inverse_folding/...
                          /<strategy_tag>/seed_summary.csv

For baseline kind, <EXP> is suffixed with _<remasking_strategy> (e.g.
"pretrained_hf_uncond", "baseline-full-1e4_lr-fs0_no_remask"). Sweep the set
via REMASKINGS_BY_KIND below.

Each per-seed job rsyncs its seed_<s>/ to NFS and re-runs
aggregate_seed_metrics.py against the strategy dir; the last finisher in a
(ckpt, ds, strategy) group writes the complete seed_summary.csv.

A (ckpt, ds, strategy, seed) combo is skipped if its inverse_fold_metrics.csv
already exists on NFS.

Usage:
    # Edit NOISE_CKPTS / BASELINE_CKPTS / HYBRID_CKPTS below.
    python scripts/sweep_eval_per_seed.py
"""
import os
import sys

from slurm_launcher.sbatch_launcher import launch_tasks

PYTHON_BIN = "/data_fast/home/sihun/diffprotein/dplm/.venv/bin/python"
PART_TO_PY = {"rtx3090": PYTHON_BIN, "ada": PYTHON_BIN}

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/run_eval_single_seed.sh"

# Path under generation-results/ for this sweep. Each max_iter gets its own
# subdir: "<RESULT_SUBDIR>_{max_iter}iter".
RESULT_SUBDIR = "eval_per_seed_sweep"


def result_subdir(max_iter: int) -> str:
    return f"{RESULT_SUBDIR}_{max_iter}iter"


def result_base(max_iter: int) -> str:
    return f"{PROJECT_DIR}/generation-results/{result_subdir(max_iter)}"

# ─── Checkpoints by model kind ────────────────────────────────────────────────
# Absolute paths to .ckpt files. Leave a list empty to skip that kind entirely.
NOISE_CKPTS: list[str] = [
    # f"{PROJECT_DIR}/train_logs/.../checkpoints/step_*.ckpt",
    # f"{PROJECT_DIR}/train_logs/noiseFT/noise-full-1e4_lr-fs0-cutL0/checkpoints/step_8202.0-loss_0.70.ckpt", # ON
    # f"{PROJECT_DIR}/train_logs/noiseFT2/noise-full-1e4_lr-linear/checkpoints/step_9025.0-loss_0.22.ckpt", # ON
    # Cross-eval: hybrid-trained ckpts run through noise inference. ckpt
    # hardlinked into cross_eval/ with a noise-style .hydra/config.yaml
    # (hybrid: block stripped, sample_noise_every_step added). Same weights as
    # the corresponding entry in HYBRID_CKPTS — isolates train- vs inference-
    # time gain of hybrid.
    f"{PROJECT_DIR}/train_logs/cross_eval/FT9_as_noise/checkpoints/step_9843.0-loss_0.68.ckpt",
    f"{PROJECT_DIR}/train_logs/cross_eval/FTO3_as_noise/checkpoints/step_10256.0-loss_0.70.ckpt",
]

BASELINE_CKPTS: list[str] = [
    # f"{PROJECT_DIR}/train_logs/.../checkpoints/step_*.ckpt",
    # "hf:<org>/<name>"  -> evaluate untuned pretrained HF model (sanity / repro)
    # f"{PROJECT_DIR}/train_logs/noiseFT/baseline-full-1e4_lr-fs0/checkpoints/step_6971.0-loss_0.22.ckpt", # ON
    # f"{PROJECT_DIR}/train_logs/noiseFT/baseline-full-1e5_lr-fs0/checkpoints/step_7792.0-loss_0.22.ckpt",
    # f"{PROJECT_DIR}/train_logs/noiseFT2/baseline-full-1e4_lr-constant/checkpoints/step_7176.0-loss_0.70.ckpt", # ON
    # "hf:airkingbd/dplm2_650m",
]

HYBRID_CKPTS: list[str] = [
    # f"{PROJECT_DIR}/train_logs/.../checkpoints/step_*.ckpt",
    # f"{PROJECT_DIR}/train_logs/FT6/emb-high_noise-full-ema_lr_745-fs0-normemb/checkpoints/step_9005.0-loss_0.67.ckpt", #FT6 vs FT9 -> FT9
    # f"{PROJECT_DIR}/train_logs/FT9/emb-high_noise-full-1e4_lr/checkpoints/step_9843.0-loss_0.68.ckpt", # ON
    # f"{PROJECT_DIR}/train_logs/FT10/emb-high_noise-full-3e5_lr-linear/checkpoints/step_15995.0-loss_0.20.ckpt",
    # f"{PROJECT_DIR}/train_logs/FT10/emb-1high_noise-full-1e4_lr-linear/checkpoints/step_15995.0-loss_0.21.ckpt", # ON
    # f"{PROJECT_DIR}/train_logs/FT10/emb-xhigh_noise-full-1e4_lr-linear/checkpoints/step_15995.0-loss_0.22.ckpt"
    # f"{PROJECT_DIR}/train_logs/FT10/emb-2xhigh_noise-full-1e4_lr-linear/checkpoints/step_15995.0-loss_0.22.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO3/oh-xhigh_noise-full-3e5_lr-fs0-normemb/checkpoints/step_10256.0-loss_0.70.ckpt", # ON
    # Cross-eval: noise-trained ckpt run through hybrid inference. Same noise
    # ckpt hardlinked twice into cross_eval/ with a hybrid-style .hydra/
    # config.yaml — once with FT9-style hybrid block (embedding, sigma 0.5-5),
    # once with FTO3-style (onehot, r 0.25-0.47). Isolates train- vs inference-
    # time gain of hybrid.
    f"{PROJECT_DIR}/train_logs/cross_eval/noise_as_hybrid_FT9/checkpoints/step_8202.0-loss_0.70.ckpt",
    f"{PROJECT_DIR}/train_logs/cross_eval/noise_as_hybrid_FTO3/checkpoints/step_8202.0-loss_0.70.ckpt",
]

DATASETS = ["cameo2022", "PDB_date"]#, "cath_4.2_all", "cath_4.3_all"]

# Per-dataset batch size tuning
BATCH_SIZE_BY_DATASET = {
    "cameo2022": 5,
    "PDB_date": 5,
    "cath_4.2_all": 5,
    "cath_4.3_all": 5,
}

# ─── Sweep axes ───────────────────────────────────────────────────────────────
# Outer sweep: max_iter values (keys). Inner sweep (per max_iter): annealing
# configs (values). Each max_iter can have a different annealing-strength sweep.
# Each max_iter writes into its own result subdir ("<RESULT_SUBDIR>_{N}iter"),
# so strategy_tag stays bare (e.g. "argmax", "annealing2.0_0.1").
ANNEALING_SWEEP_BY_MAX_ITER: dict[int, list[tuple[str, str]]] = {
    1: [
        # ("annealing@1.0:0.1", "annealing1.0_0.1"),
        # ("annealing@0.7:0.1", "annealing0.7_0.1"),
        # ("annealing@0.5:0.1", "annealing0.5_0.1"),
        # ("annealing@0.1:0.1", "annealing0.1_0.1"),
    ],
    3: [
        # ("annealing@1.0:0.1", "annealing1.0_0.1"),
        # ("annealing@0.7:0.1", "annealing0.7_0.1"),
        # ("annealing@0.5:0.1", "annealing0.5_0.1"),
        # ("annealing@0.1:0.1", "annealing0.1_0.1"),
        # ("annealing@0.1:0.01", "annealing0.1_0.01"),
        # ("annealing@1.0:1.0", "annealing1.0_1.0"),
        # ("annealing@2.0:1.0", "annealing2.0_1.0"),
    ],
    5: [
        # ("annealing@1.0:0.1", "annealing1.0_0.1"),
        # ("annealing@0.7:0.1", "annealing0.7_0.1"),
        # ("annealing@0.5:0.1", "annealing0.5_0.1"),
        # ("annealing@0.3:0.1", "annealing0.3_0.1"),
        # ("annealing@0.1:0.1", "annealing0.1_0.1"),
        # ("annealing@0.1:0.01", "annealing0.1_0.01"),
        # ("annealing@1.0:1.0", "annealing1.0_1.0"),
        # ("annealing@2.0:1.0", "annealing2.0_1.0"),
    ],
    10: [
        # ("annealing@2.0:0.1", "annealing2.0_0.1"),
        # ("annealing@1.0:0.1", "annealing1.0_0.1"),
        # ("annealing@0.7:0.1", "annealing0.7_0.1"),
        # ("annealing@0.5:0.1", "annealing0.5_0.1"),
        # ("annealing@0.3:0.1", "annealing0.3_0.1"),
        # ("annealing@1.0:1.0", "annealing1.0_1.0"),
        ("annealing@2.0:1.0", "annealing2.0_1.0"),
        ("annealing@3.0:1.0", "annealing3.0_1.0"),
        # ("annealing@5.0:1.0", "annealing5.0_1.0"),
    ],
    30: [
        # ("annealing@2.0:0.1", "annealing2.0_0.1"),
        # ("annealing@1.0:0.1", "annealing1.0_0.1"),
        # ("annealing@0.7:0.1", "annealing0.7_0.1"),
        # ("annealing@0.5:0.1", "annealing0.5_0.1"),
        # ("annealing@1.0:1.0", "annealing1.0_1.0"),
        ("annealing@2.0:1.0", "annealing2.0_1.0"),
        ("annealing@3.0:1.0", "annealing3.0_1.0"),
    ],
    100: [
        # ("annealing@4.0:0.1", "annealing4.0_0.1"),
        # ("annealing@3.0:0.1", "annealing3.0_0.1"),
        # ("annealing@2.0:0.1", "annealing2.0_0.1"),
        # ("annealing@1.0:0.1", "annealing1.0_0.1"),
        # ("annealing@1.0:1.0", "annealing1.0_1.0"),
        ("annealing@2.0:1.0", "annealing2.0_1.0"),
        ("annealing@3.0:1.0", "annealing3.0_1.0"),
        # ("annealing@5.0:1.0", "annealing5.0_1.0"),
    ],
    # 500: [
    #     # ("annealing@10.0:0.1", "annealing10.0_0.1"),
    #     # ("annealing@5.0:0.1", "annealing5.0_0.1"),
    #     # ("annealing@2.0:0.1", "annealing2.0_0.1"),
    #     ("annealing@2.0:1.0", "annealing2.0_1.0"),
    #     ("annealing@5.0:1.0", "annealing5.0_1.0"),
    #     ("annealing@10.0:1.0", "annealing10.0_1.0"),
    # ],
}
MAX_ITER_SWEEP: list[int] = sorted(ANNEALING_SWEEP_BY_MAX_ITER.keys())

# Seed policy: seeds = [SEED_BASE + i for i in range(N)]
SEED_BASE = 42
NUM_SEEDS = 10


def _seeds(n: int) -> list[int]:
    return [SEED_BASE + i for i in range(n)]


# Per-kind seed assignment for argmax / annealing.
# argmax for baseline uses a single seed (deterministic up to ties);
# everything else fans out across NUM_SEEDS.
ARGMAX_SEEDS_BY_KIND: dict[str, list[int]] = {
    "noise":    _seeds(NUM_SEEDS),
    "baseline": [SEED_BASE],
    "hybrid":   _seeds(NUM_SEEDS),
}
ANNEALING_SEEDS_BY_KIND: dict[str, list[int]] = {
    "noise":    _seeds(NUM_SEEDS),
    "baseline": _seeds(NUM_SEEDS),
    "hybrid":   _seeds(NUM_SEEDS),
}

# Remasking strategy sweep — baseline only. For noise/hybrid the value is
# passed through but ignored (the bash script only forwards it when
# MODEL_KIND=baseline and only suffixes EXP_NAME for baseline). Allowed
# values: "uncond", "cond", "no_remask".
REMASKINGS_BY_KIND: dict[str, list[str]] = {
    "noise":    ["no_remask"],          # ignored; single placeholder
    "baseline": ["uncond"],             # "no_remask", "uncond"
    "hybrid":   ["no_remask"],          # ignored; single placeholder
}

# sample_noise_every_step sweep — noise only. Inference-time knob that toggles
# whether the Gaussian noise at still-masked positions is redrawn each step
# ("true" -> EXP suffix "_redraw") or kept fixed from init ("false" -> "_freeze").
# For baseline/hybrid the value is passed through but ignored (the bash script
# only forwards/suffixes it when MODEL_KIND=noise). Use "" to inherit from the
# ckpt's training cfg without applying a suffix.
NOISE_SAMPLE_MODES_BY_KIND: dict[str, list[str]] = {
    "noise":    ["false"],      # sweep both
    "baseline": [""],                   # ignored; single placeholder
    "hybrid":   [""],                   # ignored; single placeholder
}


def build_plan(kind: str) -> list[tuple[str, str, list[int], int]]:
    """(sampling, strategy_tag, seeds, max_iter) tuples for `kind`.

    max_iter goes into the result subdir, so strategy_tag stays bare
    (e.g. "argmax", "annealing2.0_0.1").
    """
    plan: list[tuple[str, str, list[int], int]] = []
    for max_iter in MAX_ITER_SWEEP:
        # plan.append(("argmax", "argmax",
        #              ARGMAX_SEEDS_BY_KIND[kind], max_iter))
        for sampling, base_tag in ANNEALING_SWEEP_BY_MAX_ITER[max_iter]:
            plan.append((sampling, base_tag,
                         ANNEALING_SEEDS_BY_KIND[kind], max_iter))
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
    ckpt_path: str, kind: str, remasking: str, noise_mode: str,
) -> tuple[str, str]:
    if is_hf(ckpt_path):
        # hf:airkingbd/dplm2_650m -> ("pretrained_hf", "dplm2_650m")
        exp = "pretrained_hf"
        base = ckpt_path[len(HF_PREFIX):].split("/")[-1]
    else:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        exp = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    if kind == "baseline":
        # Mirrors the suffix applied in run_eval_single_seed.sh
        exp = f"{exp}_{remasking}"
    if kind == "noise":
        # Mirrors the suffix applied in run_eval_single_seed.sh
        exp = f"{exp}{_NOISE_MODE_SUFFIX[noise_mode]}"
    return exp, base


def metric_csv_path(ckpt_path: str, ds: str, strategy_tag: str, seed: int,
                    max_iter: int, kind: str, remasking: str,
                    noise_mode: str) -> str:
    exp, base = exp_and_base(ckpt_path, kind, remasking, noise_mode)
    return os.path.join(
        result_base(max_iter), exp, ds, base, strategy_tag,
        f"seed_{seed}", "inverse_folding", "aatype", "eval",
        "inverse_fold_metrics.csv",
    )


def run_slurm():
    all_ckpts = [(k, p) for k, lst in GROUPS for p in lst]
    if not all_ckpts:
        print("No checkpoints — edit NOISE_CKPTS / BASELINE_CKPTS / HYBRID_CKPTS "
              "in scripts/sweep_eval_per_seed.py.")
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
    for kind, ckpts in GROUPS:
        if not ckpts:
            continue
        plan = build_plan(kind)
        plan_summary = ", ".join(f"{tag}x{len(seeds)}" for _, tag, seeds, _ in plan)
        remaskings = REMASKINGS_BY_KIND[kind]
        rmsk_summary = (
            f" × remask{remaskings}" if kind == "baseline" else ""
        )
        noise_modes = NOISE_SAMPLE_MODES_BY_KIND[kind]
        noise_summary = (
            f" × sample_noise_every_step{noise_modes}" if kind == "noise" else ""
        )
        print(
            f"  [{kind}] {len(ckpts)} ckpts × {len(DATASETS)} datasets × "
            f"({plan_summary}){rmsk_summary}{noise_summary}"
        )

    combos: list[str] = []
    skipped = 0
    for kind, ckpts in GROUPS:
        plan = build_plan(kind)
        remaskings = REMASKINGS_BY_KIND[kind]
        noise_modes = NOISE_SAMPLE_MODES_BY_KIND[kind]
        for ckpt_path in ckpts:
            for ds in DATASETS:
                bs = BATCH_SIZE_BY_DATASET.get(ds, 50)
                for sampling, tag, seeds, max_iter in plan:
                    subdir = result_subdir(max_iter)
                    for remasking in remaskings:
                        for noise_mode in noise_modes:
                            for seed in seeds:
                                if os.path.isfile(metric_csv_path(
                                    ckpt_path, ds, tag, seed, max_iter,
                                    kind, remasking, noise_mode,
                                )):
                                    skipped += 1
                                    continue
                                # Empty noise_mode (non-noise kinds) -> omit
                                # the trailing arg; the bash script's ${11:-}
                                # defaults it to empty (= inherit / ignored).
                                tail = f" {noise_mode}" if noise_mode else ""
                                cmd = (
                                    f"{SCRIPT_PATH} {kind} {ckpt_path} {ds} {seed} "
                                    f"{sampling} {max_iter} {bs} {subdir} {tag} "
                                    f"{remasking}{tail}"
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
        partition="rtx3090",
        exclude="radish",
        qos="normal",
        timeout="5-0",
        job_name="eval_per_seed",
        max_job_num=80,
        part_to_py=PART_TO_PY,
    )


if __name__ == "__main__":
    run_slurm()
