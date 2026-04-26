"""
SLURM launcher for noise-schedule sweep of *base* DPLM2-650M (no finetuning)
with hybrid diffusion inference.

The swept knob depends on noise_space:
  - embedding -> sigma_min / sigma_max (VE-SDE schedule)
  - onehot    -> r_min / r_max (Eq. 10; both must be in (0, 0.5))

Submits one job per (noise_space, noise_min, noise_max, dataset) combination.
Skips combinations whose result directory already exists.

Usage:
    python scripts/sweep_hybrid_eval_sigma.py
"""
import itertools
import os

from slurm_launcher.sbatch_launcher import launch_tasks

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
PYTHON_BIN = f"{PROJECT_DIR}/.venv/bin/python"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/run_hybrid_eval_sigma.sh"

PART_TO_PY = {
    "rtx3090": PYTHON_BIN,
    "ada": PYTHON_BIN,
}

# Base model for the sweep (no ckpt, no finetuning).
MODEL_NAME = "airkingbd/dplm2_650m"
MODEL_TAG = os.path.basename(MODEL_NAME)

# Results on NFS (rsync target inside run_hybrid_eval_sigma.sh).
# Must match RESULT_NS in run_hybrid_eval_sigma.sh.
RESULT_BASE = f"{PROJECT_DIR}/generation-results/hybrid_noft_sigma_sweep2"

DATASETS = ["cameo2022", "PDB_date"] # "cameo2022", "PDB_date"
SAMPLING = "argmax"
MAX_ITER = 100
BATCH_SIZE = 20

# ─── Sweep axes ───────────────────────────────────────────────────────────────

# Whether to apply the masked-residual layer-0 attention patch
# (matches cutoff_layer0_attn_residual in the training yaml).
# Set e.g. [False, True] to sweep on/off.
CUTOFF_L0_RESIDUALS = [True] # True, False

# Whether to row-wise L2-normalize embedding weights when used as a noise
# basis (onehot init / training) or as E_Y0 in the ODE step.
# Matches hybrid.use_normal_emb in the model config.
USE_NORMAL_EMBS = [True] # True, False

NOISE_SPACES = ["onehot"]  # "embedding", "onehot"

# Per-noise-space schedules. The (min, max) tuples mean different things:
#   embedding -> (sigma_min, sigma_max), VE-SDE
#   onehot    -> (r_min, r_max), Eq. 10  (both must lie in (0, 0.5))
NOISE_SCHEDULE_CONFIG = {
    "embedding": [
        (0.2,  2.0,  "low"),
        (0.5,  5.0,  "default"),
        (1.0,  10.0, "high"),
        (1.0,  15.0, "xhigh"),
        (2.0,  20.0, "xxhigh"),
    ],
    "onehot": [
        (0.005, 0.05, "xxlow"),
        (0.01, 0.10, "xlow"),
        (0.01, 0.25, "default"),
        (0.10, 0.49, "high"),
        (0.20, 0.49, "xhigh"),
        (0.20, 0.495, "xxhigh"),
        (0.01, 0.49, "wide"),
    ],
}

# Prefix used in the run-directory name, must match run_hybrid_eval_sigma.sh.
NOISE_PREFIX = {"embedding": "s", "onehot": "r"}


def run_name(
    noise_space: str, tag: str, n_min: float, n_max: float,
    cutoff_l0: bool, use_normal_emb: bool,
) -> str:
    """Must match RUN_NAME logic in run_hybrid_eval_sigma.sh."""
    l0_suffix = "_cutL0" if cutoff_l0 else ""
    norm_suffix = "_normE" if use_normal_emb else ""
    prefix = NOISE_PREFIX[noise_space]
    return (
        f"{MODEL_TAG}_{noise_space}_{tag}_{prefix}{n_min}-{n_max}"
        f"{l0_suffix}{norm_suffix}"
    )


def is_done(
    noise_space: str, tag: str, n_min: float, n_max: float,
    cutoff_l0: bool, use_normal_emb: bool, ds: str,
) -> bool:
    name = run_name(noise_space, tag, n_min, n_max, cutoff_l0, use_normal_emb)
    result_dir = os.path.join(RESULT_BASE, name, ds, "inverse_folding")
    return os.path.isdir(result_dir)


def build_commands():
    cmds = []
    for noise_space in NOISE_SPACES:
        schedule_list = NOISE_SCHEDULE_CONFIG[noise_space]
        for (n_min, n_max, tag), cutoff_l0, use_normal_emb, ds in itertools.product(
            schedule_list, CUTOFF_L0_RESIDUALS, USE_NORMAL_EMBS, DATASETS
        ):
            if is_done(noise_space, tag, n_min, n_max, cutoff_l0, use_normal_emb, ds):
                print(
                    f"  [skip] {run_name(noise_space, tag, n_min, n_max, cutoff_l0, use_normal_emb)} "
                    f"/ {ds} (already done)"
                )
                continue
            # Positional args: noise_min noise_max dataset sampling max_iter batch_size
            #                  noise_space tag cutoff_l0 use_normal_emb
            # (noise_min/noise_max are r_min/r_max for onehot, sigma_min/sigma_max for embedding;
            #  the bash launcher dispatches the correct CLI flags.)
            cmd = (
                f"{SCRIPT_PATH} {n_min} {n_max} {ds} "
                f"{SAMPLING} {MAX_ITER} {BATCH_SIZE} {noise_space} {tag} "
                f"{1 if cutoff_l0 else 0} {1 if use_normal_emb else 0}"
            )
            cmds.append(cmd)
    return cmds


def main():
    cmds = build_commands()
    print(f"Total jobs to submit: {len(cmds)}")
    if not cmds:
        print("Nothing to do.")
        return

    for c in cmds:
        print(f"  {c}")

    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": cmds},
        partition="rtx3090",
        qos="normal",
        timeout="1-0",
        job_name="hybrid_sigma_eval",
        max_job_num=180,
        part_to_py=PART_TO_PY,
    )


if __name__ == "__main__":
    main()
