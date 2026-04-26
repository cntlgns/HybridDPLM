"""
SLURM launcher: for each checkpoint in CKPTS (below) x each dataset,
submit one job that sweeps multiple seeds for hybrid-diffusion
inverse-folding evaluation.

Each job:
  - loads the model ONCE on a single GPU
  - generates with N seeds (bundled inside run_hybrid_eval_seeds.sh)
  - runs evaluator per seed
  - writes seed_summary.csv aggregating metrics across seeds
  - rsyncs to NFS

A (ckpt, dataset) combo is skipped if its seed_summary.csv already exists on NFS.

Usage:
    # Edit CKPTS below to pick which checkpoints to evaluate.
    python scripts/sweep_hybrid_eval_seeds.py [num_seeds]
"""
import os
import sys

from slurm_launcher.sbatch_launcher import launch_tasks

PYTHON_BIN = "/data_fast/home/sihun/diffprotein/dplm/.venv/bin/python"
PART_TO_PY = {"rtx3090": PYTHON_BIN, "ada": PYTHON_BIN}

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/run_hybrid_eval_seeds.sh"
# Path under generation-results/ for this sweep. Passed to the shell script so we
# can rename result folders without editing the script (which would break already-
# queued SLURM jobs that re-read the script when they start).
RESULT_SUBDIR = "hybrid_invfold_FT7final_refine_100iter_seedsweep"   # e.g. "hybrid_invfold_FT7_seedsweep"
RESULT_BASE = f"{PROJECT_DIR}/generation-results/{RESULT_SUBDIR}"

# ─── Fill in the checkpoints you want to evaluate ─────────────────────────────
# Absolute paths to .ckpt files. One SLURM job is submitted per (ckpt, dataset).
CKPTS: list[str] = [
    # e.g.
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-default_noise-16ln-orig_lr-fs0/checkpoints/step_5199.0-loss_0.47.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-default_noise-16ln-orig_lr-fs0-cutL0/checkpoints/step_5199.0-loss_0.47.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-default_noise-16ln-orig_lr-fs0-cutL0-normemb/checkpoints/step_5199.0-loss_0.48.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-default_noise-16ln-orig_lr-fs0-normemb/checkpoints/step_5199.0-loss_0.47.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-high_noise-16ln-orig_lr-fs0/checkpoints/step_5199.0-loss_0.64.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-high_noise-16ln-orig_lr-fs0-cutL0/checkpoints/step_5199.0-loss_0.65.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-high_noise-16ln-orig_lr-fs0-cutL0-normemb/checkpoints/step_5199.0-loss_0.64.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-high_noise-16ln-orig_lr-fs0-normemb/checkpoints/step_5199.0-loss_0.64.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-low_noise-16ln-orig_lr-fs0/checkpoints/step_5199.0-loss_0.28.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-low_noise-16ln-orig_lr-fs0-cutL0/checkpoints/step_5199.0-loss_0.28.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-low_noise-16ln-orig_lr-fs0-cutL0-normemb/checkpoints/step_5199.0-loss_0.28.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-low_noise-16ln-orig_lr-fs0-normemb/checkpoints/step_5199.0-loss_0.28.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-wide_noise-16ln-orig_lr-fs0/checkpoints/step_5199.0-loss_0.58.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-wide_noise-16ln-orig_lr-fs0-cutL0/checkpoints/step_5199.0-loss_0.59.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-wide_noise-16ln-orig_lr-fs0-cutL0-normemb/checkpoints/step_5199.0-loss_0.59.ckpt",
    # f"{PROJECT_DIR}/train_logs/FTO1/oh-wide_noise-16ln-orig_lr-fs0-normemb/checkpoints/step_5199.0-loss_0.58.ckpt",
    f"{PROJECT_DIR}/train_logs/FT7/emb-2xhigh_noise-full-low_lr-fs0/checkpoints/step_26017.0-loss_0.72.ckpt",
    f"{PROJECT_DIR}/train_logs/FT7/emb-high_noise-full-low_lr-fs0-normemb/checkpoints/step_15009.0-loss_0.71.ckpt",
    f"{PROJECT_DIR}/train_logs/FT7/emb-xhigh_noise-full-low_lr-fs0/checkpoints/step_26017.0-loss_0.72.ckpt",
    f"{PROJECT_DIR}/train_logs/FT7/emb-xhigh_noise-full-low_lr-fs0-normemb/checkpoints/step_26017.0-loss_0.72.ckpt",
    f"{PROJECT_DIR}/train_logs/FT6/emb-high_noise-full-ema_lr_745-fs0-normemb/checkpoints/step_9005.0-loss_0.67.ckpt",
    f"{PROJECT_DIR}/train_logs/FT6/emb-xhigh_noise-full-ema_lr_745-fs0-normemb/checkpoints/step_5003.0-loss_0.71.ckpt",

]

DATASETS = ["cameo2022", "PDB_date"] # "cameo2022", "PDB_date"

# Seed policy: seeds = [SEED_BASE + i for i in range(NUM_SEEDS)]
SEED_BASE = 42
NUM_SEEDS_DEFAULT = 10

# Per-dataset batch size tuning (matches run_hybrid_eval.py convention)
BATCH_SIZE_BY_DATASET = {"cameo2022": 5, "PDB_date": 5}
SAMPLING = "argmax"
MAX_ITER = 100


def exp_and_base(ckpt_path: str) -> tuple[str, str]:
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    exp = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    return exp, base


def summary_path(ckpt_path: str, ds: str) -> str:
    exp, base = exp_and_base(ckpt_path)
    return os.path.join(RESULT_BASE, exp, ds, base, "seed_summary.csv")


def existing_seeds(summary_csv: str) -> set[int]:
    """Read seed_summary.csv and return the set of per-seed integer rows."""
    if not os.path.isfile(summary_csv):
        return set()
    seeds = set()
    with open(summary_csv, "r") as f:
        header = f.readline().rstrip("\n").split(",")
        try:
            seed_idx = header.index("seed")
        except ValueError:
            return set()
        for line in f:
            cells = line.rstrip("\n").split(",")
            if seed_idx >= len(cells):
                continue
            tok = cells[seed_idx].strip()
            if not tok or tok.startswith("_"):
                continue
            try:
                seeds.add(int(tok))
            except ValueError:
                continue
    return seeds


def make_seeds_csv(num_seeds: int) -> str:
    return ",".join(str(SEED_BASE + i) for i in range(num_seeds))


def run_slurm(num_seeds: int):
    if not CKPTS:
        print("CKPTS list is empty — edit scripts/sweep_hybrid_eval_seeds.py "
              "to add checkpoint paths.")
        return

    # Validate all ckpt paths up front so SLURM doesn't waste jobs
    missing = [p for p in CKPTS if not os.path.isfile(p)]
    if missing:
        print("The following CKPTS do not exist:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)

    seeds_csv = make_seeds_csv(num_seeds)
    requested_seeds = {SEED_BASE + i for i in range(num_seeds)}
    print(f"Seeds: {seeds_csv}")
    print(f"Checkpoints: {len(CKPTS)}   Datasets: {DATASETS}")

    combos = []
    for ckpt_path in CKPTS:
        for ds in DATASETS:
            sp = summary_path(ckpt_path, ds)
            done = existing_seeds(sp)
            todo = requested_seeds - done
            if not todo:
                print(f"  [skip] {exp_and_base(ckpt_path)[1]} / {ds} "
                      f"(all requested seeds already in seed_summary.csv)")
                continue
            if done:
                print(f"  [add ] {exp_and_base(ckpt_path)[1]} / {ds} "
                      f"(have {sorted(done)}, adding {sorted(todo)})")
            bs = BATCH_SIZE_BY_DATASET.get(ds, 50)
            cmd = (
                f"{SCRIPT_PATH} {ckpt_path} {ds} {seeds_csv} "
                f"{SAMPLING} {MAX_ITER} {bs} {RESULT_SUBDIR}"
            )
            combos.append(cmd)

    if not combos:
        print("Nothing to submit.")
        return

    print(f"Submitting {len(combos)} jobs")
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": combos},
        partition="rtx3090",
        # exclude="peach,quiznos,udon", #"kiwi,lemon,mango,nutella,peach,quiznos,radish,tomato,udon,watermelon,xoi,yogurt,vanilla",
        qos="normal",
        timeout="3-0",
        job_name="hybrid_seedsweep",
        max_job_num=180,
        part_to_py=PART_TO_PY,
    )


if __name__ == "__main__":
    num_seeds = int(sys.argv[1]) if len(sys.argv) >= 2 else NUM_SEEDS_DEFAULT
    run_slurm(num_seeds)
