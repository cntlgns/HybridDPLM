"""
SLURM sweep over (dataset, sampling, remasking, max_iter) for DPLM2 inverse
folding, running multiple seeds per (combo) job.

Wraps scripts/run_dplm2_if_seeds.sh. For each (dataset, sampling, remasking,
max_iter) combo, one SLURM job:
  - loads the model ONCE on a single GPU
  - generates with N seeds (bundled inside run_dplm2_if_seeds.sh)
  - runs evaluator per seed
  - writes seed_summary.csv via aggregate_seed_metrics.py

Output layout:
    generation-results/<RESULT_SUBDIR>/<dataset>/<sampling>_<remasking>_iter<N>/
        seed_<S>/inverse_folding/aatype/{eval,...}
        seed_summary.csv

A combo is skipped if its seed_summary.csv already contains all requested seeds.

Usage:
    # Edit the sweep axes below, then:
    python scripts/sweep_dplm2_if_seeds.py [num_seeds]
"""
import itertools
import os
import sys

from slurm_launcher.sbatch_launcher import launch_tasks

PYTHON_BIN = "/data_fast/home/sihun/diffprotein/dplm/.venv/bin/python"
PART_TO_PY = {"rtx3090": PYTHON_BIN, "ada": PYTHON_BIN, "a100": PYTHON_BIN}

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/run_dplm2_if_seeds.sh"

# ─── Output root (mirrors hybrid sweep convention) ────────────────────────────
RESULT_SUBDIR = "dplm2_650m_invfold_seedsweep"
RESULT_BASE = f"{PROJECT_DIR}/generation-results/{RESULT_SUBDIR}"

# ─── Fixed axes ───────────────────────────────────────────────────────────────
MODEL_NAME = "dplm2_650m"

# ─── Sweep axes ───────────────────────────────────────────────────────────────
DATASETS = ["cameo2022", "PDB_date"]
SAMPLINGS = ["annealing@2.0:0.1"]
REMASKINGS = ["uncond", "no_remask"]
MAX_ITERS = [1, 5, 30, 100]

# Per-dataset batch size tuning (matches run_hybrid_eval.py convention)
BATCH_SIZE_BY_DATASET = {"cameo2022": 20, "PDB_date": 20}

# Seed policy: seeds = [SEED_BASE + i for i in range(NUM_SEEDS)]
SEED_BASE = 42
NUM_SEEDS_DEFAULT = 10


def combo_dir(sampling: str, remasking: str, max_iter: int) -> str:
    return f"{sampling}_{remasking}_iter{max_iter}"


def summary_path(dataset: str, sampling: str, remasking: str, max_iter: int) -> str:
    return os.path.join(
        RESULT_BASE, dataset, combo_dir(sampling, remasking, max_iter),
        "seed_summary.csv",
    )


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
    seeds_csv = make_seeds_csv(num_seeds)
    requested = {SEED_BASE + i for i in range(num_seeds)}
    print(f"Seeds: {seeds_csv}")
    print(f"Model: {MODEL_NAME}   Datasets: {DATASETS}")
    print(f"Samplings: {SAMPLINGS}   Remaskings: {REMASKINGS}   MaxIters: {MAX_ITERS}")

    cmds = []
    for ds, sampling, remask, max_iter in itertools.product(
        DATASETS, SAMPLINGS, REMASKINGS, MAX_ITERS
    ):
        cd = combo_dir(sampling, remask, max_iter)
        sp = summary_path(ds, sampling, remask, max_iter)
        done = existing_seeds(sp)
        todo = requested - done
        if not todo:
            print(f"  [skip] {ds} / {cd} (all requested seeds present)")
            continue
        if done:
            print(f"  [add ] {ds} / {cd} "
                  f"(have {sorted(done)}, adding {sorted(todo)})")
        else:
            print(f"  [run ] {ds} / {cd}")
        bs = BATCH_SIZE_BY_DATASET.get(ds, 50)
        cmd = (
            f"{SCRIPT_PATH} {ds} {MODEL_NAME} {seeds_csv} "
            f"{sampling} {remask} {max_iter} {bs} {RESULT_SUBDIR}"
        )
        cmds.append(cmd)

    if not cmds:
        print("Nothing to submit.")
        return

    print(f"Submitting {len(cmds)} jobs")
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": cmds},
        partition="rtx3090",
        qos="normal",
        timeout="2-0",
        job_name="dplm2_if_seedsweep",
        max_job_num=64,
        part_to_py=PART_TO_PY,
    )


if __name__ == "__main__":
    num_seeds = int(sys.argv[1]) if len(sys.argv) >= 2 else NUM_SEEDS_DEFAULT
    run_slurm(num_seeds)
