"""
SLURM sweep over max_iter and remasking_strategy for DPLM2 inverse folding.

Wraps scripts/run_dplm2_if.sh. Each submitted job runs one (dataset, max_iter,
remasking_strategy) combo with fixed exp_name / model / sampling_strategy.

Results land under:
    generation-results/<EXP_NAME>/<DATASET>/<MODEL_NAME>/<SAMPLING_STRATEGY>/<REMASKING>[/iter<N>]

A (dataset, max_iter, remasking) combo is skipped if its row already exists in
    generation-results/<EXP_NAME>/summary.csv

Usage:
    # Edit the sweep axes below, then:
    python scripts/sweep_dplm2_if.py
"""

import itertools
import os
import sys

import pandas as pd

from slurm_launcher.sbatch_launcher import launch_tasks

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/run_dplm2_if.sh"
PART_TO_BASH = {"rtx3090": "/bin/bash", "ada": "/bin/bash", "a100": "/bin/bash"}

# ─── Fixed axes ───────────────────────────────────────────────────────────────
EXP_NAME = "3B_invfold_reproduction"
MODEL_NAME = "dplm2_3b"
SAMPLING_STRATEGY = "argmax"
BATCH_SIZE_BY_DATASET = {"cameo2022": 1, "PDB_date": 1}

# ─── Sweep axes ───────────────────────────────────────────────────────────────
DATASETS = ["cameo2022", "PDB_date"]  # add "PDB_date" if desired
MAX_ITERS = [1, 5, 10, 30, 50, 100, 500]
REMASKING_STRATEGIES = ["uncond", "no_remask"]


def already_done(summary_csv: str, dataset: str, max_iter: int, remasking: str) -> bool:
    if not os.path.isfile(summary_csv):
        return False
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return False
    required = {"dataset", "max_iter", "remasking_strategy",
                "sampling_strategy", "model_name"}
    if not required.issubset(df.columns):
        return False
    match = df[
        (df["dataset"] == dataset)
        & (df["max_iter"].astype(int) == int(max_iter))
        & (df["remasking_strategy"] == remasking)
        & (df["sampling_strategy"] == SAMPLING_STRATEGY)
        & (df["model_name"] == MODEL_NAME)
    ]
    return len(match) > 0


def generate_commands() -> list[str]:
    summary_csv = os.path.join(PROJECT_DIR, "generation-results", EXP_NAME, "summary.csv")

    cmds = []
    for dataset, max_iter, remasking in itertools.product(
        DATASETS, MAX_ITERS, REMASKING_STRATEGIES
    ):
        if already_done(summary_csv, dataset, max_iter, remasking):
            print(f"  [skip] {dataset} iter={max_iter} remask={remasking}")
            continue
        bs = BATCH_SIZE_BY_DATASET.get(dataset, 50)
        cmd = (
            f"{SCRIPT_PATH} {EXP_NAME} {dataset} {MODEL_NAME} "
            f"{SAMPLING_STRATEGY} {remasking} {bs} {max_iter}"
        )
        cmds.append(cmd)
    return cmds


def main():
    cmds = generate_commands()
    print(f"Total jobs to submit: {len(cmds)}")
    for c in cmds:
        print(f"  {c}")

    if not cmds:
        print("Nothing to submit.")
        return

    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": cmds},
        partition="rtx3090",
        qos="normal",
        timeout="1-0",
        job_name="if_sweep",
        max_job_num=64,
        part_to_py=PART_TO_BASH,
    )


if __name__ == "__main__":
    main()
