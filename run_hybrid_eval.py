"""
SLURM launcher for hybrid eval experiments.
Finds all .ckpt files under CKPT_ROOT, skips those already evaluated,
and submits jobs for the rest on rtx3090 x1.

Usage:
    python run_hybrid_eval.py <ckpt_root_dir>

Example:
    python run_hybrid_eval.py /data_fast/home/sihun/diffprotein/dplm/train_logs/candi_const_weight_ft5/checkpoints
"""
import sys
import os
import glob

from slurm_launcher.sbatch_launcher import launch_tasks

PYTHON_BIN = "/data_fast/home/sihun/diffprotein/dplm/.venv/bin/python"
PART_TO_PY = {
    'rtx3090': PYTHON_BIN,
}

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
SCRIPT_PATH = f"{PROJECT_DIR}/run_hybrid_eval.sh"
RESULT_BASE = f"{PROJECT_DIR}/generation-results/hybrid_invfold_test"

DATASETS = ["cameo2022", "PDB_date"] # "cameo2022", "PDB_date"
EXCLUDE_CKPTS = {"best.ckpt", "last.ckpt"}
EXCLUDE_PREFIXES = [
    # "step_999.0",
    # "step_1999.0",
    # "step_2999.0",
    # "step_3999.0",
    # "step_4999.0",
    # "step_5999.0",
    # "step_6999.0",
    # "step_7999.0",
    "step_8999.0",
    "step_9999.0",
    "step_10999.0",
    "step_11999.0",
    "step_12999.0",
    "step_13999.0",
    "step_14999.0",
    "step_15999.0",
    "step_16999.0",
    "step_17999.0",
    "step_18999.0",
    "step_19999.0",
    "step_20999.0",
    "step_21999.0",
    "step_22999.0",
    "step_23999.0",
    "step_24999.0",
    "step_25999.0",
    "step_26999.0",
    "step_27999.0",
    "step_28999.0",
    "step_29999.0",
    "step_30999.0",
    "step_31999.0",
    "step_32999.0",
    "step_33999.0",
]


def find_ckpts(ckpt_root: str):
    """Find all .ckpt files recursively under ckpt_root, excluding best/last and specific prefixes."""
    pattern = os.path.join(ckpt_root, "**", "*.ckpt")
    return sorted(
        p for p in glob.glob(pattern, recursive=True)
        if os.path.basename(p) not in EXCLUDE_CKPTS
        and not any(os.path.basename(p).startswith(prefix) for prefix in EXCLUDE_PREFIXES)
    )


def is_already_done(ckpt_path: str, datasets: list[str]) -> bool:
    """
    Check if this checkpoint has already been evaluated for ALL datasets.
    Result path convention: {RESULT_BASE}/{exp_name}/{dataset}/{ckpt_basename}/inverse_folding/
    """
    ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_dir = os.path.dirname(ckpt_path)
    exp_dir = os.path.dirname(ckpt_dir)
    exp_name = os.path.basename(exp_dir)

    for ds in datasets:
        result_dir = os.path.join(RESULT_BASE, exp_name, ds, ckpt_basename, "inverse_folding")
        if not os.path.isdir(result_dir):
            return False
    return True


def run_slurm(ckpt_root: str):
    ckpts = find_ckpts(ckpt_root)
    print(f"Found {len(ckpts)} checkpoints under {ckpt_root}")

    # Filter out already-evaluated checkpoints
    new_ckpts = [c for c in ckpts if not is_already_done(c, DATASETS)]
    skipped = len(ckpts) - len(new_ckpts)
    print(f"Skipping {skipped} already-evaluated checkpoints")
    print(f"Submitting {len(new_ckpts)} new checkpoints")

    if not new_ckpts:
        print("Nothing to do.")
        return

    # Each checkpoint x each dataset = one job
    combos = []
    for ckpt_path in new_ckpts:
        for ds in DATASETS:
            # Check per-dataset: skip if this specific dataset is already done
            ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0]
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
            result_dir = os.path.join(RESULT_BASE, exp_name, ds, ckpt_basename, "inverse_folding")
            if os.path.isdir(result_dir):
                print(f"  [skip] {ckpt_basename} / {ds} (already done)")
                continue
            cmd = f"{SCRIPT_PATH} {ckpt_path} {ds}"
            combos.append(cmd)

    if not combos:
        print("All dataset-checkpoint combinations already done.")
        return

    print(f"Total jobs to submit: {len(combos)}")
    PARAM_DICT = {"": combos}

    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict=PARAM_DICT,
        partition="rtx3090",
        exclude="pearl,nobel,ohm,poincare,quant,rene,radish,kiwi",
        qos="normal",
        timeout="3-0",
        job_name="hybrid_eval",
        max_job_num=50,
        part_to_py=PART_TO_PY,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_hybrid_eval.py <ckpt_root_dir>")
        print("Example: python run_hybrid_eval.py train_logs/candi_const_weight_ft5/checkpoints")
        sys.exit(1)

    ckpt_root = sys.argv[1]
    if not os.path.isdir(ckpt_root):
        print(f"Error: {ckpt_root} is not a directory")
        sys.exit(1)

    run_slurm(ckpt_root)
