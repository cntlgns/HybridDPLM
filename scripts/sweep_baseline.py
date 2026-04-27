"""
DPLM2 baseline (vanilla discrete diffusion) 650M lr sweep on a100.

Counterpart to sweep_hybrid.py / sweep_noise.py for the third arm of the
ablation: standard DPLM2 mask-based discrete diffusion finetuning. Used to
disentangle whether the gain reported by hybrid diffusion comes from its
hybrid score machinery or from feeding the model noisy embeddings instead
of a single shared <mask> embedding.

Sweep axes:
  - lr schedule: primary axis the user wants to explore.
  - lora: full FT vs LoRA (default: full FT, matching the noise/hybrid a100 runs).
  - fullseq_loss_weight: kept as a knob for symmetry with the other sweeps.

Reuses ``scripts/train_and_sync.sh`` as-is.

Usage:
    python scripts/sweep_baseline.py
"""

import itertools
from slurm_launcher.sbatch_launcher import launch_tasks

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
PYTHON_BIN = f"{PROJECT_DIR}/.venv/bin/python"
TRAIN_AND_SYNC_SCRIPT = f"{PROJECT_DIR}/scripts/train_and_sync.sh"

LOCAL_LOG_BASE = "/data_large/unsynced_store/sihun/diffprotein/dplm/train_logs"

PART_TO_BASH = {
    "rtx3090": "/bin/bash",
    "ada": "/bin/bash",
    "a100": "/bin/bash",
}

# ─── Sweep axes ───────────────────────────────────────────────────────────────

# (enable, rank, train_layer_norm, tag)
LORA_CONFIGS = [
    # (True,  16, True,  "16ln"),
    # (True,  64, True,  "64ln"),
    (False, None, None, "full"),
]

# (warmup_init_lr, lr, lr_end, warmup_steps, max_steps, tag)
LR_SCHEDULE_CONFIGS = [
    # (1e-7, 5e-4, 1e-7, 2000, 41000, "5e4_lr"),
    # (1e-7, 2e-4, 1e-7, 2000, 41000, "2e4_lr"),
    (1e-7, 1e-4, 1e-7, 2000, 41000, "1e4_lr"),
    (1e-7, 1e-5, 1e-7, 2000, 41000, "1e5_lr"),
]

# (fullseq_loss_weight, tag)
FULLSEQ_LOSS_WEIGHTS = [
    (0.0, "fs0"),
    # (0.1, "fs0.1"),
]


def make_job_name(lora_tag, lr_tag, fs_tag):
    """e.g. baseline-full-1e4_lr-fs0"""
    return f"baseline-{lora_tag}-{lr_tag}-{fs_tag}"


def make_overrides(name, group,
                   lora_enable, lora_rank, lora_train_ln,
                   warmup_init_lr, lr, lr_end, warmup_steps, max_steps,
                   fullseq_loss_weight):
    local_log_dir = f"{LOCAL_LOG_BASE}/{name}"
    ovs = [
        f"experiment=dplm2/dplm2_650m_baseline",
        f"name={name}",
        f"paths.log_dir={local_log_dir}",
        f"logger=wandb",
        f"logger.wandb.group={group}",
    ]

    if lora_enable:
        ovs += [
            "model.lora.enable=true",
            f"model.lora.lora_rank={lora_rank}",
            f"model.lora.train_layer_norm={'true' if lora_train_ln else 'false'}",
            "datamodule.max_tokens=1600",
            "trainer.accumulate_grad_batches=40",
            "callbacks.model_checkpoint.every_n_train_steps=400",
        ]
    else:
        ovs += [
            "model.lora.enable=false",
        ]

    ovs += [
        f"task.lr_scheduler.warmup_init_lr={warmup_init_lr}",
        f"train.lr={lr}",
        f"task.lr_scheduler.lr_end={lr_end}",
        f"task.lr_scheduler.warmup_steps={warmup_steps}",
        f"trainer.max_steps={max_steps}",
        f"+task.learning.fullseq_loss_weight={fullseq_loss_weight}",
        # Match noise/hybrid sweep cadence so val curves line up and a100 hours
        # aren't wasted on baseline's default 500-step validations.
        # Checkpoint saving piggybacks on validation, so this also aligns saves.
        "trainer.val_check_interval=3280",
    ]

    return " ".join(ovs)


def generate_commands():
    lora_cmds = []  # rtx3090
    full_cmds = []  # a100

    for lora_cfg, lr_cfg, fs_cfg in itertools.product(
        LORA_CONFIGS, LR_SCHEDULE_CONFIGS, FULLSEQ_LOSS_WEIGHTS
    ):
        lora_enable, lora_rank, lora_train_ln, lora_tag = lora_cfg
        warmup_init_lr, lr, lr_end, warmup_steps, max_steps, lr_tag = lr_cfg
        fullseq_loss_weight, fs_tag = fs_cfg

        name = make_job_name(lora_tag, lr_tag, fs_tag)
        group = "baseline"

        override_str = make_overrides(
            name, group,
            lora_enable, lora_rank, lora_train_ln,
            warmup_init_lr, lr, lr_end, warmup_steps, max_steps,
            fullseq_loss_weight,
        )

        cmd = f"{TRAIN_AND_SYNC_SCRIPT} {override_str}"

        if lora_enable:
            lora_cmds.append(cmd)
        else:
            full_cmds.append(cmd)

    return lora_cmds, full_cmds


def main():
    lora_cmds, full_cmds = generate_commands()
    print(f"LoRA jobs (rtx3090): {len(lora_cmds)}")
    print(f"Full FT jobs (a100): {len(full_cmds)}")
    print(f"Total: {len(lora_cmds) + len(full_cmds)}")

    if lora_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": lora_cmds},
            partition="rtx3090",
            exclude="alpaca",
            qos="normal",
            timeout="5-0",
            job_name="baseft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )

    if full_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": full_cmds},
            partition="a100",
            exclude="alpaca",
            qos="normal",
            timeout="5-0",
            job_name="baseft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )


if __name__ == "__main__":
    main()
