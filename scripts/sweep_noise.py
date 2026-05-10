"""
DPLM2 noise-input (hybrid ablation) 650M hyperparameter sweep.

Sweep axes (vs sweep_hybrid.py):
  - hybrid axes (noise_space / sigma / r / lambda) are dropped — there is no
    noise schedule in this model.
  - lora: same axis as hybrid sweep (full FT / LoRA variants).
  - lr schedule: same axis.
  - fullseq_loss_weight: same axis.

Reuses ``scripts/train_and_sync.sh`` as-is (it's hybrid-agnostic — runs train.py
with whatever overrides and rsyncs ``paths.log_dir`` to NFS afterward).

Usage:
    python scripts/sweep_noise.py
"""

import itertools
from slurm_launcher.sbatch_launcher import launch_tasks

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
PYTHON_BIN = f"{PROJECT_DIR}/.venv/bin/python"
TRAIN_AND_SYNC_SCRIPT = f"{PROJECT_DIR}/scripts/train_and_sync.sh"

# Save checkpoints to server-local disk during training (avoid NFS overhead).
# After training, train_and_sync.sh rsyncs results to NFS and cleans up.
LOCAL_LOG_BASE = "/data_large/unsynced_store/sihun/diffprotein/dplm/train_logs"

PART_TO_BASH = {
    "rtx3090": "/bin/bash",
    "ada": "/bin/bash",
    "a100": "/bin/bash",
}

# ─── Sweep axes ───────────────────────────────────────────────────────────────

# (enable, rank, train_layer_norm, tag)
LORA_CONFIGS = [
    # (True,  16, False, "16"),
    # (True,  16, True,  "16ln"),
    # (True,  64, False, "64"),
    # (True,  64, True,  "64ln"),
    (False, None, None, "full"),
]

# (warmup_init_lr, lr, lr_end, warmup_steps, max_steps, tag)
LR_SCHEDULE_CONFIGS = [
    # (1e-7, 5e-4, 1e-7, 2000, 41000, "5e4_lr"),
    # (1e-7, 2e-4, 1e-7, 2000, 41000, "2e4_lr"),
    (1e-7, 1e-4, 1e-7, 2000, 24600, "1e4_lr"),
    (1e-7, 3e-5, 1e-7, 2000, 24600, "3e5_lr"),
]

# (fullseq_loss_weight, tag)
FULLSEQ_LOSS_WEIGHTS = [
    (0.0, "fs0"),
    # (0.1, "fs0.1"),
]

# (cutoff_layer0_attn_residual, tag)
# Default for the noise ablation is True (matches the hybrid setup the user wants
# to compare against). Add False here if you want the layer-0 cutoff itself to
# become a sweep axis.
CUTOFF_L0_CONFIGS = [
    (True, "cutL0"),
    # (False, "noCutL0"),
]


def make_job_name(lora_tag, lr_tag, fs_tag, cut_tag):
    """e.g. noise-full-1e4_lr-fs0-cutL0"""
    return f"noise-{lora_tag}-{lr_tag}-linear"


def make_overrides(name, group,
                   lora_enable, lora_rank, lora_train_ln,
                   warmup_init_lr, lr, lr_end, warmup_steps, max_steps,
                   fullseq_loss_weight, cutoff_l0):
    local_log_dir = f"{LOCAL_LOG_BASE}/{name}"
    ovs = [
        f"experiment=dplm2/dplm2_noise_650m",
        f"name={name}",
        f"paths.log_dir={local_log_dir}",
        f"logger=wandb",
        f"logger.wandb.group={group}",
        f"model.cutoff_layer0_attn_residual={'true' if cutoff_l0 else 'false'}",
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
        f"task.learning.fullseq_loss_weight={fullseq_loss_weight}",
    ]

    return " ".join(ovs)


def generate_commands():
    lora_cmds = []  # rtx3090
    full_cmds = []  # a100/ada

    for lora_cfg, lr_cfg, fs_cfg, cut_cfg in itertools.product(
        LORA_CONFIGS, LR_SCHEDULE_CONFIGS, FULLSEQ_LOSS_WEIGHTS, CUTOFF_L0_CONFIGS
    ):
        lora_enable, lora_rank, lora_train_ln, lora_tag = lora_cfg
        warmup_init_lr, lr, lr_end, warmup_steps, max_steps, lr_tag = lr_cfg
        fullseq_loss_weight, fs_tag = fs_cfg
        cutoff_l0, cut_tag = cut_cfg

        name = make_job_name(lora_tag, lr_tag, fs_tag, cut_tag)
        group = f"noise-{cut_tag}"

        override_str = make_overrides(
            name, group,
            lora_enable, lora_rank, lora_train_ln,
            warmup_init_lr, lr, lr_end, warmup_steps, max_steps,
            fullseq_loss_weight, cutoff_l0,
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
            job_name="noiseft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )

    if full_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": full_cmds},
            partition="ada",
            exclude="alpaca",
            qos="normal",
            timeout="5-0",
            job_name="noiseft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )


if __name__ == "__main__":
    main()
