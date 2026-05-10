"""
DPLM2 Hybrid 650M hyperparameter sweep (90 jobs).

Sweep axes:
  - noise_space: embedding, onehot (2)
  - sigma_min/max: 3 sets per noise_space
  - lora: 5 configs
  - lr schedule: 3 configs

Usage:
    python scripts/sweep_hybrid.py
"""

import itertools
from slurm_launcher.sbatch_launcher import launch_tasks

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
PYTHON_BIN = f"{PROJECT_DIR}/.venv/bin/python"
TRAIN_AND_SYNC_SCRIPT = f"{PROJECT_DIR}/scripts/train_and_sync.sh"

# Save checkpoints to server-local disk during training (avoid NFS overhead)
# After training, train_and_sync.sh rsyncs results to NFS and cleans up local copy.
LOCAL_LOG_BASE = "/data_large/unsynced_store/sihun/diffprotein/dplm/train_logs"

PART_TO_BASH = {
    "rtx3090": "/bin/bash",
    "ada": "/bin/bash",
    "a100": "/bin/bash",
}

# ─── Sweep axes ───────────────────────────────────────────────────────────────

NOISE_SPACES = ["embedding"] # "embedding", "onehot"

# (sigma_min, sigma_max, tag)
NOISE_SCHEDULE_CONFIG = {
    # "embedding": [(0.001, 0.4, "xlow_noise"), (0.01, 0.4, "low_noise"), (0.05, 0.5, "mid_noise"), (0.1, 0.5, "high_noise")],
    "embedding": [(0.5, 5.0, "high_noise"), (1.0, 5.0, "1high_noise"), (1.0, 10.0, "xhigh_noise"), (2.0, 10.0, "2xhigh_noise")], #FT10(linear)
    # "embedding": [(1.0, 15.0, "xhigh_noise"), (1.0, 20.0, "2xhigh_noise")], #FT5
    # "embedding": [(0.5, 5.0, "high_noise")],#, (1.0, 10.0, "xhigh_noise")], #FT6, FT8, FT9
    # "onehot":    [(0.1, 0.45, "high_noise"), (0.25, 0.47, "xhigh_noise"), (0.25, 0.48, "2xhigh_noise"), (0.40, 0.48, "hard_noise")], #FTO2 (0.5, 5.0), (1.0, 10.0), (1.0, 14.0) for FT6(normemb)
    # "onehot":    [(0.1, 0.45, "high_noise"), (0.25, 0.47, "xhigh_noise"), (0.25, 0.48, "2xhigh_noise"), (0.25, 0.49, "3xhigh_noise")], #FTO2 (0.5, 5.0), (1.0, 10.0), (1.0, 14.0), (1.0, 28.0) for FT7(pureemb)
    # "onehot":    [(0.01, 0.10, "low_noise"), (0.01, 0.25, "default_noise"), (0.10, 0.40, "high_noise"), (0.01, 0.40, "wide_noise")], # FTO1
    # "onehot":    [(0.25, 0.47, "xhigh_noise")], #FTO3 (1.0, 10.0), (1.0, 14.0) for FT6(normemb)
}

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
    # (1e-7, 1e-5, 1e-7, 1000, 30000,  "my_lr"),
    # (1e-7, 1e-4, 1e-5, 2000, 100000, "orig_lr"),
    # (1e-7, 1e-5, 1e-6, 2000, 100000, "low_lr"),
    # (1e-5, 1e-5, 1e-5, 0, 100000, "ema_lr"),
    # (1e-7, 1e-4, 1e-5, 2000, 100000, "ema_lr_745"),
    # (1e-6, 1e-3, 1e-4, 2000, 100000, "ema_lr_634"),
    # (1e-4, 1e-4, 1e-4, 0, 100000, "ema_lora_lr"),
    
    # (1e-7, 1e-4, 1e-7, 2000, 41000, "1e4_lr"),
    # (1e-7, 3e-5, 1e-7, 2000, 41000, "3e5_lr"), FT9

    (1e-7, 1e-4, 1e-7, 2000, 24600, "1e4_lr"),
    (1e-7, 3e-5, 1e-7, 2000, 24600, "3e5_lr"),
]

# (fullseq_loss_weight, tag)
FULLSEQ_LOSS_WEIGHTS = [
    (0.0, "fs0"),
    # (0.1, "fs0.1"),
    # (0.2, "fs0.2"),
    # (0.3, "fs0.3"),
]


def make_job_name(ns_tag, noise_tag, lora_tag, lr_tag, fs_tag):
    """e.g. emb-mid_noise-64ln-my_lr-fs0.1"""
    # return f"{ns_tag}-{noise_tag}-{lora_tag}-{lr_tag}-{fs_tag}-cutL0-normemb"
    # return f"{ns_tag}-{noise_tag}-{lora_tag}-{lr_tag}-{fs_tag}-cutL0"
    # return f"{ns_tag}-{noise_tag}-{lora_tag}-{lr_tag}-{fs_tag}-normemb"
    # return f"{ns_tag}-{noise_tag}-{lora_tag}-{lr_tag}-{fs_tag}"
    return f"{ns_tag}-{noise_tag}-{lora_tag}-{lr_tag}-linear"


def make_overrides(name, group, noise_space, noise_min, noise_max,
                   lora_enable, lora_rank, lora_train_ln,
                   warmup_init_lr, lr, lr_end, warmup_steps, max_steps,
                   fullseq_loss_weight):
    local_log_dir = f"{LOCAL_LOG_BASE}/{name}"
    ovs = [
        f"experiment=dplm2/dplm2_hybrid_650m",
        f"name={name}",
        f"paths.log_dir={local_log_dir}",
        f"logger=wandb",
        f"logger.wandb.group={group}",
        f"model.hybrid.noise_space={noise_space}",
    ]
    if noise_space == "onehot":
        ovs += [
            f"model.hybrid.r_min={noise_min}",
            f"model.hybrid.r_max={noise_max}",
        ]
    else:
        ovs += [
            f"model.hybrid.sigma_min={noise_min}",
            f"model.hybrid.sigma_max={noise_max}",
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
    full_cmds = []  # ada

    for noise_space in NOISE_SPACES:
        ns_tag = "emb" if noise_space == "embedding" else "oh"
        noise_list = NOISE_SCHEDULE_CONFIG[noise_space]
        for (noise_min, noise_max, noise_tag), lora_cfg, lr_cfg, fs_cfg in itertools.product(
            noise_list, LORA_CONFIGS, LR_SCHEDULE_CONFIGS, FULLSEQ_LOSS_WEIGHTS
        ):
            lora_enable, lora_rank, lora_train_ln, lora_tag = lora_cfg
            warmup_init_lr, lr, lr_end, warmup_steps, max_steps, lr_tag = lr_cfg
            fullseq_loss_weight, fs_tag = fs_cfg

            name = make_job_name(ns_tag, noise_tag, lora_tag, lr_tag, fs_tag)

            group = f"{ns_tag}-{noise_tag}"
            override_str = make_overrides(
                name, group, noise_space, noise_min, noise_max,
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
    print(f"Full FT jobs (ada): {len(full_cmds)}")
    print(f"Total: {len(lora_cmds) + len(full_cmds)}")

    # Submit LoRA jobs on rtx3090
    if lora_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": lora_cmds},
            partition="rtx3090",
            exclude="alpaca", #"kiwi,lemon,mango,nutella,peach,quiznos,radish,tomato,udon,watermelon,xoi,yogurt,vanilla",
            qos="normal",
            timeout="5-0",
            job_name="hybridft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )

    # Submit full finetuning jobs on ada
    if full_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": full_cmds},
            partition="a100",
            exclude="alpaca",
            qos="normal",
            timeout="5-0",
            job_name="hybridft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )


if __name__ == "__main__":
    main()
