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
}

# ─── Sweep axes ───────────────────────────────────────────────────────────────

NOISE_SPACES = ["embedding"] # "embedding", "onehot"

# (sigma_min, sigma_max, tag)
NOISE_SCHEDULE_CONFIG = {
    "embedding": [(0.5, 5.0, "mid_noise"), (0.01, 2.0, "low_noise"), (1.0, 10.0, "high_noise")],
    "onehot":    [(0.01, 0.25, "low_noise"), (0.1, 1.0, "mid_noise"), (0.5, 2.0, "high_noise")],
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
    (1e-7, 1e-5, 1e-7, 1000, 30000,  "my_lr"),
    # (1e-7, 1e-4, 1e-5, 2000, 100000, "orig_lr"),
    (1e-8, 1e-5, 1e-6, 2000, 100000, "low_lr"),
]


def make_job_name(ns_tag, noise_tag, lora_tag, lr_tag):
    """e.g. emb-mid_noise-64ln-my_lr"""
    return f"{ns_tag}-{noise_tag}-{lora_tag}-{lr_tag}"


def make_overrides(name, group, noise_space, sigma_min, sigma_max,
                   lora_enable, lora_rank, lora_train_ln,
                   warmup_init_lr, lr, lr_end, warmup_steps, max_steps):
    local_log_dir = f"{LOCAL_LOG_BASE}/{name}"
    ovs = [
        f"experiment=dplm2/dplm2_hybrid_650m",
        f"name={name}",
        f"paths.log_dir={local_log_dir}",
        f"logger=wandb",
        f"logger.wandb.group={group}",
        f"model.hybrid.noise_space={noise_space}",
        f"model.hybrid.sigma_min={sigma_min}",
        f"model.hybrid.sigma_max={sigma_max}",
    ]

    if lora_enable:
        ovs += [
            "model.lora.enable=true",
            f"model.lora.lora_rank={lora_rank}",
            f"model.lora.train_layer_norm={'true' if lora_train_ln else 'false'}",
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
    ]

    return " ".join(ovs)


def generate_commands():
    lora_cmds = []  # rtx3090
    full_cmds = []  # ada

    for noise_space in NOISE_SPACES:
        ns_tag = "emb" if noise_space == "embedding" else "oh"
        sigma_list = NOISE_SCHEDULE_CONFIG[noise_space]
        for (sigma_min, sigma_max, noise_tag), lora_cfg, lr_cfg in itertools.product(
            sigma_list, LORA_CONFIGS, LR_SCHEDULE_CONFIGS
        ):
            lora_enable, lora_rank, lora_train_ln, lora_tag = lora_cfg
            warmup_init_lr, lr, lr_end, warmup_steps, max_steps, lr_tag = lr_cfg

            name = make_job_name(ns_tag, noise_tag, lora_tag, lr_tag)

            group = f"{ns_tag}-{noise_tag}"
            override_str = make_overrides(
                name, group, noise_space, sigma_min, sigma_max,
                lora_enable, lora_rank, lora_train_ln,
                warmup_init_lr, lr, lr_end, warmup_steps, max_steps,
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
            # exclude="", #"kiwi,lemon,mango,nutella,peach,quiznos,radish,tomato,udon,watermelon,xoi,yogurt,vanilla",
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
            partition="ada",
            qos="normal",
            timeout="5-0",
            job_name="hybridft",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )


if __name__ == "__main__":
    main()
