"""
Resume a list of previously-archived runs under SLURM.

For each NFS source dir in RESUME_SOURCES:
  1) resume_from_nfs.sh copies NFS -> local disk and parses the wandb id
  2) train_and_sync.sh runs training and rsyncs back to NFS at the end

The per-run hyperparameter overrides are reconstructed from the run name
(same tag convention as sweep_hybrid.py:
    {ns}-{noise_tag}-{lora}-{lr_tag}-{fs_tag}
).

Usage:
    python scripts/resume_sweep.py
"""

from slurm_launcher.sbatch_launcher import launch_tasks

PROJECT_DIR = "/data_fast/home/sihun/diffprotein/dplm"
RESUME_SCRIPT = f"{PROJECT_DIR}/scripts/resume_from_nfs.sh"

PART_TO_BASH = {
    "rtx3090": "/bin/bash",
    "ada": "/bin/bash",
}

# ─── Sources to resume ────────────────────────────────────────────────────────
RESUME_SOURCES = [
    "/storage/sihun/diffprotein/dplm/train_logs/emb-high_noise-full-ema_lr-fs0",
    "/storage/sihun/diffprotein/dplm/train_logs/emb-high_noise-full-ema_lr-fs0.1",
    "/storage/sihun/diffprotein/dplm/train_logs/emb-wide_noise-full-ema_lr-fs0",
    "/storage/sihun/diffprotein/dplm/train_logs/emb-wide_noise-full-ema_lr-fs0.1",
]

# ─── Tag → hyperparameter lookup tables (mirror sweep_hybrid.py) ──────────────
NS_TAG_TO_SPACE = {"emb": "embedding", "oh": "onehot"}

# noise_tag → (sigma_min, sigma_max) per noise_space
NOISE_TAG_TO_SIGMAS = {
    "embedding": {
        # "xlow_noise":  (0.001, 0.4),
        # "low_noise":   (0.01,  0.4),
        # "mid_noise":   (0.05,  0.5),
        "high_noise":  (1.0,  10.0),
        "wide_noise":  (0.3,  15.0),
        # "xhigh_noise": (1.0,  15.0),
    },
    "onehot": {
        "low_noise":  (0.01, 0.25),
        "mid_noise":  (0.1,  1.0),
        "high_noise": (0.5,  2.0),
    },
}

# lora_tag → (enable, rank, train_layer_norm)
LORA_TAG_TO_CFG = {
    "full": (False, None, None),
    # "16":   (True, 16, False),
    # "16ln": (True, 16, True),
    # "64":   (True, 64, False),
    # "64ln": (True, 64, True),
}

# lr_tag → (warmup_init_lr, lr, lr_end, warmup_steps, max_steps)
LR_TAG_TO_CFG = {
    # "my_lr":       (1e-7, 1e-5, 1e-7, 1000, 30000),
    # "orig_lr":     (1e-7, 1e-4, 1e-5, 2000, 100000),
    # "low_lr":      (1e-8, 1e-5, 1e-6, 2000, 100000),
    "ema_lr":      (1e-5, 1e-5, 1e-5, 0,    100000),
    # "ema_lora_lr": (1e-4, 1e-4, 1e-4, 0,    100000),
}

# fs_tag → fullseq_loss_weight
FS_TAG_TO_WEIGHT = {
    "fs0":   0.0,
    "fs0.1": 0.1,
    # "fs0.3": 0.3,
}


def parse_name(name):
    """
    '{ns}-{noise_tag}-{lora}-{lr_tag}-{fs_tag}' → dict of resolved hparams.

    `noise_tag` itself contains an underscore (e.g. 'high_noise') but no dash,
    so splitting on '-' recovers the 5 segments cleanly.
    """
    parts = name.split("-")
    if len(parts) != 5:
        raise ValueError(f"Unexpected run name format: {name!r}")
    ns_tag, noise_tag, lora_tag, lr_tag, fs_tag = parts

    noise_space = NS_TAG_TO_SPACE[ns_tag]
    sigma_min, sigma_max = NOISE_TAG_TO_SIGMAS[noise_space][noise_tag]
    lora_enable, lora_rank, lora_train_ln = LORA_TAG_TO_CFG[lora_tag]
    warmup_init_lr, lr, lr_end, warmup_steps, max_steps = LR_TAG_TO_CFG[lr_tag]
    fullseq_loss_weight = FS_TAG_TO_WEIGHT[fs_tag]

    return {
        "ns_tag": ns_tag,
        "noise_tag": noise_tag,
        "noise_space": noise_space,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "lora_enable": lora_enable,
        "lora_rank": lora_rank,
        "lora_train_ln": lora_train_ln,
        "warmup_init_lr": warmup_init_lr,
        "lr": lr,
        "lr_end": lr_end,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "fullseq_loss_weight": fullseq_loss_weight,
    }


def make_overrides(h):
    group = f"{h['ns_tag']}-{h['noise_tag']}"
    ovs = [
        "experiment=dplm2/dplm2_hybrid_650m",
        f"logger.wandb.group={group}",
        f"model.hybrid.noise_space={h['noise_space']}",
        f"model.hybrid.sigma_min={h['sigma_min']}",
        f"model.hybrid.sigma_max={h['sigma_max']}",
    ]
    if h["lora_enable"]:
        ovs += [
            "model.lora.enable=true",
            f"model.lora.lora_rank={h['lora_rank']}",
            f"model.lora.train_layer_norm={'true' if h['lora_train_ln'] else 'false'}",
            "datamodule.max_tokens=1600",
            "trainer.accumulate_grad_batches=40",
            "callbacks.model_checkpoint.every_n_train_steps=400",
        ]
    else:
        ovs += ["model.lora.enable=false"]

    ovs += [
        f"task.lr_scheduler.warmup_init_lr={h['warmup_init_lr']}",
        f"train.lr={h['lr']}",
        f"task.lr_scheduler.lr_end={h['lr_end']}",
        f"task.lr_scheduler.warmup_steps={h['warmup_steps']}",
        f"trainer.max_steps={h['max_steps']}",
        f"task.learning.fullseq_loss_weight={h['fullseq_loss_weight']}",
    ]
    return " ".join(ovs)


def generate_commands():
    lora_cmds, full_cmds = [], []
    for nfs_src in RESUME_SOURCES:
        name = nfs_src.rstrip("/").split("/")[-1]
        h = parse_name(name)
        override_str = make_overrides(h)
        cmd = f"{RESUME_SCRIPT} {nfs_src} {override_str}"
        (lora_cmds if h["lora_enable"] else full_cmds).append(cmd)
    return lora_cmds, full_cmds


def main():
    lora_cmds, full_cmds = generate_commands()
    print(f"LoRA resume jobs (rtx3090): {len(lora_cmds)}")
    print(f"Full FT resume jobs (ada): {len(full_cmds)}")
    for c in lora_cmds + full_cmds:
        print(" ", c)
    print(f"Total: {len(lora_cmds) + len(full_cmds)}")

    if lora_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": lora_cmds},
            partition="rtx3090",
            qos="normal",
            timeout="5-0",
            job_name="hybridft_resume",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )

    if full_cmds:
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": full_cmds},
            partition="ada",
            qos="normal",
            timeout="5-0",
            job_name="hybridft_resume",
            max_job_num=150,
            part_to_py=PART_TO_BASH,
        )


if __name__ == "__main__":
    main()
