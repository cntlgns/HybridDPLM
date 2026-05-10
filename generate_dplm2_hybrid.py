"""
Hybrid diffusion generation script for DPLM2.

Usage examples:

  # Inverse folding (struct → AA) from a finetuned hybrid checkpoint
  python generate_dplm2_hybrid.py \
      --ckpt_path logs/hybrid_invfold/checkpoints/last.ckpt \
      --task inverse_folding \
      --input_fasta_path data/test_struct.fasta \
      --max_iter 50 \
      --saveto results/hybrid_invfold

  # Co-generation with HuggingFace DPLM2 weights (no Hybrid finetuning)
  python generate_dplm2_hybrid.py \
      --model_name airkingbd/dplm2_650m \
      --task co_generation \
      --num_seqs 10 --seq_lens 100 \
      --max_iter 100
"""

import argparse
import os


def _optional_bool(value):
    """Parse a tri-state bool flag ('', None -> None; true/false strings -> bool)."""
    if value is None or value == "":
        return None
    v = value.strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean-like value, got {value!r}")

os.environ.setdefault("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

import torch
from peft.peft_model import PeftModel
from tqdm import tqdm

from generate_dplm2 import (
    initialize_conditional_generation,
    initialize_generation,
    save_results,
)

from byprot.models.dplm2.dplm2_hybrid import HybridDiffusionProteinLanguageModel


def _try_load_ema_weights(model, ckpt_path):
    """If the checkpoint contains EMA weights, copy them into the model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    # Training-time LitEma was built on the LightningModule (self.model_ema = LitEma(self)),
    # so shadow buffers are saved as "model_ema.<flat>" where flat is the LightningModule
    # param name with dots removed — i.e., "model" + <inner_flat>, because self.model is the
    # inner HybridDPLM. Inference-time LitEma will be built on the inner model, whose
    # flat names are just <inner_flat>. Strip the extra leading "model" to match.
    ema_prefix = "model_ema."
    ema_keys = [k for k in state_dict if k.startswith(ema_prefix)]
    if not ema_keys:
        print("No EMA weights found in checkpoint, using training weights.")
        return

    from byprot.models.structok.modules.ema import LitEma
    ema = LitEma(model, decay=0.999, use_num_upates=False)
    ema_state = {}
    for k, v in state_dict.items():
        if not k.startswith(ema_prefix):
            continue
        tail = k[len(ema_prefix):]
        if tail in ("decay", "num_updates"):
            ema_state[tail] = v
        elif tail.startswith("model"):
            ema_state[tail[len("model"):]] = v
        else:
            ema_state[tail] = v
    missing, unexpected = ema.load_state_dict(ema_state, strict=False)
    ema.copy_to(model)
    print(
        f"Loaded EMA weights from checkpoint ({len(ema_keys)} buffers; "
        f"missing={len(missing)}, unexpected={len(unexpected)})."
    )


def load_hybrid_model(args):
    """Load a hybrid model from either a local checkpoint or HuggingFace."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt_path:
        # Load from a local training checkpoint (.ckpt)
        print(f"Loading hybrid model from checkpoint: {args.ckpt_path}")
        model = HybridDiffusionProteinLanguageModel.from_pretrained(
            args.ckpt_path, from_huggingface=False
        )
        if args.use_ema:
            _try_load_ema_weights(model, args.ckpt_path)
        # Allow overriding use_normal_emb at inference time (no schedule rebuild needed).
        if args.use_normal_emb is not None:
            print(
                f"Overriding hybrid.use_normal_emb at inference: "
                f"{getattr(model.cfg.hybrid, 'use_normal_emb', False)} -> {args.use_normal_emb}"
            )
            model.cfg.hybrid.use_normal_emb = args.use_normal_emb

        # Allow overriding noise schedule at inference time (rebuilds noise_schedule).
        # For onehot noise space the schedule is driven by r_min/r_max (Eq. 10);
        # for embedding it's driven by sigma_min/sigma_max (VE-SDE).
        if (
            args.sigma_min is not None or args.sigma_max is not None
            or args.r_min is not None or args.r_max is not None
        ):
            from byprot.models.dplm2.dplm2_hybrid import HybridNoiseSchedule
            sigma_min = args.sigma_min if args.sigma_min is not None else model.cfg.hybrid.sigma_min
            sigma_max = args.sigma_max if args.sigma_max is not None else model.cfg.hybrid.sigma_max
            r_min = args.r_min if args.r_min is not None else model.cfg.hybrid.r_min
            r_max = args.r_max if args.r_max is not None else model.cfg.hybrid.r_max
            print(
                f"Overriding noise schedule at inference: "
                f"r_min={r_min}, r_max={r_max}, sigma_min={sigma_min}, sigma_max={sigma_max}."
            )
            model.cfg.hybrid.sigma_min = sigma_min
            model.cfg.hybrid.sigma_max = sigma_max
            model.cfg.hybrid.r_min = r_min
            model.cfg.hybrid.r_max = r_max
            model.noise_schedule = HybridNoiseSchedule(
                num_timesteps=model.cfg.num_diffusion_timesteps,
                r_min=r_min,
                r_max=r_max,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
    else:
        # Load from HuggingFace DPLM2 and wrap with Hybrid
        # (uses default Hybrid config; no Hybrid-specific finetuning)
        print(f"Loading base DPLM2 model: {args.model_name}")
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "num_diffusion_timesteps": 500,
            "gradient_ckpt": False,
            "training_stage": "finetune_from_dplm2_hf",
            "single_modality_ratio": 0.0,
            "folding_loss_ratio": 0.0,
            "inverse_folding_loss_ratio": 1.0,
            "joint_loss_ratio": 0.0,
            "independent_loss_ratio": 0.0,
            "hybrid": {
                "noise_space": args.noise_space,
                "r_min": args.r_min if args.r_min is not None else 0.01,
                "r_max": args.r_max if args.r_max is not None else 0.25,
                "sigma_min": args.sigma_min if args.sigma_min is not None else 0.5,
                "sigma_max": args.sigma_max if args.sigma_max is not None else 5.0,
                "lambda_bias": 1.0,
                "use_sigma_embed": False,
                "loss_weight": "hybrid",
                "use_normal_emb": bool(args.use_normal_emb) if args.use_normal_emb is not None else False,
            },
            "lora": {
                "enable": False,
                "lora_rank": 16,
                "lora_dropout": 0.1,
                "lora_target_module": "",
                "modules_to_save": "",
            },
            "net": {
                "arch_type": "esm",
                "name": args.model_name,
                "dropout": 0.1,
                "pretrain": False,
                "pretrained_model_name_or_path": args.model_name,
            },
            "cutoff_layer0_attn_residual": args.cutoff_layer0_attn_residual,
            "self_mixup": {"enable": False, "with_original_loss": False},
            "tokenizer": {"vocab_file": args.model_name, "vocab_size": 33},
            "struct_tokenizer": {
                "enable": True,
                "exp_path": "airkingbd/struct_tokenizer",
            },
        })
        model = HybridDiffusionProteinLanguageModel(cfg)

    model = model.eval().to(device)
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    return model, device


def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_seeds(args):
    """Return a list[int] of seeds if --seeds was provided, else None."""
    raw = getattr(args, "seeds", None)
    if raw is None or raw == "":
        return None
    seeds = [int(s.strip()) for s in raw.split(",") if s.strip()]
    return seeds or None


def _run_conditional_once(args, model, device, tokenizer, save_dir):
    """Generate once under the currently-set RNG state, saving under save_dir."""
    batches, name_lists = initialize_conditional_generation(
        args.input_fasta_path, tokenizer, device, args=args, model=model
    )

    for i, batch in enumerate(tqdm(batches, desc=f"Hybrid {args.task}")):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=args.max_iter,
                temperature=args.temperature,
                partial_masks=batch["partial_mask"],
                sampling_strategy=args.sampling_strategy,
            )

        save_results(
            outputs=outputs,
            task=args.task,
            save_dir=os.path.join(save_dir, args.task),
            headers=name_lists[i],
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=args.save_pdb,
            continue_write=True,
        )


def conditional_generate(args):
    """Generate sequences conditioned on structure (inverse folding) or vice versa.

    If ``--seeds`` is given, iterate each seed reseeding between runs and
    writing to ``<saveto>/seed_<seed>``. The model is loaded only once.
    Otherwise run once with ``--seed`` writing to ``<saveto>``.
    """
    model, device = load_hybrid_model(args)
    tokenizer = model.tokenizer

    seeds = _parse_seeds(args)
    if seeds is None:
        _set_seed(args.seed)
        _run_conditional_once(args, model, device, tokenizer, args.saveto)
        return

    for seed in seeds:
        seed_save_dir = os.path.join(args.saveto, f"seed_{seed}")
        gen_marker = os.path.join(seed_save_dir, args.task, "aatype.fasta")
        if args.skip_if_generated and os.path.exists(gen_marker):
            print(f"[skip-gen] seed={seed}: {gen_marker} already exists")
            continue
        print(f"=== Generating seed={seed} -> {seed_save_dir} ===")
        _set_seed(seed)
        _run_conditional_once(args, model, device, tokenizer, seed_save_dir)


def unconditional_generate(args):
    """Unconditional generation (co-generation, backbone, sequence)."""
    model, device = load_hybrid_model(args)
    tokenizer = model.tokenizer

    for seq_len in args.seq_lens:
        input_tokens_batch = initialize_generation(
            task=args.task,
            num_seqs=args.num_seqs,
            length=seq_len,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )

        all_outputs = {}
        for input_tokens in input_tokens_batch:
            _struct_tokens, _aatype_tokens = input_tokens.chunk(2, dim=1)
            if args.task == "backbone_generation":
                input_tokens = _struct_tokens
            if args.task == "sequence_generation":
                input_tokens = _aatype_tokens

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    input_tokens=input_tokens,
                    max_iter=args.max_iter,
                    temperature=args.temperature,
                    sampling_strategy=args.sampling_strategy,
                )

            if args.task == "backbone_generation":
                outputs["output_tokens"] = torch.cat(
                    [outputs["output_tokens"], _aatype_tokens], dim=1
                )
            for k, v in outputs.items():
                if k in all_outputs:
                    all_outputs[k] = torch.cat([all_outputs[k], v], dim=0)
                else:
                    all_outputs[k] = v

        save_results(
            outputs=all_outputs,
            task=args.task,
            save_dir=os.path.join(args.saveto, args.task, f"length_{seq_len}"),
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=args.save_pdb,
        )


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "1"):
        return True
    if s in ("no", "false", "f", "0", ""):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid diffusion generation for DPLM2"
    )

    # Model loading
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds (e.g. '42,43,44'). When set, "
        "overrides --seed and runs generation once per seed, writing each "
        "run to <saveto>/seed_<seed>. Model is loaded only once.",
    )
    parser.add_argument(
        "--skip_if_generated",
        action="store_true",
        help="When used with --seeds, skip seeds whose <saveto>/seed_<s>/"
        "<task>/aatype.fasta already exists.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to finetuned hybrid checkpoint (.ckpt). "
        "If empty, uses --model_name from HuggingFace.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="airkingbd/dplm2_650m",
        help="HuggingFace model name (used when --ckpt_path is empty).",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights from checkpoint for inference.",
    )

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="annealing@2.0:0.1",
    )
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--save_pdb", type=_str2bool, default=True)

    # Hybrid diffusion noise schedule (overrides model cfg when provided)
    parser.add_argument(
        "--sigma_min", type=float, default=None,
        help="Override hybrid.sigma_min at inference time (used for noise_space=embedding).",
    )
    parser.add_argument(
        "--sigma_max", type=float, default=None,
        help="Override hybrid.sigma_max at inference time (used for noise_space=embedding).",
    )
    parser.add_argument(
        "--r_min", type=float, default=None,
        help="Override hybrid.r_min at inference time (used for noise_space=onehot).",
    )
    parser.add_argument(
        "--r_max", type=float, default=None,
        help="Override hybrid.r_max at inference time (used for noise_space=onehot).",
    )
    parser.add_argument(
        "--noise_space", type=str, default="embedding",
        choices=["embedding", "onehot"],
        help="Hybrid noise space (only used for the HF-no-ckpt path).",
    )
    parser.add_argument(
        "--cutoff_layer0_attn_residual", action="store_true",
        help="Apply masked-residual layer-0 attention patch "
             "(only used for the HF-no-ckpt path; ckpt path inherits this from training cfg).",
    )
    parser.add_argument(
        "--use_normal_emb", type=_optional_bool, default=None,
        help="Row-wise L2-normalize embedding rows when used as noise basis or "
             "as E_Y0 in the ODE step. Pass true/false to override; omit to "
             "inherit from the ckpt's training cfg (defaults to false for HF-no-ckpt).",
    )

    # Task
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "backbone_generation",
            "sequence_generation",
            "co_generation",
            "folding",
            "inverse_folding",
        ],
        default="inverse_folding",
    )

    # Conditional generation input
    parser.add_argument("--input_fasta_path", type=str, default="")

    # Unconditional generation params
    parser.add_argument("--num_seqs", type=int, default=10)
    parser.add_argument("--seq_lens", nargs="*", type=int, default=[100])

    # Output
    parser.add_argument(
        "--saveto", type=str, default="generate-results/hybrid_generate"
    )

    args = parser.parse_args()

    # When --seeds is given, conditional_generate() reseeds per iteration.
    # Otherwise honor --seed up-front.
    if not _parse_seeds(args):
        _set_seed(args.seed)

    if args.task in ["folding", "inverse_folding"]:
        conditional_generate(args)
    elif args.task in [
        "backbone_generation",
        "sequence_generation",
        "co_generation",
    ]:
        unconditional_generate(args)
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
