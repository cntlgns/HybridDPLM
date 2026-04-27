"""
Noise-input diffusion generation for DPLM2 (ablation of hybrid).

Compared to generate_dplm2_hybrid.py, this script drops every hybrid-specific
knob (no sigma_min/max, no r_min/max, no noise_space, no use_normal_emb): the
masked positions are simply replaced with fresh Gaussian noise. The only new
flag is ``--sample_noise_every_step`` which controls whether the noise vector
at still-masked positions is redrawn at each decoding step (True) or sampled
once at init and reused until that position is unmasked (False).

Usage:
  python generate_dplm2_noise.py \\
      --ckpt_path logs/noise_invfold/checkpoints/last.ckpt \\
      --task inverse_folding \\
      --input_fasta_path data/test_struct.fasta \\
      --max_iter 50 \\
      --sample_noise_every_step true \\
      --saveto results/noise_invfold
"""

import argparse
import os


def _optional_bool(value):
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

from byprot.models.dplm2.dplm2_noise import (
    NoiseInputDiffusionProteinLanguageModel,
)


def _try_load_ema_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
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


def load_noise_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt_path:
        print(f"Loading noise-input model from checkpoint: {args.ckpt_path}")
        model = NoiseInputDiffusionProteinLanguageModel.from_pretrained(
            args.ckpt_path, from_huggingface=False
        )
        if args.use_ema:
            _try_load_ema_weights(model, args.ckpt_path)
    else:
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
            "token_dropout": False,
            "cutoff_layer0_attn_residual": args.cutoff_layer0_attn_residual,
            "sample_noise_every_step": True,  # overridden per-call below
            "self_mixup": {"enable": False, "with_original_loss": False},
            "tokenizer": {"vocab_file": args.model_name, "vocab_size": 33},
            "struct_tokenizer": {
                "enable": True,
                "exp_path": "airkingbd/struct_tokenizer",
            },
        })
        model = NoiseInputDiffusionProteinLanguageModel(cfg)

    if args.sample_noise_every_step is not None:
        print(
            f"Overriding sample_noise_every_step at inference: "
            f"{getattr(model.cfg, 'sample_noise_every_step', None)} "
            f"-> {args.sample_noise_every_step}"
        )
        model.cfg.sample_noise_every_step = args.sample_noise_every_step

    model = model.eval().to(device)
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    return model, device


def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_seeds(args):
    raw = getattr(args, "seeds", None)
    if raw is None or raw == "":
        return None
    seeds = [int(s.strip()) for s in raw.split(",") if s.strip()]
    return seeds or None


def _run_conditional_once(args, model, device, tokenizer, save_dir):
    batches, name_lists = initialize_conditional_generation(
        args.input_fasta_path, tokenizer, device, args=args, model=model
    )

    for i, batch in enumerate(tqdm(batches, desc=f"Noise {args.task}")):
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
    model, device = load_noise_model(args)
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
    model, device = load_noise_model(args)
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


def main():
    parser = argparse.ArgumentParser(
        description="Noise-input diffusion generation for DPLM2"
    )

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
        help="Path to finetuned noise-input checkpoint (.ckpt). "
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

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="annealing@2.0:0.1",
    )
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--save_pdb", type=bool, default=True)

    parser.add_argument(
        "--cutoff_layer0_attn_residual", action="store_true",
        help="Apply masked-residual layer-0 attention patch "
             "(only used for the HF-no-ckpt path; ckpt path inherits this from training cfg).",
    )
    parser.add_argument(
        "--sample_noise_every_step", type=_optional_bool, default=None,
        help="When True, redraw Gaussian noise at every step for still-masked "
             "positions. When False, sample once at init and keep it fixed at "
             "still-masked positions. Omit to inherit from the ckpt's training cfg.",
    )

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

    parser.add_argument("--input_fasta_path", type=str, default="")

    parser.add_argument("--num_seqs", type=int, default=10)
    parser.add_argument("--seq_lens", nargs="*", type=int, default=[100])

    parser.add_argument(
        "--saveto", type=str, default="generate-results/noise_generate"
    )

    args = parser.parse_args()

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
