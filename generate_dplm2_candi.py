"""
CANDI hybrid diffusion generation script for DPLM2.

Usage examples:

  # Inverse folding (struct → AA) from a finetuned CANDI checkpoint
  python generate_dplm2_candi.py \
      --ckpt_path logs/candi_invfold/checkpoints/last.ckpt \
      --task inverse_folding \
      --input_fasta_path data/test_struct.fasta \
      --max_iter 50 \
      --saveto results/candi_invfold

  # Co-generation with HuggingFace DPLM2 weights (no CANDI finetuning)
  python generate_dplm2_candi.py \
      --model_name airkingbd/dplm2_650m \
      --task co_generation \
      --num_seqs 10 --seq_lens 100 \
      --max_iter 100
"""

import argparse
import os

os.environ.setdefault("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

import torch
from peft.peft_model import PeftModel
from tqdm import tqdm

from generate_dplm2 import (
    initialize_conditional_generation,
    initialize_generation,
    save_results,
)

from byprot.models.dplm2.dplm2_candi import CANDIDiffusionProteinLanguageModel


def load_candi_model(args):
    """Load a CANDI model from either a local checkpoint or HuggingFace."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt_path:
        # Load from a local training checkpoint (.ckpt)
        print(f"Loading CANDI model from checkpoint: {args.ckpt_path}")
        model = CANDIDiffusionProteinLanguageModel.from_pretrained(
            args.ckpt_path, from_huggingface=False
        )
    else:
        # Load from HuggingFace DPLM2 and wrap with CANDI
        # (uses default CANDI config; no CANDI-specific finetuning)
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
            "candi": {
                "noise_space": "embedding",
                "r_min": 0.01,
                "r_max": 0.25,
                "sigma_min": 0.01,
                "sigma_max": 2.0,
                "lambda_bias": 0.5,
                "use_sigma_embed": True,
                "loss_weight": "candi",
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
            "self_mixup": {"enable": False, "with_original_loss": False},
            "tokenizer": {"vocab_file": args.model_name, "vocab_size": 33},
            "struct_tokenizer": {
                "enable": True,
                "exp_path": "airkingbd/struct_tokenizer",
            },
        })
        model = CANDIDiffusionProteinLanguageModel(cfg)

    model = model.eval().to(device)
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    return model, device


def conditional_generate(args):
    """Generate sequences conditioned on structure (inverse folding) or vice versa."""
    model, device = load_candi_model(args)
    tokenizer = model.tokenizer

    batches, name_lists = initialize_conditional_generation(
        args.input_fasta_path, tokenizer, device, args=args, model=model
    )

    for i, batch in enumerate(tqdm(batches, desc=f"CANDI {args.task}")):
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
            save_dir=os.path.join(args.saveto, args.task),
            headers=name_lists[i],
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=args.save_pdb,
            continue_write=True,
        )


def unconditional_generate(args):
    """Unconditional generation (co-generation, backbone, sequence)."""
    model, device = load_candi_model(args)
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
        description="CANDI hybrid diffusion generation for DPLM2"
    )

    # Model loading
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to finetuned CANDI checkpoint (.ckpt). "
        "If empty, uses --model_name from HuggingFace.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="airkingbd/dplm2_650m",
        help="HuggingFace model name (used when --ckpt_path is empty).",
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
    parser.add_argument("--save_pdb", type=bool, default=True)

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
        "--saveto", type=str, default="results/candi_generate"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
