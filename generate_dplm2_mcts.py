"""ProtInvTree-style MCTS generation for DPLM2.

Wraps the three DPLM2 inverse-folding generators (baseline / hybrid /
noise) with a reward-guided Monte Carlo Tree Search. The decoding
primitives (``init_decoding_state``, ``decoding_step``,
``one_shot_complete``) live on the model classes; this script only
loads the chosen variant, builds an adapter, and drives the search per
target.

Output layout matches ``generate_dplm2_{baseline,hybrid,noise}.py`` so
the existing evaluator and per-seed sweep scripts can consume the
results unchanged:

    <saveto>/seed_<s>/inverse_folding/aatype.fasta
    <saveto>/seed_<s>/inverse_folding/struct_token.fasta

Usage (single seed):

    python generate_dplm2_mcts.py \\
        --variant baseline --ckpt_path <path>.ckpt \\
        --task inverse_folding \\
        --input_fasta_path data-bin/cath_4.2_all/struct.fasta \\
        --metadata_csv data-bin/metadata/cath_4.2_all.csv \\
        --metadata_data_dir data-bin \\
        --max_iter 100 --batch_size 1 \\
        --mcts_iterations 50 --mcts_expansions 4 --mcts_c_uct 0.01 \\
        --mcts_reward_threshold 0.99 --mcts_num_outputs 10 \\
        --seeds 42,43,44 --saveto results/mcts_run
"""

import argparse
import os

# Saved FT checkpoints contain hydra configs that interpolate
# ${oc.env:PROJECT_ROOT}; ensure it's set before any byprot import resolves it.
os.environ.setdefault("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
import tree
from Bio import SeqIO
from peft.peft_model import PeftModel
from tqdm import tqdm

from byprot.datamodules.pdb_dataset import utils as du
from byprot.models.dplm2.mcts import (
    FoldRewardModel,
    MCTSConfig,
    build_adapter,
    mcts_search,
)


# ----------------------------------------------------------------------
# Loader dispatch (re-uses code from generate_dplm2_*.py)
# ----------------------------------------------------------------------
def _load_model(args):
    """Load the chosen variant. Returns (model, tokenizer, device)."""
    if args.variant == "baseline":
        from generate_dplm2 import _load_baseline_model
        model = _load_baseline_model(args)
    elif args.variant == "hybrid":
        from generate_dplm2_hybrid import load_hybrid_model
        model, _device = load_hybrid_model(args)
    elif args.variant == "noise":
        from generate_dplm2_noise import load_noise_model
        model, _device = load_noise_model(args)
    else:
        raise ValueError(f"Unknown variant: {args.variant!r}")

    model = model.eval().cuda()
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()
    device = next(model.parameters()).device
    return model, model.tokenizer, device


def _build_adapter(args, model):
    """Build a ModelAdapter, baking variant-specific gen_kwargs in."""
    gen_kwargs = dict(
        sampling_strategy=args.sampling_strategy,
        temperature=args.temperature,
    )
    if args.variant == "baseline":
        gen_kwargs.update(
            unmasking_strategy=args.unmasking_strategy,
            remasking_strategy=args.remasking_strategy,
            decoding_strategy=args.decoding_strategy,
            feedforward_mode=args.feedforward_mode,
            mask_emb_mode=args.mask_emb_mode,
        )
    elif args.variant == "noise":
        if args.sample_noise_every_step is not None:
            gen_kwargs["sample_noise_every_step"] = args.sample_noise_every_step
    # hybrid: nothing extra (sampling_strategy / temperature suffice).
    return build_adapter(args.variant, model, **gen_kwargs)


# ----------------------------------------------------------------------
# Target metadata loading
# ----------------------------------------------------------------------
def _load_target_index(metadata_csv: str, data_dir: str) -> pd.DataFrame:
    """Load metadata CSV and resolve relative path columns.

    Mirrors ``EvalRunner.load_metadata`` so we accept any of the existing
    metadata CSVs (cath_*, pdb_afdb_cameo, pdb_date) without massaging.
    """
    if not os.path.isfile(metadata_csv):
        raise FileNotFoundError(metadata_csv)
    df = pd.read_csv(metadata_csv)
    for column in ("processed_path", "raw_path", "pdb_path"):
        if column in df:
            df[column] = df[column].map(
                lambda x: os.path.join(data_dir, x) if isinstance(x, str) else x
            )
    return df


def _register_target(reward_model: FoldRewardModel, header: str, df: pd.DataFrame):
    """Parse the target row for ``header`` and cache its features."""
    rows = df[df.pdb_name == header]
    if rows.empty:
        raise KeyError(
            f"Header {header!r} not found in metadata CSV. "
            "Make sure --metadata_csv matches the input fasta."
        )
    row = rows.iloc[0]
    gt_aa_seq = row["aa_seq"] if "aa_seq" in row and isinstance(row["aa_seq"], str) else None

    # Prefer raw pickle (already has atom_positions in the same shape that
    # parse_pdb_feats produces — see byprot/utils/protein/utils.py:process_folded_outputs).
    # PdbDataset.process_chain renames keys to all_atom_*, so we DON'T call it here.
    if "processed_path" in row and isinstance(row["processed_path"], str) \
            and os.path.isfile(row["processed_path"]):
        raw_chain_feats = du.read_pkl(row["processed_path"])
        reward_model.register_target_from_feats(header, raw_chain_feats, gt_aa_seq=gt_aa_seq)
    elif "pdb_path" in row and isinstance(row["pdb_path"], str) \
            and os.path.isfile(row["pdb_path"]):
        reward_model.register_target(header, row["pdb_path"], gt_aa_seq=gt_aa_seq)
    else:
        raise FileNotFoundError(
            f"No usable processed_path / pdb_path for {header!r} in metadata."
        )


# ----------------------------------------------------------------------
# Per-target MCTS run
# ----------------------------------------------------------------------
def _build_single_target_batch(record, tokenizer, args, model, device):
    """Build a B=1 input batch from one fasta record (inverse_folding only)."""
    if args.task != "inverse_folding":
        raise NotImplementedError(
            "generate_dplm2_mcts.py currently supports --task inverse_folding only."
        )
    aatype = tokenizer.aa_mask_token * len(str(record.seq).split(","))
    aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
    struct_tokens = "".join(str(record.seq).split(","))
    struct_tokens = (
        tokenizer.struct_cls_token + struct_tokens + tokenizer.struct_eos_token
    )

    batch_struct = tokenizer.batch_encode_plus(
        [struct_tokens], add_special_tokens=False, padding="longest", return_tensors="pt"
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aatype], add_special_tokens=False, padding="longest", return_tensors="pt"
    )
    input_tokens = torch.concat(
        [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
    ).to(device)

    aa_type = 1
    struct_type = 0
    non_special = model.get_non_special_symbol_mask(input_tokens)
    type_ids = model.get_modality_type(input_tokens)

    # Inverse folding: mask the AA half.
    input_tokens.masked_fill_(
        (type_ids == aa_type) & non_special,
        tokenizer._token_to_id[tokenizer.aa_mask_token],
    )
    partial_masks = type_ids == struct_type
    return input_tokens, partial_masks


def _decode_aa_string(tokenizer, output_tokens) -> str:
    """Decode the AA half of a [1, L] output_tokens tensor to a string."""
    _struct, aatype = output_tokens.chunk(2, dim=-1)
    decoded = tokenizer.batch_decode(aatype, skip_special_tokens=True)[0]
    return "".join(decoded.split(" "))


def _decode_struct_string(tokenizer, output_tokens) -> str:
    _struct, _aa = output_tokens.chunk(2, dim=-1)
    decoded = tokenizer.batch_decode(_struct, skip_special_tokens=True)[0]
    return ",".join(decoded.split(" "))


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def _save_fasta(path: str, headers, seqs, struct: bool, mode: str):
    with open(path, mode) as fp:
        for header, seq in zip(headers, seqs):
            fp.write(f">{header}\n{seq}\n")


def _save_targets(save_dir: str, header: str, results, struct_token_str: str,
                  continue_write: bool):
    """Append one target's results to the canonical fasta files.

    Layout:
        <save_dir>/aatype.fasta            (one fasta record per output)
        <save_dir>/struct_token.fasta      (paired struct tokens)
        <save_dir>/mcts_rewards.csv        (per-output reward log)
    """
    os.makedirs(save_dir, exist_ok=True)
    aa_path = os.path.join(save_dir, "aatype.fasta")
    struct_path = os.path.join(save_dir, "struct_token.fasta")
    csv_path = os.path.join(save_dir, "mcts_rewards.csv")

    out_headers = []
    out_aa = []
    out_struct = []
    rows = []
    for i, item in enumerate(results):
        # Backward-compat: search.py used to return 3-tuples (tokens, reward, aa).
        # It now returns 4-tuples with a per-output metrics dict.
        if len(item) == 4:
            _tokens, reward, aa_str, metrics = item
        else:
            _tokens, reward, aa_str = item
            metrics = {}
        # When num_outputs > 1, suffix with _<i> so eval scripts treat
        # them as distinct records but can group via a common prefix.
        out_h = header if len(results) == 1 else f"{header}_o{i}"
        out_headers.append(out_h)
        out_aa.append(aa_str)
        out_struct.append(struct_token_str)
        row = {"pdb_name": header, "output_idx": i, "reward": reward,
               "fasta_header": out_h}
        # Mirror eval-pipeline metric columns when available.
        for k in ("bb_tmscore", "bb_rmsd", "ca_rmsd", "mean_plddt",
                  "inv_fold_seq_recovery", "length"):
            if k in metrics:
                row[k] = metrics[k]
        rows.append(row)

    mode = "a" if continue_write else "w"
    _save_fasta(aa_path, out_headers, out_aa, struct=False, mode=mode)
    _save_fasta(struct_path, out_headers, out_struct, struct=True, mode=mode)

    # CSV append/write
    df_new = pd.DataFrame(rows)
    if continue_write and os.path.isfile(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
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


def _run_seed(args, model, adapter, tokenizer, reward_model, target_df,
              seed, save_dir):
    """Run MCTS over every fasta record once at the given RNG seed."""
    save_subdir = os.path.join(save_dir, args.task)
    os.makedirs(save_subdir, exist_ok=True)

    records = list(SeqIO.parse(args.input_fasta_path, "fasta"))
    # Sort by length to keep memory characteristics similar to the
    # existing generate_dplm2_*.py scripts (which length-sort batches).
    records.sort(key=lambda r: len(str(r.seq)))

    config = MCTSConfig(
        iterations=args.mcts_iterations,
        expansions_per_node=args.mcts_expansions,
        c_uct=args.mcts_c_uct,
        reward_threshold=args.mcts_reward_threshold,
        max_depth=args.mcts_max_depth if args.mcts_max_depth > 0 else None,
        num_outputs=args.mcts_num_outputs,
        rng_seed_base=seed,
        verbose=args.mcts_verbose,
    )

    continue_write = False
    for record in tqdm(records, desc=f"[seed={seed}] MCTS"):
        header = record.name
        # Make sure the target is cached for reward.
        try:
            _register_target(reward_model, header, target_df)
        except (KeyError, FileNotFoundError) as e:
            print(f"[skip-target] {header}: {e}")
            continue

        input_tokens, partial_masks = _build_single_target_batch(
            record, tokenizer, args, model, model.device
        )
        struct_token_str = _decode_struct_string(tokenizer, input_tokens)

        def reward_fn(tokens):
            aa_str = _decode_aa_string(tokenizer, tokens)
            metrics = reward_model.compute_metrics(aa_str, header)
            return metrics["bb_tmscore"], metrics

        def decode_to_str(tokens):
            return _decode_aa_string(tokenizer, tokens)

        # Re-seed before each target so seed semantics match the
        # existing generate_dplm2_*.py scripts (one global seed per run).
        _set_seed(seed)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            results = mcts_search(
                adapter=adapter,
                input_tokens=input_tokens,
                partial_masks=partial_masks,
                max_iter=args.max_iter,
                reward_fn=reward_fn,
                decode_to_str=decode_to_str,
                config=config,
            )

        _save_targets(save_subdir, header, results, struct_token_str,
                      continue_write=continue_write)
        continue_write = True


def main():
    parser = argparse.ArgumentParser(
        description="ProtInvTree-style MCTS generation for DPLM2."
    )

    # Variant + model loading -------------------------------------------------
    parser.add_argument("--variant", choices=("baseline", "hybrid", "noise"),
                        required=True)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="airkingbd/dplm2_650m")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--bit_model", action="store_true",
                        help="Baseline only: use DPLM2Bit class.")

    # Hybrid-specific (forwarded into load_hybrid_model)
    parser.add_argument("--use_normal_emb", default=None,
                        help="Hybrid only: override hybrid.use_normal_emb.")
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument("--r_min", type=float, default=None)
    parser.add_argument("--r_max", type=float, default=None)
    parser.add_argument("--noise_space", type=str, default="embedding",
                        choices=("embedding", "onehot"))
    parser.add_argument("--cutoff_layer0_attn_residual", action="store_true")

    # Noise-specific
    def _opt_bool(v):
        if v is None or v == "":
            return None
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        raise argparse.ArgumentTypeError(v)

    parser.add_argument("--sample_noise_every_step", type=_opt_bool, default=None,
                        help="Noise only: redraw Gaussian noise per step.")

    # Generation common -------------------------------------------------------
    parser.add_argument("--task", type=str, default="inverse_folding",
                        choices=("inverse_folding",),
                        help="Currently MCTS only supports inverse_folding.")
    parser.add_argument("--input_fasta_path", type=str, required=True)
    parser.add_argument("--saveto", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling_strategy", type=str, default="annealing@2.0:0.1")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Number of denoising steps for the policy.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Currently fixed to 1 (MCTS runs per-target).")
    parser.add_argument("--save_pdb", action="store_true",
                        help="Reserved; PDB writing not yet implemented for MCTS outputs.")

    # Baseline-only sampling knobs (mirrors generate_dplm2.py)
    parser.add_argument("--unmasking_strategy", type=str, default="stochastic1.0")
    parser.add_argument("--remasking_strategy", type=str, default="uncond",
                        choices=("uncond", "cond", "no_remask"))
    parser.add_argument("--decoding_strategy", type=str, default=None)
    parser.add_argument("--feedforward_mode", type=str, default="discrete")
    parser.add_argument("--mask_emb_mode", type=str, default="add",
                        choices=("add", "replace"))

    # Reward model ------------------------------------------------------------
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="Metadata CSV (cath_*, pdb_date, pdb_afdb_cameo) "
                             "providing pdb_name → pdb_path / processed_path.")
    parser.add_argument("--metadata_data_dir", type=str, required=True,
                        help="Root directory that relative columns in --metadata_csv "
                             "are joined against.")
    parser.add_argument("--reward_metric", type=str, default="sc_tmscore",
                        choices=("sc_tmscore", "tmscore_to_target"))
    parser.add_argument("--target_chain_id", type=str, default="A")
    parser.add_argument("--fold_backend", type=str, default="esmfold",
                        choices=("esmfold",))

    # MCTS hyperparameters ----------------------------------------------------
    parser.add_argument("--mcts_iterations", type=int, default=50,
                        help="M in Algorithm 1.")
    parser.add_argument("--mcts_expansions", type=int, default=4,
                        help="K children per Selection.")
    parser.add_argument("--mcts_c_uct", type=float, default=0.01,
                        help="UCT exploration weight (Eq. 3).")
    parser.add_argument("--mcts_reward_threshold", type=float, default=0.99,
                        help="τ: early-accept reward threshold for leaves.")
    parser.add_argument("--mcts_max_depth", type=int, default=0,
                        help="Tree depth cap; 0 = use --max_iter.")
    parser.add_argument("--mcts_num_outputs", type=int, default=1,
                        help="Number of top-reward sequences to save per target.")
    parser.add_argument("--mcts_verbose", action="store_true")

    # Seeds -------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated list of seeds. When set, "
                             "overrides --seed and writes one run per seed "
                             "to <saveto>/seed_<s>.")
    parser.add_argument("--skip_if_generated", action="store_true",
                        help="Skip a seed if its aatype.fasta already exists.")

    args = parser.parse_args()

    if args.batch_size != 1:
        print(f"[warn] --batch_size={args.batch_size} ignored; MCTS runs per-target.")

    # Set up model + adapter (loaded once, reused across seeds & targets).
    model, tokenizer, _device = _load_model(args)
    adapter = _build_adapter(args, model)

    # Set up reward model.
    reward_model = FoldRewardModel(
        device="cuda",
        reward_metric=args.reward_metric,
        target_chain_id=args.target_chain_id,
    )
    target_df = _load_target_index(args.metadata_csv, args.metadata_data_dir)

    seeds = _parse_seeds(args) or [args.seed]
    for seed in seeds:
        seed_save_dir = os.path.join(args.saveto, f"seed_{seed}")
        gen_marker = os.path.join(seed_save_dir, args.task, "aatype.fasta")
        if args.skip_if_generated and os.path.exists(gen_marker):
            print(f"[skip-gen] seed={seed}: {gen_marker} already exists")
            continue
        print(f"=== MCTS seed={seed} -> {seed_save_dir} ===")
        os.makedirs(seed_save_dir, exist_ok=True)
        _run_seed(args, model, adapter, tokenizer, reward_model, target_df,
                  seed, seed_save_dir)


if __name__ == "__main__":
    main()
