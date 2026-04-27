import argparse
import os

from tokenizers import AddedToken
import torch
import tree
from Bio import SeqIO
from peft.peft_model import PeftModel
from tqdm import tqdm

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.models.dplm2 import DPLM2Bit
from byprot.models.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)


def initialize_conditional_generation(
    fasta_path, tokenizer, device, args, model=None
):
    input_data_aatype = []
    input_data_struct_tokens = []
    input_data_name = []

    # setup input data depending on args.task
    # ex.) folding: input aa sequence, mask struct tokens
    # ex.) inverse folding: input struct sequence, mask aa tokens
    # records = len of sequences in the input fasta file
    # = len(input_data_aatype) = len(input_data_struct_tokens) = len(input_data_name)
    for record in SeqIO.parse(fasta_path, "fasta"):
        input_data_name.append(record.name)
        if args.task == "folding":
            aatype = str(record.seq)
            aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
            struct_tokens = tokenizer.struct_mask_token * len(record.seq)
            struct_tokens = (
                tokenizer.struct_cls_token
                + struct_tokens
                + tokenizer.struct_eos_token
            )
        elif args.task == "inverse_folding":
            aatype = tokenizer.aa_mask_token * len(record.seq.split(","))
            aatype = tokenizer.aa_cls_token + aatype + tokenizer.aa_eos_token
            struct_tokens = "".join(str(record.seq).split(","))
            struct_tokens = (
                tokenizer.struct_cls_token
                + struct_tokens
                + tokenizer.struct_eos_token
            )
        else:
            raise NotImplementedError
        input_data_aatype.append(aatype)
        input_data_struct_tokens.append(struct_tokens)

    # sorted by length
    len_input = [len(seq) for seq in input_data_aatype]
    sorted_batch = sorted(
        zip(
            len_input,
            input_data_aatype,
            input_data_struct_tokens,
            input_data_name,
        )
    )
    _, aa, struct, name = zip(*sorted_batch)
    input_data_aatype = list(aa)
    input_data_struct_tokens = list(struct)
    input_data_name = list(name)

#     tokenizer = DPLM2Tokenizer(name_or_path='airkingbd/dplm2_650m', vocab_size=8229, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'aa_cls_token': '<cls_aa>', 'aa_eos_token': '<eos_aa>', 'aa_unk_token': '<unk_aa>', 'aa_mask_token': '<mask_aa>', 'struct_cls_token': '<cls_struct>', 'struct_eos_token': '<eos_struct>', 'struct_unk_token': '<unk_struct>', 'struct_mask_token': '<mask_struct>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
#         0: AddedToken("<cls_aa>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         2: AddedToken("<eos_aa>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         3: AddedToken("<unk_aa>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32: AddedToken("<mask_aa>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         33: AddedToken("<cls_struct>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         34: AddedToken("<eos_struct>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         35: AddedToken("<unk_struct>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         8229: AddedToken("<mask_struct>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#     })

    def build_batch(input_data_aatype, input_data_struct_tokens):
        batch_struct = tokenizer.batch_encode_plus(
            input_data_struct_tokens,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_aa = tokenizer.batch_encode_plus(
            input_data_aatype,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        input_tokens = torch.concat(
            [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
        )
        input_tokens = input_tokens.to(device)

        aa_type = 1
        struct_type = 0
        non_special = model.get_non_special_symbol_mask(input_tokens)
        type_ids = model.get_modality_type(input_tokens)

        # folding
        if args.task == "folding":
            # mask struct token
            input_tokens.masked_fill_(
                (type_ids == struct_type) & non_special,
                tokenizer._token_to_id[tokenizer.struct_mask_token],
            )
            mask_type = aa_type
        # inverse folding
        elif args.task == "inverse_folding":
            # mask aa token
            input_tokens.masked_fill_(
                (type_ids == aa_type) & non_special,
                tokenizer._token_to_id[tokenizer.aa_mask_token],
            )
            mask_type = struct_type

        # construct batch
        batch = {}
        batch["input_tokens"] = input_tokens
        batch["partial_mask"] = type_ids == mask_type

        return batch

    batches = []
    input_data_name_list = []
    # split batch according to args.batch_size
    if args.batch_size > 0:
        num = len(input_data_aatype)
        start = 0
        end = start + args.batch_size
        while end < num + args.batch_size:
            input_data_aa_batch = input_data_aatype[start:end]
            input_data_struct_batch = input_data_struct_tokens[start:end]
            new_batch = build_batch(
                input_data_aa_batch, input_data_struct_batch
            )
            batches.append(new_batch)
            input_data_name_list.append(input_data_name[start:end])
            start += args.batch_size
            end += args.batch_size
    else:
        batches = [build_batch(input_data_aatype, input_data_struct_tokens)]
        input_data_name_list = [input_data_name]

    return batches, input_data_name_list


def initialize_generation(
    task, num_seqs, length, tokenizer, device, batch_size=50
):
    def create_init_seq(length):
        if task == "sequence_generation":
            seq = tokenizer.aa_mask_token * length
            seq = tokenizer.aa_cls_token + seq + tokenizer.aa_eos_token
        elif task in ["co_generation", "backbone_generation"]:
            seq_struct = tokenizer.all_tokens[50] * length
            seq_aa = "A" * length
            seq_struct = (
                tokenizer.struct_cls_token
                + seq_struct
                + tokenizer.struct_eos_token
            )
            seq_aa = tokenizer.aa_cls_token + seq_aa + tokenizer.aa_eos_token
            seq = (seq_struct, seq_aa)
        else:
            raise NotImplementedError

        return seq

    init_struct_list = []
    init_aa_list = []
    for _ in range(num_seqs):
        seq = create_init_seq(length)
        if type(seq) == tuple:
            seq_struct, seq_aa = seq
            seq = seq_struct + seq_aa
        init_struct_list.append(seq_struct)
        init_aa_list.append(seq_aa)

    input_tokens_batch = []
    start = 0
    end = start + batch_size
    while end < num_seqs + batch_size:
        input_data_struct_tokens = init_struct_list[start:end]
        input_data_aatype = init_aa_list[start:end]
        batch_struct = tokenizer.batch_encode_plus(
            input_data_struct_tokens,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_aatype = tokenizer.batch_encode_plus(
            input_data_aatype,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        input_tokens = torch.concat(
            [batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1
        )
        input_tokens = input_tokens.to(device)
        input_tokens_batch.append(input_tokens)
        start += batch_size
        end += batch_size

    return input_tokens_batch


def unconditional_generate(args):
    if args.bit_model:
        model = DPLM2Bit.from_pretrained(args.model_name)
    else:
        model = DPLM2.from_pretrained(args.model_name)

    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

    for seq_len in args.seq_lens:
        max_iter = args.max_iter
        input_tokens_batch = initialize_generation(
            task=args.task,
            num_seqs=args.num_seqs,
            length=seq_len,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            all_outputs = {}
            for input_tokens in input_tokens_batch:
                _struct_tokens, _aatype_tokens = input_tokens.chunk(2, dim=1)
                if args.task == "backbone_generation":
                    input_tokens = _struct_tokens
                if args.task == "sequence_generation":
                    input_tokens = _aatype_tokens
                outputs = model.generate(
                    input_tokens=input_tokens,
                    max_iter=max_iter,
                    temperature=args.temperature,
                    unmasking_strategy=args.unmasking_strategy,
                    sampling_strategy=args.sampling_strategy,
                    remasking_strategy=args.remasking_strategy,
                    decoding_strategy=args.decoding_strategy,
                    feedforward_mode=args.feedforward_mode,
                    mask_emb_mode=args.mask_emb_mode,
                )
                if args.task == "backbone_generation":
                    outputs["output_tokens"] = torch.cat(
                        [outputs["output_tokens"], _aatype_tokens], dim=1
                    )
                for k, v in outputs.items():
                    if k in all_outputs:
                        all_outputs[k] = torch.concat(
                            [all_outputs[k], v], dim=0
                        )
                    else:
                        all_outputs[k] = v

        print("final:")
        if args.task == "backbone_generation":
            print(
                [
                    ",".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        all_outputs["output_tokens"], skip_special_tokens=False
                    )
                ]
            )
        elif args.task == "sequence_generation":
            print(
                [
                    "".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        all_outputs["output_tokens"], skip_special_tokens=False
                    )
                ]
            )
        elif args.task == "co_generation":
            print(
                [
                    ",".join(seq.split(" "))
                    for seq in tokenizer.batch_decode(
                        all_outputs["output_tokens"], skip_special_tokens=False
                    )
                ]
            )
        else:
            raise NotImplementedError

        # save
        save_results(
            outputs=all_outputs,
            task=args.task,
            save_dir=os.path.join(args.saveto, args.task, f"length_{seq_len}"),
            tokenizer=tokenizer,
            struct_tokenizer=model.struct_tokenizer,
            save_pdb=args.save_pdb,
        )


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

    for i, batch in enumerate(tqdm(batches, desc=f"{args.task}")):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=args.max_iter,
                temperature=args.temperature,
                unmasking_strategy=args.unmasking_strategy,
                sampling_strategy=args.sampling_strategy,
                partial_masks=batch["partial_mask"],
                remasking_strategy=args.remasking_strategy,
                decoding_strategy=args.decoding_strategy,
                feedforward_mode=args.feedforward_mode,
                mask_emb_mode=args.mask_emb_mode,
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


def conditional_generate_from_fasta(args):
    if args.bit_model:
        model = DPLM2Bit.from_pretrained(args.model_name)
    else:
        model = DPLM2.from_pretrained(args.model_name)

    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()

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


def save_fasta(
    save_name,
    output_results,
    struct_tokens=False,
    headers=None,
    continue_write=False,
):
    fp_save = (
        open(save_name, "w") if not continue_write else open(save_name, "a")
    )
    for idx, seq in enumerate(output_results):
        if headers is not None:
            fp_save.write(f">{headers[idx]}\n")
        else:
            fp_save.write(f">SEQUENCE_{idx}\n")
        seq = seq.split(" ")
        if struct_tokens:
            fp_save.write(f"{','.join(seq)}\n")
        else:
            fp_save.write(f"{''.join(seq)}\n")
    fp_save.close()


def save_results(
    tokenizer,
    struct_tokenizer,
    save_dir,
    task,
    outputs,
    headers=None,
    save_pdb=True,
    continue_write=False,
):
    # save to fasta
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}...")
    if headers is None:
        headers = [f"sample_{i}" for i in range(len(outputs["output_tokens"]))]

    if task in ["sequence_generation"]:
        aatype_tokens = outputs["output_tokens"]
        aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
        aatype_strings = list(
            map(
                lambda s: "".join(s.split()),
                tokenizer.batch_decode(
                    aatype_tokens, skip_special_tokens=True
                ),
            )
        )
        save_fasta(
            save_name=aatype_fasta_path,
            output_results=aatype_strings,
            headers=headers,
            continue_write=continue_write,
        )

    elif task in [
        "backbone_generation",
        "co_generation",
        "folding",
        "inverse_folding",
    ]:
        output_tokens = outputs["output_tokens"]
        struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)
        struct_token_fasta_path = os.path.join(save_dir, "struct_token.fasta")
        aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
        struct_tokens_strings = list(
            map(
                lambda s: ",".join(s.split()),
                tokenizer.batch_decode(
                    struct_tokens, skip_special_tokens=True
                ),
            )
        )
        aatype_strings = list(
            map(
                lambda s: "".join(s.split()),
                tokenizer.batch_decode(
                    aatype_tokens, skip_special_tokens=True
                ),
            )
        )
        save_fasta(
            save_name=struct_token_fasta_path,
            output_results=struct_tokens_strings,
            headers=headers,
            continue_write=continue_write,
        )
        save_fasta(
            save_name=aatype_fasta_path,
            output_results=aatype_strings,
            headers=headers,
            continue_write=continue_write,
        )
        if save_pdb:
            pdb_save_dir = os.path.join(save_dir, "pdb")
            os.makedirs(pdb_save_dir, exist_ok=True)
            for idx, (header, aatype_str, struct_tokens_str) in enumerate(
                zip(headers, aatype_strings, struct_tokens_strings)
            ):
                # import ipdb; ipdb.set_trace()
                # bit_model
                # ipdb> outputs['output_tokens'].shape
                # torch.Size([50, 260])
                # ipdb> outputs['res_mask'].shape
                # torch.Size([50, 128])
                # ipdb> outputs['res_mask'].sum(dim=1)
                # tensor([ 15,  31,  31,  35,  38,  38,  41,  41,  44,  44,  46,  49,  50,  54,
                #         55,  56,  60,  60,  61,  62,  63,  63,  63,  64,  65,  67,  69,  70,
                #         72,  77,  79,  87,  89,  96,  96,  98, 102, 103, 108, 116, 117, 118,
                #         118, 119, 119, 120, 121, 123, 123, 128], device='cuda:0')
                # outputs['res_mask'] is a binary mask indicating the valid length of the generated structure tokens for each sequence in the batch.
                # ipdb> outputs['final_struct_feature'].shape
                # torch.Size([50, 128, 13])
                # ipdb> outputs['final_struct_feature'][0][0]
                # tensor([ 1.,  1.,  1.,  1.,  1., -1., -1.,  1.,  1., -1.,  1., -1., -1.], device='cuda:0')
                # outputs['final_struct_feature'] contains the final generated structural features for each sequence in the batch, 
                # where the valid features are indicated by the corresponding positions in outputs['res_mask'].
                # valid structural features are represented by 13-dimensional vectors in outputs['final_struct_feature'].
                (
                    aatype_tensor,
                    struct_tokens_tensor,
                ) = struct_tokenizer.string_to_tensor(
                    aatype_str, struct_tokens_str
                )
                if "final_struct_feature" in outputs: # bit_model has "final_struct_feature" in outputs
                    actual_len = int(outputs["res_mask"][idx].sum().item())
                    decoder_out = struct_tokenizer.detokenize(
                        struct_tokens=outputs["final_struct_feature"][idx][:actual_len].unsqueeze(0),
                        res_mask=outputs["res_mask"][idx][:actual_len].unsqueeze(0),
                    )
                else:
                    decoder_out = struct_tokenizer.detokenize(
                        struct_tokens_tensor
                    )

                decoder_out["aatype"] = aatype_tensor
                decoder_out["header"] = [header]

                struct_tokenizer.output_to_pdb(
                    decoder_out, output_dir=pdb_save_dir
                )
    else:
        raise NotImplementedError

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds (e.g. '42,43,44'). When set, "
        "overrides --seed and runs generation once per seed, writing each "
        "run to <saveto>/seed_<seed>. Model is loaded only once. "
        "Conditional tasks only.",
    )
    parser.add_argument(
        "--skip_if_generated",
        action="store_true",
        help="When used with --seeds, skip seeds whose <saveto>/seed_<s>/"
        "<task>/aatype.fasta already exists.",
    )
    parser.add_argument(
        "--model_name", type=str, default="airkingbd/dplm_150m"
    )
    parser.add_argument("--num_seqs", type=int, default=40)
    parser.add_argument("--seq_lens", nargs="*", type=int)
    parser.add_argument("--saveto", type=str, default="gen.fasta")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling_strategy", type=str, default="annealing@2.0:0.1"
    )
    parser.add_argument(
        "--unmasking_strategy", type=str, default="stochastic1.0"
    )
    parser.add_argument(
        "--remasking_strategy",
        type=str,
        default="uncond",
        choices=["uncond", "cond", "no_remask"],
        help=(
            "Conditioning mode for reparameterized decoding. "
            "'uncond': standard; any token can be re-masked. "
            "'cond': conservative re-masking (only when score drops and token unchanged). "
            "'no_remask': once a token is unmasked it is never re-masked."
        ),
    )
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--save_pdb", type=bool, default=True)
    parser.add_argument("--bit_model", action="store_true")
    parser.add_argument(
        "--decoding_strategy",
        type=str,
        default=None,
        help=(
            "New decoding strategy. If None, uses legacy reparam decoding. "
            "Options: dinfer_threshold@0.8, klass@0.01:0.9:2:1, "
            "dinfer_credit@0.8:0.8:0.2:0.7, dinfer_hierarchical@0.92:0.62, punt@0.04"
        ),
    )
    parser.add_argument(
        "--feedforward_mode",
        type=str,
        default="discrete",
        help=(
            "Soft embedding feedforward mode. "
            "discrete: no soft embedding (default). "
            "linear: dInfer IterSmooth style (default params). "
            "linear@init:growth:preset: linear with custom params. "
            "entropy: LRD style."
        ),
    )
    parser.add_argument(
        "--mask_emb_mode",
        type=str,
        default="add",
        choices=["add", "replace"],
        help=(
            "How to mix mask embedding with predicted token embedding. "
            "add: e_mask + alpha*E[e_v] (dInfer). "
            "replace: (1-alpha)*e_mask + alpha*E[e_v] (LRD)."
        ),
    )

    # generation options
    ## task option
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
        default="co_generation",
    )
    ## conditional testset
    parser.add_argument("--input_fasta_path", type=str, default="")

    args = parser.parse_args()

    if not _parse_seeds(args):
        _set_seed(args.seed)

    if args.task in [
        "backbone_generation",
        "sequence_generation",
        "co_generation",
    ]:
        unconditional_generate(args)
    elif args.task in ["folding", "inverse_folding"]:
        conditional_generate_from_fasta(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
