"""
Build a CATH inverse-folding eval dataset matching the cameo2022/PDB_date layout.

Outputs (per <out_root>/cath_<ver>_<subset>/):
  pdbs/<name>.pdb         backbone-only PDBs reconstructed from chain_set.jsonl
  struct.fasta            struct tokens (comma-separated 4-digit IDs)
  aatype.fasta            AA sequences

Also writes <out_root>/metadata/cath_<ver>_<subset>.csv with columns:
  pdb_name, pdb_path, seq_len, modeled_seq_len, aa_seq, struct_seq, plddt

Filesystem-safe naming: chain ids like "3fkf.A" are written as "3fkf_A".

Subsets follow the Ingraham 2019 convention:
  all           the full chain_set_splits.json["test"] list
  short         test entries with sequence length <= 100
  single_chain  test entries whose CATH topology appears in cath_nodes only once
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biotite.sequence.io import fasta
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from byprot.datamodules.pdb_dataset import residue_constants as rc  # noqa: E402
from byprot.datamodules.pdb_dataset import utils as du  # noqa: E402
from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn  # noqa: E402
from byprot.models.utils import get_struct_tokenizer  # noqa: E402
from byprot.utils import recursive_to  # noqa: E402
from byprot.utils.protein.tokenize_pdb import load_from_pdb  # noqa: E402
from byprot.utils.protein.utils import write_prot_to_pdb  # noqa: E402


# atom37 indices for the four backbone atoms we have in chain_set.jsonl
ATOM37_INDEX = {"N": 0, "CA": 1, "C": 2, "O": 4}


def select_subset(splits, jsonl_index, subset, max_length):
    test_names = splits["test"]

    out = []
    for name in test_names:
        if name not in jsonl_index:
            continue
        entry = jsonl_index[name]
        seq_len = len(entry["seq"])
        if seq_len > max_length:
            continue
        if subset == "short" and seq_len > 100:
            continue
        if subset == "single_chain":
            # Single-domain test chains. Use the per-entry CATH topology list
            # (present in both 4.2 and 4.3); 4.2's splits also has a redundant
            # "cath_nodes" mapping but per-entry is the single source of truth.
            topos = entry.get("CATH", [])
            if len(topos) != 1:
                continue
        out.append(entry)
    return out


def jsonl_entry_to_pdb(entry, out_path):
    """Write a backbone-only PDB from a chain_set.jsonl entry. Returns True on success."""
    seq = entry["seq"]
    L = len(seq)

    aatype = np.array(
        [rc.restype_order.get(a, rc.unk_restype_index) for a in seq],
        dtype=np.int64,
    )

    coords = entry["coords"]
    pos37 = np.zeros((L, 37, 3), dtype=np.float32)
    mask37 = np.zeros((L, 37), dtype=np.float32)

    for atom_name, idx in ATOM37_INDEX.items():
        arr = np.asarray(coords[atom_name], dtype=np.float32)
        if arr.shape != (L, 3):
            return False
        valid = ~np.isnan(arr).any(axis=-1)
        pos37[valid, idx, :] = arr[valid]
        mask37[valid, idx] = 1.0

    res_valid = mask37[:, [0, 1, 2, 4]].all(axis=-1)
    if not res_valid.any():
        return False

    pos37[~res_valid] = 0.0
    mask37[~res_valid] = 0.0

    write_prot_to_pdb(
        prot_pos=pos37,
        file_path=str(out_path),
        aatype=aatype,
        no_indexing=True,
        overwrite=True,
        atom37_mask=mask37.astype(bool),
        omit_missing_residue=True,
    )
    return True


def tokenize(struct_tokenizer, pdb_dir, out_dir, names):
    all_data = []
    skipped = []
    for name in names:
        pdb_path = pdb_dir / f"{name}.pdb"
        if not pdb_path.exists():
            skipped.append((name, "no pdb"))
            continue
        try:
            feats = load_from_pdb(
                str(pdb_path), process_chain=struct_tokenizer.process_chain
            )
        except Exception as e:
            skipped.append((name, f"load_from_pdb: {type(e).__name__}: {e}"))
            continue
        feats["pdb_path"] = str(pdb_path)
        feats["header"] = name
        feats["pdb_name"] = name
        all_data.append(feats)

    dataloader = torch.utils.data.DataLoader(
        all_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    device = next(struct_tokenizer.parameters()).device
    struct_pairs, aa_pairs = [], []
    pbar = tqdm(dataloader, desc="tokenize")
    for batch in pbar:
        pdb_name = batch["pdb_name"][0]
        pbar.set_description(f"tokenize: {pdb_name} (L={batch['seq_length'][0]})")
        batch = recursive_to(batch, device)
        with torch.no_grad():
            struct_ids = struct_tokenizer.tokenize(
                batch["all_atom_positions"], batch["res_mask"], batch["seq_length"]
            )
        ids = struct_ids.cpu().tolist()[0]
        struct_seq = struct_tokenizer.struct_ids_to_seq(ids)
        aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])
        struct_pairs.append((pdb_name, struct_seq))
        aa_pairs.append((pdb_name, aa_seq))

    fasta.FastaFile.write_iter(str(out_dir / "struct.fasta"), struct_pairs)
    fasta.FastaFile.write_iter(str(out_dir / "aatype.fasta"), aa_pairs)
    return struct_pairs, aa_pairs, skipped


def build_metadata(struct_pairs, aa_pairs, dataset_name, pdb_subdir="pdbs"):
    aa_dict = dict(aa_pairs)
    rows = []
    for name, struct_seq in struct_pairs:
        aa_seq = aa_dict[name]
        rows.append(
            {
                "pdb_name": name,
                "pdb_path": f"{dataset_name}/{pdb_subdir}/{name}.pdb",
                "seq_len": len(aa_seq),
                "modeled_seq_len": len(aa_seq),
                "aa_seq": aa_seq,
                "struct_seq": struct_seq,
                "plddt": 100.0,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cath_version", choices=["4.2", "4.3"], default="4.2")
    ap.add_argument(
        "--subset", choices=["all", "short", "single_chain"], default="all"
    )
    ap.add_argument("--out_root", default=str(REPO / "data-bin"))
    ap.add_argument("--max_length", type=int, default=500)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of test entries (sanity testing).",
    )
    ap.add_argument(
        "--skip_tokenize",
        action="store_true",
        help="Write PDBs only; skip struct-tokenization + metadata.",
    )
    args = ap.parse_args()

    cath_dir = Path(args.out_root) / f"cath_{args.cath_version}"
    jsonl_path = cath_dir / "chain_set.jsonl"
    # 4.2 ships "chain_set_splits.json"; 4.3 ships "splits.json".
    splits_path = None
    for candidate in ("chain_set_splits.json", "splits.json"):
        p = cath_dir / candidate
        if p.exists():
            splits_path = p
            break
    if not jsonl_path.exists() or splits_path is None:
        sys.exit(
            f"Missing data in {cath_dir}.\n"
            f"  chain_set.jsonl: {'ok' if jsonl_path.exists() else 'MISSING'}\n"
            f"  splits file:     MISSING (looked for chain_set_splits.json / splits.json)\n"
            f"Run `bash scripts/download_cath.sh` first."
        )

    print(f"[load] {jsonl_path}")
    jsonl_index = {}
    with open(jsonl_path) as f:
        for line in f:
            e = json.loads(line)
            jsonl_index[e["name"]] = e
    print(f"  {len(jsonl_index)} entries in jsonl")

    with open(splits_path) as f:
        splits = json.load(f)
    print(
        f"  splits: train={len(splits['train'])} "
        f"val={len(splits['validation'])} test={len(splits['test'])}"
    )

    entries = select_subset(splits, jsonl_index, args.subset, args.max_length)
    print(
        f"[filter] subset={args.subset} max_len={args.max_length} "
        f"-> {len(entries)} entries"
    )
    if args.limit:
        entries = entries[: args.limit]
        print(f"  limited to {len(entries)}")

    dataset_name = f"cath_{args.cath_version}_{args.subset}"
    out_dir = Path(args.out_root) / dataset_name
    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    safe_names = []
    failed = []
    print(f"[write pdb] -> {pdb_dir}")
    for entry in tqdm(entries, desc="pdb"):
        safe_name = entry["name"].replace(".", "_")
        out_path = pdb_dir / f"{safe_name}.pdb"
        if out_path.exists() and out_path.stat().st_size > 0:
            safe_names.append(safe_name)
            continue
        try:
            ok = jsonl_entry_to_pdb(entry, out_path)
        except Exception as e:
            failed.append((entry["name"], f"{type(e).__name__}: {e}"))
            ok = False
        if ok:
            safe_names.append(safe_name)
        else:
            failed.append((entry["name"], "all-NaN or shape mismatch"))
    print(f"  wrote {len(safe_names)}/{len(entries)} pdbs ({len(failed)} skipped)")
    for n, why in failed[:5]:
        print(f"    - {n}: {why}")

    if args.skip_tokenize:
        print("[skip tokenize] --skip_tokenize set; done.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] struct_tokenizer ({device})")
    tok = get_struct_tokenizer().to(device).eval()

    struct_pairs, aa_pairs, skipped = tokenize(tok, pdb_dir, out_dir, safe_names)
    print(f"[tokenize] {len(struct_pairs)} ok, {len(skipped)} skipped")
    for n, why in skipped[:5]:
        print(f"    - {n}: {why}")

    meta = build_metadata(struct_pairs, aa_pairs, dataset_name)
    meta_dir = Path(args.out_root) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{dataset_name}.csv"
    meta.to_csv(meta_path, index=False)
    print(f"[metadata] wrote {meta_path} ({len(meta)} rows)")
    print(f"\nDataset: {dataset_name}")
    print(f"  struct.fasta: {out_dir / 'struct.fasta'}")
    print(f"  aatype.fasta: {out_dir / 'aatype.fasta'}")
    print(f"  metadata:     {meta_path}")


if __name__ == "__main__":
    main()
