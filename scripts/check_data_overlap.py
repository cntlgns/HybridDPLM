"""
Train/Valid와 Test(PDB_date, cameo2022) 간 서열 수준 겹침 확인 스크립트.

Usage:
    python scripts/check_data_overlap.py
"""

import pandas as pd


DATA_DIR = "data-bin"
TRAIN_PARQUETS = [
    f"{DATA_DIR}/pdb_swissprot/train/train-00000-of-00002.parquet",
    f"{DATA_DIR}/pdb_swissprot/train/train-00001-of-00002.parquet",
]
VALID_PARQUET = f"{DATA_DIR}/pdb_swissprot/valid/train-00000-of-00001.parquet"
PDB_DATE_FASTA = f"{DATA_DIR}/PDB_date/aatype.fasta"
PDB_DATE_STRUCT_FASTA = f"{DATA_DIR}/PDB_date/struct.fasta"
CAMEO_FASTA = f"{DATA_DIR}/cameo2022/aatype.fasta"
CAMEO_STRUCT_FASTA = f"{DATA_DIR}/cameo2022/struct.fasta"


def read_fasta(path: str) -> dict[str, str]:
    """FASTA 파일에서 {name: sequence} dict 반환."""
    seqs = {}
    name, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(parts)
                name, parts = line[1:], []
            else:
                parts.append(line)
        if name:
            seqs[name] = "".join(parts)
    return seqs


def load_train() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in TRAIN_PARQUETS])


def load_valid() -> pd.DataFrame:
    return pd.read_parquet(VALID_PARQUET)


def check_overlap(
    name_a: str,
    seqs_a: dict[str, str],
    name_b: str,
    seqs_b: dict[str, str],
    show_examples: int = 5,
):
    set_a = set(seqs_a.values())
    set_b = set(seqs_b.values())
    overlap = set_a & set_b

    print(f"\n{'='*60}")
    print(f"{name_a} ({len(set_a)} unique) vs {name_b} ({len(set_b)} unique)")
    print(f"동일 서열 수: {len(overlap)}")
    print(f"{'='*60}")

    if not overlap:
        return overlap

    # 겹치는 서열의 이름 매핑
    inv_a = {}
    for n, s in seqs_a.items():
        inv_a.setdefault(s, []).append(n)
    inv_b = {}
    for n, s in seqs_b.items():
        inv_b.setdefault(s, []).append(n)

    shown = 0
    for seq in overlap:
        if shown >= show_examples:
            print(f"  ... 외 {len(overlap) - show_examples}개")
            break
        print(f"  {name_a}: {inv_a[seq][:3]}  <->  {name_b}: {inv_b[seq][:3]}")
        shown += 1

    return overlap


def check_overlap_aa_and_struct(
    name_a: str,
    aa_seqs_a: dict[str, str],
    struct_seqs_a: dict[str, str],
    name_b: str,
    aa_seqs_b: dict[str, str],
    struct_seqs_b: dict[str, str],
    show_examples: int = 5,
):
    """AA 서열 + 구조 토큰 시퀀스가 모두 일치하는 겹침을 확인."""
    pairs_a = {n: (aa_seqs_a[n], struct_seqs_a[n]) for n in aa_seqs_a if n in struct_seqs_a}
    pairs_b = {n: (aa_seqs_b[n], struct_seqs_b[n]) for n in aa_seqs_b if n in struct_seqs_b}

    set_a = set(pairs_a.values())
    set_b = set(pairs_b.values())
    overlap = set_a & set_b

    print(f"\n{'='*60}")
    print(f"[AA+구조] {name_a} ({len(set_a)} unique) vs {name_b} ({len(set_b)} unique)")
    print(f"AA+구조 모두 동일한 수: {len(overlap)}")
    print(f"{'='*60}")

    if not overlap:
        return overlap

    inv_a = {}
    for n, pair in pairs_a.items():
        inv_a.setdefault(pair, []).append(n)
    inv_b = {}
    for n, pair in pairs_b.items():
        inv_b.setdefault(pair, []).append(n)

    shown = 0
    for pair in overlap:
        if shown >= show_examples:
            print(f"  ... 외 {len(overlap) - show_examples}개")
            break
        print(f"  {name_a}: {inv_a[pair][:3]}  <->  {name_b}: {inv_b[pair][:3]}")
        shown += 1

    return overlap


def main():
    # 데이터 로드
    print("데이터 로드 중...")
    train_df = load_train()
    valid_df = load_valid()
    pdb_date = read_fasta(PDB_DATE_FASTA)
    cameo = read_fasta(CAMEO_FASTA)
    import ipdb; ipdb.set_trace()

    pdb_date_struct = read_fasta(PDB_DATE_STRUCT_FASTA)
    cameo_struct = read_fasta(CAMEO_STRUCT_FASTA)

    train_seqs = dict(zip(train_df["pdb_name"], train_df["aa_seq"]))
    valid_seqs = dict(zip(valid_df["pdb_name"], valid_df["aa_seq"]))
    train_struct_seqs = dict(zip(train_df["pdb_name"], train_df["struct_seq"]))
    valid_struct_seqs = dict(zip(valid_df["pdb_name"], valid_df["struct_seq"]))

    print(f"Train: {len(train_df)}개 (unique 서열: {len(set(train_seqs.values()))})")
    print(f"Valid: {len(valid_df)}개 (unique 서열: {len(set(valid_seqs.values()))})")
    print(f"PDB_date: {len(pdb_date)}개 (unique 서열: {len(set(pdb_date.values()))})")
    print(f"cameo2022: {len(cameo)}개 (unique 서열: {len(set(cameo.values()))})")

    # 겹침 확인
    check_overlap("PDB_date", pdb_date, "Train", train_seqs)
    check_overlap("PDB_date", pdb_date, "Valid", valid_seqs)
    check_overlap("cameo2022", cameo, "Train", train_seqs)
    check_overlap("cameo2022", cameo, "Valid", valid_seqs)
    check_overlap("PDB_date", pdb_date, "cameo2022", cameo)

    # 구조 토큰만 일치하는 겹침 확인
    print("\n\n" + "#"*60)
    print("# 구조 토큰 시퀀스만 일치하는 겹침 확인")
    print("#"*60)
    check_overlap("PDB_date(struct)", pdb_date_struct, "Train(struct)", train_struct_seqs)
    check_overlap("PDB_date(struct)", pdb_date_struct, "Valid(struct)", valid_struct_seqs)
    check_overlap("cameo2022(struct)", cameo_struct, "Train(struct)", train_struct_seqs)
    check_overlap("cameo2022(struct)", cameo_struct, "Valid(struct)", valid_struct_seqs)
    check_overlap("PDB_date(struct)", pdb_date_struct, "cameo2022(struct)", cameo_struct)

    # AA + 구조 토큰 모두 일치하는 겹침 확인
    print("\n\n" + "#"*60)
    print("# AA + 구조 토큰 시퀀스 동시 일치 겹침 확인")
    print("#"*60)
    check_overlap_aa_and_struct("PDB_date", pdb_date, pdb_date_struct, "Train", train_seqs, train_struct_seqs)
    check_overlap_aa_and_struct("PDB_date", pdb_date, pdb_date_struct, "Valid", valid_seqs, valid_struct_seqs)
    check_overlap_aa_and_struct("cameo2022", cameo, cameo_struct, "Train", train_seqs, train_struct_seqs)
    check_overlap_aa_and_struct("cameo2022", cameo, cameo_struct, "Valid", valid_seqs, valid_struct_seqs)
    check_overlap_aa_and_struct("PDB_date", pdb_date, pdb_date_struct, "cameo2022", cameo, cameo_struct)


if __name__ == "__main__":
    main()
