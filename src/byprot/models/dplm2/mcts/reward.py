"""Folding-based reward for ProtInvTree-style MCTS.

Implements Eq. 11 of the paper: ``R(s_t, a_t) = TMScore(f(x~_T), c)``
where ``f`` is a folding model (ESMFold) and ``c`` is the target
backbone structure. Folds the designed amino-acid sequence in-process
(no temp files) and returns a float TM-score against a cached target.

We support two reward modes selectable via ``reward_metric``:

* ``"tmscore_to_target"`` — fold designed seq → TM-score vs. target PDB
  backbone (= the structure conditioning ``c``). Closest match to Eq. 11.
* ``"sc_tmscore"`` — fold both designed seq and ground-truth seq with
  the same model → TM-score between the two folded structures. Matches
  the paper's ``calc_tm_score`` default and the eval pipeline's
  ``bb_tmscore`` column. Requires the GT sequence to be supplied per
  target.

Targets are loaded once per PDB and cached by header so a single search
over many MCTS iterations only pays the parsing cost once.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from byprot.datamodules.pdb_dataset import utils as du
from byprot.utils.protein.utils import calc_tm_score
from openfold.utils.superimposition import superimpose


VALID_METRICS = ("tmscore_to_target", "sc_tmscore")


@dataclass
class TargetEntry:
    """Cached features for one target backbone structure."""
    bb_positions: np.ndarray   # [L*3, 3] backbone N/Ca/C atoms flattened
    aa_seq: str                # ground-truth AA sequence (str, length L)
    folded_bb: Optional[np.ndarray] = None  # [L*3, 3] cached fold of GT seq
    folded_plddt: Optional[float] = None     # mean plddt of fold(GT) (cached)


class FoldRewardModel:
    """ESMFold-backed sc-TMScore reward.

    Thread-safe via a lock around the ESMFold call (the ``esm.infer``
    path is not safe to call concurrently from threads sharing weights).
    """

    def __init__(
        self,
        device: str = "cuda",
        reward_metric: str = "sc_tmscore",
        target_chain_id: str = "A",
    ):
        if reward_metric not in VALID_METRICS:
            raise ValueError(
                f"reward_metric={reward_metric!r}; expected one of {VALID_METRICS}"
            )
        self.device = device
        self.reward_metric = reward_metric
        self.target_chain_id = target_chain_id
        self._esmf = None
        self._esmf_lock = threading.Lock()
        self._target_cache: Dict[str, TargetEntry] = {}

    # ------------------------------------------------------------------
    # ESMFold lifecycle
    # ------------------------------------------------------------------
    def _ensure_esmf(self):
        if self._esmf is not None:
            return
        import esm  # imported lazily to avoid loading on import

        with self._esmf_lock:
            if self._esmf is None:
                model = esm.pretrained.esmfold_v1().eval()
                # Match the dtype guidance used by FoldingModel.
                model = model.to(self.device)
                self._esmf = model

    @torch.no_grad()
    def _fold(self, aa_seq: str) -> Tuple[np.ndarray, float]:
        """Fold ``aa_seq``; return (backbone N/Ca/C positions [L*3, 3], mean_plddt)."""
        self._ensure_esmf()
        seq = aa_seq.replace("X", "A")
        if len(seq) == 0:
            return np.zeros((0, 3), dtype=np.float64), 0.0
        # Disable any ambient autocast (e.g. the bfloat16 autocast that
        # generate_dplm2_mcts.py wraps mcts_search in for the policy
        # forward passes). ESMFold's output_to_pdb calls .numpy() which
        # does not support BFloat16, so we force fp32 here.
        with self._esmf_lock, torch.cuda.amp.autocast(enabled=False):
            esmf_outputs = self._esmf.infer(seq)
            pdb_str = self._esmf.output_to_pdb(esmf_outputs)[0]
            mean_plddt = float(esmf_outputs["mean_plddt"][0].item())
        return _bb_positions_from_pdb_str(pdb_str), mean_plddt

    # ------------------------------------------------------------------
    # Target loading / caching
    # ------------------------------------------------------------------
    def register_target(
        self,
        header: str,
        pdb_path: str,
        gt_aa_seq: Optional[str] = None,
    ) -> TargetEntry:
        """Parse + cache a target PDB for `header`.

        Returns the cached TargetEntry. Idempotent.
        """
        if header in self._target_cache:
            return self._target_cache[header]
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"Target PDB not found: {pdb_path}")
        feats = du.parse_pdb_feats(header, pdb_path, chain_id=self.target_chain_id)
        bb = feats["atom_positions"][:, :3].reshape(-1, 3)
        if gt_aa_seq is None:
            gt_aa_seq = du.aatype_to_seq(feats["aatype"])
        entry = TargetEntry(bb_positions=bb, aa_seq=gt_aa_seq)
        self._target_cache[header] = entry
        return entry

    def register_target_from_feats(
        self,
        header: str,
        chain_feats: dict,
        gt_aa_seq: Optional[str] = None,
    ) -> TargetEntry:
        """Cache a target from a pre-loaded chain-feats dict.

        Mirrors the dict that ``parse_pdb_feats`` returns. If the dict
        carries a ``modeled_idx`` array (cameo / PDB_date pkls do — only
        a subset of the chain has resolved coordinates) we restrict the
        cached backbone + GT sequence to those positions, since both the
        model input and the designed sequence are also only over the
        modeled portion.
        """
        if header in self._target_cache:
            return self._target_cache[header]

        atom_positions = np.asarray(chain_feats["atom_positions"])
        aatype = np.asarray(chain_feats["aatype"])

        modeled_idx = chain_feats.get("modeled_idx")
        if modeled_idx is not None and len(modeled_idx) > 0 \
                and len(modeled_idx) != atom_positions.shape[0]:
            modeled_idx = np.asarray(modeled_idx)
            atom_positions = atom_positions[modeled_idx]
            aatype = aatype[modeled_idx]

        bb = atom_positions[:, :3].reshape(-1, 3)
        if gt_aa_seq is None:
            gt_aa_seq = du.aatype_to_seq(aatype)
        entry = TargetEntry(bb_positions=bb, aa_seq=gt_aa_seq)
        self._target_cache[header] = entry
        return entry

    def get_target(self, header: str) -> TargetEntry:
        if header not in self._target_cache:
            raise KeyError(
                f"Target {header!r} not registered. Call register_target() first."
            )
        return self._target_cache[header]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, aa_seq: str, header: str) -> float:
        """Reward for ``aa_seq`` against the target stored for ``header``.

        Thin wrapper over :meth:`compute_metrics` returning just the
        TM-score (the field that ``self.reward_metric`` selects).
        """
        return self.compute_metrics(aa_seq, header)["bb_tmscore"]

    def compute_metrics(self, aa_seq: str, header: str) -> Dict[str, float]:
        """Fold ``aa_seq`` once and return the full eval-pipeline metric dict.

        Mirrors the columns produced per-sample by the inverse-folding
        eval pipeline (``process_folded_outputs`` + the seq-recovery line
        in ``evaluator_dplm2.evaluate_inverse_folding``):

        * ``bb_tmscore``, ``bb_rmsd``, ``ca_rmsd`` — fold(designed) vs the
          reference selected by ``self.reward_metric``. For
          ``tmscore_to_target`` (default match to eval) the reference is
          the GT structure; for ``sc_tmscore`` it is fold(GT_seq).
        * ``mean_plddt`` — ESMFold's per-residue plddt averaged over the
          designed sequence.
        * ``inv_fold_seq_recovery`` — char-match between designed and GT
          AA sequence (over the common length).
        * ``length`` — designed sequence length.
        """
        target = self.get_target(header)
        L_des_seq = len(aa_seq)
        zero = {
            "bb_tmscore": 0.0,
            "bb_rmsd": 0.0,
            "ca_rmsd": 0.0,
            "mean_plddt": 0.0,
            "inv_fold_seq_recovery": 0.0,
            "length": float(L_des_seq),
        }
        if not aa_seq or all(c == "X" for c in aa_seq):
            return zero

        designed_bb, mean_plddt = self._fold(aa_seq)

        if self.reward_metric == "tmscore_to_target":
            ref_bb = target.bb_positions
            ref_seq = target.aa_seq
        else:  # sc_tmscore: fold the GT too (cached)
            if target.folded_bb is None:
                target.folded_bb, target.folded_plddt = self._fold(target.aa_seq)
            ref_bb = target.folded_bb
            ref_seq = target.aa_seq

        L_des = designed_bb.shape[0] // 3
        L_ref = ref_bb.shape[0] // 3
        if L_des == 0 or L_ref == 0:
            zero["mean_plddt"] = mean_plddt
            return zero

        # Truncate to common length so calc_tm_score's input arrays line up.
        L = min(L_des, L_ref)
        des_bb = designed_bb[: L * 3]
        ref_bb_t = ref_bb[: L * 3]
        des_ca = des_bb.reshape(-1, 3, 3)[:, 1, :]   # CA atoms only
        ref_ca = ref_bb_t.reshape(-1, 3, 3)[:, 1, :]
        seq_a = aa_seq[:L].replace("X", "A")
        seq_b = ref_seq[:L].replace("X", "A")

        # Mask out residues with all-zero reference coords (unmodeled in PDB).
        ref_ca_t = torch.from_numpy(ref_ca).float()
        res_mask = (ref_ca_t.abs().sum(-1) > 1e-7).float()
        if res_mask.sum() < 1:
            zero["mean_plddt"] = mean_plddt
            return zero

        # bb_tmscore — calc_tm_score wants [L, 3, 3]
        try:
            tm1, _ = calc_tm_score(
                des_bb.reshape(-1, 3, 3),
                ref_bb_t.reshape(-1, 3, 3),
                seq_a,
                seq_b,
            )
            bb_tmscore = float(tm1) if (tm1 is not None and not math.isnan(float(tm1))) else 0.0
        except Exception:  # pragma: no cover - tmtools alignment failure
            bb_tmscore = 0.0

        # ca_rmsd — superimpose CA atoms only
        try:
            _, ca_rmsd_t = superimpose(
                torch.from_numpy(ref_ca).float()[None],
                torch.from_numpy(des_ca).float()[None],
                res_mask,
            )
            ca_rmsd = float(ca_rmsd_t.item())
        except Exception:  # pragma: no cover
            ca_rmsd = 0.0

        # bb_rmsd — superimpose all backbone atoms (mask broadcast across N/Ca/C)
        try:
            bb_mask = res_mask[:, None].repeat(1, 3).reshape(-1)
            _, bb_rmsd_t = superimpose(
                torch.from_numpy(ref_bb_t).float()[None],
                torch.from_numpy(des_bb).float()[None],
                bb_mask,
            )
            bb_rmsd = float(bb_rmsd_t.item())
        except Exception:  # pragma: no cover
            bb_rmsd = 0.0

        # Sequence recovery vs GT AA seq (over the common length L).
        recovery_pairs = list(zip(aa_seq[:L], target.aa_seq[:L]))
        if recovery_pairs:
            inv_fold_seq_recovery = sum(a == b for a, b in recovery_pairs) / len(recovery_pairs)
        else:
            inv_fold_seq_recovery = 0.0

        return {
            "bb_tmscore": bb_tmscore,
            "bb_rmsd": bb_rmsd,
            "ca_rmsd": ca_rmsd,
            "mean_plddt": mean_plddt,
            "inv_fold_seq_recovery": float(inv_fold_seq_recovery),
            "length": float(L_des_seq),
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _bb_positions_from_pdb_str(pdb_str: str) -> np.ndarray:
    """Extract N/Ca/C backbone positions from a PDB string written by ESMFold.

    Returns a float64 array of shape [L*3, 3] where rows are interleaved
    [N_0, Ca_0, C_0, N_1, Ca_1, C_1, ...]. Uses Biopython's PDBParser
    for robustness (matches what ``parse_pdb_feats`` does on disk).
    """
    import io
    from Bio import PDB as _PDB

    parser = _PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("folded", io.StringIO(pdb_str))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # skip hetero
                    continue
                try:
                    n = residue["N"].get_coord()
                    ca = residue["CA"].get_coord()
                    c = residue["C"].get_coord()
                except KeyError:
                    continue
                coords.extend([n, ca, c])
        break  # only model 1
    if not coords:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)
