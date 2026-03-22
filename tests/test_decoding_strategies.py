"""
Unit tests for decoding strategies.

These tests verify the decoding strategies work correctly on synthetic tensors
without requiring a full model. Each strategy is tested for:
1. Correct unmasking behavior (tokens get unmasked, not re-masked)
2. Early stopping safety (no changes when all tokens unmasked)
3. Fallback behavior (at least one token unmasked per step)
4. Strategy-specific logic
"""

import math
import pytest
import torch

from byprot.models.dplm2.decoding_strategies import (
    decode_dinfer_threshold,
    decode_dinfer_hierarchical,
    decode_dinfer_credit,
    decode_klass,
    decode_punt,
    KLASSState,
    CreditState,
    parse_strategy_name,
    parse_strategy_kwargs,
    _find_contiguous_spans,
    _fallback_topk,
)


# ---- Fixtures ----

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def basic_tensors(device):
    """Create a basic set of tensors simulating a batch of 2 sequences, length 10."""
    B, L = 2, 10
    MASK_ID = 32

    # Previous output: positions 3,4,5,6,7 are masked (MASK_ID), rest are unmasked
    output_tokens = torch.tensor([
        [1, 2, 3, MASK_ID, MASK_ID, MASK_ID, MASK_ID, MASK_ID, 8, 9],
        [10, 11, MASK_ID, MASK_ID, MASK_ID, 15, 16, MASK_ID, 18, 19],
    ], device=device)
    output_scores = torch.full((B, L), -math.inf, device=device)
    # Unmasked positions have some score
    output_scores[0, :3] = torch.log(torch.tensor([0.9, 0.8, 0.7]))
    output_scores[0, 8:] = torch.log(torch.tensor([0.6, 0.5]))
    output_scores[1, :2] = torch.log(torch.tensor([0.9, 0.8]))
    output_scores[1, 5:7] = torch.log(torch.tensor([0.7, 0.6]))
    output_scores[1, 8:] = torch.log(torch.tensor([0.5, 0.4]))

    # Current predictions for all positions
    cur_tokens = torch.tensor([
        [1, 2, 3, 20, 21, 22, 23, 24, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ], device=device)
    # cur_scores = log-prob of predicted token (higher = more confident)
    cur_scores = torch.log(torch.tensor([
        [0.9, 0.8, 0.7, 0.95, 0.3, 0.6, 0.85, 0.4, 0.6, 0.5],
        [0.9, 0.8, 0.7, 0.2, 0.5, 0.7, 0.6, 0.88, 0.5, 0.4],
    ], device=device))

    # xt_neq_x0: True where masked
    xt_neq_x0 = output_tokens.eq(MASK_ID)

    # non_special_sym_mask: True for all valid positions (exclude pos 0 as "special")
    non_special_sym_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    non_special_sym_mask[:, 0] = False  # position 0 is "special" (e.g. BOS)

    return {
        "output_tokens": output_tokens,
        "output_scores": output_scores,
        "cur_tokens": cur_tokens,
        "cur_scores": cur_scores,
        "xt_neq_x0": xt_neq_x0,
        "non_special_sym_mask": non_special_sym_mask,
        "mask_id": MASK_ID,
        "B": B,
        "L": L,
    }


@pytest.fixture
def log_probs_tensors(device):
    """Create fake log-probability tensors for strategies that need full distributions."""
    B, L, V = 2, 10, 50
    # Random log-probs (log-softmax)
    logits = torch.randn(B, L, V, device=device)
    log_probs = logits.log_softmax(dim=-1)
    return log_probs, V


# ---- Test: parse functions ----

class TestParsing:
    def test_parse_reparam(self):
        assert parse_strategy_name("reparam-uncond-stochastic1.0-linear") == "reparam"

    def test_parse_dinfer_threshold(self):
        assert parse_strategy_name("dinfer_threshold") == "dinfer_threshold"
        assert parse_strategy_name("dinfer_threshold@0.9") == "dinfer_threshold"

    def test_parse_klass(self):
        assert parse_strategy_name("klass@0.01:0.9:2:1") == "klass"

    def test_parse_kwargs_threshold(self):
        kw = parse_strategy_kwargs("dinfer_threshold@0.9")
        assert kw["threshold"] == 0.9

    def test_parse_kwargs_threshold_default(self):
        kw = parse_strategy_kwargs("dinfer_threshold")
        assert kw["threshold"] == 0.8

    def test_parse_kwargs_klass(self):
        kw = parse_strategy_kwargs("klass@0.005:0.85:3:2")
        assert kw["epsilon_kl"] == 0.005
        assert kw["tau"] == 0.85
        assert kw["n"] == 3
        assert kw["fallback_count"] == 2

    def test_parse_kwargs_credit(self):
        kw = parse_strategy_kwargs("dinfer_credit@0.7:0.8:0.6:2.0")
        assert kw["threshold"] == 0.7
        assert kw["beta"] == 0.8
        assert kw["gamma"] == 0.6
        assert kw["alpha"] == 2.0

    def test_parse_kwargs_hierarchical(self):
        kw = parse_strategy_kwargs("dinfer_hierarchical@0.9:0.5")
        assert kw["upper_threshold"] == 0.9
        assert kw["lower_threshold"] == 0.5

    def test_parse_kwargs_punt(self):
        kw = parse_strategy_kwargs("punt@0.1")
        assert kw["epsilon"] == 0.1


# ---- Test: dInfer Threshold ----

class TestDinferThreshold:
    def test_basic_unmasking(self, basic_tensors):
        """High-confidence masked tokens should be unmasked."""
        t = basic_tensors
        new_mask, out_tok, out_sc = decode_dinfer_threshold(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            threshold=0.8,
        )
        # Sample 0: positions 3 (conf=0.95) and 6 (conf=0.85) should be unmasked
        assert not new_mask[0, 3], "Position 3 (conf=0.95) should be unmasked"
        assert not new_mask[0, 6], "Position 6 (conf=0.85) should be unmasked"
        assert out_tok[0, 3] == 20
        assert out_tok[0, 6] == 23

        # Position 4 (conf=0.3) should stay masked
        assert new_mask[0, 4], "Position 4 (conf=0.3) should stay masked"

    def test_fallback_unmasks_at_least_one(self, basic_tensors):
        """When no token passes threshold, at least one should be unmasked."""
        t = basic_tensors
        new_mask, _, _ = decode_dinfer_threshold(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            threshold=0.999,  # Very high: nothing should pass
        )
        # At least one token per sample should be unmasked
        for b in range(t["B"]):
            was_masked = t["xt_neq_x0"][b] & t["non_special_sym_mask"][b]
            if was_masked.any():
                newly_unmasked = was_masked & ~new_mask[b]
                assert newly_unmasked.any(), f"Sample {b}: at least one should be unmasked"

    def test_early_stopping_safety(self, basic_tensors):
        """When nothing is masked, tokens should not change."""
        t = basic_tensors
        # Set all to unmasked
        xt_neq_x0 = torch.zeros_like(t["xt_neq_x0"])
        prev_tokens = t["output_tokens"].clone()

        new_mask, out_tok, _ = decode_dinfer_threshold(
            output_tokens=prev_tokens.clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=xt_neq_x0,
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            threshold=0.5,
        )
        assert (out_tok == prev_tokens).all(), "Tokens should not change when nothing is masked"

    def test_no_remasking(self, basic_tensors):
        """Previously unmasked tokens should never become masked again."""
        t = basic_tensors
        initially_unmasked = ~t["xt_neq_x0"] & t["non_special_sym_mask"]

        new_mask, _, _ = decode_dinfer_threshold(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            threshold=0.8,
        )
        # No previously unmasked position should become masked
        remasked = initially_unmasked & new_mask
        assert not remasked.any(), "No remasking should occur"


# ---- Test: KLASS ----

class TestKLASS:
    def test_basic_flow(self, basic_tensors, log_probs_tensors):
        """KLASS should unmask stable + confident tokens after enough history."""
        t = basic_tensors
        log_probs, V = log_probs_tensors
        klass_state = KLASSState(n=2)

        # Step 1: not enough history -> fallback
        new_mask1, out1, _ = decode_klass(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            klass_state=klass_state,
            cur_log_probs=log_probs,
            prev_log_probs=None,
            epsilon_kl=0.01,
            tau=0.5,
            fallback_count=1,
        )
        # Should unmask at least 1 via fallback
        was_masked = t["xt_neq_x0"] & t["non_special_sym_mask"]
        newly_unmasked = was_masked & ~new_mask1
        assert newly_unmasked.any()

    def test_per_sample_fallback(self, device):
        """When only some samples have ready tokens, other samples must still get fallback."""
        B, L, V = 2, 3, 10
        MASK_ID = 99
        klass_state = KLASSState(n=1)

        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        cur_tokens = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Sample 0: high confidence -> will qualify as ready
        # Sample 1: low confidence -> won't qualify, needs fallback
        cur_scores = torch.log(torch.tensor([
            [0.95, 0.3, 0.4],
            [0.2, 0.3, 0.25],
        ], device=device))

        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        # Build identical log_probs so KL is ~0 (stable)
        lp = torch.randn(B, L, V, device=device).log_softmax(dim=-1)

        # First call: seeds the history (no prev)
        decode_klass(
            output_tokens=output_tokens.clone(), output_scores=output_scores.clone(),
            cur_tokens=cur_tokens, cur_scores=cur_scores,
            xt_neq_x0=xt_neq_x0.clone(), non_special_sym_mask=non_special,
            mask_id=MASK_ID, klass_state=klass_state,
            cur_log_probs=lp, prev_log_probs=None,
            epsilon_kl=0.01, tau=0.9, fallback_count=1,
        )

        # Second call: now has history, KL ~0 since same log_probs
        new_mask, out_tok, _ = decode_klass(
            output_tokens=output_tokens.clone(), output_scores=output_scores.clone(),
            cur_tokens=cur_tokens, cur_scores=cur_scores,
            xt_neq_x0=xt_neq_x0.clone(), non_special_sym_mask=non_special,
            mask_id=MASK_ID, klass_state=klass_state,
            cur_log_probs=lp, prev_log_probs=lp,
            epsilon_kl=0.01, tau=0.9, fallback_count=1,
        )

        # Sample 0: should unmask pos 0 (conf=0.95 > tau=0.9)
        s0_unmasked = (xt_neq_x0[0] & ~new_mask[0]).nonzero(as_tuple=True)[0]
        assert len(s0_unmasked) >= 1, f"Sample 0 should have unmasked tokens, got {s0_unmasked}"

        # Sample 1: no token passes tau=0.9, but fallback should unmask top-1
        s1_unmasked = (xt_neq_x0[1] & ~new_mask[1]).nonzero(as_tuple=True)[0]
        assert len(s1_unmasked) >= 1, f"Sample 1 must get fallback unmask, got {s1_unmasked}"

    def test_kl_state_accumulation(self, device):
        """KLASSState should correctly track KL history."""
        state = KLASSState(n=2)
        B, L, V = 1, 5, 10

        lp1 = torch.randn(B, L, V, device=device).log_softmax(dim=-1)
        lp2 = lp1.clone()  # identical -> KL = 0

        assert not state.update_and_get_kl(lp1, None)  # no prev
        assert not state.update_and_get_kl(lp2, lp1)   # only 1 entry
        # Need one more for n=2
        lp3 = lp1.clone()
        assert state.update_and_get_kl(lp3, lp2)       # now 2 entries

        stable = state.is_stable(epsilon_kl=0.01)
        assert stable is not None
        # Since lp1==lp2==lp3, KL should be ~0 everywhere
        assert stable.all(), "Identical distributions should be stable"

    def test_no_remasking(self, basic_tensors, log_probs_tensors):
        """KLASS should never re-mask tokens."""
        t = basic_tensors
        log_probs, _ = log_probs_tensors
        klass_state = KLASSState(n=1)

        initially_unmasked = ~t["xt_neq_x0"] & t["non_special_sym_mask"]

        new_mask, _, _ = decode_klass(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            klass_state=klass_state,
            cur_log_probs=log_probs,
            prev_log_probs=None,
            epsilon_kl=0.01,
            tau=0.5,
            fallback_count=1,
        )
        remasked = initially_unmasked & new_mask
        assert not remasked.any()


# ---- Test: dInfer Credit ----

class TestDinferCredit:
    def test_basic_unmasking(self, basic_tensors, log_probs_tensors):
        """Credit decoding should unmask tokens."""
        t = basic_tensors
        log_probs, V = log_probs_tensors
        B, L = t["B"], t["L"]
        credit_state = CreditState(B, L, V, t["output_tokens"].device)

        new_mask, out_tok, _ = decode_dinfer_credit(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            credit_state=credit_state,
            cur_log_probs=log_probs,
            threshold=0.5,
        )
        was_masked = t["xt_neq_x0"] & t["non_special_sym_mask"]
        newly_unmasked = was_masked & ~new_mask
        assert newly_unmasked.any()

    def test_credit_accumulation_full_matrix(self, device):
        """Credits should accumulate in the full [B,L,V] matrix and change argmax."""
        B, L, V = 1, 5, 10
        state = CreditState(B, L, V, device, beta=0.8, gamma=0.2)

        masked = torch.ones(B, L, dtype=torch.bool, device=device)

        # Step 1: token 3 is top
        logits = torch.zeros(B, L, V, device=device)
        logits[:, :, 3] = 5.0
        fused1 = state.update_and_fuse(logits, masked, alpha=0.7)

        # Credit for token 3 should be non-zero
        assert (state.credit_mat[:, :, 3] > 0).all()
        # Other tokens should have zero credit
        assert (state.credit_mat[:, :, 0] == 0).all()

    def test_credit_can_change_argmax(self, device):
        """Credit fusion should be able to change which token is top (the key fix)."""
        B, L, V = 1, 1, 3
        state = CreditState(B, L, V, device, beta=0.99, gamma=0.5)
        masked = torch.ones(B, L, dtype=torch.bool, device=device)

        # Accumulate credit heavily for token 0 over many steps
        logits_tok0 = torch.zeros(B, L, V, device=device)
        logits_tok0[:, :, 0] = 2.0  # token 0 is top
        for _ in range(20):
            state.update_and_fuse(logits_tok0, masked, alpha=5.0)

        # Now token 1 has slightly higher raw logit than token 0
        logits_tok1_top = torch.zeros(B, L, V, device=device)
        logits_tok1_top[:, :, 0] = 1.9
        logits_tok1_top[:, :, 1] = 2.0  # token 1 is now raw top
        fused = state.update_and_fuse(logits_tok1_top, masked, alpha=5.0)

        # After fusion, token 0 should regain the top because of accumulated credit
        fused_top = fused.argmax(dim=-1)
        assert fused_top[0, 0].item() == 0, (
            f"Credit should flip argmax back to token 0, got {fused_top[0, 0].item()}"
        )

    def test_no_remasking(self, basic_tensors, log_probs_tensors):
        t = basic_tensors
        log_probs, V = log_probs_tensors
        credit_state = CreditState(t["B"], t["L"], V, t["output_tokens"].device)
        initially_unmasked = ~t["xt_neq_x0"] & t["non_special_sym_mask"]

        new_mask, _, _ = decode_dinfer_credit(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            credit_state=credit_state,
            cur_log_probs=log_probs,
            threshold=0.5,
        )
        assert not (initially_unmasked & new_mask).any()


# ---- Test: dInfer Hierarchical ----

class TestDinferHierarchical:
    def test_basic_unmasking(self, basic_tensors):
        t = basic_tensors
        new_mask, out_tok, _ = decode_dinfer_hierarchical(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            upper_threshold=0.5,
            lower_threshold=0.3,
        )
        was_masked = t["xt_neq_x0"] & t["non_special_sym_mask"]
        newly_unmasked = was_masked & ~new_mask
        assert newly_unmasked.any(), "Hierarchical should unmask some tokens"

    def test_segment_max_selection(self, device):
        """Reference behavior: pick max-confidence per segment, filter by low_threshold,
        OR-in positions above upper_threshold, always keep global top-1."""
        B, L = 1, 10
        MASK_ID = 32

        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        cur_tokens = torch.arange(L, device=device).unsqueeze(0)
        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        # Two segments: [0..4] and [6..9], position 5 is unmasked
        xt_neq_x0[0, 5] = False
        output_tokens[0, 5] = 55  # already unmasked
        non_special[0, 5] = True

        # Confidence: segment [0..4] max at pos 2, segment [6..9] max at pos 8
        conf_vals = [0.3, 0.4, 0.7, 0.5, 0.2, 0.0, 0.1, 0.3, 0.6, 0.15]
        cur_scores = torch.log(torch.tensor([conf_vals], device=device))

        new_mask, out_tok, _ = decode_dinfer_hierarchical(
            output_tokens=output_tokens.clone(),
            output_scores=output_scores.clone(),
            cur_tokens=cur_tokens,
            cur_scores=cur_scores,
            xt_neq_x0=xt_neq_x0.clone(),
            non_special_sym_mask=non_special,
            mask_id=MASK_ID,
            upper_threshold=0.92,  # nothing above this
            lower_threshold=0.5,
        )

        unmasked = (xt_neq_x0[0] & ~new_mask[0]).nonzero(as_tuple=True)[0].tolist()
        # Segment [0..4] max is pos 2 (conf 0.7 > low_threshold 0.5) -> unmasked
        assert 2 in unmasked, f"Segment max pos 2 should be unmasked, got {unmasked}"
        # Segment [6..9] max is pos 8 (conf 0.6 > low_threshold 0.5) -> unmasked
        assert 8 in unmasked, f"Segment max pos 8 should be unmasked, got {unmasked}"

    def test_uniform_conf_single_span(self, device):
        """Reproducer from the bug report: length-7 span, all conf=0.5,
        upper=0.9, lower=0.1. Reference picks 1 segment-max + global top-1."""
        B, L = 1, 7
        MASK_ID = 32

        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        cur_tokens = torch.arange(L, device=device).unsqueeze(0)
        cur_scores = torch.log(torch.full((B, L), 0.5, device=device))
        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        new_mask, _, _ = decode_dinfer_hierarchical(
            output_tokens=output_tokens.clone(),
            output_scores=output_scores.clone(),
            cur_tokens=cur_tokens,
            cur_scores=cur_scores,
            xt_neq_x0=xt_neq_x0.clone(),
            non_special_sym_mask=non_special,
            mask_id=MASK_ID,
            upper_threshold=0.9,
            lower_threshold=0.1,
        )

        unmasked = (xt_neq_x0[0] & ~new_mask[0]).nonzero(as_tuple=True)[0].tolist()
        # Single segment -> 1 segment-max. 0.5 > low_threshold=0.1 so it passes.
        # Global top-1 is the same position. upper_threshold=0.9 > 0.5 so nothing extra.
        # Result: exactly 1 position (NOT 4 like the old recursive implementation)
        assert len(unmasked) == 1, (
            f"Single span, uniform conf: should unmask exactly 1 (segment-max = global top-1), "
            f"got {len(unmasked)}: {unmasked}"
        )

    def test_find_contiguous_spans(self, device):
        indices = torch.tensor([2, 3, 4, 7, 8, 12])
        spans = _find_contiguous_spans(indices)
        assert spans == [(2, 4), (7, 8), (12, 12)]

    def test_no_remasking(self, basic_tensors):
        t = basic_tensors
        initially_unmasked = ~t["xt_neq_x0"] & t["non_special_sym_mask"]
        new_mask, _, _ = decode_dinfer_hierarchical(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
        )
        assert not (initially_unmasked & new_mask).any()

    def test_early_stopping_safety(self, basic_tensors):
        t = basic_tensors
        xt_neq_x0 = torch.zeros_like(t["xt_neq_x0"])
        prev_tokens = t["output_tokens"].clone()
        new_mask, out_tok, _ = decode_dinfer_hierarchical(
            output_tokens=prev_tokens.clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=xt_neq_x0,
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
        )
        assert (out_tok == prev_tokens).all()


# ---- Test: PUNT ----

class TestPUNT:
    def test_basic_unmasking(self, basic_tensors, log_probs_tensors):
        t = basic_tensors
        log_probs, V = log_probs_tensors

        def mock_forward_fn(input_ids):
            # Return post-processed log-probs (simulates forward_decoder pipeline)
            B, L = input_ids.shape
            return {"logits": torch.randn(B, L, V).log_softmax(dim=-1)}

        new_mask, out_tok, _ = decode_punt(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            baseline_log_probs=log_probs,
            forward_fn=mock_forward_fn,
            epsilon=0.04,
        )
        was_masked = t["xt_neq_x0"] & t["non_special_sym_mask"]
        newly_unmasked = was_masked & ~new_mask
        assert newly_unmasked.any(), "PUNT should unmask at least some tokens"

    def test_independent_tokens_pass(self, device):
        """When forward_fn returns same distribution regardless of unmasked tokens,
        all tokens should be considered independent and unmasked."""
        B, L, V = 1, 5, 10
        MASK_ID = 32

        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        cur_tokens = torch.arange(L, device=device).unsqueeze(0)
        cur_scores = torch.log(torch.full((B, L), 0.9, device=device))
        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        # Baseline log-probs (already post-processed)
        baseline = torch.randn(B, L, V, device=device).log_softmax(dim=-1)

        # forward_fn always returns the SAME post-processed distribution
        def constant_forward_fn(input_ids):
            return {"logits": baseline.clone()}

        new_mask, _, _ = decode_punt(
            output_tokens=output_tokens.clone(),
            output_scores=output_scores.clone(),
            cur_tokens=cur_tokens,
            cur_scores=cur_scores,
            xt_neq_x0=xt_neq_x0.clone(),
            non_special_sym_mask=non_special,
            mask_id=MASK_ID,
            baseline_log_probs=baseline,
            forward_fn=constant_forward_fn,
            epsilon=0.04,
        )
        # All tokens should be unmasked since distributions don't change
        assert not new_mask.any(), "All tokens should be unmasked when independent"

    def test_dependent_tokens_rejected(self, device):
        """When forward_fn returns very different distribution after unmasking anchors,
        dependent tokens should be rejected (stay masked)."""
        B, L, V = 1, 4, 10
        MASK_ID = 32

        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        cur_tokens = torch.arange(L, device=device).unsqueeze(0)
        cur_scores = torch.log(torch.full((B, L), 0.9, device=device))
        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        baseline = torch.zeros(B, L, V, device=device)
        baseline[:, :, 0] = 10.0  # peaked at token 0
        baseline = baseline.log_softmax(dim=-1)

        def shifting_forward_fn(input_ids):
            # When any anchor is unmasked, shift distribution drastically
            shifted = torch.zeros(1, L, V, device=device)
            shifted[:, :, V - 1] = 10.0  # peaked at last token
            return {"logits": shifted.log_softmax(dim=-1)}

        new_mask, _, _ = decode_punt(
            output_tokens=output_tokens.clone(),
            output_scores=output_scores.clone(),
            cur_tokens=cur_tokens,
            cur_scores=cur_scores,
            xt_neq_x0=xt_neq_x0.clone(),
            non_special_sym_mask=non_special,
            mask_id=MASK_ID,
            baseline_log_probs=baseline,
            forward_fn=shifting_forward_fn,
            epsilon=0.04,
        )
        # Most tokens should be rejected due to large KL
        # But at least 1 always gets unmasked (fallback)
        n_unmasked = (~new_mask).sum().item()
        n_still_masked = new_mask.sum().item()
        assert n_still_masked > 0, "Dependent tokens should stay masked"
        assert n_unmasked >= 1, "Fallback should unmask at least 1"

    def test_no_remasking(self, basic_tensors, log_probs_tensors):
        t = basic_tensors
        log_probs, V = log_probs_tensors
        initially_unmasked = ~t["xt_neq_x0"] & t["non_special_sym_mask"]

        def mock_forward_fn(input_ids):
            B, L = input_ids.shape
            return {"logits": torch.randn(B, L, V).log_softmax(dim=-1)}

        new_mask, _, _ = decode_punt(
            output_tokens=t["output_tokens"].clone(),
            output_scores=t["output_scores"].clone(),
            cur_tokens=t["cur_tokens"],
            cur_scores=t["cur_scores"],
            xt_neq_x0=t["xt_neq_x0"].clone(),
            non_special_sym_mask=t["non_special_sym_mask"],
            mask_id=t["mask_id"],
            baseline_log_probs=log_probs,
            forward_fn=mock_forward_fn,
            epsilon=0.04,
        )
        assert not (initially_unmasked & new_mask).any()


# ---- Test: helpers ----

class TestHelpers:
    def test_fallback_topk(self, device):
        B, L = 2, 10
        conf = torch.tensor([
            [0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.0],
            [0.5, 0.3, 0.9, 0.1, 0.4, 0.8, 0.2, 0.7, 0.6, 0.0],
        ], device=device)
        masked = torch.ones(B, L, dtype=torch.bool, device=device)
        conf_masked = conf.masked_fill(~masked, -1.0)

        result = _fallback_topk(conf_masked, masked, k=2)
        # Each sample should have exactly 2 positions marked
        assert result[0].sum() == 2
        assert result[1].sum() == 2
        # Should be the top-2 positions
        assert result[0, 1]  # conf=0.9
        assert result[0, 4]  # conf=0.8


# ---- Test: multi-step simulation ----

class TestMultiStep:
    def test_threshold_converges(self, device):
        """Run multiple steps of threshold decoding and verify all tokens get unmasked."""
        B, L = 1, 20
        MASK_ID = 99
        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        for step in range(100):
            if not xt_neq_x0.any():
                break

            # Simulate model predictions with random confidence
            cur_tokens = torch.randint(0, 30, (B, L), device=device)
            cur_scores = torch.log(torch.rand(B, L, device=device) * 0.5 + 0.5)

            xt_neq_x0, output_tokens, output_scores = decode_dinfer_threshold(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0,
                non_special_sym_mask=non_special,
                mask_id=MASK_ID,
                threshold=0.8,
            )

        assert not xt_neq_x0.any(), f"All tokens should be unmasked, but {xt_neq_x0.sum()} remain"
        assert (output_tokens != MASK_ID).all()

    def test_hierarchical_converges(self, device):
        """Hierarchical should also converge to full unmasking."""
        B, L = 1, 20
        MASK_ID = 99
        output_tokens = torch.full((B, L), MASK_ID, device=device)
        output_scores = torch.full((B, L), -math.inf, device=device)
        xt_neq_x0 = torch.ones(B, L, dtype=torch.bool, device=device)
        non_special = torch.ones(B, L, dtype=torch.bool, device=device)

        for step in range(100):
            if not xt_neq_x0.any():
                break

            cur_tokens = torch.randint(0, 30, (B, L), device=device)
            cur_scores = torch.log(torch.rand(B, L, device=device) * 0.5 + 0.5)

            xt_neq_x0, output_tokens, output_scores = decode_dinfer_hierarchical(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0,
                non_special_sym_mask=non_special,
                mask_id=MASK_ID,
                upper_threshold=0.7,
                lower_threshold=0.3,
            )

        assert not xt_neq_x0.any(), f"All tokens should be unmasked, but {xt_neq_x0.sum()} remain"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
