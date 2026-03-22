"""
Decoding strategies for masked diffusion protein language models.

Implements strategies from:
- dInfer (Ma et al., 2025): threshold, hierarchical, credit decoding
- KLASS (Kim et al., 2025): KL-Adaptive Stability Sampling
- PUNT (Azangulov et al., 2025): Parallel Unmasking with Non-influence Tests
- LRD (Anonymous, 2026): Latent Refinement Decoding
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Strategy 1: dInfer Threshold Decoding
# ---------------------------------------------------------------------------

def decode_dinfer_threshold(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    xt_neq_x0,
    non_special_sym_mask,
    mask_id,
    threshold=0.8,
):
    """dInfer threshold decoding: unmask tokens whose confidence exceeds threshold.

    Args:
        output_tokens: [B, L] previous output tokens
        output_scores: [B, L] previous output scores
        cur_tokens: [B, L] currently predicted tokens (from forward_decoder)
        cur_scores: [B, L] currently predicted scores (log-prob of predicted token)
        xt_neq_x0: [B, L] bool, True = position is still masked
        non_special_sym_mask: [B, L] bool, True = valid (non-special) position
        mask_id: int, the mask token id
        threshold: float, confidence threshold (0-1) for unmasking

    Returns:
        (new_xt_neq_x0, output_tokens, output_scores)
    """
    masked_positions = xt_neq_x0 & non_special_sym_mask

    # If nothing is masked, return as-is (early stopping safety)
    if not masked_positions.any():
        return xt_neq_x0, output_tokens, output_scores

    # cur_scores are log-probs; convert to probabilities for confidence
    confidence = cur_scores.exp()  # [B, L]

    # Only consider masked & valid positions
    confidence_masked = confidence.masked_fill(~masked_positions, -1.0)

    # Positions that pass the threshold
    to_unmask = masked_positions & (confidence_masked >= threshold)

    # Fallback: if no position passes for a sample, unmask top-1 per sample
    for b in range(to_unmask.size(0)):
        if masked_positions[b].any() and not to_unmask[b].any():
            top_idx = confidence_masked[b].argmax()
            to_unmask[b, top_idx] = True

    # Apply unmasking: update tokens and scores at to_unmask positions
    output_tokens[to_unmask] = cur_tokens[to_unmask]
    output_scores[to_unmask] = cur_scores[to_unmask]

    # Update mask state: unmasked positions become False
    new_xt_neq_x0 = xt_neq_x0.clone()
    new_xt_neq_x0[to_unmask] = False

    return new_xt_neq_x0, output_tokens, output_scores


# ---------------------------------------------------------------------------
# Strategy 2: KLASS (KL-Adaptive Stability Sampling)
# ---------------------------------------------------------------------------

class KLASSState:
    """Maintains per-token probability history for KLASS decoding."""

    def __init__(self, n=2):
        self.n = n  # history window length
        self.kl_buffer = []  # list of [B, L] KL values

    def update_and_get_kl(self, cur_log_probs, prev_log_probs,
                          valid_vocab_mask=None):
        """Compute KL divergence and update history buffer.

        Args:
            cur_log_probs: [B, L, V] current log-probabilities
            prev_log_probs: [B, L, V] previous log-probabilities (or None)
            valid_vocab_mask: [V] bool, which vocab indices to include in KL

        Returns:
            bool: True if we have enough history to make decisions
        """
        if prev_log_probs is None:
            return False

        cur_lp = cur_log_probs
        prev_lp = prev_log_probs
        if valid_vocab_mask is not None:
            cur_lp = cur_lp[..., valid_vocab_mask]
            prev_lp = prev_lp[..., valid_vocab_mask]
            # Re-normalize after slicing to valid vocab
            cur_lp = cur_lp.log_softmax(dim=-1)
            prev_lp = prev_lp.log_softmax(dim=-1)

        # KL(cur || prev) = sum_v cur(v) * (log cur(v) - log prev(v))
        cur_probs = cur_lp.exp()
        # import ipdb; ipdb.set_trace()
        # kl = (cur_probs * (cur_lp - prev_lp)) if cur_probs > 0 else 0, sum over vocab
        kl = torch.where(
            cur_probs > 0,
            cur_probs * (cur_lp - prev_lp),
            torch.zeros_like(cur_probs),
        ).sum(dim=-1)  # [B, L]

        kl = kl.clamp(min=0.0)  # numerical safety

        self.kl_buffer.append(kl)
        if len(self.kl_buffer) > self.n:
            self.kl_buffer.pop(0)

        return len(self.kl_buffer) >= self.n

    def is_stable(self, epsilon_kl):
        """Check if all recent KL values are below threshold.

        Returns:
            [B, L] bool tensor, True = stable
        """
        if len(self.kl_buffer) < self.n:
            return None
        stacked = torch.stack(self.kl_buffer[-self.n:], dim=0)  # [n, B, L]
        return stacked.lt(epsilon_kl).all(dim=0)  # [B, L]

    def reset(self):
        self.kl_buffer = []


def decode_klass(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    xt_neq_x0,
    non_special_sym_mask,
    mask_id,
    klass_state,
    cur_log_probs,
    prev_log_probs,
    epsilon_kl=0.01,
    tau=0.9,
    fallback_count=1,
    valid_vocab_mask=None,
):
    """KLASS: unmask tokens that are both confident and KL-stable.

    Args:
        output_tokens, output_scores, cur_tokens, cur_scores, xt_neq_x0,
        non_special_sym_mask, mask_id: same as threshold strategy
        klass_state: KLASSState object tracking history
        cur_log_probs: [B, L, V] current log-softmax output
        prev_log_probs: [B, L, V] previous step's log-softmax (or None)
        epsilon_kl: KL threshold for stability
        tau: confidence threshold
        fallback_count: how many tokens to unmask when no token qualifies

    Returns:
        (new_xt_neq_x0, output_tokens, output_scores)
    """
    masked_positions = xt_neq_x0 & non_special_sym_mask

    if not masked_positions.any():
        return xt_neq_x0, output_tokens, output_scores

    confidence = cur_scores.exp()  # [B, L]
    confidence_masked = confidence.masked_fill(~masked_positions, -1.0)

    has_enough_history = klass_state.update_and_get_kl(
        cur_log_probs, prev_log_probs, valid_vocab_mask
    )

    if has_enough_history:
        stable_kl = klass_state.is_stable(epsilon_kl)  # [B, L]
        high_conf = confidence_masked >= tau
        ready = stable_kl & high_conf & masked_positions

        # Per-sample fallback: samples where no token qualifies get top-k fallback
        to_unmask = ready.clone()
        needs_fallback = masked_positions.any(dim=-1) & ~ready.any(dim=-1)  # [B]
        if needs_fallback.any():
            fallback = _fallback_topk(confidence_masked, masked_positions, fallback_count)
            to_unmask[needs_fallback] = fallback[needs_fallback]
    else:
        # Not enough history: use fallback
        to_unmask = _fallback_topk(confidence_masked, masked_positions, fallback_count)

    output_tokens[to_unmask] = cur_tokens[to_unmask]
    output_scores[to_unmask] = cur_scores[to_unmask]

    new_xt_neq_x0 = xt_neq_x0.clone()
    new_xt_neq_x0[to_unmask] = False

    return new_xt_neq_x0, output_tokens, output_scores


# ---------------------------------------------------------------------------
# Strategy 3: dInfer Credit Decoding
# ---------------------------------------------------------------------------

class CreditState:
    """Maintains full [B, L, V] credit matrix for credit decoding.

    Matches the reference dInfer CreditThresholdParallelDecoder:
    - EMA decay on the full credit matrix each step
    - scatter_add top-1 enhanced probability onto the matrix
    - Fuse: fused_logits = logits + alpha * log(credit_mat + 1)
    """

    def __init__(self, B, L, V, device, beta=0.8, gamma=0.2):
        self.beta = beta
        self.gamma = gamma
        self.credit_mat = torch.zeros(B, L, V, dtype=torch.float32, device=device)
        self.iter = 0

    def update_and_fuse(self, logits, masked_positions, alpha=0.7,
                        valid_vocab_mask=None):
        """Update credit matrix and return fused logits.

        Follows reference: _apply_credit_fusion()
          1. Decay: mat *= beta  (skip on iter 0)
          2. Compute top-1 prob^gamma on masked positions
          3. scatter_add onto mat at top-1 indices
          4. fused_logits = logits + alpha * log(mat + 1)

        Args:
            logits: [B, L, V] raw logits (NOT log-softmax)
            masked_positions: [B, L] bool
            alpha: fusion weight
            valid_vocab_mask: [V] bool, restrict softmax/argmax to valid vocab

        Returns:
            fused_logits: [B, L, V]
        """
        if self.iter > 0:
            self.credit_mat.mul_(self.beta)

        # Mask invalid vocab before softmax so they don't affect probs
        working_logits = logits.float()
        if valid_vocab_mask is not None:
            working_logits = working_logits.masked_fill(
                ~valid_vocab_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )

        probs = F.softmax(working_logits, dim=-1)
        top1_probs, top1_idx = probs.max(dim=-1)  # [B, L]
        enhanced = top1_probs.pow(self.gamma).to(self.credit_mat.dtype)
        update_vals = enhanced * masked_positions.float()
        self.credit_mat.scatter_add_(
            2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1)
        )

        fused_logits = logits + alpha * torch.log(self.credit_mat + 1)
        self.iter += 1
        return fused_logits


def decode_dinfer_credit(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    xt_neq_x0,
    non_special_sym_mask,
    mask_id,
    credit_state,
    cur_log_probs,
    threshold=0.8,
    alpha=0.7,
    valid_vocab_mask=None,
):
    """dInfer credit decoding: fuse accumulated credit into logits, then threshold.

    Matches reference CreditThresholdParallelDecoder:
    1. Update credit matrix (EMA decay + scatter_add top-1 boost)
    2. Fuse: fused_logits = logits + alpha * log(credit_mat + 1)
    3. Compute confidence from fused logits
    4. Unmask where confidence > threshold
    5. Re-derive committed tokens from fused_logits argmax

    Args:
        output_tokens, output_scores, cur_tokens, cur_scores, xt_neq_x0,
        non_special_sym_mask, mask_id: same as threshold strategy
        credit_state: CreditState object (with full [B,L,V] matrix)
        cur_log_probs: [B, L, V] current log-softmax output
        threshold: confidence threshold for unmasking
        alpha: credit fusion weight

    Returns:
        (new_xt_neq_x0, output_tokens, output_scores)
    """
    masked_positions = xt_neq_x0 & non_special_sym_mask

    if not masked_positions.any():
        return xt_neq_x0, output_tokens, output_scores

    # Convert log-softmax back to raw logits for credit fusion
    # (credit fusion adds to raw logits, not log-probs)
    raw_logits = cur_log_probs  # close enough: log_softmax ~ logits + const

    # Update credit matrix and get fused logits
    fused_logits = credit_state.update_and_fuse(
        raw_logits, masked_positions, alpha, valid_vocab_mask
    )

    # Derive tokens and confidence from fused logits (mask invalid vocab)
    working_fused = fused_logits.float()
    if valid_vocab_mask is not None:
        working_fused = working_fused.masked_fill(
            ~valid_vocab_mask.unsqueeze(0).unsqueeze(0), float('-inf')
        )
    fused_probs = F.softmax(working_fused, dim=-1)
    fused_conf, fused_tokens = fused_probs.max(dim=-1)  # [B, L]
    fused_scores = fused_probs.gather(-1, fused_tokens.unsqueeze(-1)).squeeze(-1).log()

    fused_conf_masked = fused_conf.masked_fill(~masked_positions, -1.0)

    # Threshold decode on fused confidence
    to_unmask = masked_positions & (fused_conf_masked >= threshold)

    # Fallback: at least one per sample
    for b in range(to_unmask.size(0)):
        if masked_positions[b].any() and not to_unmask[b].any():
            top_idx = fused_conf_masked[b].argmax()
            to_unmask[b, top_idx] = True

    # Use fused-logits-derived tokens (credit can change the argmax!)
    output_tokens[to_unmask] = fused_tokens[to_unmask]
    output_scores[to_unmask] = fused_scores[to_unmask]

    new_xt_neq_x0 = xt_neq_x0.clone()
    new_xt_neq_x0[to_unmask] = False

    return new_xt_neq_x0, output_tokens, output_scores


# ---------------------------------------------------------------------------
# Strategy 4: dInfer Hierarchical Decoding
# ---------------------------------------------------------------------------

def decode_dinfer_hierarchical(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    xt_neq_x0,
    non_special_sym_mask,
    mask_id,
    upper_threshold=0.92,
    lower_threshold=0.62,
):
    """dInfer hierarchical decoding matching the reference HierarchyDecoder.

    Algorithm (per sample):
    1. Find contiguous masked segments
    2. For each segment, select the max-confidence position
    3. Filter out segment-max positions below low_threshold
    4. OR-in all positions above upper_threshold
    5. Always guarantee the global top-1 masked position is unmasked

    Args:
        output_tokens, output_scores, cur_tokens, cur_scores, xt_neq_x0,
        non_special_sym_mask, mask_id: same as threshold strategy
        upper_threshold: positions above this confidence are always unmasked
        lower_threshold: segment-max positions below this are filtered out

    Returns:
        (new_xt_neq_x0, output_tokens, output_scores)
    """
    masked_positions = xt_neq_x0 & non_special_sym_mask

    if not masked_positions.any():
        return xt_neq_x0, output_tokens, output_scores

    confidence = cur_scores.exp()  # [B, L]
    B, L = xt_neq_x0.shape
    neg_inf = torch.finfo(confidence.dtype).min

    # confidence only on masked positions
    conf_masked = torch.where(masked_positions, confidence, neg_inf)

    to_unmask = torch.zeros_like(xt_neq_x0)

    for b in range(B):
        mask_b = masked_positions[b]
        if not mask_b.any():
            continue
        conf_b = conf_masked[b]

        # --- Step 1-2: find segments and their max-confidence positions ---
        mask_int = mask_b.int()
        # Detect segment starts and ends via diff
        padded = torch.cat([mask_int[:1] * 0, mask_int, mask_int[-1:] * 0])
        diff = torch.diff(padded)
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]

        # For each segment, pick the max-confidence position
        seg_max_indices = []
        if len(starts) > 0:
            for s, e in zip(starts.tolist(), ends.tolist()):
                seg_conf = conf_b[s:e]
                local_max = seg_conf.argmax().item()
                seg_max_indices.append(s + local_max)
            to_unmask[b, seg_max_indices] = True

        # --- Step 3: filter segment-maxes below low_threshold ---
        if lower_threshold is not None:
            to_unmask[b] = to_unmask[b] & (conf_b > lower_threshold)

        # --- Step 4: OR-in all positions above upper_threshold ---
        if upper_threshold is not None:
            to_unmask[b] = to_unmask[b] | (conf_b > upper_threshold)

        # --- Step 5: always guarantee global top-1 ---
        top1_idx = conf_b.argmax()
        to_unmask[b, top1_idx] = True

    output_tokens[to_unmask] = cur_tokens[to_unmask]
    output_scores[to_unmask] = cur_scores[to_unmask]

    new_xt_neq_x0 = xt_neq_x0.clone()
    new_xt_neq_x0[to_unmask] = False

    return new_xt_neq_x0, output_tokens, output_scores


def _find_contiguous_spans(indices):
    """Find contiguous spans in sorted index tensor.

    Args:
        indices: 1D sorted tensor of indices

    Returns:
        list of (start, end) tuples
    """
    if len(indices) == 0:
        return []
    spans = []
    start = indices[0].item()
    prev = start
    for i in range(1, len(indices)):
        cur = indices[i].item()
        if cur != prev + 1:
            spans.append((start, prev))
            start = cur
        prev = cur
    spans.append((start, prev))
    return spans


# ---------------------------------------------------------------------------
# Strategy 5: PUNT (Parallel Unmasking with Non-influence Tests)
# ---------------------------------------------------------------------------

def decode_punt(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    xt_neq_x0,
    non_special_sym_mask,
    mask_id,
    baseline_log_probs,
    forward_fn,
    epsilon=0.04,
    valid_vocab_mask=None,
):
    """PUNT: identify conditionally independent tokens via binary testing.

    Uses O(log|M|) extra forward passes per step to test which tokens
    can be safely unmasked in parallel.

    Args:
        output_tokens, output_scores, cur_tokens, cur_scores, xt_neq_x0,
        non_special_sym_mask, mask_id: same as threshold strategy
        baseline_log_probs: [B, L, V] post-processed log-probs (baseline)
        forward_fn: callable(input_ids) -> {"logits": [B, L, V]} post-processed
            log-probs. Must apply the same post-processing as forward_decoder
            (modality masking, special-token masking, top-p filtering, log_softmax).
        epsilon: KL divergence threshold for independence test

    Returns:
        (new_xt_neq_x0, output_tokens, output_scores)
    """
    masked_positions = xt_neq_x0 & non_special_sym_mask

    if not masked_positions.any():
        return xt_neq_x0, output_tokens, output_scores

    confidence = cur_scores.exp()  # [B, L]
    B, L = output_tokens.shape
    to_unmask = torch.zeros_like(xt_neq_x0)

    # Process each sample in the batch independently
    for b in range(B):
        masked_idx = masked_positions[b].nonzero(as_tuple=True)[0]
        if len(masked_idx) == 0:
            continue

        M = len(masked_idx)
        if M == 1:
            to_unmask[b, masked_idx[0]] = True
            continue

        # Sort by confidence (descending) - highest confidence = rank 0
        conf_vals = confidence[b, masked_idx]
        sorted_order = conf_vals.argsort(descending=True)
        sorted_masked_idx = masked_idx[sorted_order]

        # Initialize R = all masked indices (by rank)
        R_mask = torch.ones(M, dtype=torch.bool, device=output_tokens.device)

        num_rounds = math.ceil(math.log2(M)) if M > 1 else 1

        for bit in range(num_rounds):
            # Partition by b-th bit of binary encoding of rank
            ranks = torch.arange(M, device=output_tokens.device)
            anchor_bits = ((ranks >> bit) & 1) == 0  # S0: bit is 0
            test_bits = ~anchor_bits  # S1: bit is 1

            # Only consider positions still in R
            S0_in_R = anchor_bits & R_mask
            S1_in_R = test_bits & R_mask

            if not S1_in_R.any() or not S0_in_R.any():
                continue

            # Create modified sequence: unmask S0 positions with candidates
            test_input = output_tokens[b:b+1].clone()  # [1, L]
            for rank_idx in S0_in_R.nonzero(as_tuple=True)[0]:
                pos = sorted_masked_idx[rank_idx]
                test_input[0, pos] = cur_tokens[b, pos]

            # Forward pass (forward_fn returns post-processed log-probs)
            with torch.no_grad():
                test_out = forward_fn(test_input)
            test_log_probs = test_out["logits"]  # [1, L, V] already post-processed

            # Check independence for S1 positions
            for rank_idx in S1_in_R.nonzero(as_tuple=True)[0]:
                pos = sorted_masked_idx[rank_idx]
                # KL(baseline || test) restricted to valid vocab
                base_lp = baseline_log_probs[b, pos]
                test_lp = test_log_probs[0, pos]
                if valid_vocab_mask is not None:
                    base_lp = base_lp[valid_vocab_mask]
                    test_lp = test_lp[valid_vocab_mask]
                    # Re-normalize to proper distributions
                    base_lp = base_lp.log_softmax(dim=-1)
                    test_lp = test_lp.log_softmax(dim=-1)
                base_p = base_lp.exp()
                kl = (base_p * (base_lp - test_lp))[base_p > 0].sum()
                kl = kl.clamp(min=0.0)
                # import ipdb; ipdb.set_trace()
                if kl.item() > epsilon:
                    R_mask[rank_idx] = False

        # Unmask surviving set
        for rank_idx in R_mask.nonzero(as_tuple=True)[0]:
            pos = sorted_masked_idx[rank_idx]
            to_unmask[b, pos] = True

    # Ensure at least one token per sample
    for b in range(B):
        if masked_positions[b].any() and not to_unmask[b].any():
            conf_masked = confidence[b].masked_fill(~masked_positions[b], -1.0)
            top_idx = conf_masked.argmax()
            to_unmask[b, top_idx] = True

    output_tokens[to_unmask] = cur_tokens[to_unmask]
    output_scores[to_unmask] = cur_scores[to_unmask]

    new_xt_neq_x0 = xt_neq_x0.clone()
    new_xt_neq_x0[to_unmask] = False

    return new_xt_neq_x0, output_tokens, output_scores


# ---------------------------------------------------------------------------
# Strategy 6: LRD (Latent Refinement Decoding) - Phase 2 Predictive Feedback
# ---------------------------------------------------------------------------

class LRDState:
    """Tracks KL divergence for LRD's early stopping criterion.

    Monitors mean KL(p_t || p_{t-1}) per sample across masked positions.
    When a sample's mean KL drops below tau_decode, that sample is converged.
    """

    def __init__(self):
        self.prev_log_probs = None  # [B, L, V] from previous step
        self.converged = None       # [B] bool tensor, per-sample convergence

    def update(self, cur_log_probs, masked_positions,
               tau_decode=0.1, valid_vocab_mask=None):
        """Update KL tracking and check convergence per sample.

        Args:
            cur_log_probs: [B, L, V] current step log-probabilities
            masked_positions: [B, L] bool, currently masked positions
            tau_decode: KL threshold for early stopping
            valid_vocab_mask: [V] bool, restrict KL to valid vocab

        Returns:
            converged: [B] bool tensor
        """
        B = cur_log_probs.shape[0]
        device = cur_log_probs.device

        if self.prev_log_probs is None:
            self.prev_log_probs = cur_log_probs.detach().clone()
            self.converged = torch.zeros(B, dtype=torch.bool, device=device)
            return self.converged

        cur_lp = cur_log_probs
        prev_lp = self.prev_log_probs
        if valid_vocab_mask is not None:
            cur_lp = cur_lp[..., valid_vocab_mask].log_softmax(dim=-1)
            prev_lp = prev_lp[..., valid_vocab_mask].log_softmax(dim=-1)

        # KL(cur || prev) per position
        cur_p = cur_lp.exp()
        kl_per_pos = torch.where(cur_p > 0, cur_p * (cur_lp - prev_lp), torch.zeros_like(cur_p)).sum(dim=-1)  # [B, L]
        kl_per_pos = kl_per_pos.clamp(min=0.0)

        # Mean KL per sample over masked positions only
        kl_masked = kl_per_pos * masked_positions.float()            # [B, L]
        n_masked = masked_positions.float().sum(dim=-1).clamp(min=1.0)  # [B]
        mean_kl_per_sample = kl_masked.sum(dim=-1) / n_masked        # [B]

        # Samples with no masked positions are considered converged
        no_masked = ~masked_positions.any(dim=-1)                     # [B]
        self.converged = (mean_kl_per_sample < tau_decode) | no_masked

        self.prev_log_probs = cur_log_probs.detach().clone()
        # if (mean_kl_per_sample < tau_decode).sum() > 0:
        #     import ipdb; ipdb.set_trace()
        return self.converged


def decode_lrd(
    output_tokens,
    output_scores,
    cur_tokens,
    cur_scores,
    xt_neq_x0,
    non_special_sym_mask,
    mask_id,
    cur_log_probs,
    lrd_state,
    tau_decode=0.1,
    k=1,
    valid_vocab_mask=None,
):
    """LRD Phase 2: unmask top-k lowest-entropy masked positions per step.

    Entropy-based selection: positions with lowest entropy (most confident)
    are committed first. KL monitoring provides early stopping signal.

    Args:
        output_tokens, output_scores, cur_tokens, cur_scores, xt_neq_x0,
        non_special_sym_mask, mask_id: same as other strategies
        cur_log_probs: [B, L, V] current log-softmax output
        lrd_state: LRDState object for KL tracking
        tau_decode: KL threshold for convergence detection
        k: number of lowest-entropy positions to unmask per step
        valid_vocab_mask: [V] bool, restrict entropy/KL to valid vocab

    Returns:
        (new_xt_neq_x0, output_tokens, output_scores)
    """
    masked_positions = xt_neq_x0 & non_special_sym_mask

    if not masked_positions.any():
        return xt_neq_x0, output_tokens, output_scores

    # Update KL tracking (for early stopping signal, per-sample)
    sample_converged = lrd_state.update(
        cur_log_probs, masked_positions, tau_decode, valid_vocab_mask
    )  # [B]

    # For converged samples, force-unmask all remaining masked positions
    converged_mask = sample_converged.unsqueeze(-1) & masked_positions  # [B, L]

    # For non-converged samples, do entropy-based top-k selection
    non_converged_masked = ~sample_converged.unsqueeze(-1) & masked_positions  # [B, L]

    # Compute entropy per position over valid vocab only
    if valid_vocab_mask is not None:
        valid_lp = cur_log_probs[..., valid_vocab_mask].log_softmax(dim=-1)
    else:
        valid_lp = cur_log_probs
    valid_p = valid_lp.exp()
    entropy = -(valid_p * valid_lp).sum(dim=-1)  # [B, L]

    # Set entropy to +inf for non-selectable positions
    entropy = entropy.masked_fill(~non_converged_masked, float('inf'))

    # Select top-k lowest entropy positions per non-converged sample
    B = output_tokens.shape[0]
    topk_unmask = torch.zeros_like(xt_neq_x0)
    for b in range(B):
        if sample_converged[b] or not non_converged_masked[b].any():
            continue
        n_masked = non_converged_masked[b].sum().item()
        actual_k = min(k, n_masked)
        _, topk_idx = (-entropy[b]).topk(actual_k)
        topk_unmask[b, topk_idx] = True

    # Combine: converged samples unmask all, others unmask top-k
    to_unmask = (converged_mask | topk_unmask) & masked_positions

    output_tokens[to_unmask] = cur_tokens[to_unmask]
    output_scores[to_unmask] = cur_scores[to_unmask]

    new_xt_neq_x0 = xt_neq_x0.clone()
    new_xt_neq_x0[to_unmask] = False

    return new_xt_neq_x0, output_tokens, output_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_topk(confidence_masked, masked_positions, k):
    """Fallback: unmask top-k by confidence for each sample.

    Args:
        confidence_masked: [B, L] confidence with non-masked set to -1
        masked_positions: [B, L] bool
        k: number of tokens to unmask per sample

    Returns:
        [B, L] bool tensor
    """
    B, L = confidence_masked.shape
    to_unmask = torch.zeros_like(masked_positions)
    for b in range(B):
        if not masked_positions[b].any():
            continue
        n_masked = masked_positions[b].sum().item()
        actual_k = min(k, n_masked)
        _, topk_idx = confidence_masked[b].topk(actual_k)
        to_unmask[b, topk_idx] = True
        to_unmask[b] &= masked_positions[b]
    return to_unmask


# ---------------------------------------------------------------------------
# Dispatcher: parse decoding_strategy string and call appropriate function
# ---------------------------------------------------------------------------

STRATEGY_PREFIX = "dinfer_threshold"  # just for reference

def parse_strategy_name(decoding_strategy):
    """Extract the strategy name from decoding_strategy string.

    Expected formats:
        "reparam-uncond-stochastic1.0-linear"  -> "reparam" (legacy)
        "dinfer_threshold"                      -> "dinfer_threshold"
        "dinfer_threshold@0.8"                  -> "dinfer_threshold"
        "klass@0.01:0.9:2:1"                   -> "klass"
        "dinfer_credit@0.8:0.9:0.5:1.0"        -> "dinfer_credit"
        "dinfer_hierarchical@0.92:0.62"         -> "dinfer_hierarchical"
        "punt@0.04"                             -> "punt"
        "lrd@0.1:1"                             -> "lrd"

    Returns:
        strategy_name: str
    """
    base = decoding_strategy.split("@")[0]
    if base.startswith("reparam"):
        return "reparam"
    return base


def parse_strategy_kwargs(decoding_strategy):
    """Parse hyperparameters from decoding_strategy string.

    Formats:
        "dinfer_threshold"          -> {"threshold": 0.8}
        "dinfer_threshold@0.9"      -> {"threshold": 0.9}
        "klass@0.01:0.9:2:1"       -> {"epsilon_kl": 0.01, "tau": 0.9, "n": 2, "fallback_count": 1}
        "dinfer_credit@0.8:0.9:0.5:1.0" -> {"threshold": 0.8, "beta": 0.9, "gamma": 0.5, "alpha": 1.0}
        "dinfer_hierarchical@0.92:0.62"  -> {"upper_threshold": 0.92, "lower_threshold": 0.62}
        "punt@0.04"                 -> {"epsilon": 0.04}
        "lrd@0.1:1"                 -> {"tau_decode": 0.1, "k": 1}
    """
    parts = decoding_strategy.split("@")
    name = parts[0]

    if name == "dinfer_threshold":
        if len(parts) > 1:
            return {"threshold": float(parts[1])}
        return {"threshold": 0.8}

    elif name == "klass":
        if len(parts) > 1:
            vals = parts[1].split(":")
            kwargs = {"epsilon_kl": float(vals[0])}
            if len(vals) > 1:
                kwargs["tau"] = float(vals[1])
            if len(vals) > 2:
                kwargs["n"] = int(vals[2])
            if len(vals) > 3:
                kwargs["fallback_count"] = int(vals[3])
            return kwargs
        return {"epsilon_kl": 0.01, "tau": 0.9, "n": 2, "fallback_count": 1}

    elif name == "dinfer_credit":
        if len(parts) > 1:
            vals = parts[1].split(":")
            kwargs = {"threshold": float(vals[0])}
            if len(vals) > 1:
                kwargs["beta"] = float(vals[1])
            if len(vals) > 2:
                kwargs["gamma"] = float(vals[2])
            if len(vals) > 3:
                kwargs["alpha"] = float(vals[3])
            return kwargs
        return {"threshold": 0.8, "beta": 0.8, "gamma": 0.2, "alpha": 0.7}

    elif name == "dinfer_hierarchical":
        if len(parts) > 1:
            vals = parts[1].split(":")
            kwargs = {"upper_threshold": float(vals[0])}
            if len(vals) > 1:
                kwargs["lower_threshold"] = float(vals[1])
            return kwargs
        return {"upper_threshold": 0.92, "lower_threshold": 0.62}

    elif name == "punt":
        if len(parts) > 1:
            return {"epsilon": float(parts[1])}
        return {"epsilon": 0.04}

    elif name == "lrd":
        if len(parts) > 1:
            vals = parts[1].split(":")
            kwargs = {"tau_decode": float(vals[0])}
            if len(vals) > 1:
                kwargs["k"] = int(vals[1])
            return kwargs
        return {"tau_decode": 0.1, "k": 1}

    return {}


# # Baseline (reparam)
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --saveto generation-results/decoding_test/baseline

# # KLASS
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --decoding_strategy "klass@0.01:0.9:2:1" \
#     --saveto generation-results/decoding_test/klass

# # PUNT
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --decoding_strategy "punt@0.1" \
#     --saveto generation-results/decoding_test/punt

# # dInfer Threshold
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --decoding_strategy "dinfer_threshold@0.9" \
#     --saveto generation-results/decoding_test/dinfer_threshold

# # dInfer Hierarchical
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --decoding_strategy "dinfer_hierarchical@0.92:0.62" \
#     --saveto generation-results/decoding_test/dinfer_hierarchical

# # dInfer Credit
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --decoding_strategy "dinfer_credit@0.8:0.8:0.2:0.7" \
#     --saveto generation-results/decoding_test/dinfer_credit

# # LRD (Latent Refinement Decoding)
# python generate_dplm2.py \
#     --model_name airkingbd/dplm2_650m \
#     --task inverse_folding \
#     --input_fasta_path data-bin/cameo2022/struct.fasta \
#     --max_iter 100 \
#     --unmasking_strategy deterministic \
#     --sampling_strategy argmax \
#     --decoding_strategy "lrd@0.1:1" \
#     --saveto generation-results/decoding_test/lrd
