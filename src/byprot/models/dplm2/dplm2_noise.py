# Noise-input diffusion finetuning for DPLM2.
#
# Ablation of dplm2_hybrid: instead of "clean embedding + sigma * eps" with a
# noise schedule, masked positions simply receive a fresh Gaussian noise vector
# (no addition to the ground-truth embedding, no sigma scheduling).
#
# Purpose: isolate whether the gain reported by hybrid diffusion comes from the
# hybrid ODE/score machinery, or merely from feeding the model noisy random
# vectors at masked positions instead of a single shared <mask> token embedding
# — i.e. an input-augmentation effect.
#
# The training recipe and inference loop otherwise mirror dplm2_hybrid:
#   - Multi-task split (single / folding / inverse_folding / joint / independent)
#   - Soft-embedding forward via inputs_embeds_override
#   - Layer-0 attention residual cutoff for masked positions (ported from hybrid)
#   - Iterative top-k confidence unmasking
#
# Inference exposes `sample_noise_every_step`:
#   - True  : draw fresh Gaussian noise at every step for still-masked positions
#   - False : sample noise once at init and keep it fixed at masked positions
#             (positions that get unmasked drop their cached noise)

import math
from dataclasses import dataclass, field

import torch

from byprot.models import register_model
from byprot.models.dplm2.dplm2 import (
    DPLM2Config,
    MultimodalDiffusionProteinLanguageModel,
)
from byprot.models.dplm2.dplm2_hybrid import MaskedResidualSelfOutput
from byprot.models.utils import sample_from_categorical


@dataclass
class DPLM2NoiseConfig(DPLM2Config):
    # Whether ESM's embedding layer applies token_dropout (0.88 scaling).
    # Set False because we provide soft embeddings, not discrete tokens, so
    # there is no <mask> token in the actual input to scale around.
    token_dropout: bool = field(default=False)
    # Cut residual connection in layer[0] attention for masked positions only,
    # matching the hybrid setup (the residual would otherwise leak raw noise
    # straight through layer 0).
    cutoff_layer0_attn_residual: bool = field(default=True)
    # Inference-only: resample noise every step (True) or keep the initial
    # noise sample at still-masked positions (False).
    sample_noise_every_step: bool = field(default=True)


@register_model("dplm2_noise")
class NoiseInputDiffusionProteinLanguageModel(
    MultimodalDiffusionProteinLanguageModel
):
    _default_cfg = DPLM2NoiseConfig()

    def __init__(self, cfg, net=None):
        super().__init__(cfg, net)

        if hasattr(cfg, "token_dropout"):
            emb = self.net.esm.embeddings
            if hasattr(emb, "original_module"):
                emb.original_module.token_dropout = cfg.token_dropout
                for mod in emb.modules_to_save.values():
                    mod.token_dropout = cfg.token_dropout
            else:
                emb.token_dropout = cfg.token_dropout

        if self.cfg.cutoff_layer0_attn_residual:
            layer0_attn = self._get_layer0_attention()
            layer0_attn.output = MaskedResidualSelfOutput(layer0_attn.output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_word_embeddings(self) -> torch.Tensor:
        """Return the (PEFT-aware) trainable word embedding weight tensor."""
        try:
            emb = self.net.base_model.model.esm.embeddings
            if hasattr(emb, "modules_to_save"):
                active = emb.modules_to_save[emb.active_adapter]
                return active.word_embeddings.weight
            return emb.word_embeddings.weight
        except AttributeError:
            return self.net.esm.embeddings.word_embeddings.weight

    def _get_layer0_attention(self):
        try:
            return self.net.base_model.model.esm.encoder.layer[0].attention
        except AttributeError:
            return self.net.esm.encoder.layer[0].attention

    # ------------------------------------------------------------------
    # Noising: pure Gaussian replacement at masked positions
    # ------------------------------------------------------------------
    def noise_q_sample(self, x_0, t, maskable_mask):
        """
        Pure-noise masking: mark positions with prob t/T and replace their
        embedding with a fresh N(0, I) vector. No noise schedule, no sigma.

        Args:
            x_0:           [B, L] clean token indices
            t:             [B]    discrete timesteps in {0, ..., T}
            maskable_mask: [B, L] bool, positions eligible for corruption

        Returns:
            soft_embeds: [B, L, d] soft embeddings (clean rows untouched,
                                   corrupted rows replaced by Gaussian noise)
            mask_t:      [B, L]    bool, True for corrupted positions
        """
        B, L = x_0.shape
        device = x_0.device

        mask_prob = t.float() / self.cfg.num_diffusion_timesteps  # [B]
        u = torch.rand(B, L, device=device)
        mask_t = (u < mask_prob[:, None]) & maskable_mask  # True = corrupted

        W = self._get_word_embeddings()  # [V, d]
        soft_embeds = W[x_0].clone()  # [B, L, d]

        if mask_t.any():
            n = mask_t.sum().item()
            d = soft_embeds.shape[-1]
            eps = torch.randn(n, d, device=device, dtype=soft_embeds.dtype)
            soft_embeds[mask_t] = eps

        return soft_embeds, mask_t

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def construct_x_t(self, struct_target, aatype_target):
        """Sample timesteps, build multi-task split, return soft embeddings."""
        bsz = struct_target.size(0)
        device = struct_target.device

        struct_t = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1, (bsz,), device=device
        )
        aatype_t = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1, (bsz,), device=device
        )

        assert (
            self.cfg.single_modality_ratio
            + self.cfg.folding_loss_ratio
            + self.cfg.inverse_folding_loss_ratio
            + self.cfg.joint_loss_ratio
            + self.cfg.independent_loss_ratio
            == 1.0
        )

        split_sizes = [
            int(bsz * self.cfg.single_modality_ratio),
            int(bsz * self.cfg.folding_loss_ratio),
            int(bsz * self.cfg.inverse_folding_loss_ratio),
            int(bsz * self.cfg.independent_loss_ratio),
            int(bsz * self.cfg.joint_loss_ratio),
        ]
        split_sizes[-1] = bsz - sum(split_sizes[:-1])

        rand_index = torch.randperm(bsz).type_as(struct_target)
        int_index_list = torch.split(rand_index, split_sizes)

        bool_index_list = []
        for int_index in int_index_list:
            bool_index = torch.zeros(bsz, dtype=torch.bool, device=device)
            bool_index[int_index] = True
            bool_index_list.append(bool_index)

        (
            single_modality_index,
            folding_index,
            inverse_folding_index,
            independent_index,
            joint_index,
        ) = bool_index_list

        struct_t = struct_t.masked_fill(inverse_folding_index, 0)
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])

        struct_soft, struct_mask = self.noise_q_sample(
            struct_target,
            struct_t,
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )
        aa_soft, aa_mask = self.noise_q_sample(
            aatype_target,
            aatype_t,
            maskable_mask=self.get_non_special_symbol_mask(aatype_target),
        )

        return (
            {"t": struct_t, "soft_embeds": struct_soft, "mask": struct_mask},
            {"t": aatype_t, "soft_embeds": aa_soft, "mask": aa_mask},
            single_modality_index,
        )

    def compute_loss(self, batch, weighting="linear"):
        """Forward with soft embeddings; reuse parent's loss bookkeeping."""
        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(struct_target, aatype_target)

        soft_embeds = torch.cat(
            [struct_noised["soft_embeds"], aatype_noised["soft_embeds"]],
            dim=1,
        )
        ref_input_ids = torch.cat([struct_target, aatype_target], dim=1)

        if self.cfg.cutoff_layer0_attn_residual:
            combined_mask = torch.cat(
                [struct_noised["mask"], aatype_noised["mask"]], dim=1
            )
            self._get_layer0_attention().output.set_residual_mask(combined_mask)

        model_outputs = self.forward(
            input_ids=ref_input_ids,
            single_modality=single_modality_index,
            inputs_embeds_override=soft_embeds,
        )

        struct_logits, aatype_logits = model_outputs["logits"].chunk(2, dim=1)

        num_ts = self.cfg.num_diffusion_timesteps
        if weighting == "linear":
            struct_w = (
                (num_ts - (struct_noised["t"] - 1)).float() / num_ts
            )[:, None]
            aa_w = (
                (num_ts - (aatype_noised["t"] - 1)).float() / num_ts
            )[:, None]
        else:  # constant
            struct_w = torch.ones(
                struct_target.size(0), 1, device=struct_target.device
            )
            aa_w = torch.ones(
                aatype_target.size(0), 1, device=aatype_target.device
            )

        struct_w = struct_w.expand_as(struct_target).float()
        aa_w = aa_w.expand_as(aatype_target).float()

        return (
            {"aatype": aatype_logits, "struct": struct_logits},
            {"aatype": aatype_target, "struct": struct_target},
            {"aatype": aatype_noised["mask"], "struct": struct_noised["mask"]},
            {"aatype": aa_w, "struct": struct_w},
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _init_noise_state(self, output_tokens, output_masks):
        """Sample initial noise for all maskable positions; mark clean mask."""
        B, L = output_tokens.shape
        device = output_tokens.device
        W = self._get_word_embeddings()
        d = W.shape[1]
        noise_cache = torch.randn(B, L, d, device=device, dtype=W.dtype)
        clean_mask = ~output_masks
        return noise_cache, clean_mask

    def _noise_forward_step(
        self,
        output_tokens,
        noise_cache,
        clean_mask,
        output_masks,
        temperature,
        ref_input_ids,
    ):
        """Build Y_t (clean rows from W[token], corrupted rows from noise),
        run the model, return sampled tokens and scores."""
        W = self._get_word_embeddings()
        clean_embeds = W[output_tokens]
        clean_f = clean_mask.unsqueeze(-1).to(clean_embeds.dtype)
        Y_t = clean_f * clean_embeds + (1 - clean_f) * noise_cache

        if self.cfg.cutoff_layer0_attn_residual:
            corrupted_mask = ~clean_mask & output_masks
            self._get_layer0_attention().output.set_residual_mask(corrupted_mask)

        net_out = self.forward(
            input_ids=ref_input_ids,
            inputs_embeds_override=Y_t,
        )
        logits = net_out["logits"]

        type_ids = self.get_modality_type(output_tokens)
        idx_aa = torch.where(type_ids.eq(self.aa_type) & output_masks)
        idx_struct = torch.where(type_ids.eq(self.struct_type) & output_masks)
        logits[idx_aa[0], idx_aa[1], 33:] = -math.inf
        logits[idx_struct[0], idx_struct[1], :33] = -math.inf
        logits[..., self.special_token_list] = -math.inf

        log_probs = logits.log_softmax(dim=-1)
        _tokens, _scores = sample_from_categorical(
            log_probs, temperature=temperature
        )
        return _tokens, _scores

    def _select_topk_to_unmask(self, _scores, still_corrupted, k_per_sample):
        B, L = _scores.shape
        device = _scores.device
        scores_for_topk = _scores.clone()
        scores_for_topk[~still_corrupted] = -1e9
        _, sorted_idx = scores_for_topk.sort(dim=1, descending=True)
        j_range = torch.arange(L, device=device).unsqueeze(0)
        topk_mask_sorted = j_range < k_per_sample.unsqueeze(1)
        newly_clean = torch.zeros_like(still_corrupted)
        newly_clean.scatter_(1, sorted_idx, topk_mask_sorted)
        return newly_clean & still_corrupted

    # ------------------------------------------------------------------
    # MCTS-friendly decoding API (mirrors DPLM2 / hybrid)
    # ------------------------------------------------------------------
    def init_decoding_state(
        self,
        input_tokens,
        max_iter,
        partial_masks=None,
        sampling_strategy="annealing@2.0:0.1",
        temperature=1.0,
        sample_noise_every_step=None,
        **_unused,
    ):
        """Build decoding state for the noise-input variant."""
        self.eval()

        if sample_noise_every_step is None:
            sample_noise_every_step = self.cfg.sample_noise_every_step

        output_tokens, output_scores = self.initialize_output_tokens(
            input_tokens, partial_masks=partial_masks
        )
        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )
        noise_cache, clean_mask = self._init_noise_state(
            output_tokens, output_masks
        )
        ref_tokens = output_tokens.clone()

        return {
            "output_tokens": output_tokens,
            "output_scores": output_scores,
            "output_masks": output_masks,
            "noise_cache": noise_cache,
            "clean_mask": clean_mask,
            "ref_tokens": ref_tokens,
            "step": 0,
            "max_iter": max_iter,
            "partial_masks": partial_masks,
            "sampling_strategy": sampling_strategy,
            "temperature": temperature,
            "sample_noise_every_step": sample_noise_every_step,
        }

    @staticmethod
    def clone_decoding_state(state):
        import copy as _copy

        new_state = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                new_state[k] = v.clone()
            elif isinstance(v, list):
                new_state[k] = [
                    x.clone() if torch.is_tensor(x) else _copy.copy(x)
                    for x in v
                ]
            else:
                new_state[k] = _copy.copy(v)
        return new_state

    def _compute_step_temperature(self, sampling_strategy, step, max_iter, fallback):
        if sampling_strategy == "argmax":
            return 0.0
        if sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1.0 - step / max_iter
            return min_temp + (max_temp - min_temp) * rate
        return fallback

    def decoding_step(self, state):
        step = state["step"]
        max_iter = state["max_iter"]
        is_final = step == max_iter - 1

        T = self.cfg.num_diffusion_timesteps
        t_curr = max(int((1.0 - step / max_iter) * T), 1)
        t_next = max(int((1.0 - (step + 1) / max_iter) * T), 0)
        alpha_curr = 1.0 - t_curr / T
        alpha_next = 1.0 - t_next / T

        cur_temp = self._compute_step_temperature(
            state["sampling_strategy"], step, max_iter, state["temperature"]
        )

        with torch.no_grad():
            _tokens, _scores = self._noise_forward_step(
                state["output_tokens"],
                state["noise_cache"],
                state["clean_mask"],
                state["output_masks"],
                cur_temp,
                ref_input_ids=state["ref_tokens"],
            )

        state["ref_tokens"][state["output_masks"]] = _tokens[state["output_masks"]]

        still_corrupted = ~state["clean_mask"] & state["output_masks"]
        if is_final:
            newly_clean = still_corrupted
        elif still_corrupted.any():
            unmask_rate = (alpha_next - alpha_curr) / (
                1.0 - alpha_curr + 1e-8
            )
            n_corrupted = still_corrupted.sum(dim=1)
            k_per_sample = (
                unmask_rate * n_corrupted.float()
            ).floor().long()
            newly_clean = self._select_topk_to_unmask(
                _scores, still_corrupted, k_per_sample
            )
        else:
            newly_clean = torch.zeros_like(still_corrupted)

        state["output_tokens"][newly_clean] = _tokens[newly_clean]
        state["output_scores"][newly_clean] = _scores[newly_clean]
        state["clean_mask"] = state["clean_mask"] | newly_clean

        if state["sample_noise_every_step"]:
            still_corrupted = ~state["clean_mask"] & state["output_masks"]
            if still_corrupted.any():
                fresh = torch.randn_like(state["noise_cache"])
                sc = still_corrupted.unsqueeze(-1)
                state["noise_cache"] = torch.where(
                    sc, fresh, state["noise_cache"]
                )

        state["step"] = step + 1
        return state

    def one_shot_complete(self, state):
        """Force-unmask every still-corrupted position in one forward pass."""
        step = state["step"]
        max_iter = state["max_iter"]

        cur_temp = self._compute_step_temperature(
            state["sampling_strategy"], step, max_iter, state["temperature"]
        )

        still_corrupted = ~state["clean_mask"] & state["output_masks"]
        if not still_corrupted.any():
            return state

        with torch.no_grad():
            _tokens, _scores = self._noise_forward_step(
                state["output_tokens"],
                state["noise_cache"],
                state["clean_mask"],
                state["output_masks"],
                cur_temp,
                ref_input_ids=state["ref_tokens"],
            )

        state["output_tokens"][still_corrupted] = _tokens[still_corrupted]
        state["output_scores"][still_corrupted] = _scores[still_corrupted]
        state["clean_mask"] = state["clean_mask"] | still_corrupted
        state["ref_tokens"][state["output_masks"]] = _tokens[state["output_masks"]]
        state["step"] = step + 1
        return state

    def generate(
        self,
        input_tokens,
        max_iter=None,
        temperature=1.0,
        partial_masks=None,
        sampling_strategy="annealing@2.0:0.1",
        sample_noise_every_step=None,
        **kwargs,
    ):
        """Iterative noise-input decoding.

        Args:
            sample_noise_every_step: if None, falls back to
                ``self.cfg.sample_noise_every_step``. When True, the noise at
                still-masked positions is redrawn at every step. When False,
                the initial sample is reused until each position is unmasked.
        """
        state = self.init_decoding_state(
            input_tokens=input_tokens,
            max_iter=max_iter,
            partial_masks=partial_masks,
            sampling_strategy=sampling_strategy,
            temperature=temperature,
            sample_noise_every_step=sample_noise_every_step,
        )
        for _ in range(max_iter):
            state = self.decoding_step(state)
        return {"output_tokens": state["output_tokens"]}
