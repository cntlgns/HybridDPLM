# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import reduce

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.models.dplm2.dplm2 import DPLM2Config
from byprot.models.dplm2.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)
from byprot.models.dplm2.modules.dplm2_modeling_esm import *
from byprot.models.utils import *
from byprot.models.dplm2.decoding_strategies import (
    parse_strategy_name,
    parse_strategy_kwargs,
    parse_feedforward_mode,
    decode_dinfer_threshold,
    decode_dinfer_hierarchical,
    decode_dinfer_credit,
    decode_klass,
    decode_punt,
    decode_lrd,
    KLASSState,
    CreditState,
    LRDState,
)


@dataclass
class BitConfig:
    load_from_pretrained: bool = field(default=False)
    load_path: str = field(default="")
    # quantized feature is a 13-dimensional vector, resulting in 2^13 struct tokens
    codebook_embed_dim: int = field(default=13)


@dataclass
class DPLM2BitConfig(DPLM2Config):
    ## bit dplm2 config
    bit: BitConfig = field(default=BitConfig())


@register_model("dplm2_bit")
class DPLM2Bit(DPLM2):
    _default_cfg = DPLM2BitConfig()

    def __init__(self, cfg, net=None):
        nn.Module.__init__(self)
        self._update_cfg(cfg)
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file
        )
        self._struct_tokenizer = None
        # binary classification for each dimension of quant feature
        self.cfg.bit.codebook_embed_dim = (
            self.struct_tokenizer.codebook_embed_dim
        )
        if net is None:
            self.net = get_net_dplm2_bit(self.cfg)
        else:
            if "bit" not in net.config.dplm_type:
                raise ValueError(
                    f"The loaded net is not a bit model, which can not be loaded by DPLM2Bit."
                )
            self.net = net
        self._prepare_special_token()

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

        if self.cfg.bit.load_from_pretrained:
            pretrained_state_dict = torch.load(
                self.cfg.bit.load_path, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()

            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            self.load_state_dict(new_pretrained_state_dict, strict=True)
            print(
                f"Successfully load pretrained dplm2 bit model from {self.cfg.bit.load_path}!"
            )

    def _prepare_special_token(self):
        super()._prepare_special_token()
        # HACK: struct tokens and amino acid tokens are in the same vocabulary,
        # there are 33 amino acid tokens, 3 special struct tokens (bos, eos, unk)
        # so the first index of normal struct token is 36
        self.struct_vocab_offset = 36

    def forward(self, input_ids, **kwargs):
        input_mask = input_ids.ne(self.pad_id)

        type_ids = self.get_modality_type(input_ids)

        L = input_ids.shape[1]
        num_heads = self.net.config.num_attention_heads
        # [B, num_heads, L+2, L+2]
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, input_ids.shape
            ).repeat(1, num_heads, L, 1)
        )  # -inf for padding positions, 0 otherwise

        if "single_modality" in kwargs:
            single_modality_index = kwargs["single_modality"]
            struct_attention_bias, aa_attention_bias = attention_bias.chunk(
                2, dim=-2
            )
            struct_attention_bias[
                single_modality_index, :, :, L // 2 :
            ] = -math.inf
            aa_attention_bias[
                single_modality_index, :, :, : L // 2
            ] = -math.inf
            attention_bias = torch.concat(
                [struct_attention_bias, aa_attention_bias], dim=-2
            )

        # Allow external soft embedding override (for feedforward diffusion)
        if "inputs_embeds_override" in kwargs:
            input_embeds = kwargs["inputs_embeds_override"]
        else:
            ######## construct the input embedding
            # [B, L, d_model]
            input_struct_ids, input_aatype_ids = input_ids.chunk(2, dim=1)
            input_struct_mask, input_aatype_mask = input_mask.chunk(2, dim=1)
            input_aatype_embeds = self.net.esm.embeddings(
                input_aatype_ids, attention_mask=input_aatype_mask
            )
            input_struct_embeds = torch.zeros_like(input_aatype_embeds)
            quant = self.struct_tokenizer.quantize.get_codebook_entry(
                input_struct_ids - self.struct_vocab_offset
            )

            input_struct_embeds = self.net.quant2emb(quant).float()
            input_struct_embeds[:, 0] = self.net.struct_bos_emb
            eos_position = input_struct_ids == self.struct_eos_id
            input_struct_embeds[eos_position] = self.net.struct_eos_emb
            mask_position = input_struct_ids == self.struct_mask_id
            input_struct_embeds[mask_position] = self.net.struct_mask_emb

            input_embeds = torch.concat(
                [input_struct_embeds, input_aatype_embeds], dim=1
            )

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            output_hidden_states=True,
            type_ids=type_ids,
        )

        return outputs

    def compute_loss(self, batch, weighting="linear"):
        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        bsz, seq_len = struct_target.shape

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(struct_target, aatype_target)
        x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)
        if self.cfg.self_mixup.enable:
            model_outputs, mixup_loss_mask = self.self_mixup(
                x_t, single_modality_index, bsz, seq_len
            )
            (
                struct_noised["mask"],
                aatype_noised["mask"],
            ) = mixup_loss_mask.chunk(2, dim=1)
        else:
            model_outputs = self.forward(
                input_ids=x_t,
                single_modality=single_modality_index,
            )

        aatype_logits = model_outputs["aatype_logits"]
        struct_logits = model_outputs["struct_logits"].reshape(
            bsz, seq_len, -1, 2
        )
        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_weight = {
            "linear": (
                num_timesteps - (struct_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        struct_target = (
            self.struct_tokenizer.quantize.get_codebook_entry(
                struct_target - self.struct_vocab_offset
            )
            > 0
        ).long()
        assert struct_target.shape == struct_logits.shape[:3]
        struct_weight = struct_weight[:, :, None].expand(struct_target.size())

        aatype_weight = {
            "linear": (
                num_timesteps - (aatype_noised["t"] - 1)
            ),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(aatype_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        aatype_weight = aatype_weight.expand(aatype_target.size())

        return (
            {
                "aatype": aatype_logits,
                "struct": struct_logits,
            },  # model pred logits
            {
                "aatype": aatype_target,
                "struct": struct_target,
            },  # training targets
            {  # training loss mask
                "aatype": aatype_noised["mask"],
                "struct": struct_noised["mask"],
            },
            {
                "aatype": aatype_weight,
                "struct": struct_weight,
            },  # training loss weight
        )

    def self_mixup(self, x_t, single_modality_index, bsz, seq_len):
        # 1. first part: masked prediction
        with torch.no_grad():
            model_outputs = self.forward(
                input_ids=x_t, single_modality=single_modality_index
            )
            aatype_logits = model_outputs["aatype_logits"]
            struct_logits = model_outputs["struct_logits"].reshape(
                bsz, seq_len, -1, 2
            )
        # 2. mixup: alternate mask with model prediction and gt with masks
        prev_input_ids = x_t
        non_special_sym_mask = self.get_non_special_symbol_mask(prev_input_ids)
        model_pred = torch.where(
            non_special_sym_mask,
            self.sample_from_logits(
                aatype_logits, struct_logits, temperature=0.0
            ),
            prev_input_ids,
        )
        mixup_xt, mixup_loss_mask = self.get_mixup_xt(
            input_ids=prev_input_ids,
            model_pred=model_pred,
            non_special_sym_mask=non_special_sym_mask,
        )

        # # 3. second part: denoising + masked prediction
        model_outputs = self.forward(
            input_ids=mixup_xt, single_modality=single_modality_index
        )
        return model_outputs, mixup_loss_mask

    def forward_decoder(
        self,
        prev_decoder_out,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@1.1:0.1",
        feedforward_mode="discrete",
        mask_emb_mode="add",
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        # Compute soft embeddings if feedforward_mode is not discrete
        forward_kwargs = {}
        prev_aatype_logits = prev_decoder_out.get("prev_aatype_logits")
        prev_struct_logits = prev_decoder_out.get("prev_struct_logits")
        has_prev_logits = prev_aatype_logits is not None or prev_struct_logits is not None
        if not feedforward_mode.startswith("discrete") and has_prev_logits:
            soft_embeds = self._compute_bit_soft_embeds(
                output_tokens=output_tokens,
                prev_aatype_logits=prev_aatype_logits,
                prev_struct_logits=prev_struct_logits,
                feedforward_mode=feedforward_mode,
                mask_emb_mode=mask_emb_mode,
                step=step,
                max_step=max_step,
            )
            if soft_embeds is not None:
                forward_kwargs["inputs_embeds_override"] = soft_embeds

        net_out = self.forward(input_ids=output_tokens, **forward_kwargs)

        aatype_logits = net_out["aatype_logits"]
        struct_logits = net_out["struct_logits"]
        attentions = net_out["attentions"] if need_attn_weights else None

        if aatype_logits.dtype != output_scores.dtype:
            aatype_logits = aatype_logits.type_as(output_scores)
            struct_logits = struct_logits.type_as(output_scores)

        bsz, seq_len = aatype_logits.shape[:2]
        aatype_logits[:, :, :4] = -math.inf
        aatype_logits[:, :, 24:] = -math.inf
        struct_logits = struct_logits.reshape(bsz, seq_len, -1, 2)

        aatype_logits = top_k_top_p_filtering(aatype_logits, top_p=0.95)

        if sampling_strategy == "argmax":
            _tokens, _scores = self.sample_from_logits(
                aatype_logits, struct_logits, temperature=0.0
            )
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = self.sample_from_logits(
                aatype_logits, struct_logits, temperature=temperature
            )
        else:
            _tokens, _scores = self.sample_from_logits(
                aatype_logits, struct_logits, temperature=temperature
            )

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        # Build combined log-probs for new decoding strategies
        # aatype_logits: [B, L, V_aa], struct_logits: [B, L, C, 2] -> concat log-softmax
        combined_logits = torch.cat([
            struct_logits.reshape(bsz, seq_len, -1),  # [B, L, C*2]
            aatype_logits,  # [B, L, V_aa]
        ], dim=-1).log_softmax(dim=-1)

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,  # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
            all_hidden_states=net_out["all_hidden_states"],
            logits=combined_logits,
            aatype_logits=aatype_logits,  # [B, L, V_aa] for soft embeds
            struct_logits=struct_logits,  # [B, L, C, 2] for soft embeds
        )

    def sample_from_logits(
        self, aatype_logits, struct_logits, temperature=1.0
    ):
        _aatype_tokens, _aatype_scores = sample_from_categorical(
            aatype_logits, temperature=temperature
        )
        _struct_bits, _struct_scores = sample_from_categorical(
            struct_logits, temperature=temperature
        )
        _struct_tokens = reduce(
            _struct_bits * self.struct_tokenizer.quantize.mask.int(),
            "b n c -> b n",
            "sum",
        )
        _struct_scores = _struct_scores.sum(dim=-1)
        # IMPORTANT: add struct_vocab_offset to _struct_tokens
        _struct_tokens += self.struct_vocab_offset
        assert _struct_tokens.shape == _struct_scores.shape
        _tokens = torch.concat([_struct_tokens, _aatype_tokens], dim=1)
        _scores = torch.concat([_struct_scores, _aatype_scores], dim=1)
        return _tokens, _scores

    def _compute_bit_soft_embeds(
        self,
        output_tokens,
        prev_aatype_logits=None,
        prev_struct_logits=None,
        feedforward_mode="linear",
        mask_emb_mode="add",
        step=0,
        max_step=1,
    ):
        """Compute soft embeddings for the bit model.

        For AA tokens: uses standard word embedding expected value (same as parent).
        For struct tokens: uses per-bit probabilities to compute expected binary
        codes, then passes through quant2emb.

        Args:
            output_tokens: [B, L] current tokens (L = struct_half + aa_half)
            prev_aatype_logits: [B, half_L, V_aa] from previous forward_decoder
            prev_struct_logits: [B, half_L, C, 2] from previous forward_decoder
            feedforward_mode: "linear" or "entropy"
            mask_emb_mode: "add" or "replace"
            step: current decoding step
            max_step: total decoding steps

        Returns:
            soft_embeds: [B, L, D] embeddings with soft mixing at masked positions
        """
        mode_name, mode_kwargs = parse_feedforward_mode(feedforward_mode)
        if mode_name == "discrete":
            return None  # no soft embeddings needed

        input_mask = output_tokens.ne(self.pad_id)
        input_struct_ids, input_aatype_ids = output_tokens.chunk(2, dim=1)
        input_struct_mask, input_aatype_mask = input_mask.chunk(2, dim=1)

        # Build base embeddings (same as forward())
        input_aatype_embeds = self.net.esm.embeddings(
            input_aatype_ids, attention_mask=input_aatype_mask
        )
        quant = self.struct_tokenizer.quantize.get_codebook_entry(
            input_struct_ids - self.struct_vocab_offset
        )
        input_struct_embeds = self.net.quant2emb(quant).float()
        input_struct_embeds[:, 0] = self.net.struct_bos_emb
        eos_position = input_struct_ids == self.struct_eos_id
        input_struct_embeds[eos_position] = self.net.struct_eos_emb
        mask_position_struct = input_struct_ids == self.struct_mask_id
        input_struct_embeds[mask_position_struct] = self.net.struct_mask_emb

        base_embeds = torch.concat(
            [input_struct_embeds, input_aatype_embeds], dim=1
        )

        B, L, D = base_embeds.shape
        half_L = L // 2  # struct half length

        # Identify masked positions per modality
        aa_masked = input_aatype_ids.eq(self.aa_mask_id)  # [B, half_L]
        struct_masked = mask_position_struct  # [B, half_L]

        if not aa_masked.any() and not struct_masked.any():
            return base_embeds

        soft_embeds = base_embeds.clone()

        # --- AA part: standard word embedding expected value ---
        if aa_masked.any() and prev_aatype_logits is not None:
            aa_log_probs = prev_aatype_logits.log_softmax(dim=-1)  # [B, half_L, V_aa]
            aa_probs = aa_log_probs.exp()

            W_aa = self.net.esm.embeddings.word_embeddings.weight[:aa_probs.shape[-1]].detach()  # [V_aa, D]

            # Compute expected AA embedding for masked positions
            expected_aa_emb = torch.zeros(B, half_L, D,
                                          dtype=base_embeds.dtype,
                                          device=base_embeds.device)
            for b in range(B):
                mask_b = aa_masked[b]
                if mask_b.any():
                    expected_aa_emb[b, mask_b] = (
                        aa_probs[b, mask_b].to(W_aa.dtype) @ W_aa
                    ).to(expected_aa_emb.dtype)

            # Compute alpha for AA
            alpha_aa = self._compute_alpha(
                mode_name, mode_kwargs, aa_probs, aa_masked,
                step, max_step, base_embeds.dtype, base_embeds.device, B, half_L
            )
            alpha_3d = alpha_aa.unsqueeze(-1)  # [B, half_L, 1]
            mask_3d = aa_masked.unsqueeze(-1)

            aa_embeds = soft_embeds[:, half_L:]  # AA is the second half
            if mask_emb_mode == "add":
                soft_aa = aa_embeds + alpha_3d * expected_aa_emb
            else:  # replace
                soft_aa = (1 - alpha_3d) * aa_embeds + alpha_3d * expected_aa_emb
            soft_embeds[:, half_L:] = torch.where(mask_3d, soft_aa, aa_embeds)

        # --- Struct part: per-bit soft codes through quant2emb ---
        if struct_masked.any() and prev_struct_logits is not None:
            # prev_struct_logits: [B, half_L, C, 2]
            struct_probs = prev_struct_logits.softmax(dim=-1)  # [B, half_L, C, 2]
            # Soft binary code: probability of bit=1
            soft_bits = struct_probs[..., 1]  # [B, half_L, C]

            # Compute expected struct embedding via quant2emb
            expected_struct_emb = self.net.quant2emb(soft_bits).float()  # [B, half_L, D]

            # For entropy-based alpha, compute entropy from per-bit distributions
            if mode_name == "entropy":
                # Per-bit entropy, averaged over codebook dims
                bit_entropy = -(struct_probs * (struct_probs + 1e-10).log()).sum(dim=-1)  # [B, half_L, C]
                avg_entropy = bit_entropy.mean(dim=-1)  # [B, half_L]
                H_max = math.log(2)  # max entropy for binary
                H_norm = (avg_entropy / H_max).clamp(0.0, 1.0)
                rf = mode_kwargs.get("rf", 0.2)
                alpha_struct = (rf * (1.0 - H_norm)).to(base_embeds.dtype)
            else:
                alpha_struct = self._compute_alpha(
                    mode_name, mode_kwargs, None, struct_masked,
                    step, max_step, base_embeds.dtype, base_embeds.device, B, half_L
                )

            alpha_3d = alpha_struct.unsqueeze(-1)  # [B, half_L, 1]
            mask_3d = struct_masked.unsqueeze(-1)

            struct_embeds = soft_embeds[:, :half_L]  # struct is the first half
            if mask_emb_mode == "add":
                soft_struct = struct_embeds + alpha_3d * expected_struct_emb
            else:  # replace
                soft_struct = (1 - alpha_3d) * struct_embeds + alpha_3d * expected_struct_emb
            soft_embeds[:, :half_L] = torch.where(mask_3d, soft_struct, struct_embeds)

        return soft_embeds

    def _compute_alpha(self, mode_name, mode_kwargs, probs, masked_positions,
                       step, max_step, dtype, device, B, L):
        """Compute alpha mixing weight for soft embeddings."""
        if mode_name == "linear":
            alpha_init = mode_kwargs.get("init", 0.1)
            alpha_growth = mode_kwargs.get("growth", 0.001)
            alpha_preset = mode_kwargs.get("preset", 0.3)
            alpha = min(alpha_init + alpha_growth * step, alpha_preset)
            return torch.full((B, L), alpha, dtype=dtype, device=device)
        elif mode_name == "entropy" and probs is not None:
            rf = mode_kwargs.get("rf", 0.2)
            log_probs = (probs + 1e-10).log()
            entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]
            V_size = probs.shape[-1]
            H_norm = (entropy / math.log(V_size)).clamp(0.0, 1.0) if V_size > 1 else entropy
            return (rf * (1.0 - H_norm)).to(dtype)
        else:
            # Fallback: linear default
            alpha = min(0.1 + 0.001 * step, 0.3)
            return torch.full((B, L), alpha, dtype=dtype, device=device)

    def _apply_new_strategy(
        self,
        strategy_name,
        strategy_kwargs,
        strategy_state,
        prev_tokens,
        prev_scores,
        cur_tokens,
        cur_scores,
        cur_log_probs,
        xt_neq_x0,
        type_ids,
        non_special_sym_mask,
        step,
        cur_aatype_logits=None,
        cur_struct_logits=None,
        prev_aatype_logits=None,
        prev_struct_logits=None,
    ):
        """Override for bit model: handle separate struct/AA logit spaces.

        The bit model has fundamentally different logit structures per modality:
        - AA: [B, L/2, V_aa] categorical logits
        - Struct: [B, L/2, C, 2] per-bit binary logits

        This override splits all tensors into per-modality halves, calls strategy
        functions with consistently-shaped half-length tensors, then combines results.
        """
        output_tokens = prev_tokens
        output_scores = prev_scores
        new_xt_neq_x0 = xt_neq_x0.clone()

        B, L = prev_tokens.shape
        half_L = L // 2

        for modality in ["struct", "aa"]:
            if modality == "struct":
                sl = slice(0, half_L)
                mask_id = self.struct_mask_id
                # Reshape struct logits: [B, half_L, C, 2] -> [B, half_L, C*2]
                if cur_struct_logits is not None:
                    mod_cur_lp = cur_struct_logits.reshape(B, half_L, -1).log_softmax(dim=-1)
                else:
                    mod_cur_lp = None
                if prev_struct_logits is not None:
                    mod_prev_lp = prev_struct_logits.reshape(B, half_L, -1).log_softmax(dim=-1)
                else:
                    mod_prev_lp = None
            else:
                sl = slice(half_L, L)
                mask_id = self.aa_mask_id
                if cur_aatype_logits is not None:
                    mod_cur_lp = cur_aatype_logits.log_softmax(dim=-1)
                else:
                    mod_cur_lp = None
                if prev_aatype_logits is not None:
                    mod_prev_lp = prev_aatype_logits.log_softmax(dim=-1)
                else:
                    mod_prev_lp = None

            # Extract half-length tensors for this modality
            mod_out_tokens = output_tokens[:, sl].clone()
            mod_out_scores = output_scores[:, sl].clone()
            mod_cur_tokens = cur_tokens[:, sl]
            mod_cur_scores = cur_scores[:, sl]
            mod_xt = xt_neq_x0[:, sl]
            mod_nsm = non_special_sym_mask[:, sl]

            if not (mod_xt & mod_nsm).any():
                continue

            # No valid_vocab_mask needed: each modality already has its own logit space
            if strategy_name == "dinfer_threshold":
                new_mod_xt, mod_out_tokens, mod_out_scores = decode_dinfer_threshold(
                    output_tokens=mod_out_tokens,
                    output_scores=mod_out_scores,
                    cur_tokens=mod_cur_tokens,
                    cur_scores=mod_cur_scores,
                    xt_neq_x0=mod_xt,
                    non_special_sym_mask=mod_nsm,
                    mask_id=mask_id,
                    **strategy_kwargs,
                )

            elif strategy_name == "klass":
                klass_key = f"klass_{modality}"
                klass_st = strategy_state[klass_key]
                prev_lp_key = f"prev_log_probs_{modality}"
                mod_prev_lp_for_klass = strategy_state.get(prev_lp_key)

                kw = {k: v for k, v in strategy_kwargs.items() if k != "n"}
                new_mod_xt, mod_out_tokens, mod_out_scores = decode_klass(
                    output_tokens=mod_out_tokens,
                    output_scores=mod_out_scores,
                    cur_tokens=mod_cur_tokens,
                    cur_scores=mod_cur_scores,
                    xt_neq_x0=mod_xt,
                    non_special_sym_mask=mod_nsm,
                    mask_id=mask_id,
                    klass_state=klass_st,
                    cur_log_probs=mod_cur_lp,
                    prev_log_probs=mod_prev_lp_for_klass,
                    valid_vocab_mask=None,
                    **kw,
                )

            elif strategy_name == "dinfer_credit":
                credit_key = f"credit_{modality}"
                credit_st = strategy_state[credit_key]
                kw = {k: v for k, v in strategy_kwargs.items()
                      if k not in ("beta", "gamma")}
                new_mod_xt, mod_out_tokens, mod_out_scores = decode_dinfer_credit(
                    output_tokens=mod_out_tokens,
                    output_scores=mod_out_scores,
                    cur_tokens=mod_cur_tokens,
                    cur_scores=mod_cur_scores,
                    xt_neq_x0=mod_xt,
                    non_special_sym_mask=mod_nsm,
                    mask_id=mask_id,
                    credit_state=credit_st,
                    cur_log_probs=mod_cur_lp,
                    valid_vocab_mask=None,
                    **kw,
                )

            elif strategy_name == "dinfer_hierarchical":
                new_mod_xt, mod_out_tokens, mod_out_scores = decode_dinfer_hierarchical(
                    output_tokens=mod_out_tokens,
                    output_scores=mod_out_scores,
                    cur_tokens=mod_cur_tokens,
                    cur_scores=mod_cur_scores,
                    xt_neq_x0=mod_xt,
                    non_special_sym_mask=mod_nsm,
                    mask_id=mask_id,
                    **strategy_kwargs,
                )

            elif strategy_name == "punt":
                _sl = sl  # capture for closure
                # Capture the other modality's tokens for reconstructing full input
                if modality == "struct":
                    _other_tokens = cur_tokens[:, half_L:]  # AA half
                else:
                    _other_tokens = cur_tokens[:, :half_L]  # struct half

                def forward_fn(half_input_ids, _self=self, _sl=_sl,
                               _other=_other_tokens, _mod=modality):
                    """Reconstruct full input, forward, extract half logits."""
                    if _mod == "struct":
                        full_input = torch.cat([half_input_ids, _other[:half_input_ids.shape[0]]], dim=1)
                    else:
                        full_input = torch.cat([_other[:half_input_ids.shape[0]], half_input_ids], dim=1)
                    net_out = _self.forward(input_ids=full_input)
                    if _mod == "struct":
                        logits = net_out["struct_logits"]
                        bsz, seq_len = logits.shape[:2]
                        logits = logits.reshape(bsz, seq_len, -1).log_softmax(dim=-1)
                    else:
                        logits = net_out["aatype_logits"]
                        logits[:, :, :4] = -math.inf
                        logits[:, :, 24:] = -math.inf
                        logits = top_k_top_p_filtering(logits, top_p=0.95)
                        logits = logits.log_softmax(dim=-1)
                    return {"logits": logits}

                new_mod_xt, mod_out_tokens, mod_out_scores = decode_punt(
                    output_tokens=mod_out_tokens,
                    output_scores=mod_out_scores,
                    cur_tokens=mod_cur_tokens,
                    cur_scores=mod_cur_scores,
                    xt_neq_x0=mod_xt,
                    non_special_sym_mask=mod_nsm,
                    mask_id=mask_id,
                    baseline_log_probs=mod_cur_lp,
                    forward_fn=forward_fn,
                    valid_vocab_mask=None,
                    **strategy_kwargs,
                )

            elif strategy_name == "lrd":
                lrd_key = f"lrd_{modality}"
                lrd_st = strategy_state[lrd_key]
                new_mod_xt, mod_out_tokens, mod_out_scores = decode_lrd(
                    output_tokens=mod_out_tokens,
                    output_scores=mod_out_scores,
                    cur_tokens=mod_cur_tokens,
                    cur_scores=mod_cur_scores,
                    xt_neq_x0=mod_xt,
                    non_special_sym_mask=mod_nsm,
                    mask_id=mask_id,
                    cur_log_probs=mod_cur_lp,
                    lrd_state=lrd_st,
                    valid_vocab_mask=None,
                    **strategy_kwargs,
                )
            else:
                raise ValueError(f"Unknown decoding strategy: {strategy_name}")

            # Write back half-length results into full-length tensors
            output_tokens[:, sl] = mod_out_tokens
            output_scores[:, sl] = mod_out_scores
            new_xt_neq_x0[:, sl] = new_mod_xt

            # Store per-modality prev_log_probs for KLASS
            if strategy_name == "klass" and mod_cur_lp is not None:
                strategy_state[f"prev_log_probs_{modality}"] = mod_cur_lp.clone()

        return new_xt_neq_x0, output_tokens, output_scores

    def generate(
        self,
        input_tokens,
        max_iter=None,
        temperature=None,
        partial_masks=None,
        unmasking_strategy="stochastic1.0",  # [stochastic{temperature}, deterministic]
        sampling_strategy="annealing@1.1:0.1",
        remasking_strategy="uncond",  # [uncond, cond, no_remask]
        decoding_strategy=None,  # e.g. "dinfer_threshold@0.8", "klass@0.01:0.9:2:1", etc.
        feedforward_mode="discrete",  # [discrete, linear, entropy]
        mask_emb_mode="add",  # [add, replace]
    ):
        self.eval()
        max_iter = max_iter
        temperature = temperature

        if not feedforward_mode.startswith("discrete"):
            self.net.esm.embeddings.token_dropout = False

        # Determine which decoding path to use
        if decoding_strategy is None:
            strategy_name = "reparam"
        else:
            strategy_name = parse_strategy_name(decoding_strategy)

        # 0) encoding
        encoder_out = self.forward_encoder(input_tokens)
        # 1) initialized from all mask tokens
        (
            initial_output_tokens,
            initial_output_scores,
        ) = self.initialize_output_tokens(
            input_tokens, encoder_out=encoder_out, partial_masks=partial_masks
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            type_ids=self.get_modality_type(initial_output_tokens),
        )

        prev_decoder_out["output_masks"] = self.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        # --- Initialize strategy-specific state ---
        strategy_kwargs = parse_strategy_kwargs(decoding_strategy) if decoding_strategy else {}
        strategy_state = {}
        B, L = initial_output_tokens.shape

        half_L = L // 2

        if strategy_name == "klass":
            n = strategy_kwargs.get("n", 2)
            strategy_state["klass_aa"] = KLASSState(n=n)
            strategy_state["klass_struct"] = KLASSState(n=n)
        elif strategy_name == "dinfer_credit":
            # AA vocab size (number of AA tokens)
            V_aa = 33
            # Struct vocab: C*2 (binary logits per codebook dim)
            C = self.cfg.bit.codebook_embed_dim
            V_struct = C * 2
            strategy_state["credit_aa"] = CreditState(
                B, half_L, V_aa, initial_output_tokens.device,
                beta=strategy_kwargs.get("beta", 0.8),
                gamma=strategy_kwargs.get("gamma", 0.2),
            )
            strategy_state["credit_struct"] = CreditState(
                B, half_L, V_struct, initial_output_tokens.device,
                beta=strategy_kwargs.get("beta", 0.8),
                gamma=strategy_kwargs.get("gamma", 0.2),
            )
        elif strategy_name == "lrd":
            tau_refine = strategy_kwargs.get("tau_refine", 0.1)
            T_refine = strategy_kwargs.get("T_refine", 20)
            strategy_state["lrd_aa"] = LRDState(
                tau_refine=tau_refine, T_refine=T_refine,
            )
            strategy_state["lrd_struct"] = LRDState(
                tau_refine=tau_refine, T_refine=T_refine,
            )

        for step in tqdm(range(max_iter), desc="Decoding"):
            # Early stopping: if nothing is masked, stop
            if strategy_name != "reparam":
                if not prev_decoder_out["output_masks"].any():
                    break

            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    feedforward_mode=feedforward_mode,
                    mask_emb_mode=mask_emb_mode,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: re-mask / unmask
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            if strategy_name == "reparam":
                (
                    output_masks,
                    result_tokens,
                    result_scores,
                ) = self._reparam_decoding(
                    output_tokens=prev_decoder_out["output_tokens"].clone(),
                    output_scores=prev_decoder_out["output_scores"].clone(),
                    cur_tokens=output_tokens.clone(),
                    cur_scores=output_scores.clone(),
                    decoding_strategy=f"reparam-{remasking_strategy}-{unmasking_strategy}-linear",
                    xt_neq_x0=prev_decoder_out["output_masks"],
                    type_ids=prev_decoder_out["type_ids"].clone(),
                    non_special_sym_mask=non_special_sym_mask,
                    t=step + 1,
                    max_step=max_iter,
                )
            else:
                (
                    output_masks,
                    result_tokens,
                    result_scores,
                ) = self._apply_new_strategy(
                    strategy_name=strategy_name,
                    strategy_kwargs=strategy_kwargs,
                    strategy_state=strategy_state,
                    prev_tokens=prev_decoder_out["output_tokens"].clone(),
                    prev_scores=prev_decoder_out["output_scores"].clone(),
                    cur_tokens=output_tokens.clone(),
                    cur_scores=output_scores.clone(),
                    cur_log_probs=decoder_out.get("logits"),
                    xt_neq_x0=prev_decoder_out["output_masks"],
                    type_ids=prev_decoder_out["type_ids"].clone(),
                    non_special_sym_mask=non_special_sym_mask,
                    step=step,
                    cur_aatype_logits=decoder_out.get("aatype_logits"),
                    cur_struct_logits=decoder_out.get("struct_logits"),
                    prev_aatype_logits=prev_decoder_out.get("prev_aatype_logits"),
                    prev_struct_logits=prev_decoder_out.get("prev_struct_logits"),
                )

            # Final step: force-unmask remaining
            # (LRD convergence is handled per-sample inside decode_lrd)
            is_final = (step == max_iter - 1)
            if strategy_name != "reparam" and is_final:
                still_masked = output_masks & non_special_sym_mask
                if still_masked.any():
                    raw_tokens = decoder_out["output_tokens"]
                    raw_scores = decoder_out["output_scores"]
                    result_tokens[still_masked] = raw_tokens[still_masked]
                    result_scores[still_masked] = raw_scores[still_masked]
                    output_masks = output_masks & ~still_masked

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
                all_hidden_states=decoder_out["all_hidden_states"],
                prev_log_probs=decoder_out.get("logits"),
                prev_aatype_logits=decoder_out.get("aatype_logits"),
                prev_struct_logits=decoder_out.get("struct_logits"),
            )

        decoder_out = prev_decoder_out

        decoder_out = self.prepare_for_struct_tokenizer(
            decoder_out, non_special_sym_mask
        )
        return {
            "output_tokens": decoder_out["output_tokens"],
            "res_mask": decoder_out["res_mask"],
            "final_struct_feature": decoder_out["final_struct_feature"],
        }

    def prepare_for_struct_tokenizer(self, decoder_out, non_special_sym_mask):
        lm_output_struct_tokens = decoder_out["output_tokens"].chunk(2, dim=1)[
            0
        ]
        non_bos_eos_mask = lm_output_struct_tokens.ne(
            self.struct_eos_id
        ) & lm_output_struct_tokens.ne(self.struct_bos_id)
        bsz, max_len = non_bos_eos_mask.shape

        res_mask = (
            lm_output_struct_tokens[non_bos_eos_mask]
            .view(bsz, max_len - 2)
            .ne(self.pad_id)
            .int()
        )
        struct_tokens = (
            lm_output_struct_tokens[non_bos_eos_mask].view(bsz, max_len - 2)
            - self.struct_vocab_offset
        )
        struct_tokens[~res_mask.bool()] = 0
        quant = self.struct_tokenizer.quantize.get_codebook_entry(
            struct_tokens
        )
        decoder_out["res_mask"] = res_mask
        decoder_out["final_struct_feature"] = quant

        return decoder_out
