# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# dplm2_lm_heads.py
#
# Ablation model for isolating the contribution of two design choices in dplm2_bit:
#   (A) Separate lm_heads for struct and AA tokens
#   (B) Bitwise struct token encoding (via VQ codebook binary features)
#
# This model adopts (A) but NOT (B):
#   - Separate lm_head (AA) and lm_head_struct (struct)   ← same as dplm2_bit
#   - Struct tokens remain discrete integers (no quant2emb) ← same as dplm2
#   - Standard token embedding for struct tokens           ← same as dplm2
#
# Comparison grid:
#   dplm2          → single lm_head,    integer struct tokens
#   dplm2_lm_heads → separate lm_heads, integer struct tokens  [this model]
#   dplm2_bit      → separate lm_heads, bitwise struct encoding
#
# dplm2 vs dplm2_lm_heads      → effect of separate lm_heads
# dplm2_lm_heads vs dplm2_bit  → effect of bitwise struct representation


import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm import tqdm

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.models.dplm2.dplm2 import DPLM2Config
from byprot.models.dplm2.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)
from byprot.models.dplm2.dplm2_bit import DPLM2Bit
from byprot.models.dplm2.modules.dplm2_modeling_esm import *
from byprot.models.utils import *


@dataclass
class DPLM2LMHeadsConfig(DPLM2Config):
    """Config for dplm2_lm_heads.

    struct_vocab_size: number of discrete struct token classes (= codebook size).
    Defaults to 8192 (= 2^13, matching the VQ codebook used in dplm2/dplm2_bit).
    """
    struct_vocab_size: int = field(default=8192)


@register_model("dplm2_lm_heads")
class DPLM2LMHeads(DPLM2Bit):
    """DPLM2 with separate lm_heads but integer (non-bitwise) struct tokens.

    Inherits the iterative masked-diffusion decoding loop from DPLM2Bit,
    but replaces:
      - quant2emb + struct special embeddings  →  standard token embedding
      - binary struct logits [B,L,13,2]        →  categorical struct logits [B,L,struct_vocab_size]
      - binary struct targets                  →  integer struct targets in [0, struct_vocab_size)

    Methods inherited unchanged from DPLM2Bit:
      generate(), prepare_for_struct_tokenizer(), _reparam_decoding(),
      get_non_special_symbol_mask(), initialize_output_tokens(), ...
    """

    _default_cfg = DPLM2LMHeadsConfig()

    def __init__(self, cfg, net=None):
        nn.Module.__init__(self)
        self._update_cfg(cfg)
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file
        )
        self._struct_tokenizer = None

        if net is None:
            self.net = get_net_dplm2_lm_heads(self.cfg)
        else:
            self.net = net

        self._prepare_special_token()

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

    def _prepare_special_token(self):
        super()._prepare_special_token()
        # Same offset as dplm2_bit:
        #   0-32  → 33 AA tokens
        #   33-35 → 3 special struct tokens (bos, eos, unk)
        #   36+   → normal struct tokens (codebook entries)
        self.struct_vocab_offset = 36

    # ------------------------------------------------------------------
    # forward: standard token embedding for ALL tokens (no quant2emb)
    # ------------------------------------------------------------------

    def forward(self, input_ids, **kwargs):
        """Run the model with standard token embedding for struct tokens.

        Unlike dplm2_bit, struct tokens are looked up in the shared embedding
        table just like AA tokens — no quant2emb linear projection is applied.
        The output is routed to separate lm_heads by EsmForDPLM2LMHeads.
        """
        input_mask = input_ids.ne(self.pad_id)
        type_ids = self.get_modality_type(input_ids)

        L = input_ids.shape[1]
        num_heads = self.net.config.num_attention_heads
        attention_bias: torch.FloatTensor = (
            self.net.esm.get_extended_attention_mask(
                input_mask, input_ids.shape
            ).repeat(1, num_heads, L, 1)
        )

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

        # Standard token embedding for all tokens (struct IDs 36+ are valid
        # entries in the shared embedding table just like AA tokens).
        input_embeds = self.net.esm.embeddings(
            input_ids, attention_mask=input_mask
        )

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            output_hidden_states=True,
            type_ids=type_ids,
        )
        return outputs

    # ------------------------------------------------------------------
    # compute_loss: categorical cross-entropy for struct (not binary)
    # ------------------------------------------------------------------

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
        # [B, L, struct_vocab_size] — categorical, no reshape needed
        struct_logits = model_outputs["struct_logits"]

        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_weight = {
            "linear": (num_timesteps - (struct_noised["t"] - 1)),
            "constant": num_timesteps * torch.ones_like(struct_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        struct_weight = struct_weight.expand(struct_target.size())

        aatype_weight = {
            "linear": (num_timesteps - (aatype_noised["t"] - 1)),
            "constant": num_timesteps * torch.ones_like(aatype_noised["t"]),
        }[weighting][:, None].float() / num_timesteps
        aatype_weight = aatype_weight.expand(aatype_target.size())

        # Shift struct targets from unified vocab space [36, 36+8192)
        # to head output space [0, struct_vocab_size).
        # clamp(min=0) ensures bos/eos positions (IDs < 36) don't give
        # negative indices; their loss contribution is zeroed by struct_noised["mask"].
        struct_target_shifted = (struct_target - self.struct_vocab_offset).clamp(
            min=0
        )

        return (
            {"aatype": aatype_logits, "struct": struct_logits},
            {"aatype": aatype_target, "struct": struct_target_shifted},
            {
                "aatype": aatype_noised["mask"],
                "struct": struct_noised["mask"],
            },
            {"aatype": aatype_weight, "struct": struct_weight},
        )

    # ------------------------------------------------------------------
    # self_mixup: corrected version (no binary reshape, no tuple bug)
    # ------------------------------------------------------------------

    def self_mixup(self, x_t, single_modality_index, bsz, seq_len):
        # 1. first pass: predict tokens with current noised input
        with torch.no_grad():
            model_outputs = self.forward(
                input_ids=x_t, single_modality=single_modality_index
            )
            aatype_logits = model_outputs["aatype_logits"]
            struct_logits = model_outputs["struct_logits"]  # [B,L,struct_vocab_size]

        # 2. mixup: replace masked positions with model predictions
        prev_input_ids = x_t
        non_special_sym_mask = self.get_non_special_symbol_mask(prev_input_ids)

        # sample_from_logits returns (_tokens, _scores); take only tokens
        _tokens, _ = self.sample_from_logits(
            aatype_logits, struct_logits, temperature=0.0
        )
        model_pred = torch.where(non_special_sym_mask, _tokens, prev_input_ids)

        mixup_xt, mixup_loss_mask = self.get_mixup_xt(
            input_ids=prev_input_ids,
            model_pred=model_pred,
            non_special_sym_mask=non_special_sym_mask,
        )

        # 3. second pass: denoise the mixed input
        model_outputs = self.forward(
            input_ids=mixup_xt, single_modality=single_modality_index
        )
        return model_outputs, mixup_loss_mask

    # ------------------------------------------------------------------
    # forward_decoder: categorical struct sampling (no binary reshape)
    # ------------------------------------------------------------------

    def forward_decoder(
        self,
        prev_decoder_out,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@1.1:0.1",
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        net_out = self.forward(input_ids=output_tokens)

        aatype_logits = net_out["aatype_logits"]
        struct_logits = net_out["struct_logits"]  # [B, L, struct_vocab_size]
        attentions = net_out["attentions"] if need_attn_weights else None

        if aatype_logits.dtype != output_scores.dtype:
            aatype_logits = aatype_logits.type_as(output_scores)
            struct_logits = struct_logits.type_as(output_scores)

        # Restrict AA logits to the 20 standard amino acids (indices 4–23)
        aatype_logits[:, :, :4] = -math.inf
        aatype_logits[:, :, 24:] = -math.inf

        # top-p filtering for both modalities
        aatype_logits = top_k_top_p_filtering(aatype_logits, top_p=0.95)
        struct_logits = top_k_top_p_filtering(struct_logits, top_p=0.95)

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

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,
            step=step + 1,
            max_step=max_step,
            history=history,
            all_hidden_states=net_out["all_hidden_states"],
        )

    # ------------------------------------------------------------------
    # sample_from_logits: simple categorical for both modalities
    # ------------------------------------------------------------------

    def sample_from_logits(
        self, aatype_logits, struct_logits, temperature=1.0
    ):
        """Categorical sampling for AA and struct tokens.

        Unlike dplm2_bit, struct is sampled from a single (struct_vocab_size)-way
        categorical distribution. No bit-wise combination is needed.
        """
        _aatype_tokens, _aatype_scores = sample_from_categorical(
            aatype_logits, temperature=temperature
        )
        # struct_logits: [B, L, struct_vocab_size] → sample token index in [0, struct_vocab_size)
        _struct_tokens, _struct_scores = sample_from_categorical(
            struct_logits, temperature=temperature
        )
        # Shift struct token indices back to unified vocab space
        _struct_tokens = _struct_tokens + self.struct_vocab_offset

        _tokens = torch.concat([_struct_tokens, _aatype_tokens], dim=1)
        _scores = torch.concat([_struct_scores, _aatype_scores], dim=1)
        return _tokens, _scores

    # ------------------------------------------------------------------
    # generate() and prepare_for_struct_tokenizer() are inherited from
    # DPLM2Bit unchanged: both operate on integer struct token IDs with
    # vocab offset, which is exactly what this model produces.
    # ------------------------------------------------------------------
