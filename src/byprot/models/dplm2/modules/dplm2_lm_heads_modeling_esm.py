# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# dplm2_lm_heads_modeling_esm.py
# Ablation model: separate struct/AA lm_heads (like dplm2_bit),
# but struct tokens remain as discrete integers (like dplm2, no quant2emb).

from copy import deepcopy

import torch
import torch.nn as nn
from transformers.models.esm.modeling_esm import EsmForMaskedLM, EsmLMHead, EsmPreTrainedModel

from byprot.models import register_model

# Reuse all modified ESM internals (rotary emb, attention, encoder, etc.)
from byprot.models.dplm2.modules.dplm2_bit_modeling_esm import ModifiedEsmModel


@register_model("dplm2_lm_heads_esm")
class EsmForDPLM2LMHeads(EsmForMaskedLM):
    """ESM backbone with **separate** lm_heads for AA and struct tokens.

    Differences from EsmForDPLM2 (single-head):
        - Separate lm_head (AA) and lm_head_struct (struct).
        - forward() splits sequence output and routes each half to its head.

    Differences from EsmForDPLM2Bit (bit-model):
        - No quant2emb linear layer.
        - No learned struct_bos/eos/mask embeddings.
        - lm_head_struct outputs integer-level logits (struct_vocab_size classes),
          NOT binary (codebook_embed_dim * 2 classes).
        - Input struct tokens are embedded via the standard token embedding table.
    """

    def __init__(self, config, dropout: float = 0.1, struct_vocab_size: int = 8192):
        config.hidden_dropout_prob = dropout
        config.tie_word_embeddings = False
        EsmPreTrainedModel.__init__(self, config)

        self.esm = ModifiedEsmModel(config, add_pooling_layer=False)

        # AA output head — keep the canonical name "lm_head" for
        # compatibility with pretrained DPLM weight loading.
        self.lm_head = EsmLMHead(config)

        # Struct output head — categorical over struct_vocab_size tokens.
        struct_config = deepcopy(config)
        struct_config.vocab_size = struct_vocab_size
        self.lm_head_struct = EsmLMHead(struct_config)

        self.init_weights()
        self.pad_id = config.pad_token_id
        self.contact_head = None
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        type_ids=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)

        outputs = self.esm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            type_ids=type_ids,
        )

        sequence_output = outputs[0]  # [B, 2L, hidden]

        # First half = struct positions, second half = AA positions
        # (same layout convention as dplm2_bit)
        struct_hidden, aa_hidden = sequence_output.chunk(2, dim=1)
        logits_aatype = self.lm_head(aa_hidden)        # [B, L, aa_vocab]
        logits_struct = self.lm_head_struct(struct_hidden)  # [B, L, struct_vocab_size]

        result = {
            "aatype_logits": logits_aatype,
            "struct_logits": logits_struct,
            "last_hidden_state": sequence_output,
        }
        if output_hidden_states:
            result["all_hidden_states"] = outputs["hidden_states"]
        return result
