# Hybrid discrete-continuous diffusion for DPLM2.
#
# Implements hybrid diffusion (Continuous And Discrete diffusion, Pynadath et al. 2025)
# on top of DPLM2's multimodal protein language model.
#
# Key changes from standard DPLM2:
#   1. Hybrid noising: discrete masking + continuous Gaussian noise (instead of mask-only)
#   2. Soft embedding input: model receives noisy embeddings, not discrete mask tokens
#   3. Corruption bias + preconditioning for corrupted positions
#   4. Hybrid loss weighting: 1/(1 - alpha(t)) from the ELBO
#   5. Hybrid inference: continuous ODE step + discrete unmasking

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from byprot.models import register_model
from byprot.models.dplm2.hybrid_noise import HybridNoiseSchedule
from byprot.models.dplm2.dplm2 import (
    DPLM2Config,
    MultimodalDiffusionProteinLanguageModel,
)
from byprot.models.utils import sample_from_categorical

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HybridSpecificConfig:
    # Noise space: "onehot" adds Gaussian noise in one-hot space then maps
    # through the embedding table; "embedding" adds noise directly in the
    # d-dimensional embedding space.
    noise_space: str = field(default="embedding")

    # One-hot schedule params (target rank degradation range)
    r_min: float = field(default=0.01)
    r_max: float = field(default=0.25)

    # Embedding-space VE-SDE schedule params
    sigma_min: float = field(default=0.5)
    sigma_max: float = field(default=5.0)

    # Corruption bias mixing coefficient.
    # lambda=1 → pure noisy embedding; lambda=0 → pure bias (like mask token)
    lambda_bias: float = field(default=1.0)

    # Whether to add a learned sigma embedding to the input.
    # If True, a small MLP maps per-position sigma values to d-dimensional
    # vectors that are added to the embeddings (helps the model know noise level).
    use_sigma_embed: bool = field(default=False)

    # Whether to learn the corruption bias parameter.
    # If False, corruption_bias is frozen (requires_grad=False).
    learn_corruption_bias: bool = field(default=False)

    # Loss weighting: "hybrid" (ELBO-derived 1/t) or "linear" (original DPLM2)
    loss_weight: str = field(default="hybrid")


@dataclass
class DPLM2HybridConfig(DPLM2Config):
    hybrid: HybridSpecificConfig = field(default_factory=HybridSpecificConfig)
    # Whether ESM's embedding layer applies token_dropout (0.88 scaling).
    # Set False for hybrid since we provide soft embeddings, not discrete tokens.
    token_dropout: bool = field(default=False)
    # Cut residual connection in layer[0] attention for masked positions only.
    cutoff_layer0_attn_residual: bool = field(default=False)


# ---------------------------------------------------------------------------
# Masked residual module for layer-0 attention
# ---------------------------------------------------------------------------
class MaskedResidualSelfOutput(nn.Module):
    """Drop-in replacement for EsmSelfOutput that skips residual for masked positions."""

    def __init__(self, original_output):
        super().__init__()
        self.dense = original_output.dense
        self.dropout = original_output.dropout
        self._residual_mask = None  # [B, L] bool, True = masked (no residual)

    def set_residual_mask(self, mask):
        self._residual_mask = mask

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self._residual_mask is not None:
            scale = (~self._residual_mask).unsqueeze(-1).float()
            hidden_states = hidden_states + input_tensor * scale
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


# ---------------------------------------------------------------------------
# Sigma embedding module
# ---------------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding + MLP for per-position sigma conditioning."""

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half).float() / half
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Zero-init last layer so the initial output is zero → doesn't
        # disturb pre-trained weights at the start of finetuning.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: [B, L] per-position sigma values
        Returns:
            [B, L, d_model] sigma embeddings
        """
        args = sigma.unsqueeze(-1) * self.freqs  # [B, L, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@register_model("dplm2_hybrid")
class HybridDiffusionProteinLanguageModel(
    MultimodalDiffusionProteinLanguageModel
):
    _default_cfg = DPLM2HybridConfig()

    def __init__(self, cfg, net=None):
        # Handle loading from HuggingFace DPLM2 directly
        if net is None and getattr(cfg, "training_stage", "") == "finetune_from_dplm2_hf":
            net = self._load_dplm2_from_hf(cfg)

        super().__init__(cfg, net)

        # Apply token_dropout setting from config
        if hasattr(cfg, "token_dropout"):
            emb = self.net.esm.embeddings
            # Handle PEFT ModulesToSaveWrapper
            if hasattr(emb, "original_module"):
                emb.original_module.token_dropout = cfg.token_dropout
                for mod in emb.modules_to_save.values():
                    mod.token_dropout = cfg.token_dropout
            else:
                emb.token_dropout = cfg.token_dropout

        d_model = self.net.config.hidden_size

        # Corruption bias: learned embedding for corrupted positions.
        # Initialized from average of mask token embeddings.
        self.corruption_bias = nn.Parameter(
            torch.zeros(d_model),
            requires_grad=self.cfg.hybrid.learn_corruption_bias,
        )
        self._init_corruption_bias()

        # Noise schedule
        self.noise_schedule = HybridNoiseSchedule(
            num_timesteps=self.cfg.num_diffusion_timesteps,
            r_min=self.cfg.hybrid.r_min,
            r_max=self.cfg.hybrid.r_max,
            sigma_min=self.cfg.hybrid.sigma_min,
            sigma_max=self.cfg.hybrid.sigma_max,
        )

        # Lambda bias annealing state

        # Optional sigma embedding
        if self.cfg.hybrid.use_sigma_embed:
            self.sigma_embedding = TimestepEmbedding(d_model)
        else:
            self.sigma_embedding = None

        # Replace layer[0] attention output with masked-residual version
        if self.cfg.cutoff_layer0_attn_residual:
            layer0_attn = self._get_layer0_attention()
            layer0_attn.output = MaskedResidualSelfOutput(layer0_attn.output)

        # import ipdb; ipdb.set_trace()  # check init

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_dplm2_from_hf(cfg):
        """Load a pre-trained DPLM2 ESM backbone from HuggingFace."""
        from byprot.models.dplm2.modules.dplm2_modeling_esm import EsmForDPLM2

        hf_name = cfg.net.pretrained_model_name_or_path
        net = EsmForDPLM2.from_pretrained(hf_name)

        if cfg.lora.enable:
            lora_target_module = cfg.lora.lora_target_module
            modules_to_save = cfg.lora.modules_to_save.split(",")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=lora_target_module,
                modules_to_save=modules_to_save,
                inference_mode=False,
                r=cfg.lora.lora_rank,
                lora_alpha=32,
                lora_dropout=cfg.lora.lora_dropout,
            )
            net = get_peft_model(net, peft_config)

            if getattr(cfg.lora, "train_layer_norm", False):
                for name, param in net.named_parameters():
                    if "LayerNorm" in name:
                        param.requires_grad = True

        return net

    def _get_word_embeddings(self) -> torch.Tensor:
        """Get word embedding weight tensor, handling PEFT wrapping.

        When LoRA's ``modules_to_save`` includes ``esm.embeddings``, PEFT
        wraps the entire EsmEmbeddings in a ``ModulesToSaveWrapper``:

            net.base_model.model.esm.embeddings  (ModulesToSaveWrapper)
            ├── .original_module.word_embeddings.weight   ← frozen original
            └── .modules_to_save["default"].word_embeddings.weight  ← trainable

        We want the **trainable** copy used in the forward pass.
        """
        try:
            emb = self.net.base_model.model.esm.embeddings
            # PEFT ModulesToSaveWrapper: get the active trainable copy
            if hasattr(emb, "modules_to_save"):
                active = emb.modules_to_save[emb.active_adapter]
                return active.word_embeddings.weight
            # PEFT without modules_to_save on embeddings
            return emb.word_embeddings.weight
        except AttributeError:
            # Non-PEFT: direct access
            return self.net.esm.embeddings.word_embeddings.weight

    def _init_corruption_bias(self):
        """Initialize corruption bias from average of mask token embeddings."""
        try:
            with torch.no_grad():
                # W = self._get_word_embeddings()
                # mask_avg = (W[self.aa_mask_id] + W[self.struct_mask_id]) / 2
                # self.corruption_bias.copy_(mask_avg)
                self.corruption_bias.zero_()  # zero init for aligning dplm2
        except Exception:
            pass  # keep zero init

    def _get_layer0_attention(self):
        """Get layer[0].attention, handling PEFT wrapping."""
        try:
            return self.net.base_model.model.esm.encoder.layer[0].attention
        except AttributeError:
            return self.net.esm.encoder.layer[0].attention

    def get_lambda_bias(self):
        """Return fixed lambda_bias from config."""
        return self.cfg.hybrid.lambda_bias

    # ------------------------------------------------------------------
    # Hybrid noising
    # ------------------------------------------------------------------
    def hybrid_q_sample(self, x_0, t, type_ids, maskable_mask):
        """
        Hybrid noising: discrete masking + continuous Gaussian noise.

        Args:
            x_0:           [B, L] clean token indices
            t:             [B]    discrete timesteps in {0, ..., T}
            type_ids:      [B, L] modality type (0=struct, 1=aa, 2=pad)
            maskable_mask: [B, L] bool, True for positions eligible for corruption

        Returns:
            soft_embeds: [B, L, d] soft embeddings for model input
            mask_t:      [B, L]    bool, True for corrupted positions (loss positions)
            sigma:       [B]       sigma values per sample
        """
        B, L = x_0.shape
        device = x_0.device
        noise_space = self.cfg.hybrid.noise_space

        # 1. Discrete masking: each maskable position is corrupted with prob 1-alpha
        alpha_t = self.noise_schedule.get_alpha(t)  # [B]
        mask_prob = 1.0 - alpha_t  # [B]
        u = torch.rand(B, L, device=device)
        mask_t = (u < mask_prob[:, None]) & maskable_mask  # True = corrupted

        # 2. Get continuous noise level
        sigma = self.noise_schedule.get_sigma(t, noise_space)  # [B]

        # 3. Get clean embeddings for all positions
        W = self._get_word_embeddings()  # [V, d]
        # W[self.aa_mask_id] = torch.zeros_like(W[self.aa_mask_id])  # align with dplm2
        clean_embeds = W[x_0]  # [B, L, d]

        # 4. Compute noisy embeddings for corrupted positions
        if noise_space == "onehot":
            soft_embeds = self._apply_onehot_noise(
                x_0, clean_embeds, mask_t, type_ids, sigma, W
            )
        else:
            soft_embeds = self._apply_embedding_noise(
                clean_embeds, mask_t, sigma
            )

        # 5. Add per-position sigma embedding
        if self.sigma_embedding is not None:
            sigma_per_pos = torch.zeros(B, L, device=device)
            sigma_per_pos[mask_t] = sigma[
                mask_t.nonzero(as_tuple=True)[0]
            ]
            soft_embeds = soft_embeds + self.sigma_embedding(sigma_per_pos)

        return soft_embeds, mask_t, sigma

    def _apply_onehot_noise(self, x_0, clean_embeds, mask_t, type_ids, sigma, W):
        """
        One-hot space noising.

        For each corrupted position with original token x:
            noisy_embed = W[x] + sigma * (eps @ W_modality)
        where eps ~ N(0, I_{|V_modality|}).

        Then apply preconditioning and corruption bias:
            output = lambda * noisy_embed / sqrt(sigma^2+1) + (1-lambda) * bias
        """
        B, L, d = clean_embeds.shape
        device = clean_embeds.device
        lam = self.get_lambda_bias()
        soft_embeds = clean_embeds.clone()

        for modality_type, vocab_start, vocab_end in [
            (self.aa_type, 0, 33),
            (self.struct_type, 33, W.shape[0]),
        ]:
            mod_mask = (type_ids == modality_type) & mask_t
            if not mod_mask.any():
                continue

            V_mod = vocab_end - vocab_start
            W_mod = W[vocab_start:vocab_end]  # [V_mod, d]

            n = mod_mask.sum().item()
            eps = torch.randn(n, V_mod, device=device, dtype=clean_embeds.dtype)
            noise_emb = eps @ W_mod  # [n, d]

            batch_idx = mod_mask.nonzero(as_tuple=True)[0]
            sig = sigma[batch_idx].unsqueeze(-1)  # [n, 1]

            noisy = soft_embeds[mod_mask] + sig * noise_emb
            c_in = 1.0 / torch.sqrt(sig**2 + 1.0)
            corrupted = lam * noisy * c_in + (1 - lam) * self.corruption_bias
            soft_embeds[mod_mask] = corrupted

        return soft_embeds

    def _apply_embedding_noise(self, clean_embeds, mask_t, sigma):
        """
        Embedding space noising.

        For each corrupted position:
            noisy_embed = W[x] + sigma * eps,  eps ~ N(0, I_d)

        Then apply preconditioning and corruption bias:
            output = lambda * noisy_embed / sqrt(sigma^2+1) + (1-lambda) * bias
        """
        B, L, d = clean_embeds.shape
        device = clean_embeds.device
        lam = self.get_lambda_bias()
        soft_embeds = clean_embeds.clone()

        if not mask_t.any():
            return soft_embeds

        n = mask_t.sum().item()
        eps = torch.randn(n, d, device=device, dtype=clean_embeds.dtype) # / 12 ###### sihun : scale down noise for embedding space

        batch_idx = mask_t.nonzero(as_tuple=True)[0]
        sig = sigma[batch_idx].unsqueeze(-1)  # [n, 1]

        noisy = soft_embeds[mask_t] + sig * eps
        c_in = 1.0 / torch.sqrt(sig**2 + 1.0)
        corrupted = lam * noisy * c_in + (1 - lam) * self.corruption_bias
        soft_embeds[mask_t] = corrupted

        return soft_embeds

    # ------------------------------------------------------------------
    # Training: construct_x_t and compute_loss
    # ------------------------------------------------------------------
    def construct_x_t(self, struct_target, aatype_target):
        """
        Override: apply Hybrid noising instead of discrete mask-only.

        Returns soft embeddings per modality (not discrete noised tokens).
        """
        bsz = struct_target.size(0)
        device = struct_target.device

        # Sample timesteps
        struct_t = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1, (bsz,), device=device
        )
        aatype_t = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1, (bsz,), device=device
        )

        # Multi-task split (identical logic to parent)
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

        # Set t=0 for conditioning modality (no noise)
        struct_t = struct_t.masked_fill(inverse_folding_index, 0)
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])

        # Apply hybrid noising to struct half
        struct_type_id = self.get_modality_type(struct_target)
        struct_soft, struct_mask, struct_sigma = self.hybrid_q_sample(
            struct_target,
            struct_t,
            struct_type_id,
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )

        # Apply hybrid noising to AA half
        aa_type_id = self.get_modality_type(aatype_target)
        aa_soft, aa_mask, aa_sigma = self.hybrid_q_sample(
            aatype_target,
            aatype_t,
            aa_type_id,
            maskable_mask=self.get_non_special_symbol_mask(aatype_target),
        )

        return (
            {
                "t": struct_t,
                "soft_embeds": struct_soft,
                "mask": struct_mask,
                "sigma": struct_sigma,
            },
            {
                "t": aatype_t,
                "soft_embeds": aa_soft,
                "mask": aa_mask,
                "sigma": aa_sigma,
            },
            single_modality_index,
        )

    def compute_loss(self, batch, weighting=None):
        """
        Override: use Hybrid soft embeddings as model input and Hybrid loss weighting.

        The return signature matches the parent so that the existing
        StructAARDMCrossEntropyLoss criterion works unchanged.
        """
        if weighting is None:
            weighting = self.cfg.hybrid.loss_weight

        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]

        (
            struct_noised,
            aatype_noised,
            single_modality_index,
        ) = self.construct_x_t(struct_target, aatype_target)

        # Concatenate soft embeddings: [struct, aa] along seq dim
        soft_embeds = torch.cat(
            [struct_noised["soft_embeds"], aatype_noised["soft_embeds"]],
            dim=1,
        )

        # Reference input_ids for type_ids / attention computation
        ref_input_ids = torch.cat([struct_target, aatype_target], dim=1)

        # Set mask for layer-0 attention residual cutoff
        if self.cfg.cutoff_layer0_attn_residual:
            combined_mask = torch.cat(
                [struct_noised["mask"], aatype_noised["mask"]], dim=1
            )
            self._get_layer0_attention().output.set_residual_mask(combined_mask)

        # Forward pass with soft embeddings
        model_outputs = self.forward(
            input_ids=ref_input_ids,
            single_modality=single_modality_index,
            inputs_embeds_override=soft_embeds,
        )

        struct_logits, aatype_logits = model_outputs["logits"].chunk(
            2, dim=1
        )

        # Loss weights
        num_ts = self.cfg.num_diffusion_timesteps
        if weighting == "hybrid":
            # Hybrid ELBO weight: 1 / (1 - alpha(t)) = T / t_discrete
            struct_w = (
                num_ts / struct_noised["t"].float().clamp(min=1)
            )[:, None]
            aa_w = (
                num_ts / aatype_noised["t"].float().clamp(min=1)
            )[:, None]
        elif weighting == "linear":
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
    # Inference: hybrid decoding (continuous ODE + discrete unmasking)
    # ------------------------------------------------------------------
    def _init_hybrid_inference_state(self, output_tokens, output_masks):
        """Initialize noisy embedding cache and clean mask for Hybrid inference."""
        B, L = output_tokens.shape
        device = output_tokens.device
        W = self._get_word_embeddings()
        d = W.shape[1]
        noise_space = self.cfg.hybrid.noise_space

        t_init = torch.tensor(
            [self.cfg.num_diffusion_timesteps], device=device
        )
        sigma_init = self.noise_schedule.get_sigma(t_init, noise_space).item()

        type_ids = self.get_modality_type(output_tokens)

        if noise_space == "onehot":
            y_cache = torch.zeros(B, L, d, device=device, dtype=W.dtype)
            for mod_type, v_start, v_end in [
                (self.aa_type, 0, 33),
                (self.struct_type, 33, W.shape[0]),
            ]:
                pos = (type_ids == mod_type) & output_masks
                if not pos.any():
                    continue
                V_mod = v_end - v_start
                n = pos.sum().item()
                eps = torch.randn(n, V_mod, device=device, dtype=W.dtype)
                y_cache[pos] = sigma_init * (eps @ W[v_start:v_end])
        else:
            y_cache = sigma_init * torch.randn(
                B, L, d, device=device, dtype=W.dtype
            ) / 30  ###### sihun: scale down noise for embedding space / case we are trained by (sigma_min, sigma_max) =(0.5, 5.0) and want to scale noise down for inference by 1/12
 
        clean_mask = ~output_masks
        return y_cache, clean_mask

    def _hybrid_forward_step(
        self, output_tokens, y_cache, clean_mask, output_masks, sigma_curr,
        temperature, ref_input_ids=None,
    ):
        """
        Single Hybrid forward pass shared by forward_decoder and generate.

        Constructs Y_t from clean/cached embeddings → model forward → sample.

        Args:
            ref_input_ids: Optional [B, L] token indices to use as input_ids
                for the forward pass. When provided, these should not contain
                mask tokens so that ESM's token_dropout applies only the 0.88
                scaling (matching training) without zeroing any positions.
                If None, falls back to output_tokens.

        Returns:
            _tokens:  [B, L] sampled token indices
            _scores:  [B, L] log-prob confidence of sampled tokens
            net_out:  dict with 'logits', 'last_hidden_state', 'log_probs'
        """
        if ref_input_ids is None:
            ref_input_ids = output_tokens

        W = self._get_word_embeddings()
        B, L = output_tokens.shape
        device = output_tokens.device

        # 1. Construct Y_t
        clean_embeds = W[output_tokens]
        clean_f = clean_mask.unsqueeze(-1).float()
        Y_t = clean_f * clean_embeds + (1 - clean_f) * y_cache

        # Preconditioning + corruption bias
        corrupted = ~clean_mask & output_masks
        if corrupted.any() and sigma_curr > 0:
            lam = self.get_lambda_bias()
            c_in = 1.0 / math.sqrt(sigma_curr**2 + 1.0)
            precond = (
                lam * Y_t[corrupted] * c_in
                + (1 - lam) * self.corruption_bias
            )
            Y_t = Y_t.clone()
            Y_t[corrupted] = precond

        # Sigma embedding (added after preconditioning, matching training)
        if self.sigma_embedding is not None:
            sigma_per_pos = torch.where(
                clean_mask,
                torch.zeros(B, L, device=device),
                torch.full((B, L), sigma_curr, device=device),
            )
            Y_t = Y_t + self.sigma_embedding(sigma_per_pos)

        # Set mask for layer-0 attention residual cutoff
        if self.cfg.cutoff_layer0_attn_residual:
            corrupted_mask = ~clean_mask & output_masks
            self._get_layer0_attention().output.set_residual_mask(corrupted_mask)

        # 2. Forward pass — use ref_input_ids (mask-free) so that ESM's
        #    token_dropout applies the 0.88 scaling without zeroing positions.
        net_out = self.forward(
            input_ids=ref_input_ids,
            inputs_embeds_override=Y_t,
        )
        logits = net_out["logits"]

        # Modality-specific vocab masking (use output_tokens for type_ids)
        type_ids = self.get_modality_type(output_tokens)
        idx_aa = torch.where(type_ids.eq(self.aa_type) & output_masks)
        idx_struct = torch.where(
            type_ids.eq(self.struct_type) & output_masks
        )
        logits[idx_aa[0], idx_aa[1], 33:] = -math.inf
        logits[idx_struct[0], idx_struct[1], :33] = -math.inf
        logits[..., self.special_token_list] = -math.inf

        # 3. Sample
        log_probs = logits.log_softmax(dim=-1)
        _tokens, _scores = sample_from_categorical(
            log_probs, temperature=temperature
        )
        net_out["log_probs"] = log_probs
        # import ipdb; ipdb.set_trace()
        return _tokens, _scores, net_out

    def _select_topk_to_unmask(self, _scores, still_corrupted, k_per_sample):
        """
        Select top-k most confident positions among corrupted ones to unmask.

        Args:
            _scores:          [B, L] log-prob confidence
            still_corrupted:  [B, L] bool mask of positions still corrupted
            k_per_sample:     [B] number of positions to unmask per sample

        Returns:
            newly_clean: [B, L] bool mask of positions to unmask this step
        """
        B, L = _scores.shape
        device = _scores.device

        scores_for_topk = _scores.clone()
        scores_for_topk[~still_corrupted] = -1e9

        sorted_scores, sorted_idx = scores_for_topk.sort(
            dim=1, descending=True
        )
        j_range = torch.arange(L, device=device).unsqueeze(0)
        topk_mask_sorted = j_range < k_per_sample.unsqueeze(1)

        newly_clean = torch.zeros_like(still_corrupted)
        newly_clean.scatter_(1, sorted_idx, topk_mask_sorted)
        newly_clean = newly_clean & still_corrupted
        return newly_clean

    def generate(
        self,
        input_tokens,
        max_iter=None,
        temperature=1.0,
        partial_masks=None,
        sampling_strategy="annealing@2.0:0.1",
        **kwargs,
    ):
        """
        Hybrid generation loop.

        At each step:
          1. Forward pass → logits → sample tokens
          2. Discrete step: unmask top-k most confident corrupted positions
          3. Continuous step: ODE-refine y_cache for still-corrupted positions

        Similar to DPLM2.generate() but replaces discrete mask-only decoding
        with hybrid continuous-discrete decoding.
        """
        self.eval()
        T = self.cfg.num_diffusion_timesteps
        noise_space = self.cfg.hybrid.noise_space
        device = input_tokens.device

        # 0. Initialize: all maskable positions → mask tokens
        output_tokens, output_scores = self.initialize_output_tokens(
            input_tokens, partial_masks=partial_masks
        )
        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        # Hybrid state
        y_cache, clean_mask = self._init_hybrid_inference_state(
            output_tokens, output_masks
        )
        history = [output_tokens.clone()]

        # ref_tokens: mask-free version of output_tokens used as input_ids
        # so that ESM's token_dropout applies 0.88 scaling (matching training)
        # without zeroing any positions. Initialized from output_tokens;
        # corrupted positions are updated with model predictions each step.
        ref_tokens = output_tokens.clone()

        for step in range(max_iter):
            is_final = step == max_iter - 1

            # Timestep mapping (decreasing from T to 0)
            t_curr = max(int((1.0 - step / max_iter) * T), 1)
            t_next = max(int((1.0 - (step + 1) / max_iter) * T), 0)
            sigma_curr = self.noise_schedule.get_sigma(
                torch.tensor([t_curr], device=device), noise_space
            ).item()
            sigma_next = self.noise_schedule.get_sigma(
                torch.tensor([t_next], device=device), noise_space
            ).item()
            alpha_curr = 1.0 - t_curr / T
            alpha_next = 1.0 - t_next / T

            # Temperature annealing
            if sampling_strategy.startswith("annealing"):
                max_temp, min_temp = map(
                    float, sampling_strategy.split("@")[1].split(":")
                )
                rate = 1.0 - step / max_iter
                cur_temp = min_temp + (max_temp - min_temp) * rate
            else:
                cur_temp = temperature

            # 1. Forward pass → logits → sample tokens
            with torch.no_grad():
                _tokens, _scores, net_out = self._hybrid_forward_step(
                    output_tokens, y_cache, clean_mask,
                    output_masks, sigma_curr, cur_temp,
                    ref_input_ids=ref_tokens,
                )

            # Update ref_tokens with predictions for all maskable positions
            ref_tokens[output_masks] = _tokens[output_masks]

            # 2. Discrete step: unmask top-k by confidence
            still_corrupted = ~clean_mask & output_masks
            # import ipdb; ipdb.set_trace()

            if is_final:
                newly_clean = still_corrupted
            elif still_corrupted.any():
                unmask_rate = (alpha_next - alpha_curr) / (
                    1.0 - alpha_curr + 1e-8
                )
                n_corrupted = still_corrupted.sum(dim=1)
                k_per_sample = (
                    unmask_rate * n_corrupted.float()
                ).ceil().long().clamp(min=1)
                newly_clean = self._select_topk_to_unmask(
                    _scores, still_corrupted, k_per_sample,
                )
            else:
                newly_clean = torch.zeros_like(still_corrupted)
            # import ipdb; ipdb.set_trace()

            # Commit tokens at newly unmasked positions
            output_tokens[newly_clean] = _tokens[newly_clean]
            output_scores[newly_clean] = _scores[newly_clean]
            clean_mask = clean_mask | newly_clean

            # 3. Continuous ODE step for still-corrupted positions
            still_corrupted = ~clean_mask & output_masks
            if still_corrupted.any() and sigma_curr > 0 and sigma_next > 0:
                W = self._get_word_embeddings()
                E_Y0 = W[_tokens]
                score = (y_cache - E_Y0) / (sigma_curr**2)
                dt = 0.5 * (sigma_curr**2 - sigma_next**2)
                # dt = 0.5 * (sigma_curr - sigma_next)
                y_cache = y_cache - dt * score

            history.append(output_tokens.clone())

        return {"output_tokens": output_tokens}
