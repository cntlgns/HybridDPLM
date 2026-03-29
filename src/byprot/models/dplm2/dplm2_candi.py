# CANDI hybrid discrete-continuous diffusion for DPLM2.
#
# Implements CANDI (Continuous And Discrete diffusion, Pynadath et al. 2025)
# on top of DPLM2's multimodal protein language model.
#
# Key changes from standard DPLM2:
#   1. Hybrid noising: discrete masking + continuous Gaussian noise (instead of mask-only)
#   2. Soft embedding input: model receives noisy embeddings, not discrete mask tokens
#   3. Corruption bias + preconditioning for corrupted positions
#   4. CANDI loss weighting: 1/(1 - alpha(t)) from the ELBO
#   5. Hybrid inference: continuous ODE step + discrete unmasking

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from byprot.models import register_model
from byprot.models.dplm2.candi_noise import CANDINoiseSchedule
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
class CANDISpecificConfig:
    # Noise space: "onehot" adds Gaussian noise in one-hot space then maps
    # through the embedding table; "embedding" adds noise directly in the
    # d-dimensional embedding space.
    noise_space: str = field(default="embedding")

    # One-hot schedule params (target rank degradation range)
    r_min: float = field(default=0.01)
    r_max: float = field(default=0.25)

    # Embedding-space VE-SDE schedule params
    sigma_min: float = field(default=0.01)
    sigma_max: float = field(default=2.0)

    # Corruption bias mixing coefficient.
    # lambda=0 → pure noisy embedding; lambda=1 → pure bias (like mask token)
    lambda_bias: float = field(default=0.5)

    # Whether to add a learned sigma embedding to the input.
    # If True, a small MLP maps per-position sigma values to d-dimensional
    # vectors that are added to the embeddings (helps the model know noise level).
    use_sigma_embed: bool = field(default=True)

    # Loss weighting: "candi" (ELBO-derived 1/t) or "linear" (original DPLM2)
    loss_weight: str = field(default="candi")


@dataclass
class DPLM2CANDIConfig(DPLM2Config):
    candi: CANDISpecificConfig = field(default_factory=CANDISpecificConfig)


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
@register_model("dplm2_candi")
class CANDIDiffusionProteinLanguageModel(
    MultimodalDiffusionProteinLanguageModel
):
    _default_cfg = DPLM2CANDIConfig()

    def __init__(self, cfg, net=None):
        # Handle loading from HuggingFace DPLM2 directly
        if net is None and getattr(cfg, "training_stage", "") == "finetune_from_dplm2_hf":
            net = self._load_dplm2_from_hf(cfg)

        super().__init__(cfg, net)

        d_model = self.net.config.hidden_size

        # Corruption bias: learned embedding for corrupted positions.
        # Initialized from average of mask token embeddings.
        self.corruption_bias = nn.Parameter(torch.zeros(d_model))
        self._init_corruption_bias()

        # Noise schedule
        self.noise_schedule = CANDINoiseSchedule(
            num_timesteps=self.cfg.num_diffusion_timesteps,
            r_min=self.cfg.candi.r_min,
            r_max=self.cfg.candi.r_max,
            sigma_min=self.cfg.candi.sigma_min,
            sigma_max=self.cfg.candi.sigma_max,
        )

        # Optional sigma embedding
        if self.cfg.candi.use_sigma_embed:
            self.sigma_embedding = TimestepEmbedding(d_model)
        else:
            self.sigma_embedding = None

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

        return net

    def _get_word_embeddings(self) -> torch.Tensor:
        """Get word embedding weight tensor, handling PEFT wrapping."""
        try:
            # PEFT-wrapped: PeftModel -> LoraModel -> original model
            return self.net.base_model.model.esm.embeddings.word_embeddings.weight
        except AttributeError:
            # Non-PEFT: direct access
            return self.net.esm.embeddings.word_embeddings.weight

    def _init_corruption_bias(self):
        """Initialize corruption bias from average of mask token embeddings."""
        try:
            with torch.no_grad():
                W = self._get_word_embeddings()
                mask_avg = (W[self.aa_mask_id] + W[self.struct_mask_id]) / 2
                self.corruption_bias.copy_(mask_avg)
        except Exception:
            pass  # keep zero init

    # ------------------------------------------------------------------
    # CANDI hybrid noising
    # ------------------------------------------------------------------
    def candi_q_sample(self, x_0, t, type_ids, maskable_mask):
        """
        CANDI hybrid noising: discrete masking + continuous Gaussian noise.

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
        noise_space = self.cfg.candi.noise_space

        # 1. Discrete masking: each maskable position is corrupted with prob 1-alpha
        alpha_t = self.noise_schedule.get_alpha(t)  # [B]
        mask_prob = 1.0 - alpha_t  # [B]
        u = torch.rand(B, L, device=device)
        mask_t = (u < mask_prob[:, None]) & maskable_mask  # True = corrupted

        # 2. Get continuous noise level
        sigma = self.noise_schedule.get_sigma(t, noise_space)  # [B]

        # 3. Get clean embeddings for all positions
        W = self._get_word_embeddings()  # [V, d]
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
            output = (1-lambda) * noisy_embed / sqrt(sigma^2+1) + lambda * bias
        """
        B, L, d = clean_embeds.shape
        device = clean_embeds.device
        lam = self.cfg.candi.lambda_bias
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
            corrupted = (1 - lam) * noisy * c_in + lam * self.corruption_bias
            soft_embeds[mod_mask] = corrupted

        return soft_embeds

    def _apply_embedding_noise(self, clean_embeds, mask_t, sigma):
        """
        Embedding space noising.

        For each corrupted position:
            noisy_embed = W[x] + sigma * eps,  eps ~ N(0, I_d)

        Then apply preconditioning and corruption bias:
            output = (1-lambda) * noisy_embed / sqrt(sigma^2+1) + lambda * bias
        """
        B, L, d = clean_embeds.shape
        device = clean_embeds.device
        lam = self.cfg.candi.lambda_bias
        soft_embeds = clean_embeds.clone()

        if not mask_t.any():
            return soft_embeds

        n = mask_t.sum().item()
        eps = torch.randn(n, d, device=device, dtype=clean_embeds.dtype)

        batch_idx = mask_t.nonzero(as_tuple=True)[0]
        sig = sigma[batch_idx].unsqueeze(-1)  # [n, 1]

        noisy = soft_embeds[mask_t] + sig * eps
        c_in = 1.0 / torch.sqrt(sig**2 + 1.0)
        corrupted = (1 - lam) * noisy * c_in + lam * self.corruption_bias
        soft_embeds[mask_t] = corrupted

        return soft_embeds

    # ------------------------------------------------------------------
    # Training: construct_x_t and compute_loss
    # ------------------------------------------------------------------
    def construct_x_t(self, struct_target, aatype_target):
        """
        Override: apply CANDI hybrid noising instead of discrete mask-only.

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

        # Apply CANDI noising to struct half
        struct_type_id = self.get_modality_type(struct_target)
        struct_soft, struct_mask, struct_sigma = self.candi_q_sample(
            struct_target,
            struct_t,
            struct_type_id,
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )

        # Apply CANDI noising to AA half
        aa_type_id = self.get_modality_type(aatype_target)
        aa_soft, aa_mask, aa_sigma = self.candi_q_sample(
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
        Override: use CANDI soft embeddings as model input and CANDI loss weighting.

        The return signature matches the parent so that the existing
        StructAARDMCrossEntropyLoss criterion works unchanged.
        """
        if weighting is None:
            weighting = self.cfg.candi.loss_weight

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
        if weighting == "candi":
            # CANDI ELBO weight: 1 / (1 - alpha(t)) = T / t_discrete
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
    def _init_candi_inference_state(self, output_tokens, output_masks):
        """
        Initialize CANDI inference state: noisy embedding cache and clean mask.

        Called on the first decoding step.
        """
        B, L = output_tokens.shape
        device = output_tokens.device
        W = self._get_word_embeddings()
        d = W.shape[1]
        noise_space = self.cfg.candi.noise_space

        # Initial sigma at t = T (max noise)
        t_init = torch.tensor(
            [self.cfg.num_diffusion_timesteps], device=device
        )
        sigma_init = self.noise_schedule.get_sigma(t_init, noise_space).item()

        # Sample initial noisy embeddings for corrupted positions
        type_ids = self.get_modality_type(output_tokens)

        if noise_space == "onehot":
            y_cache = torch.zeros(B, L, d, device=device)
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
            y_cache = sigma_init * torch.randn(B, L, d, device=device, dtype=W.dtype)

        # Clean mask: special tokens start as "clean"
        clean_mask = ~output_masks

        return y_cache, clean_mask

    def forward_decoder(
        self,
        prev_decoder_out,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@2.2:1.0",
        feedforward_mode="discrete",
        mask_emb_mode="add",
        linear_kwargs=None,
    ):
        """
        CANDI hybrid inference step.

        Combines:
        1. Continuous ODE step (refine noisy embeddings via score)
        2. Discrete unmasking step (commit tokens at selected positions)

        State is carried across steps via y_cache and clean_mask in the
        decoder output dict.
        """
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step = prev_decoder_out["step"]
        max_step = prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        # Map decoding step → CANDI timestep (decreasing from T to 0)
        T = self.cfg.num_diffusion_timesteps
        noise_space = self.cfg.candi.noise_space
        t_disc_curr = max(int((1.0 - step / max_step) * T), 1)
        t_disc_next = max(int((1.0 - (step + 1) / max_step) * T), 0)

        sigma_curr = self.noise_schedule.get_sigma(
            torch.tensor([t_disc_curr], device=output_tokens.device),
            noise_space,
        ).item()
        sigma_next = self.noise_schedule.get_sigma(
            torch.tensor([t_disc_next], device=output_tokens.device),
            noise_space,
        ).item()

        alpha_curr = 1.0 - t_disc_curr / T
        alpha_next = 1.0 - t_disc_next / T

        # Initialize state on first step
        if "y_cache" not in prev_decoder_out or prev_decoder_out.get("y_cache") is None:
            y_cache, clean_mask = self._init_candi_inference_state(
                output_tokens, output_masks
            )
        else:
            y_cache = prev_decoder_out["y_cache"]
            clean_mask = prev_decoder_out["clean_mask"]

        W = self._get_word_embeddings()

        # 1. Construct Y_t: clean positions use token embeds, corrupted use cache
        clean_embeds = W[output_tokens]  # [B, L, d]
        clean_f = clean_mask.unsqueeze(-1).float()
        Y_t = clean_f * clean_embeds + (1 - clean_f) * y_cache

        # Add sigma embedding
        if self.sigma_embedding is not None:
            B, L = output_tokens.shape
            sigma_per_pos = torch.where(
                clean_mask,
                torch.zeros(B, L, device=output_tokens.device),
                torch.full((B, L), sigma_curr, device=output_tokens.device),
            )
            Y_t = Y_t + self.sigma_embedding(sigma_per_pos)

        # Apply preconditioning + corruption bias to corrupted positions
        corrupted = ~clean_mask & output_masks
        if corrupted.any() and sigma_curr > 0:
            lam = self.cfg.candi.lambda_bias
            c_in = 1.0 / math.sqrt(sigma_curr**2 + 1.0)
            precond = (1 - lam) * Y_t[corrupted] * c_in + lam * self.corruption_bias
            Y_t = Y_t.clone()
            Y_t[corrupted] = precond

        # 2. Forward pass
        net_out = self.forward(
            input_ids=output_tokens,
            inputs_embeds_override=Y_t,
        )
        logits = net_out["logits"]

        # Modality-specific vocab masking
        type_ids = self.get_modality_type(output_tokens)
        aa_pos = type_ids.eq(self.aa_type) & output_masks
        struct_pos = type_ids.eq(self.struct_type) & output_masks
        idx_aa = torch.where(aa_pos)
        idx_struct = torch.where(struct_pos)
        logits[idx_aa[0], idx_aa[1], 33:] = -math.inf
        logits[idx_struct[0], idx_struct[1], :33] = -math.inf
        logits[..., self.special_token_list] = -math.inf

        # 3. Sample tokens from logits
        log_probs = logits.log_softmax(dim=-1)
        if sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate

        _tokens, _scores = sample_from_categorical(
            log_probs, temperature=temperature
        )

        # 4. Discrete unmasking step
        still_corrupted = ~clean_mask & output_masks
        if alpha_curr < 1.0 and t_disc_next < t_disc_curr:
            unmask_prob = (alpha_next - alpha_curr) / (1.0 - alpha_curr + 1e-8)
            u = torch.rand_like(output_tokens.float())
            newly_clean = (u < unmask_prob) & still_corrupted

            output_tokens[newly_clean] = _tokens[newly_clean]
            output_scores[newly_clean] = _scores[newly_clean]
            clean_mask = clean_mask | newly_clean

        # 5. Continuous ODE step for still-corrupted positions
        still_corrupted = ~clean_mask & output_masks
        if still_corrupted.any() and sigma_curr > 0 and sigma_next > 0:
            E_Y0 = W[_tokens]  # Monte Carlo estimate of E[Y_0|Y_t]
            score = -(y_cache - E_Y0) / (sigma_curr**2)
            dt = 0.5 * (sigma_curr**2 - sigma_next**2)
            y_cache = y_cache - dt * score

        # 6. Final step: unmask everything
        if t_disc_next == 0:
            remaining = ~clean_mask & output_masks
            output_tokens[remaining] = _tokens[remaining]
            output_scores[remaining] = _scores[remaining]
            clean_mask = clean_mask | remaining

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=None,
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out["last_hidden_state"],
            logits=log_probs,
            y_cache=y_cache,
            clean_mask=clean_mask,
            temperature=temperature,
        )
