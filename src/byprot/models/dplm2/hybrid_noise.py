# Hybrid noise schedule for hybrid discrete-continuous diffusion.
#
# Implements the noise schedule from:
#   Hybrid: Continuous And Discrete Diffusion (Pynadath et al., 2025)
#
# Key idea: decouple discrete masking (alpha) from continuous Gaussian noise (sigma)
# to avoid temporal dissonance between discrete identity corruption and
# continuous rank degradation.

import math

import torch


class HybridNoiseSchedule:
    """
    Noise schedule for hybrid diffusion.

    Manages both:
    - Discrete masking rate: alpha(t) = 1 - t/T  (log-linear schedule)
    - Continuous noise level: sigma(t) derived from target rank degradation r*(t)

    The one-hot sigma is computed from Eq. 10 in the paper:
        r*(t) = r_min + (r_max - r_min) * t
        sigma(t) = -1 / (Phi_inv(r*(t)) * sqrt(2))

    The embedding sigma uses a VE-SDE schedule:
        sigma(t) = sigma_min * (sigma_max / sigma_min)^t
    """

    def __init__(
        self,
        num_timesteps: int = 500,
        r_min: float = 0.01,
        r_max: float = 0.25,
        sigma_min: float = 0.01,
        sigma_max: float = 2.0,
    ):
        self.num_timesteps = num_timesteps
        self.r_min = r_min
        self.r_max = r_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self._sigmas_onehot = self._compute_onehot_sigmas()
        self._sigmas_embed = self._compute_embed_sigmas()

    def _compute_onehot_sigmas(self) -> torch.Tensor:
        """
        Compute sigma(t) from target continuous rank degradation r*(t).

        r*(t) = r_min + (r_max - r_min) * t_continuous
        sigma(t) = -1 / (Phi_inv(r*(t)) * sqrt(2))

        where Phi_inv(r) = sqrt(2) * erfinv(2*r - 1).

        Returns: [T+1] tensor of sigma values for t in {0, 1, ..., T}
        """
        T = self.num_timesteps
        t_cont = torch.arange(0, T + 1).double() / T  # [0, 1]

        r_star = self.r_min + (self.r_max - self.r_min) * t_cont
        r_star = torch.clamp(r_star, 1e-6, 0.5 - 1e-6)

        # Phi_inv(r) = sqrt(2) * erfinv(2*r - 1)
        phi_inv = math.sqrt(2) * torch.erfinv(2 * r_star - 1)

        # sigma = -1 / (Phi_inv(r) * sqrt(2))
        sigmas = -1.0 / (phi_inv * math.sqrt(2))
        sigmas[0] = 0.0  # t=0: no noise

        return sigmas.float()

    def _compute_embed_sigmas(self) -> torch.Tensor:
        """
        VE-SDE schedule for embedding-space diffusion.
        sigma(t) = sigma_min * (sigma_max / sigma_min)^t

        Returns: [T+1] tensor
        """
        T = self.num_timesteps
        t_cont = torch.arange(0, T + 1).float() / T
        sigmas = self.sigma_min * (self.sigma_max / self.sigma_min) ** t_cont
        sigmas[0] = 0.0
        return sigmas

    def get_sigma(
        self, t_discrete: torch.Tensor, noise_space: str = "onehot"
    ) -> torch.Tensor:
        """
        Get sigma for discrete timestep(s).

        Args:
            t_discrete: integer timesteps in {0, ..., T}, shape [B] or scalar
            noise_space: "onehot" or "embedding"

        Returns:
            sigma values with same shape as t_discrete
        """
        if noise_space == "onehot":
            sigmas = self._sigmas_onehot
        else:
            sigmas = self._sigmas_embed
        return sigmas.to(t_discrete.device)[t_discrete]

    def get_alpha(self, t_discrete: torch.Tensor) -> torch.Tensor:
        """Discrete keep rate alpha(t) = 1 - t/T."""
        return 1.0 - t_discrete.float() / self.num_timesteps
