"""
Divergence Metrics for ITO (Information-Theoretic Optimization)

This module implements various divergence calculation methods to measure
the difference between conditional and unconditional predictions in Flux Dev.
"""

import torch
import torch.nn.functional as F
from typing import Literal, Optional
import math


class DivergenceCalculator:
    """
    Calculates divergence between conditional and unconditional predictions
    for adaptive guidance scaling in Flux Dev.
    """

    def __init__(
        self,
        divergence_type: Literal["l2", "cosine", "kl_approx", "frequency", "angular", "flow_angular"] = "l2",
        smoothing_window: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize divergence calculator.

        Args:
            divergence_type: Type of divergence metric to use
            smoothing_window: Window size for exponential moving average
            device: Device to run calculations on
        """
        self.divergence_type = divergence_type
        self.smoothing_window = smoothing_window
        self.device = device

        # EMA tracking
        self.ema_divergence = None
        self.alpha = 2.0 / (smoothing_window + 1.0)

        # Statistics for normalization
        self.divergence_history = []
        self.max_divergence_seen = 1e-6
        self.min_divergence_seen = float('inf')

    def calculate(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor,
        timestep: Optional[int] = None,
        timestep_float: Optional[float] = None
    ) -> float:
        """
        Calculate divergence between conditional and unconditional predictions.

        Args:
            pred_cond: Conditional prediction from the model (velocity field for Flux)
            pred_uncond: Unconditional prediction from the model (velocity field for Flux)
            timestep: Current timestep (optional, used for some metrics)
            timestep_float: Normalized timestep [0, 1] for flow matching (optional)

        Returns:
            Divergence value as a float
        """
        if self.divergence_type == "l2":
            divergence = self._l2_divergence(pred_cond, pred_uncond)
        elif self.divergence_type == "cosine":
            divergence = self._cosine_divergence(pred_cond, pred_uncond)
        elif self.divergence_type == "kl_approx":
            divergence = self._kl_approx_divergence(pred_cond, pred_uncond)
        elif self.divergence_type == "frequency":
            divergence = self._frequency_divergence(pred_cond, pred_uncond)
        elif self.divergence_type == "angular":
            divergence = self._angular_divergence(pred_cond, pred_uncond)
        elif self.divergence_type == "flow_angular":
            divergence = self._flow_angular_divergence(pred_cond, pred_uncond, timestep_float)
        else:
            raise ValueError(f"Unknown divergence type: {self.divergence_type}")

        # Update statistics
        self._update_statistics(divergence)

        # Apply EMA smoothing
        smoothed_divergence = self._apply_ema(divergence)

        return smoothed_divergence

    def _l2_divergence(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> float:
        """
        Calculate L2 norm divergence (Euclidean distance).
        Simple and efficient, works well for most cases.
        """
        diff = pred_cond - pred_uncond
        l2_norm = torch.norm(diff.flatten(), p=2)

        # Normalize by the number of elements for scale invariance
        normalized_divergence = l2_norm / math.sqrt(diff.numel())

        return normalized_divergence.item()

    def _cosine_divergence(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> float:
        """
        Calculate cosine similarity-based divergence.
        Scale-invariant and focuses on directional differences.
        """
        # Flatten tensors
        pred_cond_flat = pred_cond.flatten()
        pred_uncond_flat = pred_uncond.flatten()

        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(
            pred_cond_flat.unsqueeze(0),
            pred_uncond_flat.unsqueeze(0),
            dim=1
        )

        # Convert to divergence: 0 = identical, 2 = opposite
        divergence = 1.0 - cosine_sim

        return divergence.item()

    def _kl_approx_divergence(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> float:
        """
        Approximate KL divergence using Gaussian assumption.
        More theoretically grounded for probability distributions.
        """
        # Treat predictions as samples from distributions
        # Use simplified KL approximation: sum((p - q)^2 / (q + eps))
        eps = 1e-8

        diff_squared = (pred_cond - pred_uncond) ** 2
        normalized_diff = diff_squared / (torch.abs(pred_uncond) + eps)

        kl_approx = torch.mean(normalized_diff)

        return kl_approx.item()

    def _frequency_divergence(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> float:
        """
        Calculate divergence in frequency domain.
        Emphasizes high-frequency differences (details).
        """
        # Apply FFT to each channel
        # Expecting shape: [batch, channels, height, width]

        # Handle different tensor shapes
        if pred_cond.dim() == 4:
            # Get original spatial dimensions BEFORE FFT
            # This is important because rfft2 changes the last dimension size
            h, w = pred_cond.shape[-2:]

            # Process in spatial frequency domain
            fft_cond = torch.fft.rfft2(pred_cond, norm="ortho")
            fft_uncond = torch.fft.rfft2(pred_uncond, norm="ortho")

            # Calculate magnitude difference
            mag_diff = torch.abs(torch.abs(fft_cond) - torch.abs(fft_uncond))

            # Weight high frequencies more (optional)
            # Create frequency weighting mask using ORIGINAL spatial dimensions
            y_freq = torch.fft.fftfreq(h, device=mag_diff.device)[:, None]
            x_freq = torch.fft.rfftfreq(w, device=mag_diff.device)[None, :]
            freq_magnitude = torch.sqrt(y_freq**2 + x_freq**2)

            # Apply weighting: emphasize high frequencies
            weight = 1.0 + freq_magnitude
            weighted_diff = mag_diff * weight

            divergence = torch.mean(weighted_diff)
        else:
            # Fallback to L2 for non-4D tensors
            divergence = self._l2_divergence(pred_cond, pred_uncond)
            return divergence

        return divergence.item()

    def _angular_divergence(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> float:
        """
        Calculate angular divergence for flow matching models.
        Measures the direction change between velocity fields.

        For flow matching (like Flux), velocity fields are vectors that guide
        the denoising trajectory. Angular divergence captures how much the
        conditional and unconditional flows point in different directions.

        Returns: 0 = same direction, 2 = opposite directions
        """
        # Flatten tensors to treat as high-dimensional vectors
        v_cond = pred_cond.flatten(1)  # [batch, features]
        v_uncond = pred_uncond.flatten(1)

        # Calculate cosine similarity between velocity vectors
        # This gives us the angular relationship
        cos_sim = F.cosine_similarity(v_cond, v_uncond, dim=1)

        # Convert to angular divergence
        # cos_sim ranges from -1 (opposite) to 1 (same direction)
        # We want: 0 (same) to 2 (opposite)
        angular_div = 1.0 - cos_sim

        # Average across batch if needed
        if angular_div.numel() > 1:
            angular_div = angular_div.mean()

        return angular_div.item()

    def _flow_angular_divergence(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor,
        timestep_float: Optional[float] = None
    ) -> float:
        """
        Combined flow-matching divergence: angular + magnitude with phase weighting.

        This is specifically designed for flow matching models like Flux Dev:
        1. Angular divergence: Direction change between velocity fields
        2. Magnitude divergence: Difference in flow speeds
        3. Phase weighting: Early timesteps prioritize direction, late prioritize magnitude

        Args:
            pred_cond: Conditional velocity field
            pred_uncond: Unconditional velocity field
            timestep_float: Normalized timestep [0, 1] where 1 = start, 0 = end

        Returns:
            Combined divergence value
        """
        # Flatten to treat as vectors
        v_cond = pred_cond.flatten(1)
        v_uncond = pred_uncond.flatten(1)

        # 1. Angular divergence (direction change)
        cos_sim = F.cosine_similarity(v_cond, v_uncond, dim=1)
        angular_div = (1.0 - cos_sim).mean()  # 0 = same, 2 = opposite

        # 2. Magnitude divergence (speed difference)
        mag_cond = torch.norm(v_cond, p=2, dim=1)
        mag_uncond = torch.norm(v_uncond, p=2, dim=1)
        mag_diff = torch.abs(mag_cond - mag_uncond).mean()

        # Normalize magnitude difference by average magnitude
        avg_mag = (mag_cond.mean() + mag_uncond.mean()) / 2.0
        mag_div = mag_diff / (avg_mag + 1e-8)

        # 3. Phase-aware weighting
        if timestep_float is not None:
            # timestep_float: 1.0 = start of flow, 0.0 = end of flow
            # Early in flow (t close to 1): direction matters more
            # Late in flow (t close to 0): magnitude matters more
            phase_weight = timestep_float
        else:
            # Default to balanced weighting
            phase_weight = 0.5

        # Combine: early = more angular, late = more magnitude
        combined_div = phase_weight * angular_div + (1.0 - phase_weight) * mag_div

        return combined_div.item()

    def _update_statistics(self, divergence: float):
        """Update running statistics for normalization."""
        self.divergence_history.append(divergence)

        # Keep history bounded
        if len(self.divergence_history) > 100:
            self.divergence_history.pop(0)

        # Update min/max
        self.max_divergence_seen = max(self.max_divergence_seen, divergence)
        self.min_divergence_seen = min(self.min_divergence_seen, divergence)

    def _apply_ema(self, divergence: float) -> float:
        """Apply exponential moving average for smoothing."""
        if self.ema_divergence is None:
            self.ema_divergence = divergence
        else:
            self.ema_divergence = (
                self.alpha * divergence + (1 - self.alpha) * self.ema_divergence
            )

        return self.ema_divergence

    def get_normalized_divergence(self) -> float:
        """
        Get the current divergence normalized to [0, 1] range.

        Returns:
            Normalized divergence value
        """
        if self.ema_divergence is None:
            return 0.0

        # Avoid division by zero
        range_val = max(self.max_divergence_seen - self.min_divergence_seen, 1e-8)

        normalized = (self.ema_divergence - self.min_divergence_seen) / range_val

        # Clamp to [0, 1]
        return max(0.0, min(1.0, normalized))

    def reset(self):
        """Reset all statistics and history."""
        self.ema_divergence = None
        self.divergence_history = []
        self.max_divergence_seen = 1e-6
        self.min_divergence_seen = float('inf')

    def get_statistics(self) -> dict:
        """Get current divergence statistics for debugging."""
        return {
            "current_ema": self.ema_divergence,
            "normalized": self.get_normalized_divergence(),
            "min_seen": self.min_divergence_seen,
            "max_seen": self.max_divergence_seen,
            "history_length": len(self.divergence_history),
            "recent_mean": sum(self.divergence_history[-10:]) / len(self.divergence_history[-10:]) if self.divergence_history else 0.0
        }
