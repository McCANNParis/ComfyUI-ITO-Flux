"""
ITO (Information-Theoretic Optimization) Sampler for Flux Dev

This module implements the core ITO algorithm that dynamically adjusts
guidance strength during the denoising process based on divergence metrics.
"""

import torch
import math
from typing import Callable, Optional, Literal, Dict, Any, Tuple
from .divergence_metrics import DivergenceCalculator


class AdaptiveGuidanceScheduler:
    """
    Manages adaptive guidance scaling based on divergence measurements.
    """

    def __init__(
        self,
        guidance_min: float = 1.0,
        guidance_max: float = 3.5,
        schedule_type: Literal["sigmoid", "linear", "exponential", "polynomial"] = "sigmoid",
        sensitivity: float = 1.0,
        warmup_steps: int = 0,
    ):
        """
        Initialize the adaptive guidance scheduler.

        Args:
            guidance_min: Minimum guidance scale to use
            guidance_max: Maximum guidance scale to use
            schedule_type: Type of schedule function
            sensitivity: How responsive guidance is to divergence (higher = more responsive)
            warmup_steps: Number of steps before adaptive guidance kicks in
        """
        self.guidance_min = guidance_min
        self.guidance_max = guidance_max
        self.schedule_type = schedule_type
        self.sensitivity = sensitivity
        self.warmup_steps = warmup_steps

        self.current_step = 0

    def get_guidance_scale(
        self,
        normalized_divergence: float,
        timestep: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """
        Calculate the adaptive guidance scale based on divergence.

        Args:
            normalized_divergence: Divergence value normalized to [0, 1]
            timestep: Current timestep
            total_steps: Total number of steps

        Returns:
            Guidance scale to use
        """
        # During warmup, use max guidance
        if self.current_step < self.warmup_steps:
            return self.guidance_max

        # Apply sensitivity
        adjusted_divergence = normalized_divergence * self.sensitivity

        # Map divergence to guidance scale using selected schedule
        if self.schedule_type == "sigmoid":
            guidance = self._sigmoid_schedule(adjusted_divergence)
        elif self.schedule_type == "linear":
            guidance = self._linear_schedule(adjusted_divergence)
        elif self.schedule_type == "exponential":
            guidance = self._exponential_schedule(adjusted_divergence)
        elif self.schedule_type == "polynomial":
            guidance = self._polynomial_schedule(adjusted_divergence)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Incorporate timestep-based weighting if available
        if timestep is not None and total_steps is not None:
            guidance = self._apply_timestep_weighting(guidance, timestep, total_steps)

        return guidance

    def _sigmoid_schedule(self, divergence: float) -> float:
        """
        Sigmoid-based smooth transition from min to max guidance.
        High divergence -> high guidance, low divergence -> low guidance.
        """
        # Sigmoid centered at 0.5, with steepness controlled by sensitivity
        # Map [0, 1] divergence to guidance range
        x = (divergence - 0.5) * 10  # Center and scale
        sigmoid = 1.0 / (1.0 + math.exp(-x))

        guidance = self.guidance_min + (self.guidance_max - self.guidance_min) * sigmoid
        return guidance

    def _linear_schedule(self, divergence: float) -> float:
        """
        Simple linear mapping from divergence to guidance.
        """
        guidance = self.guidance_min + (self.guidance_max - self.guidance_min) * divergence
        return guidance

    def _exponential_schedule(self, divergence: float) -> float:
        """
        Exponential schedule: more aggressive response to high divergence.
        """
        # Use exponential curve: guidance = min + (max - min) * exp(k * divergence)
        # Normalize so that exp(k) - 1 maps to the full range
        k = 2.0  # Exponential factor
        exp_val = (math.exp(k * divergence) - 1.0) / (math.exp(k) - 1.0)

        guidance = self.guidance_min + (self.guidance_max - self.guidance_min) * exp_val
        return guidance

    def _polynomial_schedule(self, divergence: float) -> float:
        """
        Polynomial schedule: quadratic response for more control.
        """
        # Use quadratic: divergence^2 for smoother low-end, sharper high-end
        poly_val = divergence ** 2

        guidance = self.guidance_min + (self.guidance_max - self.guidance_min) * poly_val
        return guidance

    def _apply_timestep_weighting(
        self,
        guidance: float,
        timestep: int,
        total_steps: int
    ) -> float:
        """
        Apply timestep-based weighting to guidance.
        Early steps (high noise) might benefit from higher guidance.
        """
        # Normalize timestep to [0, 1], where 0 is start, 1 is end
        t_norm = timestep / max(total_steps - 1, 1)

        # Apply subtle weighting: slightly higher guidance early on
        # weight = 1.0 + 0.2 * (1.0 - t_norm)  # 1.0 to 1.2 range
        # For now, keep it simple and return as-is
        # Users can experiment with this

        return guidance

    def step(self):
        """Increment the step counter."""
        self.current_step += 1

    def reset(self):
        """Reset the scheduler state."""
        self.current_step = 0


class ITOSampler:
    """
    ITO (Information-Theoretic Optimization) Sampler wrapper.

    This wraps around Flux Dev's sampling process and dynamically adjusts
    guidance based on conditional/unconditional divergence.
    """

    def __init__(
        self,
        divergence_type: str = "l2",
        schedule_type: str = "sigmoid",
        guidance_min: float = 1.0,
        guidance_max: float = 3.5,
        sensitivity: float = 1.0,
        warmup_steps: int = 0,
        smoothing_window: int = 3,
        debug_mode: bool = False,
    ):
        """
        Initialize ITO sampler.

        Args:
            divergence_type: Type of divergence metric ('l2', 'cosine', 'kl_approx', 'frequency')
            schedule_type: Type of guidance schedule ('sigmoid', 'linear', 'exponential', 'polynomial')
            guidance_min: Minimum guidance scale
            guidance_max: Maximum guidance scale
            sensitivity: Sensitivity to divergence changes
            warmup_steps: Steps before adaptive guidance starts
            smoothing_window: EMA smoothing window size
            debug_mode: Enable detailed logging
        """
        self.divergence_calculator = DivergenceCalculator(
            divergence_type=divergence_type,
            smoothing_window=smoothing_window
        )

        self.guidance_scheduler = AdaptiveGuidanceScheduler(
            guidance_min=guidance_min,
            guidance_max=guidance_max,
            schedule_type=schedule_type,
            sensitivity=sensitivity,
            warmup_steps=warmup_steps
        )

        self.debug_mode = debug_mode

        # Debug tracking
        self.divergence_history = []
        self.guidance_history = []
        self.timestep_history = []

    def calculate_adaptive_guidance(
        self,
        model_output_cond: torch.Tensor,
        model_output_uncond: torch.Tensor,
        timestep: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate adaptive guidance scale based on model outputs.

        Args:
            model_output_cond: Conditional model output
            model_output_uncond: Unconditional model output
            timestep: Current timestep
            total_steps: Total number of sampling steps

        Returns:
            Tuple of (guidance_scale, debug_info)
        """
        # Calculate divergence
        divergence = self.divergence_calculator.calculate(
            model_output_cond,
            model_output_uncond,
            timestep
        )

        # Get normalized divergence
        normalized_divergence = self.divergence_calculator.get_normalized_divergence()

        # Calculate adaptive guidance scale
        guidance_scale = self.guidance_scheduler.get_guidance_scale(
            normalized_divergence,
            timestep,
            total_steps
        )

        # Increment step
        self.guidance_scheduler.step()

        # Track history for debugging
        if self.debug_mode:
            self.divergence_history.append(divergence)
            self.guidance_history.append(guidance_scale)
            if timestep is not None:
                self.timestep_history.append(timestep)

        # Prepare debug info
        debug_info = {
            "raw_divergence": divergence,
            "normalized_divergence": normalized_divergence,
            "guidance_scale": guidance_scale,
            "timestep": timestep,
        }

        if self.debug_mode:
            debug_info.update(self.divergence_calculator.get_statistics())

        return guidance_scale, debug_info

    def apply_guidance(
        self,
        model_output_cond: torch.Tensor,
        model_output_uncond: torch.Tensor,
        timestep: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply adaptive guidance to model outputs.

        This is the main method to use during sampling.

        Args:
            model_output_cond: Conditional model output
            model_output_uncond: Unconditional model output
            timestep: Current timestep
            total_steps: Total number of sampling steps

        Returns:
            Tuple of (guided_output, debug_info)
        """
        # Calculate adaptive guidance scale
        guidance_scale, debug_info = self.calculate_adaptive_guidance(
            model_output_cond,
            model_output_uncond,
            timestep,
            total_steps
        )

        # Apply CFG with adaptive guidance
        # Formula: output = uncond + guidance_scale * (cond - uncond)
        guided_output = model_output_uncond + guidance_scale * (
            model_output_cond - model_output_uncond
        )

        return guided_output, debug_info

    def reset(self):
        """Reset the sampler state."""
        self.divergence_calculator.reset()
        self.guidance_scheduler.reset()
        self.divergence_history = []
        self.guidance_history = []
        self.timestep_history = []

    def get_history(self) -> Dict[str, list]:
        """Get sampling history for visualization."""
        return {
            "divergence": self.divergence_history,
            "guidance": self.guidance_history,
            "timesteps": self.timestep_history
        }
