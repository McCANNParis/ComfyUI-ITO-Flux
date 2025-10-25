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
    Flux-optimized with safeguards against guidance collapse.
    """

    def __init__(
        self,
        guidance_min: float = 1.0,
        guidance_max: float = 3.5,
        schedule_type: Literal["sigmoid", "linear", "exponential", "polynomial", "flow_aware"] = "sigmoid",
        sensitivity: float = 1.0,
        warmup_steps: int = 0,
        flux_mode: bool = True,
        absolute_min_guidance: float = 3.5,  # Raised from 3.0 to prevent guidance collapse
        divergence_scaling: float = 5.0,
    ):
        """
        Initialize the adaptive guidance scheduler.

        Args:
            guidance_min: Minimum guidance scale to use
            guidance_max: Maximum guidance scale to use
            schedule_type: Type of schedule function
            sensitivity: How responsive guidance is to divergence (higher = more responsive)
            warmup_steps: Number of steps before adaptive guidance kicks in
            flux_mode: Enable Flux-specific optimizations (default: True)
            absolute_min_guidance: Hard floor for guidance (Flux needs 3.5+ for prompts, default: 3.5)
            divergence_scaling: Multiply divergence by this factor for Flux (default: 5.0)
        """
        self.guidance_min = guidance_min
        self.guidance_max = guidance_max
        self.schedule_type = schedule_type
        self.sensitivity = sensitivity
        self.warmup_steps = warmup_steps

        # Flux-specific parameters
        self.flux_mode = flux_mode
        self.absolute_min_guidance = absolute_min_guidance
        self.divergence_scaling = divergence_scaling

        self.current_step = 0
        self.previous_guidance = None  # Track previous guidance for gradient limiting

    def get_guidance_scale(
        self,
        normalized_divergence: float,
        timestep: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """
        Calculate the adaptive guidance scale based on divergence.
        Includes Flux-specific optimizations to prevent guidance collapse.

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

        # FLUX FIX: Scale divergence up for Flux's lower natural divergences
        if self.flux_mode:
            # Flux has divergences 5-10x smaller than SDXL
            # Scale them up so the scheduler responds appropriately
            scaled_divergence = normalized_divergence * self.divergence_scaling
            # Clamp to [0, 1] after scaling
            scaled_divergence = max(0.0, min(1.0, scaled_divergence))
        else:
            scaled_divergence = normalized_divergence

        # Apply sensitivity
        adjusted_divergence = scaled_divergence * self.sensitivity

        # Map divergence to guidance scale using selected schedule
        if self.schedule_type == "sigmoid":
            guidance = self._sigmoid_schedule(adjusted_divergence)
        elif self.schedule_type == "linear":
            guidance = self._linear_schedule(adjusted_divergence)
        elif self.schedule_type == "exponential":
            guidance = self._exponential_schedule(adjusted_divergence)
        elif self.schedule_type == "polynomial":
            guidance = self._polynomial_schedule(adjusted_divergence)
        elif self.schedule_type == "flow_aware":
            guidance = self._flow_aware_schedule(adjusted_divergence, timestep, total_steps)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Incorporate timestep-based weighting if available
        if timestep is not None and total_steps is not None:
            guidance = self._apply_timestep_weighting(guidance, timestep, total_steps)

        # FLUX FIX: Enforce absolute minimum guidance
        # Flux needs at least 3.0 guidance to follow prompts properly
        if self.flux_mode:
            guidance = max(guidance, self.absolute_min_guidance)

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

    def _flow_aware_schedule(
        self,
        divergence: float,
        timestep: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> float:
        """
        Flow-matching-aware schedule with phase-based guidance.

        FIXED: Conservative phase boundaries to prevent premature guidance collapse.

        Designed specifically for flow matching models like Flux Dev.
        Uses three phases:
        1. Establishment (t > 0.6): High, consistent guidance to set trajectory (first 40%)
        2. Trajectory (0.15 < t <= 0.6): Adaptive guidance with smooth transitions (middle 45%)
        3. Arrival (t <= 0.15): Controlled reduction with hard minimum (final 15%)

        Args:
            divergence: Adjusted divergence value
            timestep: Current timestep (if None, uses balanced approach)
            total_steps: Total number of steps

        Returns:
            Phase-appropriate guidance scale
        """
        # Calculate normalized timestep (1.0 = start, 0.0 = end)
        if timestep is not None and total_steps is not None:
            # For flow matching, timesteps often go from high to low
            # Normalize to [0, 1] where 1 = start, 0 = end
            t_norm = 1.0 - (timestep / max(total_steps - 1, 1))
        else:
            # Default to middle of trajectory phase
            t_norm = 0.5

        # Phase 1: Establishment (t > 0.6) - Set correct flow direction
        # This is now first 40% instead of first 30%
        if t_norm > 0.6:
            # High, stable guidance to establish trajectory
            # Minimal divergence influence in this critical phase
            base_guidance = self.guidance_max * 0.95  # Even higher base (was 0.9)
            # Very small adjustment based on divergence
            adjustment = divergence * (self.guidance_max - base_guidance) * 0.2
            guidance = base_guidance + adjustment

        # Phase 2: Trajectory (0.15 < t <= 0.6) - Follow established flow
        # This is now middle 45% instead of middle 40%
        elif t_norm > 0.15:  # FIXED: was 0.3, now 0.15
            # Moderate to high guidance with smooth adaptation
            # More responsive to divergence but still conservative
            phase_progress = (t_norm - 0.15) / 0.45  # 0 to 1 within this phase

            # Keep guidance high throughout this phase
            # Start at 90% of max, reduce to 75% of max
            base_guidance = self.guidance_max * (0.75 + 0.15 * phase_progress)

            # Moderate divergence sensitivity
            sensitivity_factor = 0.4 + 0.3 * phase_progress
            adjustment = divergence * (self.guidance_max - base_guidance) * sensitivity_factor

            guidance = base_guidance + adjustment

        # Phase 3: Arrival (t <= 0.15) - Controlled completion
        # This is now final 15% instead of final 70% (BIG FIX!)
        else:
            # Gradual reduction but maintain strong minimum
            phase_progress = t_norm / 0.15  # 0 to 1 within this phase

            # Never go below 70% of max guidance, even in final phase
            # This prevents the catastrophic drop
            min_in_phase = self.guidance_max * 0.7  # MUCH higher than before
            max_in_phase = self.guidance_max * 0.85

            base_guidance = min_in_phase + (max_in_phase - min_in_phase) * phase_progress

            # Still responsive to divergence for detail refinement
            adjustment = divergence * (self.guidance_max - base_guidance) * 0.5

            guidance = base_guidance + adjustment

        # Ensure smooth transitions and valid range
        guidance = max(self.guidance_min, min(self.guidance_max, guidance))

        # CRITICAL FIX: Gradient limiting to prevent sudden drops
        if self.previous_guidance is not None and self.flux_mode:
            max_drop_per_step = 0.3  # Never drop more than 0.3 per step
            if guidance < self.previous_guidance - max_drop_per_step:
                guidance = self.previous_guidance - max_drop_per_step

        # Update previous guidance for next step
        self.previous_guidance = guidance

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
        self.previous_guidance = None  # Reset gradient tracking


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
        flux_mode: bool = True,
        absolute_min_guidance: float = 3.5,  # Raised from 3.0 to prevent guidance collapse
        divergence_scaling: float = 5.0,
    ):
        """
        Initialize ITO sampler with Flux-specific optimizations.

        Args:
            divergence_type: Type of divergence metric ('l2', 'cosine', 'kl_approx', 'frequency')
            schedule_type: Type of guidance schedule ('sigmoid', 'linear', 'exponential', 'polynomial')
            guidance_min: Minimum guidance scale
            guidance_max: Maximum guidance scale
            sensitivity: Sensitivity to divergence changes
            warmup_steps: Steps before adaptive guidance starts
            smoothing_window: EMA smoothing window size
            debug_mode: Enable detailed logging
            flux_mode: Enable Flux-specific optimizations (default: True)
            absolute_min_guidance: Hard floor for guidance (Flux needs 3.5+, default: 3.5)
            divergence_scaling: Scale Flux divergences by this factor (default: 5.0)
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
            warmup_steps=warmup_steps,
            flux_mode=flux_mode,
            absolute_min_guidance=absolute_min_guidance,
            divergence_scaling=divergence_scaling
        )

        self.debug_mode = debug_mode
        self.flux_mode = flux_mode

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
            model_output_cond: Conditional model output (velocity field for Flux)
            model_output_uncond: Unconditional model output (velocity field for Flux)
            timestep: Current timestep
            total_steps: Total number of sampling steps

        Returns:
            Tuple of (guidance_scale, debug_info)
        """
        # Calculate normalized timestep for flow-aware metrics
        timestep_float = None
        if timestep is not None and total_steps is not None:
            # Normalize to [0, 1] where 1 = start, 0 = end
            timestep_float = 1.0 - (timestep / max(total_steps - 1, 1))

        # Calculate divergence (with optional timestep_float for flow_angular)
        divergence = self.divergence_calculator.calculate(
            model_output_cond,
            model_output_uncond,
            timestep,
            timestep_float
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
