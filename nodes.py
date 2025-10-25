"""
ComfyUI Node Definitions for ITO Flux Sampler

This module provides the ComfyUI node interface for the ITO sampler.
"""

import torch
import comfy.samplers
import comfy.sample
import nodes
from .ito_sampler import ITOSampler
from .visualization import create_metric_plot, MetricsCollector


class ITOFluxSampler:
    """
    ITO (Information-Theoretic Optimization) Sampler Node for Flux Dev.

    Dynamically adjusts guidance strength during sampling based on
    divergence between conditional and unconditional predictions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # ITO-specific parameters
                "guidance_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "guidance_max": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01}),
                "divergence_type": (["l2", "cosine", "kl_approx", "frequency", "angular", "flow_angular"],),
                "schedule_type": (["sigmoid", "linear", "exponential", "polynomial", "flow_aware"],),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "round": 0.01}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 100}),
                "smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20}),

                # Flux-specific parameters
                "flux_mode": ("BOOLEAN", {"default": True}),
                "absolute_min_guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "divergence_scaling": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        guidance_min,
        guidance_max,
        divergence_type,
        schedule_type,
        sensitivity,
        warmup_steps,
        smoothing_window,
        flux_mode,
        absolute_min_guidance,
        divergence_scaling,
        debug_mode=False
    ):
        """
        Execute ITO sampling.
        """
        # Initialize ITO sampler
        ito_sampler = ITOSampler(
            divergence_type=divergence_type,
            schedule_type=schedule_type,
            guidance_min=guidance_min,
            guidance_max=guidance_max,
            sensitivity=sensitivity,
            warmup_steps=warmup_steps,
            smoothing_window=smoothing_window,
            debug_mode=debug_mode,
            flux_mode=flux_mode,
            absolute_min_guidance=absolute_min_guidance,
            divergence_scaling=divergence_scaling
        )

        # Create metrics collector
        metrics_collector = MetricsCollector()

        # Wrap the model to intercept CFG application
        original_model = model.model

        class ITOModelWrapper:
            """Wrapper that applies ITO guidance instead of standard CFG."""

            def __init__(self, model, ito_sampler, metrics_collector, debug_mode):
                self.model = model
                self.ito_sampler = ito_sampler
                self.metrics_collector = metrics_collector
                self.debug_mode = debug_mode
                self.step_counter = 0
                self.total_steps = steps

            def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
                """
                Intercept model application to apply ITO guidance.
                """
                # Call the model normally with full batch
                # This handles all the complex parameter passing automatically
                output = self.model.apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                                control=control, transformer_options=transformer_options, **kwargs)

                # Check if we have both cond and uncond in the batch
                if output.shape[0] == 2:
                    # Split output into uncond and cond
                    # ComfyUI convention: batch is [uncond, cond]
                    out_uncond = output[0:1]
                    out_cond = output[1:2]

                    # Apply ITO guidance
                    guided_output, debug_info = self.ito_sampler.apply_guidance(
                        out_cond,
                        out_uncond,
                        timestep=self.step_counter,
                        total_steps=self.total_steps
                    )

                    # Collect metrics
                    if self.debug_mode:
                        self.metrics_collector.add_step(debug_info)
                        from .visualization import print_debug_info
                        print_debug_info(debug_info, self.step_counter)

                    self.step_counter += 1

                    # Return batch with uncond and guided output
                    # Keep uncond unchanged, replace cond with guided
                    return torch.cat([out_uncond, guided_output], dim=0)
                else:
                    # Single input, just pass through
                    return output

            def __getattr__(self, name):
                """Pass through all other attributes to the wrapped model."""
                return getattr(self.model, name)

        # Create wrapped model
        wrapped_model = ITOModelWrapper(original_model, ito_sampler, metrics_collector, debug_mode)

        # Temporarily replace the model
        model_backup = model.model
        model.model = wrapped_model

        try:
            # Use the standard KSampler node's sample method
            ksampler = nodes.KSampler()
            result = ksampler.sample(
                model,
                seed,
                steps,
                cfg,  # This will be overridden by ITO
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise
            )

            # Print debug summary if enabled
            if debug_mode:
                summary = metrics_collector.get_summary()
                print("\n=== ITO Sampling Summary ===")
                for key, value in summary.items():
                    print(f"{key}: {value}")
                print("=" * 40 + "\n")

            return result

        finally:
            # Restore original model
            model.model = model_backup
            ito_sampler.reset()


class ITOFluxSamplerDebug:
    """
    ITO (Information-Theoretic Optimization) Sampler Node with Debug Output.

    Same as ITOFluxSampler but includes visualization output for debugging.
    Use this when you want to see the divergence and guidance curves.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # ITO-specific parameters
                "guidance_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "guidance_max": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01}),
                "divergence_type": (["l2", "cosine", "kl_approx", "frequency", "angular", "flow_angular"],),
                "schedule_type": (["sigmoid", "linear", "exponential", "polynomial", "flow_aware"],),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "round": 0.01}),
                "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 100}),
                "smoothing_window": ("INT", {"default": 3, "min": 1, "max": 20}),

                # Flux-specific parameters
                "flux_mode": ("BOOLEAN", {"default": True}),
                "absolute_min_guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "divergence_scaling": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1, "round": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "debug_plot")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        guidance_min,
        guidance_max,
        divergence_type,
        schedule_type,
        sensitivity,
        warmup_steps,
        smoothing_window,
        flux_mode,
        absolute_min_guidance,
        divergence_scaling,
    ):
        """
        Execute ITO sampling with debug visualization.
        """
        # Initialize ITO sampler with debug enabled
        ito_sampler = ITOSampler(
            divergence_type=divergence_type,
            schedule_type=schedule_type,
            guidance_min=guidance_min,
            guidance_max=guidance_max,
            sensitivity=sensitivity,
            warmup_steps=warmup_steps,
            smoothing_window=smoothing_window,
            debug_mode=True,  # Always enabled for debug node
            flux_mode=flux_mode,
            absolute_min_guidance=absolute_min_guidance,
            divergence_scaling=divergence_scaling
        )

        # Create metrics collector
        metrics_collector = MetricsCollector()

        # Wrap the model to intercept CFG application
        original_model = model.model

        class ITOModelWrapper:
            """Wrapper that applies ITO guidance instead of standard CFG."""

            def __init__(self, model, ito_sampler, metrics_collector):
                self.model = model
                self.ito_sampler = ito_sampler
                self.metrics_collector = metrics_collector
                self.step_counter = 0
                self.total_steps = steps

            def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
                """
                Intercept model application to apply ITO guidance.
                """
                # Call the model normally with full batch
                # This handles all the complex parameter passing automatically
                output = self.model.apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                                control=control, transformer_options=transformer_options, **kwargs)

                # Check if we have both cond and uncond in the batch
                if output.shape[0] == 2:
                    # Split output into uncond and cond
                    # ComfyUI convention: batch is [uncond, cond]
                    out_uncond = output[0:1]
                    out_cond = output[1:2]

                    # Apply ITO guidance
                    guided_output, debug_info = self.ito_sampler.apply_guidance(
                        out_cond,
                        out_uncond,
                        timestep=self.step_counter,
                        total_steps=self.total_steps
                    )

                    # Collect metrics
                    self.metrics_collector.add_step(debug_info)
                    from .visualization import print_debug_info
                    print_debug_info(debug_info, self.step_counter)

                    self.step_counter += 1

                    # Return batch with uncond and guided output
                    # Keep uncond unchanged, replace cond with guided
                    return torch.cat([out_uncond, guided_output], dim=0)
                else:
                    # Single input, just pass through
                    return output

            def __getattr__(self, name):
                """Pass through all other attributes to the wrapped model."""
                return getattr(self.model, name)

        # Create wrapped model
        wrapped_model = ITOModelWrapper(original_model, ito_sampler, metrics_collector)

        # Temporarily replace the model
        model_backup = model.model
        model.model = wrapped_model

        try:
            # Use the standard KSampler node's sample method
            ksampler = nodes.KSampler()
            result = ksampler.sample(
                model,
                seed,
                steps,
                cfg,  # This will be overridden by ITO
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise
            )

            # Create debug visualization
            debug_plot_tensor = metrics_collector.create_plot()
            if debug_plot_tensor is not None:
                debug_plot = debug_plot_tensor
            else:
                # Create empty tensor if visualization failed
                debug_plot = torch.zeros((1, 64, 64, 3))

            # Print summary
            summary = metrics_collector.get_summary()
            print("\n=== ITO Sampling Summary ===")
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("=" * 40 + "\n")

            return (result[0], debug_plot)

        finally:
            # Restore original model
            model.model = model_backup
            ito_sampler.reset()


class ITOFluxGuidanceSchedule:
    """
    Standalone node to visualize different ITO guidance schedules
    without running a full sampling pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guidance_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "guidance_max": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "schedule_type": (["sigmoid", "linear", "exponential", "polynomial"],),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "num_points": ("INT", {"default": 100, "min": 10, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("schedule_plot",)
    FUNCTION = "visualize"
    CATEGORY = "sampling/custom"

    def visualize(self, guidance_min, guidance_max, schedule_type, sensitivity, num_points):
        """
        Visualize the guidance schedule.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image
        except ImportError:
            # Return empty image
            return (torch.zeros((1, 64, 64, 3)),)

        from .ito_sampler import AdaptiveGuidanceScheduler

        scheduler = AdaptiveGuidanceScheduler(
            guidance_min=guidance_min,
            guidance_max=guidance_max,
            schedule_type=schedule_type,
            sensitivity=sensitivity,
            warmup_steps=0
        )

        # Generate divergence values
        divergences = np.linspace(0, 1, num_points)
        guidances = [scheduler.get_guidance_scale(d) for d in divergences]

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(divergences, guidances, linewidth=2, color='blue')
        ax.set_xlabel('Normalized Divergence', fontsize=12)
        ax.set_ylabel('Guidance Scale', fontsize=12)
        ax.set_title(f'ITO Guidance Schedule ({schedule_type})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(guidance_min - 0.5, guidance_max + 0.5)

        # Add reference lines
        ax.axhline(y=guidance_min, color='r', linestyle='--', alpha=0.5, label=f'Min: {guidance_min}')
        ax.axhline(y=guidance_max, color='r', linestyle='--', alpha=0.5, label=f'Max: {guidance_max}')
        ax.legend()

        plt.tight_layout()

        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        plt.close(fig)
        buf.close()

        return (img_tensor,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ITOFluxSampler": ITOFluxSampler,
    "ITOFluxSamplerDebug": ITOFluxSamplerDebug,
    "ITOFluxGuidanceSchedule": ITOFluxGuidanceSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ITOFluxSampler": "ITO Flux Sampler",
    "ITOFluxSamplerDebug": "ITO Flux Sampler (Debug)",
    "ITOFluxGuidanceSchedule": "ITO Guidance Schedule Visualizer",
}
