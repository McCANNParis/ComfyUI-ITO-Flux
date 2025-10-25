"""
Visualization and debugging utilities for ITO sampler.

Provides tools to visualize divergence curves, guidance schedules,
and other metrics during the sampling process.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import io


def create_metric_plot(
    divergence_history: List[float],
    guidance_history: List[float],
    timesteps: Optional[List[int]] = None,
    width: int = 512,
    height: int = 384
) -> Optional[torch.Tensor]:
    """
    Create a visualization plot of divergence and guidance over time.

    Args:
        divergence_history: List of divergence values
        guidance_history: List of guidance scales used
        timesteps: Optional list of timesteps
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Tensor containing the plot image, or None if matplotlib unavailable
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
        return None

    if not divergence_history or not guidance_history:
        return None

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), dpi=100)

    x_axis = timesteps if timesteps else list(range(len(divergence_history)))

    # Plot divergence
    ax1.plot(x_axis, divergence_history, 'b-', linewidth=2, label='Divergence')
    ax1.set_ylabel('Divergence', fontsize=10)
    ax1.set_title('ITO Metrics Over Sampling Steps', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Plot guidance scale
    ax2.plot(x_axis, guidance_history, 'r-', linewidth=2, label='Guidance Scale')
    ax2.set_xlabel('Step' if not timesteps else 'Timestep', fontsize=10)
    ax2.set_ylabel('Guidance Scale', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Add statistics text
    avg_divergence = np.mean(divergence_history)
    avg_guidance = np.mean(guidance_history)
    stats_text = f'Avg Divergence: {avg_divergence:.4f}\nAvg Guidance: {avg_guidance:.4f}'

    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Convert plot to tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    # Read image
    from PIL import Image
    img = Image.open(buf)
    img_array = np.array(img)

    # Convert to tensor [1, H, W, C]
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    plt.close(fig)
    buf.close()

    return img_tensor


def create_comparison_grid(
    image_ito: Optional[torch.Tensor],
    image_standard: Optional[torch.Tensor],
    metrics_plot: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """
    Create a comparison grid showing ITO vs standard CFG results.

    Args:
        image_ito: Image generated with ITO
        image_standard: Image generated with standard CFG
        metrics_plot: Optional metrics visualization

    Returns:
        Tensor containing the comparison grid
    """
    if image_ito is None and image_standard is None:
        return None

    images = []
    if image_standard is not None:
        images.append(image_standard)
    if image_ito is not None:
        images.append(image_ito)

    if not images:
        return None

    # Simple horizontal concatenation
    # Assuming images are [B, H, W, C]
    if len(images) == 2:
        combined = torch.cat(images, dim=2)  # Concatenate along width
    else:
        combined = images[0]

    return combined


def print_debug_info(debug_info: Dict, step: int):
    """
    Print debug information to console.

    Args:
        debug_info: Dictionary containing debug metrics
        step: Current step number
    """
    print(f"\n=== ITO Debug Info - Step {step} ===")
    print(f"Raw Divergence: {debug_info.get('raw_divergence', 0):.6f}")
    print(f"Normalized Divergence: {debug_info.get('normalized_divergence', 0):.6f}")
    print(f"Guidance Scale: {debug_info.get('guidance_scale', 0):.4f}")

    if 'current_ema' in debug_info:
        print(f"EMA Divergence: {debug_info['current_ema']:.6f}")
        print(f"Min Seen: {debug_info['min_seen']:.6f}")
        print(f"Max Seen: {debug_info['max_seen']:.6f}")

    print("=" * 40)


def export_metrics_csv(
    divergence_history: List[float],
    guidance_history: List[float],
    timesteps: Optional[List[int]] = None,
    filename: str = "ito_metrics.csv"
) -> str:
    """
    Export metrics to a CSV file.

    Args:
        divergence_history: List of divergence values
        guidance_history: List of guidance scales
        timesteps: Optional list of timesteps
        filename: Output filename

    Returns:
        Path to the exported file
    """
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        if timesteps:
            writer.writerow(['Step', 'Timestep', 'Divergence', 'Guidance'])
            for i, (div, guide, ts) in enumerate(zip(divergence_history, guidance_history, timesteps)):
                writer.writerow([i, ts, div, guide])
        else:
            writer.writerow(['Step', 'Divergence', 'Guidance'])
            for i, (div, guide) in enumerate(zip(divergence_history, guidance_history)):
                writer.writerow([i, div, guide])

    return filename


class MetricsCollector:
    """
    Collects and manages metrics during sampling for later analysis.
    """

    def __init__(self):
        self.metrics = {
            'divergence': [],
            'guidance': [],
            'timesteps': [],
            'normalized_divergence': [],
            'ema_divergence': []
        }

    def add_step(self, debug_info: Dict):
        """Add metrics from a single step."""
        self.metrics['divergence'].append(debug_info.get('raw_divergence', 0))
        self.metrics['guidance'].append(debug_info.get('guidance_scale', 0))
        self.metrics['timesteps'].append(debug_info.get('timestep', 0))
        self.metrics['normalized_divergence'].append(debug_info.get('normalized_divergence', 0))
        self.metrics['ema_divergence'].append(debug_info.get('current_ema', 0))

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics['divergence']:
            return {}

        return {
            'total_steps': len(self.metrics['divergence']),
            'avg_divergence': np.mean(self.metrics['divergence']),
            'avg_guidance': np.mean(self.metrics['guidance']),
            'min_guidance': np.min(self.metrics['guidance']),
            'max_guidance': np.max(self.metrics['guidance']),
            'guidance_std': np.std(self.metrics['guidance']),
            'divergence_range': (np.min(self.metrics['divergence']), np.max(self.metrics['divergence']))
        }

    def create_plot(self, width: int = 512, height: int = 384) -> Optional[torch.Tensor]:
        """Create visualization plot."""
        return create_metric_plot(
            self.metrics['divergence'],
            self.metrics['guidance'],
            self.metrics['timesteps'],
            width,
            height
        )

    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []
