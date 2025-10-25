"""
ComfyUI-ITO-Flux: Information-Theoretic Optimization for Flux Dev

A custom ComfyUI node that implements dynamic inference-time optimization (ITO)
for Flux Dev model, providing adaptive guidance scaling based on divergence metrics.

Author: Claude Code
License: MIT
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Version info
__version__ = "1.1.1"
__author__ = "Claude Code"

# ComfyUI expects these exports
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Print load message
print(f"\n{'='*60}")
print(f"ComfyUI-ITO-Flux v{__version__} loaded successfully!")
print(f"{'='*60}")
print("Available nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")
print(f"{'='*60}\n")
