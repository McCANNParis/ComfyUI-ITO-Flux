# Changelog

All notable changes to ComfyUI-ITO-Flux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-10-25

### Changed
- Bump project version to 3.0.0 to mark the latest release.

## [1.1.1] - 2025-10-25

### Fixed
- **CRITICAL: Guidance cliff at step 14** - Fixed catastrophic guidance collapse in `flow_aware` schedule
  - **Problem**: Arrival phase (t < 0.3) was triggering at 70% through sampling instead of final 15%
  - **Symptom**: Guidance dropped from 4.5 → 3.0 at step 14/20, causing abstract blobs
  - **Root cause**: Phase boundary at 0.3 was too high, making 70% of steps "arrival" phase
- **Phase boundary adjustments**:
  - Establishment: Changed from t > 0.7 to t > 0.6 (first 40% instead of 30%)
  - Trajectory: Changed from 0.3 < t < 0.7 to 0.15 < t < 0.6 (middle 45% instead of 40%)
  - Arrival: Changed from t < 0.3 to t < 0.15 (final 15% instead of 70%!)
- **Higher base guidance throughout**:
  - Establishment: 95% of max (was 90%)
  - Trajectory: 75-90% of max (was 70-90%)
  - Arrival: 70-85% of max (was absolute_min to 50% of max)
- **Gradient limiting**: Added max_drop_per_step = 0.3 to prevent sudden guidance cliffs
  - Tracks previous guidance and limits drops to 0.3 per step
  - Prevents the 4.5 → 3.0 cliff that was breaking images

### Changed
- **Raised default `absolute_min_guidance`** from 3.0 to 3.5
  - Flux needs at least 3.5 guidance for reliable prompt adherence
  - Prevents mode collapse even if schedule goes too low

### Technical Details
- Added `previous_guidance` tracking to AdaptiveGuidanceScheduler
- Gradient limiter only active when `flux_mode=True`
- Reset previous_guidance on scheduler.reset()
- Flow-aware schedule now maintains 70%+ of max guidance even in final phase

### Expected Behavior (20 steps, guidance_max=5.0)
```
Steps 1-8:   4.5-5.0  (Establishment: setting trajectory)
Steps 9-17:  3.8-4.5  (Trajectory: following flow)
Steps 18-20: 3.5-4.0  (Arrival: completing with high guidance)
```
No more sudden drops! Smooth, gradual guidance adjustments throughout.

## [1.1.0] - 2025-10-25

### Added
- **Flow-matching-aware ITO** - Complete redesign for Flux's flow matching architecture
  - `angular` divergence type: Measures direction changes in velocity fields (best for flow matching)
  - `flow_angular` divergence type: Combined angular + magnitude with phase weighting
  - `flow_aware` schedule type: Phase-based guidance (establishment/trajectory/arrival)
- Three-phase guidance strategy optimized for flow matching:
  - **Establishment phase** (t > 0.7): High, consistent guidance to set trajectory direction
  - **Trajectory phase** (0.3 < t < 0.7): Adaptive guidance with smooth transitions
  - **Arrival phase** (t < 0.3): Controlled reduction while maintaining minimum for prompt adherence
- Phase-aware divergence weighting: Early steps prioritize direction, late steps prioritize magnitude
- Smooth trajectory consistency: Prevents guidance jumps that break flow paths

### Changed
- **BREAKING**: Divergence now correctly interprets Flux's velocity fields vs SDXL's noise predictions
- Enhanced divergence calculator with optional `timestep_float` parameter for flow-aware metrics
- Improved documentation explaining flow matching vs diffusion differences

### Technical Details
- Flow matching uses velocity fields that guide ODE trajectories, not noise predictions
- Angular divergence (direction) is more important than L2 divergence (magnitude) for flows
- Phase-based scheduling maintains trajectory consistency critical for flow matching
- Combined metrics weight angular divergence higher early, magnitude divergence higher late

### Recommended Settings for Flux Dev
- **Divergence type**: `flow_angular` (best) or `angular` (simpler)
- **Schedule type**: `flow_aware` (optimal for flow matching)
- **Guidance range**: 3.0-5.0 (narrower than diffusion models)
- **Sensitivity**: 1.0-1.5 (lower than diffusion, flows are sensitive to changes)

### Migration from 1.0.x
- Existing workflows will continue working with improved defaults
- For best results with Flux, switch to `flow_angular` + `flow_aware`
- Non-Flux models can continue using original settings (`l2` + `sigmoid`)

## [1.0.6] - 2025-10-25

### Added
- **Flux-specific optimizations** to prevent guidance collapse and mode failure
  - `flux_mode` parameter (default: True) - enables Flux-specific fixes
  - `absolute_min_guidance` parameter (default: 3.0) - hard floor for guidance scale
  - `divergence_scaling` parameter (default: 5.0) - scales up Flux's low divergences
- All three parameters exposed in both ITOFluxSampler and ITOFluxSamplerDebug nodes

### Fixed
- **Critical: Guidance collapse with Flux Dev**
  - Flux has divergences 5-10x smaller than SDXL (0.005-0.020 vs 0.1-0.5)
  - Previous versions allowed guidance to drop to 1.0-1.5, causing prompt abandonment
  - Flux requires minimum 3.0 guidance to follow prompts correctly
- **Divergence scaling**: Now multiplies Flux divergences by 5.0 before mapping to guidance range
- **Absolute minimum enforcement**: Never allows guidance below 3.0 when flux_mode=True
- Prevents mode collapse (abstract blobs) that occurred with low guidance values

### Technical Details
- Modified `AdaptiveGuidanceScheduler.get_guidance_scale()` with two-stage fix:
  1. Scale normalized divergence by `divergence_scaling` factor before schedule calculation
  2. Enforce `absolute_min_guidance` floor after schedule calculation
- Flux's natural divergences (0.001-0.003 normalized) now map to useful guidance range (3.0+)
- Allows ITO adaptation while maintaining minimum guidance for prompt adherence

### Migration Notes
- **No breaking changes**: Existing workflows continue to work with improved defaults
- For non-Flux models: Set `flux_mode=False` to disable these optimizations
- Advanced users can tune `divergence_scaling` and `absolute_min_guidance` for their models

## [1.0.5] - 2024-10-25

### Fixed
- Fixed RuntimeError in frequency divergence: tensor size mismatch (65 vs 33)
- Bug was caused by getting dimensions from FFT output instead of input
- Now captures original spatial dimensions before applying FFT
- `rfft2` changes the last dimension size, so we need original dimensions for frequency grid
- Frequency divergence now works correctly for detail-focused sampling

## [1.0.4] - 2024-10-25

### Fixed
- Fixed RuntimeError: output shape mismatch in Flux model
- Changed approach: now call model once with full batch, then apply ITO to outputs
- No longer manually splitting inputs (which caused shape mismatches with Flux's complex conditioning)
- Properly handles Flux's batch processing and all conditioning parameters in **kwargs
- Applied fix to both ITOFluxSampler and ITOFluxSamplerDebug

### Technical Details
- Previous approach tried to split inputs and call model twice
- This failed because Flux has additional conditioning params that weren't being split
- New approach: call model once, split outputs, calculate divergence, apply guidance
- More efficient and handles all model architectures correctly

## [1.0.3] - 2024-10-25

### Fixed
- Fixed AttributeError: module 'comfy.samplers' has no attribute 'common_ksampler'
- Now uses `nodes.KSampler()` directly (the actual standard KSampler class)
- Proper integration with ComfyUI's native sampling infrastructure
- Compatible with ComfyUI 0.3.66+

## [1.0.2] - 2024-10-25

### Fixed
- Fixed critical sampling bug where wrong function was called
- Attempted to use `comfy.samplers.common_ksampler()` (function doesn't exist)
- Resolved AttributeError: 'int' object has no attribute 'shape'
- Proper parameter passing to ComfyUI sampling functions

## [1.0.1] - 2024-10-25

### Changed
- **BREAKING**: Split ITO Flux Sampler into two nodes for better compatibility
  - **ITO Flux Sampler**: Now returns only LATENT (drop-in KSampler replacement)
  - **ITO Flux Sampler (Debug)**: Returns LATENT + debug plot IMAGE
- Changed category from "sampling/custom" to "sampling" for better organization
- Debug mode parameter in main node now only controls console output

### Added
- True drop-in replacement for KSampler with identical return types
- Separate debug node for users who want visualization
- Better documentation explaining the two node types

### Fixed
- Return type compatibility with standard ComfyUI workflows
- Users can now directly replace KSampler without workflow modifications

## [1.0.0] - 2024-10-25

### Added
- Initial release of ComfyUI-ITO-Flux
- Core ITO sampler implementation for Flux Dev
- Four divergence metrics: L2, Cosine, KL-approximation, Frequency-domain
- Four guidance schedules: Sigmoid, Linear, Exponential, Polynomial
- Adaptive guidance scheduler with configurable parameters
- Debug mode with real-time metrics visualization
- ITO Guidance Schedule Visualizer node
- Comprehensive documentation and examples
- Production-ready error handling and stability features

### Features
- Dynamic guidance adjustment based on divergence
- Flux-specific optimizations for flow matching
- Exponential moving average smoothing
- Warmup steps support
- Sensitivity tuning
- Matplotlib-based visualization
- Metrics collection and export
- Console debug output

### Documentation
- Comprehensive README with usage guide
- Detailed EXAMPLES.md with practical use cases
- Parameter descriptions and recommendations
- Troubleshooting guide
- Performance notes

### Technical
- Efficient PyTorch operations
- Minimal memory overhead (~5-10% compute increase)
- Batch processing support
- Numerical stability guarantees
- Deterministic results with fixed seeds

## [Unreleased]

### Planned Features
- Additional divergence metrics (perceptual, learned)
- Auto-tuning mode to find optimal parameters
- Integration with ControlNet/IP-Adapter
- Animation/video support with temporal smoothing
- Performance optimizations (torch.compile support)
- Multi-resolution support
- Advanced schedule functions (attention-based, learned)
- Export metrics to TensorBoard
- A/B testing utilities

### Known Issues
None reported yet.

---

## Version History

- **1.0.0** (2024-10-25): Initial release

---

## Contributing

See CONTRIBUTING.md (coming soon) for guidelines on how to contribute to this project.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
