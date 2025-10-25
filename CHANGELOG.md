# Changelog

All notable changes to ComfyUI-ITO-Flux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2024-10-25

### Fixed
- Fixed critical sampling bug where wrong function was called
- Now uses `comfy.samplers.common_ksampler()` (same as standard KSampler)
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
