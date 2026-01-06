# Changelog

All notable changes to vivid-onnx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2026-01-06

Initial alpha release of the Vivid ML addon.

### Added

- ONNX Runtime integration with automatic per-platform downloads via CMake
- `ONNXModel` operator for generic ONNX model inference
- `PoseDetector` operator for MoveNet skeleton tracking (17 keypoints)
- Cross-platform CI builds (macOS arm64/x64, Linux x64, Windows x64)
- Automated release workflow triggered by version tags

### Notes

- GPU acceleration (CoreML/DirectML/CUDA) not yet implemented
- Video addon integration is optional and disabled by default
