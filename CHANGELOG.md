# Changelog

All notable changes to vivid-onnx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.3] - 2026-01-10

### Changed

- Model paths now use `AssetLoader` with `models:` prefix (e.g., `pose.model("models:movenet/singlepose-lightning.onnx")`)
- Removed hardcoded `~/.vivid` paths - now resolves relative to vivid executable
- Simplified examples by removing manual path fallback logic (~30 lines → 1 line)

### Fixed

- CI build failure on Linux: Added nlohmann/json fetch for SDK builds (not included in vivid SDK package)

## [0.1.0-alpha.2] - 2026-01-10

### Changed

- **Breaking:** Renamed namespace from `vivid::ml` to `vivid::onnx`
- **Breaking:** Headers moved from `vivid/ml/` to `vivid/onnx/` (use `#include <vivid/onnx/onnx.h>`)
- Examples now default to video file input with optional webcam (easier to run out of the box)
- Improved model path resolution - examples now search multiple locations (installed module, dev, relative)

### Fixed

- FaceDetector: Fixed BlazeFace decoding for 4-output ONNX models (split feature maps)
- FaceDetector: Corrected input normalization to [-1, 1] range as expected by BlazeFace
- FaceDetector: Fixed bounding box scale factor (128) for proper coordinate decoding
- ONNXModel: Fixed BGRA→RGB color channel conversion for texture-to-tensor
- ONNXModel: Corrected float32 tensor normalization (0-1 range)
- PoseDetector: Fixed MoveNet input to expect 0-255 range for float32 tensors

### Added

- FaceDetector now listed in module.json operators
- Release archives now include assets (ONNX models) and examples
- BlazeFace model assets for face detection

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
