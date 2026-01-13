# vivid-onnx TODO

## Architecture Decision

**Keep vivid-onnx as a separate repo** - not merged into vivid/addons.

Rationale:
- ONNX-based lightweight inference (pose, segmentation) and TensorRT-based diffusion are different subsystems
- Model distribution differs: 18MB (MoveNet) vs 2GB+ (diffusion)
- Faster iteration cycle for ML experiments without destabilizing core
- Users who don't need ML aren't penalized with build bloat

---

## Phase 1: Current State (ONNX Runtime)

- [x] ONNXModel base operator
- [x] PoseDetector (MoveNet SinglePose/MultiPose)
- [x] Cross-platform builds (macOS, Windows, Linux)
- [x] CPU inference stable
- [ ] GPU acceleration (CoreML/DirectML/CUDA) - stubbed but not active

### Asset Management

- [x] onnx pose-tracking example: `std::vector<std::string> modelPaths` isn't great. Need a better way to reference assets.
  - Now uses `AssetLoader` with simple relative paths (e.g., `pose.model("models/movenet/singlepose-lightning.onnx")`)
  - Module assets are auto-registered to search paths when module is loaded

### FaceDetector Improvements

- [ ] Aspect ratio handling: Video is 16:9 but model expects 1:1, causing distortion
  - Option 1: Letterbox input (pad to square, track padding for coordinate adjustment)
  - Option 2: Allow user to specify aspect ratio mode (stretch, fit, fill)
- [ ] False positives: Model occasionally detects non-face regions
  - Could improve with higher confidence threshold (currently 0.20)
  - NMS (non-max suppression) IoU threshold could be tuned (currently 0.3)
- [ ] BlazeFace scores are generally lower than expected (max ~0.30 vs typical 0.75+)
  - May be model-specific (PINTO converted model vs original TFLite)
  - Consider trying back camera model or different BlazeFace variant

---

## Phase 2: TensorRT Integration (Real-Time Diffusion)

Target: Windows + NVIDIA, pure C++, real-time img2img

### 2.1 TensorRT Backend

- [ ] Add TensorRT as optional backend alongside ONNX Runtime
- [ ] CMake option: `VIVID_ML_TENSORRT=ON`
- [ ] FetchContent or find_package for TensorRT SDK
- [ ] Abstract inference backend: `InferenceBackend` base class
  - `OnnxBackend` (existing)
  - `TensorRTBackend` (new)

### 2.2 DiffusionModel Operator

- [ ] New operator: `DiffusionModel` (inherits TextureOperator)
- [ ] img2img pipeline: input texture → latent encode → denoise → decode → output
- [ ] Support for few-step models:
  - SDXL Lightning (1-8 steps)
  - SDXL Turbo (1-4 steps)
  - Hyper-SD (1-8 steps)
- [ ] Parameters:
  - `strength` (0-1, how much to transform input)
  - `steps` (1-8)
  - `prompt` / `negativePrompt`
  - `seed`

### 2.3 Model Distribution

- [ ] Pre-optimized TensorRT engines (~2GB per model)
- [ ] Lazy download on first use (like MoveNet models)
- [ ] Version-specific engines (TensorRT version + GPU arch)
- [ ] Asset manifest in addon.json

### 2.4 Performance Targets

| Metric | Target |
|--------|--------|
| Latency | <100ms for 4-step SDXL Lightning @ 512x512 |
| VRAM | <8GB for single model loaded |
| GPU | RTX 3060+ (Ampere or newer) |

---

## Phase 3: Future Operators

- [ ] SegmentMask - Background/person segmentation (SAM, RMBG)
- [ ] StyleTransfer - Neural style transfer
- [ ] DepthEstimate - Monocular depth (MiDaS, Depth Anything)
- [ ] Upscaler - Real-ESRGAN, LDSR

---

## Resources

- [SDXL Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) - ByteDance, 1-8 step distilled SDXL
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) - ByteDance, SOTA few-step diffusion
- [SD 3.5 TensorRT](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-tensorrt) - Official TensorRT export
- [TensorRT 10.0](https://developer.nvidia.com/blog/nvidia-tensorrt-10-0-upgrades-usability-performance-and-ai-model-support/) - Windows C++ support
- [x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion) - Reference implementation (0.88s latency)
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) - Pipeline-level streaming (Python, for reference)
