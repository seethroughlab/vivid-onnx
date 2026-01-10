# Vivid ML Library

Machine Learning library for Vivid using ONNX Runtime.

## Build Commands

```bash
# Build as part of vivid (from vivid root)
cmake -B build -DVIVID_ADDON_ML=ON && cmake --build build

# Build standalone (for development)
cd vivid-onnx
cmake -B build -DVIVID_ROOT=/path/to/vivid && cmake --build build
```

## Machine Learning Library (ONNX)

**Goal:** ML inference for creative applications

**Library:** [ONNX Runtime](https://onnxruntime.ai/) with platform-specific acceleration

| Platform | Accelerator |
|----------|-------------|
| macOS | CoreML |
| Windows | DirectML |
| Linux | CUDA (optional) |

**Core Operator:**
```cpp
class ONNXModel : public Operator {
public:
    void setModel(const std::string& path);  // .onnx file
    void input(const std::string& node);     // Texture input

    // Output: inference result as texture or values
};
```

**Specialized Operators:**
- [x] PoseDetector - MoveNet skeleton tracking
- [x] FaceDetector - BlazeFace face detection (6 landmarks)
- [ ] SegmentMask - Background/person segmentation
- [ ] StyleTransfer - Neural style transfer
- [ ] DepthEstimate - Monocular depth estimation

## MoveNet Body Tracking Example

MoveNet Lightning detects 17 body keypoints in real-time. The model is bundled in `assets/models/movenet/`.

```cpp
// examples/pose-tracking/chain.cpp
#include <vivid/vivid.h>
#include <vivid/video/video.h>
#include <vivid/onnx/onnx.h>
#include <vivid/effects/effects.h>
#include <cstdlib>

using namespace vivid;
using namespace vivid::video;
using namespace vivid::onnx;
using namespace vivid::effects;

void setup(Context& ctx) {
    auto& chain = ctx.chain();

    // Webcam input
    auto& cam = chain.add<Webcam>("cam");
    cam.setResolution(640, 480);

    // MoveNet pose detection (Lightning = fast, Thunder = accurate)
    std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
    auto& pose = chain.add<PoseDetector>("pose");
    pose.input(&cam);
    pose.model(home + "/.vivid/modules/vivid-onnx/assets/models/movenet/singlepose-lightning.onnx");

    // Canvas for skeleton overlay
    auto& canvas = chain.add<Canvas>("skeleton");
    canvas.size(640, 480);

    // Composite webcam and skeleton
    auto& out = chain.add<Composite>("out");
    out.input(0, "cam");
    out.input(1, "skeleton");
    out.mode(BlendMode::Over);

    chain.output("out");
}

void update(Context& ctx) {
    auto& chain = ctx.chain();
    auto& pose = chain.get<PoseDetector>("pose");
    auto& canvas = chain.get<Canvas>("skeleton");

    canvas.clear(0, 0, 0, 0);

    if (pose.detected()) {
        // Access individual keypoints (normalized 0-1 coordinates)
        glm::vec2 nose = pose.keypoint(Keypoint::Nose);
        float confidence = pose.confidence(Keypoint::Nose);

        // Draw keypoints using Canvas path API
        if (confidence > 0.3f) {
            canvas.fillStyle(1, 0, 0, 1);
            canvas.beginPath();
            canvas.arc(nose.x * 640, nose.y * 480, 8.0f, 0, 6.28f);
            canvas.fill();
        }
    }
}

VIVID_CHAIN(setup, update)
```

## PoseDetector Keypoint Enum

```cpp
enum Keypoint {
    Nose = 0,
    LeftEye, RightEye,
    LeftEar, RightEar,
    LeftShoulder, RightShoulder,
    LeftElbow, RightElbow,
    LeftWrist, RightWrist,
    LeftHip, RightHip,
    LeftKnee, RightKnee,
    LeftAnkle, RightAnkle
};
```

## Tasks

- [x] ONNX Runtime integration (auto-downloads per platform via CMake)
- [ ] GPU acceleration per platform (CoreML/DirectML/CUDA)
- [ ] Texture→tensor conversion (NHWC format, 192x192 or 256x256)
- [x] Tensor→keypoint parsing (17 points × 3 values: x, y, confidence)
- [ ] Skeleton overlay rendering
- [ ] Model hot-reload
- [ ] Bundle MoveNet Lightning ONNX model

## Validation

- [ ] MoveNet detects 17 keypoints from webcam
- [ ] Inference runs at >30fps on integrated GPU
- [ ] Keypoint coordinates are correctly normalized (0-1)
- [ ] Skeleton overlay draws correctly
- [ ] examples/movenet-tracking runs on macOS and Windows
