# Vivid ML Addon

Machine Learning addon for Vivid using ONNX Runtime.

## Build Commands

```bash
# Build as part of vivid (from vivid root)
cmake -B build -DVIVID_ADDON_ML=ON && cmake --build build

# Build standalone (for development)
cd vivid-ml
cmake -B build -DVIVID_ROOT=/path/to/vivid && cmake --build build
```

## Phase 12: Machine Learning Addon (ONNX)

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
- [ ] SegmentMask - Background/person segmentation
- [ ] StyleTransfer - Neural style transfer
- [ ] DepthEstimate - Monocular depth estimation

## MoveNet Body Tracking Example

MoveNet Lightning detects 17 body keypoints in real-time. Download the ONNX model from TensorFlow Hub.

```cpp
// examples/movenet-tracking/chain.cpp
#include <vivid/vivid.h>
#include <vivid/media/webcam.h>
#include <vivid/ml/pose_detector.h>
#include <vivid/effects/effects.h>

using namespace vivid;

void setup(Context& ctx) {
    auto& chain = ctx.chain();

    // Webcam input
    auto& cam = chain.add<Webcam>("cam");
    cam.resolution(640, 480);

    // MoveNet pose detection (Lightning = fast, Thunder = accurate)
    auto& pose = chain.add<PoseDetector>("pose");
    pose.input("cam");
    pose.setModel("assets/models/movenet_lightning.onnx");

    // Visualize skeleton on top of camera feed
    auto& out = chain.add<Composite>("out");
    out.inputA("cam");
    out.inputB("pose");  // PoseDetector outputs skeleton overlay

    chain.output("out");
}

void update(Context& ctx) {
    auto& chain = ctx.chain();

    // Access individual keypoints (normalized 0-1 coordinates)
    auto& pose = chain.get<PoseDetector>("pose");

    if (pose.detected()) {
        // 17 keypoints: nose, eyes, ears, shoulders, elbows, wrists,
        //               hips, knees, ankles
        glm::vec2 nose = pose.keypoint(PoseDetector::Nose);
        glm::vec2 leftWrist = pose.keypoint(PoseDetector::LeftWrist);
        glm::vec2 rightWrist = pose.keypoint(PoseDetector::RightWrist);

        float confidence = pose.confidence(PoseDetector::Nose);

        // Use keypoints to drive effects
        if (confidence > 0.5f) {
            float handDistance = glm::distance(leftWrist, rightWrist);
            chain.get<Noise>("effect").scale = handDistance * 20.0f;
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
