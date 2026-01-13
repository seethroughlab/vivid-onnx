// Pose Tracking Example
// Demonstrates body pose detection using MoveNet via ONNX Runtime
// with skeleton visualization overlay
//
// Usage:
//   ./vivid examples/pose-tracking
//
// To use webcam instead of video file:
//   1. Comment out the VideoPlayer section below
//   2. Uncomment the Webcam section
//
// Model: MoveNet SinglePose Lightning from PINTO_model_zoo
// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/115_MoveNet

#include <vivid/vivid.h>
#include <vivid/video/video.h>
#include <vivid/onnx/onnx.h>
#include <vivid/effects/effects.h>
#include <cmath>
#include <iostream>

using namespace vivid;
using namespace vivid::video;
using namespace vivid::onnx;
using namespace vivid::effects;

// Colors for different body parts (RGBA 0-1)
static const glm::vec4 COLOR_FACE = {0.2f, 0.8f, 1.0f, 1.0f};      // Cyan
static const glm::vec4 COLOR_ARM_L = {1.0f, 0.4f, 0.4f, 1.0f};     // Red
static const glm::vec4 COLOR_ARM_R = {0.4f, 1.0f, 0.4f, 1.0f};     // Green
static const glm::vec4 COLOR_TORSO = {1.0f, 1.0f, 0.4f, 1.0f};     // Yellow
static const glm::vec4 COLOR_LEG_L = {1.0f, 0.6f, 0.2f, 1.0f};     // Orange
static const glm::vec4 COLOR_LEG_R = {0.6f, 0.4f, 1.0f, 1.0f};     // Purple

glm::vec4 getKeypointColor(Keypoint kp) {
    int idx = static_cast<int>(kp);
    if (idx <= 4) return COLOR_FACE;
    if (idx == 5 || idx == 7 || idx == 9) return COLOR_ARM_L;
    if (idx == 6 || idx == 8 || idx == 10) return COLOR_ARM_R;
    if (idx == 11 || idx == 13 || idx == 15) return COLOR_LEG_L;
    if (idx == 12 || idx == 14 || idx == 16) return COLOR_LEG_R;
    return COLOR_TORSO;
}

glm::vec4 getConnectionColor(Keypoint from, Keypoint to) {
    int f = static_cast<int>(from);
    int t = static_cast<int>(to);

    if (f <= 4 && t <= 4) return COLOR_FACE;
    if ((f == 5 || f == 7 || f == 9) && (t == 5 || t == 7 || t == 9)) return COLOR_ARM_L;
    if ((f == 6 || f == 8 || f == 10) && (t == 6 || t == 8 || t == 10)) return COLOR_ARM_R;
    if ((f == 11 || f == 13 || f == 15) && (t == 11 || t == 13 || t == 15)) return COLOR_LEG_L;
    if ((f == 12 || f == 14 || f == 16) && (t == 12 || t == 14 || t == 16)) return COLOR_LEG_R;
    return COLOR_TORSO;
}

void setup(Context& ctx) {
    auto& chain = ctx.chain();

    // --- Video file input (default) ---
    auto& video = chain.add<VideoPlayer>("source");
    video.setFile("assets/prom.mp4");
    video.setLoop(true);

    // --- Webcam input (uncomment to use instead of video) ---
    // auto& cam = chain.add<Webcam>("source");
    // cam.setResolution(1280, 720);
    // cam.setFrameRate(30);

    // Pose detector using MoveNet SinglePose Lightning
    auto& pose = chain.add<PoseDetector>("pose");
    pose.input(&video);
    pose.model("models/movenet/singlepose-lightning.onnx");
    pose.confidenceThreshold(0.3f);

    // Canvas overlay for skeleton visualization
    auto& canvas = chain.add<Canvas>("skeleton");
    canvas.size(1280, 720);

    // Composite video and skeleton overlay
    auto& comp = chain.add<Composite>("output");
    comp.input(0, "source");
    comp.input(1, "skeleton");
    comp.mode = BlendMode::Over;

    chain.output("output");

    std::cout << "Pose Tracking Example" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "Model: MoveNet SinglePose Lightning" << std::endl;
    std::cout << "Skeleton overlay shows detected body pose" << std::endl;
}

void update(Context& ctx) {
    auto& chain = ctx.chain();
    auto& pose = chain.get<PoseDetector>("pose");
    auto& canvas = chain.get<Canvas>("skeleton");

    const float width = 1280.0f;
    const float height = 720.0f;
    const float lineWidth = 4.0f;
    const float pointRadius = 8.0f;
    const float minConfidence = 0.3f;

    canvas.clear(0, 0, 0, 0);

    if (pose.detected()) {
        canvas.lineWidth(lineWidth);
        canvas.lineCap(LineCap::Round);

        // Draw skeleton lines
        for (const auto& bone : onnx::SKELETON_CONNECTIONS) {
            float conf1 = pose.confidence(bone.from);
            float conf2 = pose.confidence(bone.to);

            if (conf1 >= minConfidence && conf2 >= minConfidence) {
                glm::vec2 p1 = pose.keypoint(bone.from);
                glm::vec2 p2 = pose.keypoint(bone.to);

                float x1 = p1.x * width;
                float y1 = p1.y * height;
                float x2 = p2.x * width;
                float y2 = p2.y * height;

                glm::vec4 color = getConnectionColor(bone.from, bone.to);
                float avgConf = (conf1 + conf2) * 0.5f;
                color.a = std::min(1.0f, avgConf * 2.0f);

                canvas.strokeStyle(color.r, color.g, color.b, color.a);
                canvas.beginPath();
                canvas.moveTo(x1, y1);
                canvas.lineTo(x2, y2);
                canvas.stroke();
            }
        }

        // Draw keypoint circles
        for (int i = 0; i < static_cast<int>(Keypoint::Count); i++) {
            Keypoint kp = static_cast<Keypoint>(i);
            float conf = pose.confidence(kp);

            if (conf >= minConfidence) {
                glm::vec2 p = pose.keypoint(kp);
                float x = p.x * width;
                float y = p.y * height;

                glm::vec4 color = getKeypointColor(kp);
                color.a = std::min(1.0f, conf * 2.0f);

                canvas.fillStyle(color.r, color.g, color.b, color.a);
                canvas.beginPath();
                canvas.arc(x, y, pointRadius, 0, 2.0f * 3.14159f);
                canvas.fill();

                canvas.strokeStyle(1, 1, 1, color.a * 0.8f);
                canvas.lineWidth(2.0f);
                canvas.stroke();
            }
        }
    }
}

VIVID_CHAIN(setup, update)
