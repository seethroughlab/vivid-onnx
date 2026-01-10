// Face Detection Example
// Demonstrates face detection using BlazeFace via ONNX Runtime
// with bounding box and landmark visualization overlay
//
// Usage:
//   ./vivid examples/face-detection
//
// To use webcam instead of video file:
//   1. Comment out the VideoPlayer section below
//   2. Uncomment the Webcam section
//
// Model: BlazeFace from PINTO_model_zoo
// https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace

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

// Colors for visualization
static const glm::vec4 COLOR_BOX = {0.0f, 1.0f, 0.5f, 1.0f};        // Green
static const glm::vec4 COLOR_EYE = {0.2f, 0.8f, 1.0f, 1.0f};        // Cyan
static const glm::vec4 COLOR_NOSE = {1.0f, 0.8f, 0.2f, 1.0f};       // Yellow
static const glm::vec4 COLOR_MOUTH = {1.0f, 0.4f, 0.4f, 1.0f};      // Red
static const glm::vec4 COLOR_EAR = {0.8f, 0.4f, 1.0f, 1.0f};        // Purple

glm::vec4 getLandmarkColor(FaceLandmark lm) {
    switch (lm) {
        case FaceLandmark::RightEye:
        case FaceLandmark::LeftEye:
            return COLOR_EYE;
        case FaceLandmark::Nose:
            return COLOR_NOSE;
        case FaceLandmark::Mouth:
            return COLOR_MOUTH;
        case FaceLandmark::RightEar:
        case FaceLandmark::LeftEar:
            return COLOR_EAR;
        default:
            return COLOR_BOX;
    }
}

void setup(Context& ctx) {
    auto& chain = ctx.chain();

    // --- Video file input (default) ---
    auto& video = chain.add<VideoPlayer>("source");
    video.setFile("assets/face-demographics.mp4");
    video.setLoop(true);

    // --- Webcam input (uncomment to use instead of video) ---
    // auto& cam = chain.add<Webcam>("source");
    // cam.setResolution(1280, 720);
    // cam.setFrameRate(30);

    // Face detector using BlazeFace
    auto& faces = chain.add<FaceDetector>("faces");
    faces.input(&video);
    faces.model("models:blazeface/face_detection_front_128x128_float32.onnx");
    faces.confidenceThreshold(0.20f);  // Tuned for BlazeFace ONNX model
    faces.maxFaces(5);

    // Canvas overlay for face visualization (match video resolution)
    auto& canvas = chain.add<Canvas>("overlay");
    canvas.size(768, 432);

    // Composite video and face overlay
    auto& comp = chain.add<Composite>("output");
    comp.input(0, "source");
    comp.input(1, "overlay");
    comp.mode = BlendMode::Over;

    chain.output("output");

    std::cout << "Face Detection Example" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Model: BlazeFace 128x128" << std::endl;
    std::cout << "Detecting up to 5 faces with landmarks" << std::endl;
}

void update(Context& ctx) {
    auto& chain = ctx.chain();
    auto& faces = chain.get<FaceDetector>("faces");
    auto& canvas = chain.get<Canvas>("overlay");

    // Match video resolution (768x432)
    const float width = 768.0f;
    const float height = 432.0f;
    const float lineWidth = 3.0f;
    const float pointRadius = 6.0f;

    canvas.clear(0, 0, 0, 0);

    int numFaces = faces.faceCount();

    for (int f = 0; f < numFaces; f++) {
        float conf = faces.confidence(f);
        glm::vec4 bbox = faces.boundingBox(f);

        // Convert normalized bbox to pixel coords
        float x = bbox.x * width;
        float y = bbox.y * height;
        float w = bbox.z * width;
        float h = bbox.w * height;

        // Draw bounding box (use full alpha, not confidence-based)
        glm::vec4 boxColor = COLOR_BOX;

        canvas.strokeStyle(boxColor.r, boxColor.g, boxColor.b, 1.0f);
        canvas.lineWidth(lineWidth);
        canvas.strokeRect(x, y, w, h);

        // Draw corner accents (like FaceDetect in opencv)
        float corner = std::min(w, h) * 0.2f;
        canvas.strokeStyle(1.0f, 1.0f, 1.0f, conf);
        canvas.lineWidth(lineWidth + 1);

        // Top-left
        canvas.beginPath();
        canvas.moveTo(x, y + corner);
        canvas.lineTo(x, y);
        canvas.lineTo(x + corner, y);
        canvas.stroke();

        // Top-right
        canvas.beginPath();
        canvas.moveTo(x + w - corner, y);
        canvas.lineTo(x + w, y);
        canvas.lineTo(x + w, y + corner);
        canvas.stroke();

        // Bottom-left
        canvas.beginPath();
        canvas.moveTo(x, y + h - corner);
        canvas.lineTo(x, y + h);
        canvas.lineTo(x + corner, y + h);
        canvas.stroke();

        // Bottom-right
        canvas.beginPath();
        canvas.moveTo(x + w - corner, y + h);
        canvas.lineTo(x + w, y + h);
        canvas.lineTo(x + w, y + h - corner);
        canvas.stroke();

        // Draw confidence label background
        canvas.fillStyle(0.0f, 0.0f, 0.0f, 0.7f);
        canvas.fillRect(x, y - 24, 80, 22);

        // Draw landmarks
        for (int l = 0; l < static_cast<int>(FaceLandmark::Count); l++) {
            FaceLandmark lm = static_cast<FaceLandmark>(l);
            glm::vec2 pt = faces.landmark(f, lm);

            float px = pt.x * width;
            float py = pt.y * height;

            glm::vec4 color = getLandmarkColor(lm);
            color.a = conf;

            // Draw filled circle
            canvas.fillStyle(color.r, color.g, color.b, color.a);
            canvas.beginPath();
            canvas.arc(px, py, pointRadius, 0, 2.0f * 3.14159f);
            canvas.fill();

            // Draw outline
            canvas.strokeStyle(1.0f, 1.0f, 1.0f, color.a * 0.8f);
            canvas.lineWidth(2.0f);
            canvas.stroke();
        }

        // Draw eye line (connects both eyes)
        glm::vec2 leftEye = faces.landmark(f, FaceLandmark::LeftEye);
        glm::vec2 rightEye = faces.landmark(f, FaceLandmark::RightEye);

        canvas.strokeStyle(COLOR_EYE.r, COLOR_EYE.g, COLOR_EYE.b, conf * 0.5f);
        canvas.lineWidth(2.0f);
        canvas.beginPath();
        canvas.moveTo(leftEye.x * width, leftEye.y * height);
        canvas.lineTo(rightEye.x * width, rightEye.y * height);
        canvas.stroke();
    }

    // Display face count
    if (numFaces > 0) {
        canvas.fillStyle(0.0f, 1.0f, 0.5f, 0.9f);
        // Note: text rendering would go here if Canvas supported it
    }
}

VIVID_CHAIN(setup, update)
