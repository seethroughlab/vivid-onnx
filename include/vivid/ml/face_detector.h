// FaceDetector - BlazeFace face detection
//
// Detects faces using Google's BlazeFace model, returning bounding boxes
// and 6 facial landmarks (eyes, ears, nose, mouth).
//
// Usage:
//   chain.add<FaceDetector>("faces")
//       .input(&webcam)
//       .model("assets/models/blazeface/face_detection_front.onnx");
//
//   void update(Context& ctx) {
//       auto& faces = chain.get<FaceDetector>("faces");
//       for (int i = 0; i < faces.faceCount(); i++) {
//           auto bbox = faces.boundingBox(i);
//           glm::vec2 nose = faces.landmark(i, FaceLandmark::Nose);
//       }
//   }

#pragma once

#include "onnx_model.h"
#include <glm/glm.hpp>
#include <array>
#include <vector>

namespace vivid::ml {

/// BlazeFace landmark indices (6 points)
enum class FaceLandmark : int {
    RightEye = 0,
    LeftEye = 1,
    Nose = 2,
    Mouth = 3,
    RightEar = 4,
    LeftEar = 5,
    Count = 6
};

/// Detected face with bounding box and landmarks
struct DetectedFace {
    /// Bounding box (normalized 0-1): x, y, width, height
    glm::vec4 bbox;

    /// 6 facial landmarks (normalized 0-1 coordinates)
    std::array<glm::vec2, 6> landmarks;

    /// Detection confidence (0-1)
    float confidence;
};

class FaceDetector : public ONNXModel {
public:
    FaceDetector();
    ~FaceDetector() override;

    // Configuration
    FaceDetector& input(Operator* op);
    FaceDetector& model(const std::string& path);
    FaceDetector& confidenceThreshold(float threshold);
    FaceDetector& maxFaces(int max);

    // Detection results
    bool detected() const { return !m_faces.empty(); }
    int faceCount() const { return static_cast<int>(m_faces.size()); }

    /// Get detected face by index
    const DetectedFace& face(int index) const;

    /// Get bounding box for face (normalized 0-1: x, y, width, height)
    glm::vec4 boundingBox(int faceIndex = 0) const;

    /// Get landmark position (normalized 0-1)
    glm::vec2 landmark(int faceIndex, FaceLandmark lm) const;
    glm::vec2 landmark(int faceIndex, int landmarkIndex) const;

    /// Get face confidence
    float confidence(int faceIndex = 0) const;

    /// Get all detected faces
    const std::vector<DetectedFace>& faces() const { return m_faces; }

    // Operator interface
    std::string name() const override { return "FaceDetector"; }

protected:
    void onModelLoaded() override;
    void prepareInputTensor(Context& ctx, Tensor& tensor) override;
    void processOutputTensor(const Tensor& tensor) override;

private:
    void decodeDetections(const float* regressors, const float* scores,
                          int numAnchors);
    void nonMaxSuppression();

    float m_confidenceThreshold = 0.5f;
    int m_maxFaces = 10;

    // Detected faces
    std::vector<DetectedFace> m_faces;

    // Model input size (BlazeFace uses 128x128)
    int m_inputWidth = 128;
    int m_inputHeight = 128;

    // Anchor configuration for BlazeFace
    std::vector<std::array<float, 4>> m_anchors;
    void generateAnchors();
};

} // namespace vivid::ml
