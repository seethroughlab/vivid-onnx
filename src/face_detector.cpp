#include <vivid/ml/face_detector.h>
#include <vivid/context.h>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace vivid::ml {

// Static anchor box used for decoding
static DetectedFace s_emptyFace = {};

FaceDetector::FaceDetector() {
    generateAnchors();
}

FaceDetector::~FaceDetector() = default;

FaceDetector& FaceDetector::input(Operator* op) {
    ONNXModel::input(op);
    return *this;
}

FaceDetector& FaceDetector::model(const std::string& path) {
    ONNXModel::model(path);
    return *this;
}

FaceDetector& FaceDetector::confidenceThreshold(float threshold) {
    m_confidenceThreshold = std::clamp(threshold, 0.0f, 1.0f);
    return *this;
}

FaceDetector& FaceDetector::maxFaces(int max) {
    m_maxFaces = std::max(1, max);
    return *this;
}

const DetectedFace& FaceDetector::face(int index) const {
    if (index < 0 || index >= static_cast<int>(m_faces.size())) {
        return s_emptyFace;
    }
    return m_faces[index];
}

glm::vec4 FaceDetector::boundingBox(int faceIndex) const {
    if (faceIndex < 0 || faceIndex >= static_cast<int>(m_faces.size())) {
        return glm::vec4(0.0f);
    }
    return m_faces[faceIndex].bbox;
}

glm::vec2 FaceDetector::landmark(int faceIndex, FaceLandmark lm) const {
    return landmark(faceIndex, static_cast<int>(lm));
}

glm::vec2 FaceDetector::landmark(int faceIndex, int landmarkIndex) const {
    if (faceIndex < 0 || faceIndex >= static_cast<int>(m_faces.size()) ||
        landmarkIndex < 0 || landmarkIndex >= 6) {
        return glm::vec2(0.0f);
    }
    return m_faces[faceIndex].landmarks[landmarkIndex];
}

float FaceDetector::confidence(int faceIndex) const {
    if (faceIndex < 0 || faceIndex >= static_cast<int>(m_faces.size())) {
        return 0.0f;
    }
    return m_faces[faceIndex].confidence;
}

void FaceDetector::generateAnchors() {
    // BlazeFace front model anchors
    // Two feature maps: 16x16 (2 anchors/cell) and 8x8 (6 anchors/cell)
    // Total: 16*16*2 + 8*8*6 = 512 + 384 = 896 anchors

    m_anchors.clear();
    m_anchors.reserve(896);

    // Feature map 1: 16x16, 2 anchors per cell
    int size1 = 16;
    int anchorsPerCell1 = 2;
    for (int y = 0; y < size1; y++) {
        for (int x = 0; x < size1; x++) {
            float cx = (x + 0.5f) / size1;
            float cy = (y + 0.5f) / size1;
            for (int a = 0; a < anchorsPerCell1; a++) {
                m_anchors.push_back({cx, cy, 1.0f, 1.0f});
            }
        }
    }

    // Feature map 2: 8x8, 6 anchors per cell
    int size2 = 8;
    int anchorsPerCell2 = 6;
    for (int y = 0; y < size2; y++) {
        for (int x = 0; x < size2; x++) {
            float cx = (x + 0.5f) / size2;
            float cy = (y + 0.5f) / size2;
            for (int a = 0; a < anchorsPerCell2; a++) {
                m_anchors.push_back({cx, cy, 1.0f, 1.0f});
            }
        }
    }

    std::cout << "[FaceDetector] Generated " << m_anchors.size() << " anchors" << std::endl;
}

void FaceDetector::onModelLoaded() {
    // BlazeFace front model expects 128x128 input
    // Check model input shape and update if different
    if (!m_inputShapes.empty() && m_inputShapes[0].size() >= 4) {
        int64_t h = m_inputShapes[0][1];
        int64_t w = m_inputShapes[0][2];

        // Handle dynamic sizes
        if (h > 32 && w > 32) {
            m_inputWidth = static_cast<int>(w);
            m_inputHeight = static_cast<int>(h);
        }
    }

    std::cout << "[FaceDetector] Model input size: " << m_inputWidth << "x" << m_inputHeight << std::endl;
    std::cout << "[FaceDetector] Outputs: " << outputCount() << std::endl;
    for (size_t i = 0; i < outputCount(); i++) {
        std::cout << "  Output " << i << ": " << outputName(i) << " shape=[";
        for (size_t j = 0; j < outputShape(i).size(); j++) {
            if (j > 0) std::cout << ",";
            std::cout << outputShape(i)[j];
        }
        std::cout << "]" << std::endl;
    }
}

void FaceDetector::prepareInputTensor(Context& ctx, Tensor& tensor) {
    // BlazeFace expects NHWC format with values 0-255 or normalized
    // Shape: [1, height, width, channels]

    int channels = (tensor.shape.size() >= 4) ? static_cast<int>(tensor.shape[3]) : 3;
    tensor.shape = {1, m_inputHeight, m_inputWidth, channels};

    // Resize tensor buffer
    size_t tensorSize = tensor.size();
    if (tensor.type == TensorType::UInt8 && tensor.dataU8.size() != tensorSize) {
        tensor.dataU8.resize(tensorSize);
    } else if (tensor.type == TensorType::Int32 && tensor.dataI32.size() != tensorSize) {
        tensor.dataI32.resize(tensorSize);
    } else if (tensor.data.size() != tensorSize) {
        tensor.data.resize(tensorSize);
    }

    // Convert input texture to tensor
    bool success = textureToTensor(ctx, tensor, m_inputWidth, m_inputHeight);

    if (!success) {
        // Fill with gray placeholder if conversion fails
        if (tensor.type == TensorType::UInt8) {
            std::fill(tensor.dataU8.begin(), tensor.dataU8.end(), uint8_t(128));
        } else if (tensor.type == TensorType::Int32) {
            std::fill(tensor.dataI32.begin(), tensor.dataI32.end(), int32_t(128));
        } else {
            std::fill(tensor.data.begin(), tensor.data.end(), 0.5f);
        }
    }
}

void FaceDetector::processOutputTensor(const Tensor& tensor) {
    m_faces.clear();

    // BlazeFace outputs:
    // Output 0 (regressors): [1, 896, 16] - box offsets + landmarks
    // Output 1 (classificators): [1, 896, 1] - confidence scores

    if (m_outputTensors.size() < 2) {
        // Single output model - try to parse as combined format
        if (tensor.data.empty()) return;

        // Some models output [1, 896, 17] with confidence as last value
        int numAnchors = static_cast<int>(m_anchors.size());
        int valuesPerAnchor = static_cast<int>(tensor.data.size()) / numAnchors;

        if (valuesPerAnchor >= 17) {
            // Combined format: 16 box values + 1 confidence
            for (int i = 0; i < numAnchors && static_cast<int>(m_faces.size()) < m_maxFaces; i++) {
                const float* data = &tensor.data[i * valuesPerAnchor];
                float score = 1.0f / (1.0f + std::exp(-data[16])); // Sigmoid

                if (score >= m_confidenceThreshold) {
                    DetectedFace face;
                    face.confidence = score;

                    // Decode bounding box from anchor
                    float cx = data[0] / m_inputWidth + m_anchors[i][0];
                    float cy = data[1] / m_inputHeight + m_anchors[i][1];
                    float w = data[2] / m_inputWidth;
                    float h = data[3] / m_inputHeight;

                    face.bbox = glm::vec4(cx - w/2, cy - h/2, w, h);

                    // Decode landmarks
                    for (int l = 0; l < 6; l++) {
                        float lx = data[4 + l*2] / m_inputWidth + m_anchors[i][0];
                        float ly = data[4 + l*2 + 1] / m_inputHeight + m_anchors[i][1];
                        face.landmarks[l] = glm::vec2(lx, ly);
                    }

                    m_faces.push_back(face);
                }
            }
        }
    } else {
        // Two-output model (regressors + classificators)
        const Tensor& regressors = m_outputTensors[0];
        const Tensor& scores = m_outputTensors[1];

        if (regressors.data.empty() || scores.data.empty()) return;

        decodeDetections(regressors.data.data(), scores.data.data(),
                        static_cast<int>(m_anchors.size()));
    }

    // Apply non-max suppression
    nonMaxSuppression();
}

void FaceDetector::decodeDetections(const float* regressors, const float* scores,
                                     int numAnchors) {
    // Decode all detections above threshold
    std::vector<DetectedFace> candidates;

    for (int i = 0; i < numAnchors; i++) {
        // Apply sigmoid to score
        float score = 1.0f / (1.0f + std::exp(-scores[i]));

        if (score >= m_confidenceThreshold) {
            DetectedFace face;
            face.confidence = score;

            const float* box = &regressors[i * 16];

            // Decode bounding box
            // BlazeFace uses center-offset encoding
            float cx = box[0] / m_inputWidth + m_anchors[i][0];
            float cy = box[1] / m_inputHeight + m_anchors[i][1];
            float w = box[2] / m_inputWidth;
            float h = box[3] / m_inputHeight;

            // Convert to x,y,w,h format (normalized)
            face.bbox = glm::vec4(
                std::clamp(cx - w/2, 0.0f, 1.0f),
                std::clamp(cy - h/2, 0.0f, 1.0f),
                std::clamp(w, 0.0f, 1.0f),
                std::clamp(h, 0.0f, 1.0f)
            );

            // Decode 6 landmarks
            for (int l = 0; l < 6; l++) {
                float lx = box[4 + l*2] / m_inputWidth + m_anchors[i][0];
                float ly = box[4 + l*2 + 1] / m_inputHeight + m_anchors[i][1];
                face.landmarks[l] = glm::vec2(
                    std::clamp(lx, 0.0f, 1.0f),
                    std::clamp(ly, 0.0f, 1.0f)
                );
            }

            candidates.push_back(face);
        }
    }

    // Sort by confidence
    std::sort(candidates.begin(), candidates.end(),
              [](const DetectedFace& a, const DetectedFace& b) {
                  return a.confidence > b.confidence;
              });

    // Take top candidates before NMS
    if (static_cast<int>(candidates.size()) > m_maxFaces * 3) {
        candidates.resize(m_maxFaces * 3);
    }

    m_faces = std::move(candidates);
}

void FaceDetector::nonMaxSuppression() {
    if (m_faces.empty()) return;

    const float iouThreshold = 0.3f;
    std::vector<DetectedFace> kept;
    std::vector<bool> suppressed(m_faces.size(), false);

    for (size_t i = 0; i < m_faces.size() && static_cast<int>(kept.size()) < m_maxFaces; i++) {
        if (suppressed[i]) continue;

        kept.push_back(m_faces[i]);
        const auto& boxA = m_faces[i].bbox;

        // Suppress overlapping detections
        for (size_t j = i + 1; j < m_faces.size(); j++) {
            if (suppressed[j]) continue;

            const auto& boxB = m_faces[j].bbox;

            // Calculate IoU
            float x1 = std::max(boxA.x, boxB.x);
            float y1 = std::max(boxA.y, boxB.y);
            float x2 = std::min(boxA.x + boxA.z, boxB.x + boxB.z);
            float y2 = std::min(boxA.y + boxA.w, boxB.y + boxB.w);

            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float areaA = boxA.z * boxA.w;
            float areaB = boxB.z * boxB.w;
            float unionArea = areaA + areaB - intersection;

            float iou = (unionArea > 0) ? intersection / unionArea : 0.0f;

            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    m_faces = std::move(kept);
}

} // namespace vivid::ml
