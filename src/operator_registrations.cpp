/**
 * @file operator_registrations.cpp
 * @brief Static operator registrations for vivid-onnx addon
 *
 * This file registers all vivid-onnx operators with the global operator registry
 * so they appear in `vivid operators --json` and the VS Code extension.
 */

#include <vivid/operator_registry.h>
#include <vivid/ml/onnx_model.h>
#include <vivid/ml/pose_detector.h>

namespace vivid::ml {

// Register ONNXModel - generic ONNX model inference operator
REGISTER_ADDON_OPERATOR(ONNXModel, "ML", "Run ONNX model inference on input texture", true, "vivid-onnx");

// Register PoseDetector - MoveNet body tracking
REGISTER_ADDON_OPERATOR(PoseDetector, "ML", "Detect body poses using MoveNet model", true, "vivid-onnx");

} // namespace vivid::ml
