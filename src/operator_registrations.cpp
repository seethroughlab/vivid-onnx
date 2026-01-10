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
#include <vivid/ml/face_detector.h>

// Use type aliases to match REGISTER_OPERATOR macro pattern
using MLONNXModel = vivid::ml::ONNXModel;
using MLPoseDetector = vivid::ml::PoseDetector;
using MLFaceDetector = vivid::ml::FaceDetector;

// Register ONNXModel - generic ONNX model inference operator
REGISTER_OPERATOR(MLONNXModel, "ML", "Run ONNX model inference on input texture", true);

// Register PoseDetector - MoveNet body tracking
REGISTER_OPERATOR(MLPoseDetector, "ML", "Detect body poses using MoveNet model", true);

// Register FaceDetector - BlazeFace face detection
REGISTER_OPERATOR(MLFaceDetector, "ML", "Detect faces using BlazeFace model", true);
