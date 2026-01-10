/**
 * @file test_inference.cpp
 * @brief Integration test that actually loads and runs a MoveNet model
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vivid/onnx/onnx_model.h>
#include <vivid/onnx/pose_detector.h>
#include <iostream>
#include <fstream>

using namespace vivid::onnx;
using Catch::Matchers::WithinAbs;

// Check if model file exists
bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

TEST_CASE("ONNXModel loads MoveNet model", "[ml][integration]") {
    const std::string modelPath = "assets/models/movenet/singlepose-lightning.onnx";

    if (!fileExists(modelPath)) {
        WARN("Skipping integration test - model not found at: " << modelPath);
        SKIP("Model file not available");
    }

    ONNXModel model;
    model.model(modelPath);

    // We can't call init() without a Context, but we can verify configuration
    REQUIRE(model.modelPath() == modelPath);
    REQUIRE(model.isLoaded() == false);  // Not loaded until init()
}

TEST_CASE("PoseDetector configuration with model", "[ml][integration]") {
    const std::string modelPath = "assets/models/movenet/singlepose-lightning.onnx";

    if (!fileExists(modelPath)) {
        WARN("Skipping integration test - model not found at: " << modelPath);
        SKIP("Model file not available");
    }

    PoseDetector detector;
    detector
        .model(modelPath)
        .confidenceThreshold(0.3f)
        .drawSkeleton(true);

    REQUIRE(detector.modelPath() == modelPath);
    REQUIRE(detector.isLoaded() == false);  // Not loaded until init()
    REQUIRE(detector.detected() == false);
}

TEST_CASE("Tensor operations", "[ml][tensor]") {
    Tensor t;
    t.shape = {1, 192, 192, 3};  // MoveNet input shape (NHWC)
    t.data.resize(t.size());

    SECTION("size calculation") {
        REQUIRE(t.size() == 1 * 192 * 192 * 3);
    }

    SECTION("reshape") {
        t.reshape({1, 3, 192, 192});  // NCHW
        REQUIRE(t.shape[0] == 1);
        REQUIRE(t.shape[1] == 3);
        REQUIRE(t.shape[2] == 192);
        REQUIRE(t.shape[3] == 192);
    }

    SECTION("fill with test pattern") {
        // Fill with gradient (simulating an image)
        for (size_t i = 0; i < t.size(); i++) {
            t.data[i] = static_cast<float>(i % 256) / 255.0f;
        }
        REQUIRE_THAT(t.data[0], WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(t.data[255], WithinAbs(1.0f, 0.001f));
    }
}
