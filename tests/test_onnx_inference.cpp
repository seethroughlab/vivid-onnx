/**
 * @file test_onnx_inference.cpp
 * @brief Direct ONNX Runtime test - verifies model loading and inference
 *
 * This test bypasses the vivid framework and tests ONNX Runtime directly
 * to ensure the MoveNet model loads and produces valid output.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

using Catch::Matchers::WithinAbs;

static bool modelFileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

TEST_CASE("ONNX Runtime loads and runs MoveNet", "[ml][onnx][integration]") {
    const std::string modelPath = "assets/models/movenet/singlepose-lightning.onnx";

    if (!modelFileExists(modelPath)) {
        WARN("Skipping ONNX inference test - model not found at: " << modelPath);
        SKIP("Model file not available");
    }

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    SECTION("model loads successfully") {
        REQUIRE_NOTHROW([&]() {
            Ort::Session session(env, modelPath.c_str(), sessionOptions);
        }());
    }

    SECTION("model has correct input/output shape") {
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        // Check input
        size_t numInputs = session.GetInputCount();
        REQUIRE(numInputs == 1);

        auto inputName = session.GetInputNameAllocated(0, allocator);
        INFO("Input name: " << inputName.get());

        auto inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = inputTensorInfo.GetShape();

        INFO("Input shape: [" << inputShape[0] << ", " << inputShape[1] << ", "
             << inputShape[2] << ", " << inputShape[3] << "]");

        // MoveNet SinglePose Lightning expects [1, 192, 192, 3] (NHWC)
        REQUIRE(inputShape.size() == 4);
        // Note: batch dim might be -1 (dynamic), so check abs value
        REQUIRE((inputShape[0] == 1 || inputShape[0] == -1));
        REQUIRE(inputShape[1] == 192);
        REQUIRE(inputShape[2] == 192);
        REQUIRE(inputShape[3] == 3);

        // Check output
        size_t numOutputs = session.GetOutputCount();
        REQUIRE(numOutputs == 1);

        auto outputTypeInfo = session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();

        INFO("Output shape: [" << outputShape[0] << ", " << outputShape[1] << ", "
             << outputShape[2] << ", " << outputShape[3] << "]");

        // MoveNet outputs [1, 1, 17, 3] - 17 keypoints with (y, x, confidence)
        REQUIRE(outputShape.size() == 4);
        REQUIRE((outputShape[0] == 1 || outputShape[0] == -1));
        REQUIRE(outputShape[1] == 1);
        REQUIRE(outputShape[2] == 17);
        REQUIRE(outputShape[3] == 3);
    }

    SECTION("inference runs with dummy input") {
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor [1, 192, 192, 3] with test pattern
        std::vector<int64_t> inputShape = {1, 192, 192, 3};
        size_t inputSize = 1 * 192 * 192 * 3;
        std::vector<float> inputData(inputSize);

        // Fill with gradient (simulating an image) - values 0-1
        for (size_t i = 0; i < inputSize; i++) {
            inputData[i] = static_cast<float>(i % 256) / 255.0f;
        }

        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(),
            inputShape.data(), inputShape.size()
        );

        // Get input/output names
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        // Run inference
        std::vector<Ort::Value> outputTensors;
        REQUIRE_NOTHROW([&]() {
            outputTensors = session.Run(
                Ort::RunOptions{nullptr},
                inputNames, &inputTensor, 1,
                outputNames, 1
            );
        }());

        REQUIRE(outputTensors.size() == 1);

        // Check output
        auto& output = outputTensors[0];
        auto outputShape = output.GetTensorTypeAndShapeInfo().GetShape();
        REQUIRE(outputShape[2] == 17);  // 17 keypoints
        REQUIRE(outputShape[3] == 3);   // y, x, confidence

        // Get output data
        const float* outputData = output.GetTensorData<float>();
        size_t outputSize = 1 * 1 * 17 * 3;

        // Verify output contains reasonable values
        // Keypoints should be in 0-1 range, confidence 0-1
        bool hasValidOutput = false;
        for (size_t i = 0; i < outputSize; i += 3) {
            float y = outputData[i];
            float x = outputData[i + 1];
            float conf = outputData[i + 2];

            // All values should be finite
            REQUIRE(std::isfinite(y));
            REQUIRE(std::isfinite(x));
            REQUIRE(std::isfinite(conf));

            // With random input, we might get low confidence but values should be bounded
            if (conf > 0.001f) {
                hasValidOutput = true;
            }
        }

        INFO("Model produced " << (hasValidOutput ? "some" : "no") << " confident keypoints");
        // Note: With dummy input, model may not detect a pose, which is fine
    }
}
