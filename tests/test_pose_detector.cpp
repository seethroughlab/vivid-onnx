/**
 * @file test_pose_detector.cpp
 * @brief Unit tests for PoseDetector operator
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vivid/onnx/pose_detector.h>

using namespace vivid::onnx;
using Catch::Matchers::WithinAbs;

TEST_CASE("PoseDetector defaults", "[ml][pose]") {
    PoseDetector detector;

    SECTION("not detected initially") {
        REQUIRE(detector.detected() == false);
    }

    SECTION("name is correct") {
        REQUIRE(detector.name() == "PoseDetector");
    }

    SECTION("model not loaded initially") {
        REQUIRE(detector.isLoaded() == false);
    }
}

TEST_CASE("PoseDetector keypoint access", "[ml][pose]") {
    PoseDetector detector;

    SECTION("keypoint by enum returns zero when not detected") {
        auto kp = detector.keypoint(Keypoint::Nose);
        REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(kp.y, WithinAbs(0.0f, 0.001f));
    }

    SECTION("keypoint by index returns zero when not detected") {
        auto kp = detector.keypoint(0);
        REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(kp.y, WithinAbs(0.0f, 0.001f));
    }

    SECTION("invalid index returns zero") {
        auto kp = detector.keypoint(-1);
        REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));

        kp = detector.keypoint(17);
        REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));

        kp = detector.keypoint(100);
        REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));
    }

    SECTION("all 17 keypoints accessible") {
        for (int i = 0; i < 17; i++) {
            auto kp = detector.keypoint(i);
            // Should not crash, just return zero
            REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));
        }
    }
}

TEST_CASE("PoseDetector confidence access", "[ml][pose]") {
    PoseDetector detector;

    SECTION("confidence by enum returns zero when not detected") {
        REQUIRE_THAT(detector.confidence(Keypoint::Nose), WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(detector.confidence(Keypoint::LeftWrist), WithinAbs(0.0f, 0.001f));
    }

    SECTION("confidence by index returns zero when not detected") {
        REQUIRE_THAT(detector.confidence(0), WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(detector.confidence(10), WithinAbs(0.0f, 0.001f));
    }

    SECTION("invalid index returns zero confidence") {
        REQUIRE_THAT(detector.confidence(-1), WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(detector.confidence(17), WithinAbs(0.0f, 0.001f));
        REQUIRE_THAT(detector.confidence(100), WithinAbs(0.0f, 0.001f));
    }
}

TEST_CASE("PoseDetector configuration chaining", "[ml][pose]") {
    PoseDetector detector;

    SECTION("confidenceThreshold returns self") {
        PoseDetector& ref = detector.confidenceThreshold(0.5f);
        REQUIRE(&ref == &detector);
    }

    SECTION("drawSkeleton returns self") {
        PoseDetector& ref = detector.drawSkeleton(true);
        REQUIRE(&ref == &detector);
    }

    SECTION("model returns self") {
        PoseDetector& ref = detector.model("test.onnx");
        REQUIRE(&ref == &detector);
    }

    SECTION("chaining works") {
        PoseDetector& ref = detector
            .model("test.onnx")
            .confidenceThreshold(0.4f)
            .drawSkeleton(false);
        REQUIRE(&ref == &detector);
    }
}

TEST_CASE("Keypoint enum values", "[ml][pose]") {
    SECTION("keypoint indices are correct") {
        REQUIRE(static_cast<int>(Keypoint::Nose) == 0);
        REQUIRE(static_cast<int>(Keypoint::LeftEye) == 1);
        REQUIRE(static_cast<int>(Keypoint::RightEye) == 2);
        REQUIRE(static_cast<int>(Keypoint::LeftEar) == 3);
        REQUIRE(static_cast<int>(Keypoint::RightEar) == 4);
        REQUIRE(static_cast<int>(Keypoint::LeftShoulder) == 5);
        REQUIRE(static_cast<int>(Keypoint::RightShoulder) == 6);
        REQUIRE(static_cast<int>(Keypoint::LeftElbow) == 7);
        REQUIRE(static_cast<int>(Keypoint::RightElbow) == 8);
        REQUIRE(static_cast<int>(Keypoint::LeftWrist) == 9);
        REQUIRE(static_cast<int>(Keypoint::RightWrist) == 10);
        REQUIRE(static_cast<int>(Keypoint::LeftHip) == 11);
        REQUIRE(static_cast<int>(Keypoint::RightHip) == 12);
        REQUIRE(static_cast<int>(Keypoint::LeftKnee) == 13);
        REQUIRE(static_cast<int>(Keypoint::RightKnee) == 14);
        REQUIRE(static_cast<int>(Keypoint::LeftAnkle) == 15);
        REQUIRE(static_cast<int>(Keypoint::RightAnkle) == 16);
        REQUIRE(static_cast<int>(Keypoint::Count) == 17);
    }
}

TEST_CASE("Skeleton connections", "[ml][pose]") {
    SECTION("has 16 connections") {
        REQUIRE(SKELETON_CONNECTIONS.size() == 16);
    }

    SECTION("connections are valid keypoint pairs") {
        for (const auto& conn : SKELETON_CONNECTIONS) {
            int from = static_cast<int>(conn.from);
            int to = static_cast<int>(conn.to);
            REQUIRE(from >= 0);
            REQUIRE(from < 17);
            REQUIRE(to >= 0);
            REQUIRE(to < 17);
        }
    }
}

TEST_CASE("PoseDetector keypoints array access", "[ml][pose]") {
    PoseDetector detector;

    SECTION("keypoints() returns array of 17") {
        const auto& keypoints = detector.keypoints();
        REQUIRE(keypoints.size() == 17);
    }

    SECTION("all keypoints initialized to zero") {
        const auto& keypoints = detector.keypoints();
        for (const auto& kp : keypoints) {
            REQUIRE_THAT(kp.x, WithinAbs(0.0f, 0.001f));
            REQUIRE_THAT(kp.y, WithinAbs(0.0f, 0.001f));
            REQUIRE_THAT(kp.z, WithinAbs(0.0f, 0.001f));  // confidence
        }
    }
}
