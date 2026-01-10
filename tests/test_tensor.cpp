/**
 * @file test_tensor.cpp
 * @brief Unit tests for Tensor class
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vivid/onnx/onnx_model.h>

using namespace vivid::onnx;
using Catch::Matchers::WithinAbs;

TEST_CASE("Tensor size calculation", "[ml][tensor]") {
    Tensor tensor;

    SECTION("empty shape returns 0") {
        tensor.shape = {};
        REQUIRE(tensor.size() == 0);
    }

    SECTION("1D shape") {
        tensor.shape = {10};
        REQUIRE(tensor.size() == 10);
    }

    SECTION("2D shape") {
        tensor.shape = {3, 4};
        REQUIRE(tensor.size() == 12);
    }

    SECTION("3D shape") {
        tensor.shape = {2, 3, 4};
        REQUIRE(tensor.size() == 24);
    }

    SECTION("4D shape (NCHW)") {
        tensor.shape = {1, 3, 224, 224};
        REQUIRE(tensor.size() == 1 * 3 * 224 * 224);
    }

    SECTION("4D shape (NHWC)") {
        tensor.shape = {1, 192, 192, 3};
        REQUIRE(tensor.size() == 1 * 192 * 192 * 3);
    }
}

TEST_CASE("Tensor reshape", "[ml][tensor]") {
    Tensor tensor;
    tensor.shape = {2, 3, 4};
    tensor.data.resize(tensor.size());

    SECTION("reshape to same total size succeeds") {
        tensor.reshape({4, 6});
        REQUIRE(tensor.shape.size() == 2);
        REQUIRE(tensor.shape[0] == 4);
        REQUIRE(tensor.shape[1] == 6);
        REQUIRE(tensor.size() == 24);
    }

    SECTION("reshape to 1D") {
        tensor.reshape({24});
        REQUIRE(tensor.shape.size() == 1);
        REQUIRE(tensor.shape[0] == 24);
    }

    SECTION("reshape to 4D") {
        tensor.reshape({1, 2, 3, 4});
        REQUIRE(tensor.shape.size() == 4);
        REQUIRE(tensor.size() == 24);
    }

    SECTION("reshape to different total size throws") {
        REQUIRE_THROWS_AS(tensor.reshape({10}), std::runtime_error);
        REQUIRE_THROWS_AS(tensor.reshape({2, 2}), std::runtime_error);
    }
}

TEST_CASE("Tensor data access", "[ml][tensor]") {
    Tensor tensor;
    tensor.shape = {2, 3};
    tensor.data.resize(tensor.size());

    SECTION("operator[] write") {
        tensor[0] = 1.0f;
        tensor[5] = 5.0f;
        REQUIRE_THAT(tensor.data[0], WithinAbs(1.0f, 0.001f));
        REQUIRE_THAT(tensor.data[5], WithinAbs(5.0f, 0.001f));
    }

    SECTION("operator[] read") {
        tensor.data[2] = 3.14f;
        REQUIRE_THAT(tensor[2], WithinAbs(3.14f, 0.001f));
    }

    SECTION("const operator[]") {
        tensor.data[0] = 42.0f;
        const Tensor& constRef = tensor;
        REQUIRE_THAT(constRef[0], WithinAbs(42.0f, 0.001f));
    }
}

TEST_CASE("Tensor types", "[ml][tensor]") {
    Tensor tensor;

    SECTION("default type is Float32") {
        REQUIRE(tensor.type == TensorType::Float32);
    }

    SECTION("UInt8 tensor") {
        tensor.type = TensorType::UInt8;
        tensor.shape = {1, 192, 192, 3};
        tensor.dataU8.resize(tensor.size());
        REQUIRE(tensor.dataU8.size() == 1 * 192 * 192 * 3);
    }

    SECTION("Int32 tensor") {
        tensor.type = TensorType::Int32;
        tensor.shape = {1, 192, 192, 3};
        tensor.dataI32.resize(tensor.size());
        REQUIRE(tensor.dataI32.size() == 1 * 192 * 192 * 3);
    }
}
