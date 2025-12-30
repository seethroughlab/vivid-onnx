# vivid-ml Tests
# Unit tests for ML operators

cmake_minimum_required(VERSION 3.16)

# Fetch Catch2 v3
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Include Catch2's CMake helpers
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

add_executable(test_vivid_ml
    test_tensor.cpp
    test_pose_detector.cpp
)

target_link_libraries(test_vivid_ml PRIVATE
    Catch2::Catch2WithMain
    vivid-ml
)

# Tests need vivid-core for OperatorRegistry symbols (addon uses -undefined dynamic_lookup)
if(DEFINED VIVID_ROOT AND EXISTS "${VIVID_ROOT}/build/lib/libvivid-core.dylib")
    target_link_libraries(test_vivid_ml PRIVATE "${VIVID_ROOT}/build/lib/libvivid-core.dylib")
endif()

target_include_directories(test_vivid_ml PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Disable catch_discover_tests (crashes during build due to ONNX Runtime dylib path)
# Use add_test instead - run all tests as a single test case
add_test(NAME vivid_ml_tests COMMAND test_vivid_ml)
