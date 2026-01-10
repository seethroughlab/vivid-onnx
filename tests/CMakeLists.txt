# vivid-onnx Tests
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
    test_inference.cpp
    test_onnx_inference.cpp
)

target_link_libraries(test_vivid_ml PRIVATE
    Catch2::Catch2WithMain
    vivid-onnx
)

# Tests need vivid-core for symbol resolution (addon uses -undefined dynamic_lookup)
if(DEFINED VIVID_ROOT)
    # Find vivid-core library
    set(VIVID_CORE_FOUND FALSE)

    # Check SDK layout first, then source tree layout
    if(EXISTS "${VIVID_ROOT}/lib/libvivid-core.dylib")
        set(VIVID_CORE_LIBRARY "${VIVID_ROOT}/lib/libvivid-core.dylib")
        set(VIVID_CORE_FOUND TRUE)
    elseif(EXISTS "${VIVID_ROOT}/lib/libvivid-core.so")
        set(VIVID_CORE_LIBRARY "${VIVID_ROOT}/lib/libvivid-core.so")
        set(VIVID_CORE_FOUND TRUE)
    elseif(EXISTS "${VIVID_ROOT}/build/lib/libvivid-core.dylib")
        set(VIVID_CORE_LIBRARY "${VIVID_ROOT}/build/lib/libvivid-core.dylib")
        set(VIVID_CORE_FOUND TRUE)
    elseif(EXISTS "${VIVID_ROOT}/build/lib/libvivid-core.so")
        set(VIVID_CORE_LIBRARY "${VIVID_ROOT}/build/lib/libvivid-core.so")
        set(VIVID_CORE_FOUND TRUE)
    elseif(WIN32 AND DEFINED VIVID_CORE_LIB)
        set(VIVID_CORE_LIBRARY "${VIVID_CORE_LIB}")
        set(VIVID_CORE_FOUND TRUE)
    endif()

    if(VIVID_CORE_FOUND)
        message(STATUS "[vivid-onnx tests] Found vivid-core: ${VIVID_CORE_LIBRARY}")
        target_link_libraries(test_vivid_ml PRIVATE "${VIVID_CORE_LIBRARY}")

        # Set rpath so test can find vivid-core at runtime
        if(APPLE)
            get_filename_component(VIVID_CORE_DIR "${VIVID_CORE_LIBRARY}" DIRECTORY)
            set_target_properties(test_vivid_ml PROPERTIES
                BUILD_RPATH "${VIVID_CORE_DIR}"
                INSTALL_RPATH "${VIVID_CORE_DIR}"
            )
        endif()
    else()
        message(WARNING "[vivid-onnx tests] vivid-core not found - tests may fail at runtime")
    endif()
endif()

target_include_directories(test_vivid_ml PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${VIVID_INCLUDE_DIR}
    ${VIVID_DEP_INCLUDE_DIRS}
)

# Copy runtime DLLs to test directory on Windows
if(WIN32)
    add_custom_command(TARGET test_vivid_ml POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            "$<TARGET_FILE_DIR:test_vivid_ml>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:vivid-onnx>"
            "$<TARGET_FILE_DIR:test_vivid_ml>"
        COMMENT "Copying DLLs to test directory"
    )

    # Copy all DLLs from vivid SDK lib directory (vivid-core.dll, glfw3.dll, etc.)
    if(DEFINED VIVID_ROOT AND EXISTS "${VIVID_ROOT}/lib")
        file(GLOB VIVID_SDK_DLLS "${VIVID_ROOT}/lib/*.dll")
        foreach(DLL_FILE ${VIVID_SDK_DLLS})
            add_custom_command(TARGET test_vivid_ml POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL_FILE}"
                    "$<TARGET_FILE_DIR:test_vivid_ml>"
            )
        endforeach()
    endif()
endif()

# Disable catch_discover_tests (crashes during build due to ONNX Runtime dylib path)
# Use add_test instead - run all tests as a single test case
# Set working directory to source root so tests can find model files
add_test(NAME vivid_ml_tests COMMAND test_vivid_ml)
set_tests_properties(vivid_ml_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
