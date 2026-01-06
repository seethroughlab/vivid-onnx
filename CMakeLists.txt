cmake_minimum_required(VERSION 3.16)
project(vivid-ml VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vivid ML Addon
# Machine learning inference for creative applications
# - ONNXModel: Generic ONNX model inference
# - PoseDetector: MoveNet body tracking

include(FetchContent)

# -----------------------------------------------------------------------------
# Find vivid headers (required dependency)
# -----------------------------------------------------------------------------
# vivid-ml is a standalone addon that only needs vivid headers at compile time.
# Symbols from vivid-core are resolved at load time when the addon is loaded
# by the vivid runtime. This allows vivid-ml to be built independently.
#
# Options:
#   1. As subdirectory of main vivid repo (TARGET vivid-core exists)
#   2. Standalone build with VIVID_ROOT pointing to vivid SDK
#   3. Standalone build with find_package(vivid)

set(VIVID_BUILD_MODE "standalone")

if(TARGET vivid-core)
    # Building as part of the main vivid repo
    set(VIVID_BUILD_MODE "in-tree")
    message(STATUS "[vivid-ml] Building in-tree (vivid-core target exists)")
elseif(DEFINED VIVID_ROOT)
    # Standalone build with SDK path or source tree
    message(STATUS "[vivid-ml] Using VIVID_ROOT: ${VIVID_ROOT}")

    # Check for SDK layout (include/vivid) or source tree layout (core/include/vivid)
    if(EXISTS "${VIVID_ROOT}/include/vivid/vivid.h")
        # SDK package layout - all headers bundled in include/
        set(VIVID_INCLUDE_DIR "${VIVID_ROOT}/include")
        set(VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/include")  # GLM, GLFW, WebGPU all in include/
    elseif(EXISTS "${VIVID_ROOT}/core/include/vivid/vivid.h")
        # Source tree layout - headers spread across build/_deps
        set(VIVID_INCLUDE_DIR "${VIVID_ROOT}/core/include")

        # Find dependency headers in build/_deps
        set(VIVID_DEP_INCLUDE_DIRS "")

        # WebGPU
        if(EXISTS "${VIVID_ROOT}/build/_deps/wgpu/include/webgpu/webgpu.h")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/wgpu/include")
        else()
            message(FATAL_ERROR "[vivid-ml] WebGPU headers not found. Please build vivid first:\n"
                "  cd ${VIVID_ROOT} && cmake -B build && cmake --build build")
        endif()

        # GLM
        if(EXISTS "${VIVID_ROOT}/build/_deps/glm-src/glm/glm.hpp")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/glm-src")
        else()
            message(FATAL_ERROR "[vivid-ml] GLM headers not found. Please build vivid first.")
        endif()

        # GLFW
        if(EXISTS "${VIVID_ROOT}/build/_deps/glfw-src/include/GLFW/glfw3.h")
            list(APPEND VIVID_DEP_INCLUDE_DIRS "${VIVID_ROOT}/build/_deps/glfw-src/include")
        else()
            message(FATAL_ERROR "[vivid-ml] GLFW headers not found. Please build vivid first.")
        endif()
    else()
        message(FATAL_ERROR "[vivid-ml] VIVID_ROOT specified but headers not found.\n"
            "  Checked: ${VIVID_ROOT}/include/vivid/vivid.h\n"
            "  Checked: ${VIVID_ROOT}/core/include/vivid/vivid.h")
    endif()
    message(STATUS "[vivid-ml] Found vivid headers at: ${VIVID_INCLUDE_DIR}")
    message(STATUS "[vivid-ml] Dependency include dirs: ${VIVID_DEP_INCLUDE_DIRS}")

    # On Windows, we need the import library for linking
    if(WIN32)
        find_library(VIVID_CORE_LIB vivid-core PATHS "${VIVID_ROOT}/lib" REQUIRED)
        message(STATUS "[vivid-ml] Found vivid-core import library: ${VIVID_CORE_LIB}")
    endif()
else()
    # Try find_package
    find_package(vivid QUIET)
    if(vivid_FOUND)
        set(VIVID_BUILD_MODE "installed")
        message(STATUS "[vivid-ml] Found vivid via find_package")
    else()
        message(FATAL_ERROR
            "[vivid-ml] Could not find vivid headers. Options:\n"
            "  1. Set VIVID_ROOT to your vivid SDK installation\n"
            "  2. Install vivid and ensure it's in CMAKE_PREFIX_PATH\n"
            "  3. Build as part of the main vivid repo"
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# ONNX Runtime dependency
# -----------------------------------------------------------------------------
set(ONNXRUNTIME_VERSION "1.19.2")

# Platform-specific ONNX Runtime download
if(APPLE)
    # Detect target architecture (handles cross-compilation)
    if(CMAKE_OSX_ARCHITECTURES)
        set(TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
    else()
        set(TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    if(TARGET_ARCH MATCHES "arm64")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
    endif()
    message(STATUS "[vivid-ml] Target architecture: ${TARGET_ARCH}")
elseif(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
elseif(UNIX)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
endif()

message(STATUS "[vivid-ml] Downloading ONNX Runtime from: ${ONNXRUNTIME_URL}")

FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
)
FetchContent_MakeAvailable(onnxruntime)

# Find the ONNX Runtime headers and libraries
set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/include")
if(APPLE)
    set(ONNXRUNTIME_LIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib")
elseif(WIN32)
    set(ONNXRUNTIME_LIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll")
else()
    set(ONNXRUNTIME_LIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
endif()

message(STATUS "[vivid-ml] ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "[vivid-ml] ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# -----------------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------------
set(ML_SOURCES
    src/onnx_model.cpp
    src/pose_detector.cpp
    src/operator_registrations.cpp
)

add_library(vivid-ml SHARED ${ML_SOURCES})

target_include_directories(vivid-ml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link dependencies based on build mode
if(VIVID_BUILD_MODE STREQUAL "in-tree")
    # Building as part of main vivid repo - link directly
    target_link_libraries(vivid-ml PUBLIC vivid-core ${ONNXRUNTIME_LIB})
elseif(VIVID_BUILD_MODE STREQUAL "standalone")
    # Standalone build - use headers only, resolve symbols at load time
    target_include_directories(vivid-ml PUBLIC ${VIVID_INCLUDE_DIR} ${VIVID_DEP_INCLUDE_DIRS})
    target_link_libraries(vivid-ml PUBLIC ${ONNXRUNTIME_LIB})

    if(WIN32)
        # Windows requires import library for symbol resolution
        target_link_libraries(vivid-ml PUBLIC ${VIVID_CORE_LIB})
    else()
        # macOS/Linux: symbols resolved at load time by vivid runtime
        if(APPLE)
            target_link_options(vivid-ml PRIVATE -undefined dynamic_lookup)
        else()
            # Linux: allow undefined symbols to be resolved at load time
            target_link_options(vivid-ml PRIVATE -Wl,--allow-shlib-undefined)
        endif()
    endif()
else()
    # Installed vivid (via find_package)
    target_link_libraries(vivid-ml PUBLIC vivid::core ${ONNXRUNTIME_LIB})
endif()

# -----------------------------------------------------------------------------
# Platform-specific configuration
# -----------------------------------------------------------------------------
if(APPLE)
    set_target_properties(vivid-ml PROPERTIES
        INSTALL_RPATH "@executable_path;@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(WIN32)
    set_target_properties(vivid-ml PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS vivid-ml
    EXPORT vivid-ml-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install ONNX Runtime alongside
if(APPLE)
    install(FILES
        ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib
        ${ONNXRUNTIME_LIB}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
elseif(WIN32)
    install(FILES ${ONNXRUNTIME_DLL}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    install(FILES ${ONNXRUNTIME_LIB}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Export targets for find_package
install(EXPORT vivid-ml-targets
    FILE vivid-ml-targets.cmake
    NAMESPACE vivid::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vivid-ml
)

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
option(BUILD_TESTS "Build the test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
