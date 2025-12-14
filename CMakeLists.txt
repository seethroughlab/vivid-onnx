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
# Find vivid-core (required dependency)
# -----------------------------------------------------------------------------
# Option 1: As subdirectory of main vivid repo (VIVID_ROOT set by parent)
# Option 2: Installed vivid (find_package)
# Option 3: Manual path via VIVID_ROOT

if(NOT TARGET vivid-core)
    if(DEFINED VIVID_ROOT)
        # Manual path specified
        message(STATUS "[vivid-ml] Using VIVID_ROOT: ${VIVID_ROOT}")
        set(VIVID_INCLUDE_DIR "${VIVID_ROOT}/include")
        find_library(VIVID_CORE_LIB vivid-core PATHS "${VIVID_ROOT}/lib" REQUIRED)
    else()
        # Try find_package
        find_package(vivid QUIET)
        if(vivid_FOUND)
            message(STATUS "[vivid-ml] Found vivid via find_package")
        else()
            message(FATAL_ERROR
                "[vivid-ml] Could not find vivid-core. Options:\n"
                "  1. Set VIVID_ROOT to your vivid installation\n"
                "  2. Install vivid and ensure it's in CMAKE_PREFIX_PATH\n"
                "  3. Build as part of the main vivid repo"
            )
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
# ONNX Runtime dependency
# -----------------------------------------------------------------------------
set(ONNXRUNTIME_VERSION "1.19.2")

# Platform-specific ONNX Runtime download
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
    endif()
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
)

add_library(vivid-ml SHARED ${ML_SOURCES})

target_include_directories(vivid-ml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link against vivid-core
if(TARGET vivid-core)
    target_link_libraries(vivid-ml PUBLIC vivid-core ${ONNXRUNTIME_LIB})
elseif(DEFINED VIVID_INCLUDE_DIR)
    target_include_directories(vivid-ml PUBLIC ${VIVID_INCLUDE_DIR})
    target_link_libraries(vivid-ml PUBLIC ${VIVID_CORE_LIB} ${ONNXRUNTIME_LIB})
else()
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
