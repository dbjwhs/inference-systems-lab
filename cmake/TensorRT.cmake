# TensorRT.cmake - Detection and configuration for NVIDIA TensorRT
# 
# This module provides detection of TensorRT installation and sets up
# appropriate compile definitions and linking for GPU inference support.
#
# Variables set by this module:
#   TensorRT_FOUND - True if TensorRT is found
#   TensorRT_VERSION - Version of TensorRT found
#   TensorRT_INCLUDE_DIRS - Include directories for TensorRT headers
#   TensorRT_LIBRARIES - Libraries to link against for TensorRT
#   TENSORRT_ROOT - Root directory of TensorRT installation

# Allow user to specify TensorRT root directory
set(TENSORRT_ROOT "" CACHE PATH "Root directory of TensorRT installation")

# Try to find TensorRT in standard locations
if(NOT TENSORRT_ROOT)
    # Standard installation paths
    find_path(TENSORRT_ROOT
        NAMES include/NvInfer.h
        PATHS
            /usr/local/tensorrt
            /opt/tensorrt
            /usr/local
            /opt
            $ENV{TENSORRT_ROOT}
        DOC "Root directory of TensorRT installation"
    )
endif()

# Find TensorRT headers
find_path(TensorRT_INCLUDE_DIRS
    NAMES NvInfer.h
    PATHS
        ${TENSORRT_ROOT}/include
        /usr/local/include
        /usr/include
    DOC "TensorRT include directory"
)

# Find TensorRT libraries
find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer libnvinfer
    PATHS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
    DOC "TensorRT nvinfer library"
)

find_library(TensorRT_nvinfer_plugin_LIBRARY
    NAMES nvinfer_plugin libnvinfer_plugin
    PATHS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
    DOC "TensorRT nvinfer_plugin library"
)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser libnvonnxparser
    PATHS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
    DOC "TensorRT nvonnxparser library"
)

# Collect all TensorRT libraries
set(TensorRT_LIBRARIES
    ${TensorRT_nvinfer_LIBRARY}
    ${TensorRT_nvinfer_plugin_LIBRARY}
    ${TensorRT_nvonnxparser_LIBRARY}
)

# Try to determine TensorRT version
if(TensorRT_INCLUDE_DIRS)
    file(READ "${TensorRT_INCLUDE_DIRS}/NvInferVersion.h" TENSORRT_VERSION_FILE)
    
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" _ ${TENSORRT_VERSION_FILE})
    set(TensorRT_VERSION_MAJOR ${CMAKE_MATCH_1})
    
    string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" _ ${TENSORRT_VERSION_FILE})
    set(TensorRT_VERSION_MINOR ${CMAKE_MATCH_1})
    
    string(REGEX MATCH "#define NV_TENSORRT_PATCH ([0-9]+)" _ ${TENSORRT_VERSION_FILE})
    set(TensorRT_VERSION_PATCH ${CMAKE_MATCH_1})
    
    set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

# Check for CUDA dependency (required for TensorRT)
find_package(CUDA QUIET)

# Determine if TensorRT is found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIRS
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvinfer_plugin_LIBRARY
        TensorRT_nvonnxparser_LIBRARY
        CUDA_FOUND
    VERSION_VAR TensorRT_VERSION
    FAIL_MESSAGE "TensorRT not found. Please install TensorRT 8.5+ and set TENSORRT_ROOT if needed."
)

# Create TensorRT target if found
if(TensorRT_FOUND)
    # Create imported target for TensorRT
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT INTERFACE IMPORTED)
        
        set_target_properties(TensorRT::TensorRT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARIES}"
        )
        
        # Add CUDA dependency
        target_link_libraries(TensorRT::TensorRT INTERFACE ${CUDA_LIBRARIES})
        target_include_directories(TensorRT::TensorRT INTERFACE ${CUDA_INCLUDE_DIRS})
        
        # Add compile definition to enable TensorRT code
        target_compile_definitions(TensorRT::TensorRT INTERFACE ENABLE_TENSORRT=1)
    endif()
    
    # Print configuration information
    if(NOT TensorRT_FIND_QUIETLY)
        message(STATUS "Found TensorRT: ${TensorRT_VERSION}")
        message(STATUS "  TensorRT root: ${TENSORRT_ROOT}")
        message(STATUS "  TensorRT includes: ${TensorRT_INCLUDE_DIRS}")
        message(STATUS "  TensorRT libraries: ${TensorRT_LIBRARIES}")
        message(STATUS "  CUDA version: ${CUDA_VERSION}")
    endif()
    
    # Version compatibility check
    if(TensorRT_VERSION VERSION_LESS "8.5.0")
        message(WARNING "TensorRT ${TensorRT_VERSION} found, but 8.5.0+ is recommended for best compatibility")
    endif()
    
else()
    if(NOT TensorRT_FIND_QUIETLY)
        message(STATUS "TensorRT not found - TensorRT GPU acceleration will not be available")
        message(STATUS "  To enable TensorRT support:")
        message(STATUS "    1. Install NVIDIA TensorRT 8.5+ from https://developer.nvidia.com/tensorrt")
        message(STATUS "    2. Set TENSORRT_ROOT environment variable or CMake cache variable")
        message(STATUS "    3. Ensure CUDA Toolkit 11.8+ is installed")
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    TENSORRT_ROOT
    TensorRT_INCLUDE_DIRS
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvinfer_plugin_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
)
