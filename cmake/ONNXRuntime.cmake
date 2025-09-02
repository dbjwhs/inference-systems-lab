# ONNXRuntime.cmake - Detection and configuration for ONNX Runtime
# 
# This module provides detection of ONNX Runtime installation and sets up
# appropriate compile definitions and linking for cross-platform ML inference.
#
# Variables set by this module:
#   ONNXRuntime_FOUND - True if ONNX Runtime is found
#   ONNXRuntime_VERSION - Version of ONNX Runtime found
#   ONNXRuntime_INCLUDE_DIRS - Include directories for ONNX Runtime headers
#   ONNXRuntime_LIBRARIES - Libraries to link against for ONNX Runtime
#   ONNXRUNTIME_ROOT - Root directory of ONNX Runtime installation

# Allow user to specify ONNX Runtime root directory
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime installation")

# Try to find ONNX Runtime in standard locations
if(NOT ONNXRUNTIME_ROOT)
    # Standard installation paths
    find_path(ONNXRUNTIME_ROOT
        NAMES include/onnxruntime_cxx_api.h
        PATHS
            /usr/local/onnxruntime
            /opt/onnxruntime
            /usr/local
            /opt
            $ENV{ONNXRUNTIME_ROOT}
            # Platform-specific paths
            "C:/Program Files/onnxruntime"
            "C:/onnxruntime"
        DOC "Root directory of ONNX Runtime installation"
    )
endif()

# Find ONNX Runtime headers
find_path(ONNXRuntime_INCLUDE_DIRS
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        /usr/local/include
        /usr/include
        # Platform-specific paths
        "C:/Program Files/onnxruntime/include"
    DOC "ONNX Runtime include directory"
)

# Determine library suffix based on platform
if(WIN32)
    set(ONNX_LIB_SUFFIX ".lib")
    set(ONNX_LIB_PREFIX "")
else()
    set(ONNX_LIB_SUFFIX ".so")
    set(ONNX_LIB_PREFIX "lib")
endif()

# Find ONNX Runtime main library
find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime ${ONNX_LIB_PREFIX}onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        ${ONNXRUNTIME_ROOT}/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        # Platform-specific paths
        "C:/Program Files/onnxruntime/lib"
    DOC "ONNX Runtime main library"
)

# Set libraries list
set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})

# Try to determine ONNX Runtime version
if(ONNXRuntime_INCLUDE_DIRS)
    # Look for version in onnxruntime_config.h or similar
    find_file(ONNX_VERSION_FILE
        NAMES onnxruntime_config.h onnxruntime_version.h
        PATHS ${ONNXRuntime_INCLUDE_DIRS}
        NO_DEFAULT_PATH
    )
    
    if(ONNX_VERSION_FILE)
        file(READ ${ONNX_VERSION_FILE} ONNX_VERSION_CONTENT)
        
        # Try to extract version numbers
        string(REGEX MATCH "ORT_VERSION_MAJOR[ ]+([0-9]+)" _ ${ONNX_VERSION_CONTENT})
        if(CMAKE_MATCH_1)
            set(ONNXRuntime_VERSION_MAJOR ${CMAKE_MATCH_1})
        else()
            set(ONNXRuntime_VERSION_MAJOR "1") # Default fallback
        endif()
        
        string(REGEX MATCH "ORT_VERSION_MINOR[ ]+([0-9]+)" _ ${ONNX_VERSION_CONTENT})
        if(CMAKE_MATCH_1)
            set(ONNXRuntime_VERSION_MINOR ${CMAKE_MATCH_1})
        else()
            set(ONNXRuntime_VERSION_MINOR "16") # Default fallback
        endif()
        
        string(REGEX MATCH "ORT_VERSION_PATCH[ ]+([0-9]+)" _ ${ONNX_VERSION_CONTENT})
        if(CMAKE_MATCH_1)
            set(ONNXRuntime_VERSION_PATCH ${CMAKE_MATCH_1})
        else()
            set(ONNXRuntime_VERSION_PATCH "0") # Default fallback
        endif()
        
        set(ONNXRuntime_VERSION "${ONNXRuntime_VERSION_MAJOR}.${ONNXRuntime_VERSION_MINOR}.${ONNXRuntime_VERSION_PATCH}")
    else()
        set(ONNXRuntime_VERSION "1.16.0") # Reasonable default
    endif()
endif()

# Determine if ONNX Runtime is found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS
        ONNXRuntime_INCLUDE_DIRS
        ONNXRuntime_LIBRARY
    VERSION_VAR ONNXRuntime_VERSION
    FAIL_MESSAGE "ONNX Runtime not found. Please install ONNX Runtime 1.16+ and set ONNXRUNTIME_ROOT if needed."
)

# Create ONNX Runtime target if found
if(ONNXRuntime_FOUND)
    # Create imported target for ONNX Runtime
    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime INTERFACE IMPORTED)
        
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${ONNXRuntime_LIBRARIES}"
        )
        
        # Add compile definition to enable ONNX Runtime code
        target_compile_definitions(ONNXRuntime::ONNXRuntime INTERFACE ENABLE_ONNX_RUNTIME=1)
        
        # Platform-specific configurations
        if(WIN32)
            # Windows-specific settings
            target_compile_definitions(ONNXRuntime::ONNXRuntime INTERFACE WIN32_LEAN_AND_MEAN)
        elseif(APPLE)
            # macOS-specific settings
            target_link_libraries(ONNXRuntime::ONNXRuntime INTERFACE "-framework Foundation")
        else()
            # Linux-specific settings
            target_link_libraries(ONNXRuntime::ONNXRuntime INTERFACE dl pthread)
        endif()
    endif()
    
    # Print configuration information
    if(NOT ONNXRuntime_FIND_QUIETLY)
        message(STATUS "Found ONNX Runtime: ${ONNXRuntime_VERSION}")
        message(STATUS "  ONNX Runtime root: ${ONNXRUNTIME_ROOT}")
        message(STATUS "  ONNX Runtime includes: ${ONNXRuntime_INCLUDE_DIRS}")
        message(STATUS "  ONNX Runtime libraries: ${ONNXRuntime_LIBRARIES}")
    endif()
    
    # Version compatibility check
    if(ONNXRuntime_VERSION VERSION_LESS "1.16.0")
        message(WARNING "ONNX Runtime ${ONNXRuntime_VERSION} found, but 1.16.0+ is recommended for best compatibility")
    endif()
    
else()
    if(NOT ONNXRuntime_FIND_QUIETLY)
        message(STATUS "ONNX Runtime not found - Cross-platform ML inference will not be available")
        message(STATUS "  To enable ONNX Runtime support:")
        message(STATUS "    1. Install ONNX Runtime 1.16+ from https://onnxruntime.ai/")
        message(STATUS "    2. Set ONNXRUNTIME_ROOT environment variable or CMake cache variable")
        message(STATUS "    3. For GPU support, ensure CUDA/DirectML/CoreML is available")
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    ONNXRUNTIME_ROOT
    ONNXRuntime_INCLUDE_DIRS
    ONNXRuntime_LIBRARY
)
