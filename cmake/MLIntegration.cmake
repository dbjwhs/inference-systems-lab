# MLIntegration.cmake - Machine Learning framework integration
# 
# This module provides optional ML framework integration with proper dependency
# management, feature detection, and graceful fallbacks when frameworks are unavailable.
#
# Optional features controlled by CMake options:
#   ENABLE_TENSORRT - Enable NVIDIA TensorRT GPU acceleration (default: AUTO)
#   ENABLE_ONNX_RUNTIME - Enable ONNX Runtime cross-platform inference (default: AUTO)
#
# AUTO detection tries to find the framework but doesn't fail if unavailable
# ON requires the framework and fails if not found
# OFF explicitly disables the framework

include(CMakeDependentOption)

# ML Framework Options
option(ENABLE_TENSORRT "Enable NVIDIA TensorRT GPU acceleration" AUTO)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime cross-platform inference" AUTO)

# Helper function to handle AUTO detection logic
function(handle_auto_detection FEATURE_NAME ENABLE_VAR FOUND_VAR)
    if(${ENABLE_VAR} STREQUAL "AUTO")
        if(${FOUND_VAR})
            set(${ENABLE_VAR} ON PARENT_SCOPE)
            message(STATUS "AUTO-ENABLED ${FEATURE_NAME}: Found and enabled")
        else()
            set(${ENABLE_VAR} OFF PARENT_SCOPE)
            message(STATUS "AUTO-DISABLED ${FEATURE_NAME}: Not found, proceeding without")
        endif()
    elseif(${ENABLE_VAR})
        if(NOT ${FOUND_VAR})
            message(FATAL_ERROR "${FEATURE_NAME} explicitly enabled but not found. Please install ${FEATURE_NAME} or set ENABLE_${FEATURE_NAME}=OFF")
        else()
            message(STATUS "EXPLICITLY ENABLED ${FEATURE_NAME}: Found and enabled")
        endif()
    else()
        message(STATUS "EXPLICITLY DISABLED ${FEATURE_NAME}: Skipped by user configuration")
    endif()
endfunction()

function(configure_ml_integration)
    message(STATUS "Configuring ML framework integration...")
    
    # Initialize ML framework status
    set(ML_FRAMEWORKS_FOUND 0)
    set(ML_FRAMEWORKS_ENABLED "")
    
    # === TensorRT Integration ===
    if(NOT ENABLE_TENSORRT STREQUAL "OFF")
        message(STATUS "Checking for TensorRT...")
        
        # Try to find TensorRT (quietly for AUTO detection)
        if(ENABLE_TENSORRT STREQUAL "AUTO")
            find_package(TensorRT QUIET)
        else()
            find_package(TensorRT REQUIRED)
        endif()
        
        # Handle AUTO detection logic
        handle_auto_detection("TensorRT" ENABLE_TENSORRT TensorRT_FOUND)
        
        if(ENABLE_TENSORRT AND TensorRT_FOUND)
            math(EXPR ML_FRAMEWORKS_FOUND "${ML_FRAMEWORKS_FOUND} + 1")
            list(APPEND ML_FRAMEWORKS_ENABLED "TensorRT ${TensorRT_VERSION}")
            
            # Create ML-specific alias for easier usage with error handling
            if(NOT TARGET ML::TensorRT)
                if(TARGET TensorRT::TensorRT)
                    add_library(ML::TensorRT ALIAS TensorRT::TensorRT)
                    message(DEBUG "Created ML::TensorRT alias for TensorRT::TensorRT")
                else()
                    message(WARNING "TensorRT found but TensorRT::TensorRT target not available")
                    # Fallback: try to create interface library if we have the components
                    if(TensorRT_LIBRARIES AND TensorRT_INCLUDE_DIRS)
                        add_library(ML::TensorRT INTERFACE IMPORTED)
                        set_target_properties(ML::TensorRT PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
                            INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARIES}"
                        )
                        message(STATUS "Created fallback ML::TensorRT interface library")
                    else()
                        message(ERROR "Cannot create ML::TensorRT target - missing components")
                        set(ENABLE_TENSORRT OFF)
                        math(EXPR ML_FRAMEWORKS_FOUND "${ML_FRAMEWORKS_FOUND} - 1")
                        list(REMOVE_ITEM ML_FRAMEWORKS_ENABLED "TensorRT ${TensorRT_VERSION}")
                    endif()
                endif()
            endif()
        endif()
    endif()
    
    # === ONNX Runtime Integration ===
    if(NOT ENABLE_ONNX_RUNTIME STREQUAL "OFF")
        message(STATUS "Checking for ONNX Runtime...")
        
        # Try to find ONNX Runtime (quietly for AUTO detection)
        if(ENABLE_ONNX_RUNTIME STREQUAL "AUTO")
            find_package(ONNXRuntime QUIET)
        else()
            find_package(ONNXRuntime REQUIRED)
        endif()
        
        # Handle AUTO detection logic
        handle_auto_detection("ONNX Runtime" ENABLE_ONNX_RUNTIME ONNXRuntime_FOUND)
        
        if(ENABLE_ONNX_RUNTIME AND ONNXRuntime_FOUND)
            math(EXPR ML_FRAMEWORKS_FOUND "${ML_FRAMEWORKS_FOUND} + 1")
            list(APPEND ML_FRAMEWORKS_ENABLED "ONNX Runtime ${ONNXRuntime_VERSION}")
            
            # Create ML-specific alias for easier usage with error handling
            if(NOT TARGET ML::ONNXRuntime)
                if(TARGET ONNXRuntime::ONNXRuntime)
                    add_library(ML::ONNXRuntime ALIAS ONNXRuntime::ONNXRuntime)
                    message(DEBUG "Created ML::ONNXRuntime alias for ONNXRuntime::ONNXRuntime")
                else()
                    message(WARNING "ONNX Runtime found but ONNXRuntime::ONNXRuntime target not available")
                    # Fallback: try to create interface library if we have the components
                    if(ONNXRuntime_LIBRARIES AND ONNXRuntime_INCLUDE_DIRS)
                        add_library(ML::ONNXRuntime INTERFACE IMPORTED)
                        set_target_properties(ML::ONNXRuntime PROPERTIES
                            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
                            INTERFACE_LINK_LIBRARIES "${ONNXRuntime_LIBRARIES}"
                        )
                        message(STATUS "Created fallback ML::ONNXRuntime interface library")
                    else()
                        message(ERROR "Cannot create ML::ONNXRuntime target - missing components")
                        set(ENABLE_ONNX_RUNTIME OFF)
                        math(EXPR ML_FRAMEWORKS_FOUND "${ML_FRAMEWORKS_FOUND} - 1")
                        list(REMOVE_ITEM ML_FRAMEWORKS_ENABLED "ONNX Runtime ${ONNXRuntime_VERSION}")
                    endif()
                endif()
            endif()
        endif()
    endif()
    
    # === ML Framework Summary ===
    message(STATUS "ML Integration Summary:")
    message(STATUS "  Frameworks found: ${ML_FRAMEWORKS_FOUND}")
    
    if(ML_FRAMEWORKS_FOUND GREATER 0)
        foreach(framework IN LISTS ML_FRAMEWORKS_ENABLED)
            message(STATUS "    ✓ ${framework}")
        endforeach()
        
        # Set global compile definition indicating ML support is available
        add_compile_definitions(ENABLE_ML_FRAMEWORKS=1)
        add_compile_definitions(ML_FRAMEWORKS_COUNT=${ML_FRAMEWORKS_FOUND})
        
        # Create convenience interface library for projects that need any ML framework
        if(NOT TARGET ML::Frameworks)
            add_library(ML::Frameworks INTERFACE)
            
            if(TARGET ML::TensorRT)
                target_link_libraries(ML::Frameworks INTERFACE ML::TensorRT)
            endif()
            
            if(TARGET ML::ONNXRuntime)
                target_link_libraries(ML::Frameworks INTERFACE ML::ONNXRuntime)
            endif()
        endif()
        
    else()
        message(STATUS "    No ML frameworks enabled - inference will use CPU-only implementations")
        add_compile_definitions(ENABLE_ML_FRAMEWORKS=0)
        add_compile_definitions(ML_FRAMEWORKS_COUNT=0)
    endif()
    
    # Export configuration for parent scope
    set(ML_FRAMEWORKS_FOUND ${ML_FRAMEWORKS_FOUND} PARENT_SCOPE)
    set(ML_FRAMEWORKS_ENABLED "${ML_FRAMEWORKS_ENABLED}" PARENT_SCOPE)
    
    message(STATUS "ML framework integration configured")
endfunction()

# Helper function for displaying ML status in configuration summary
function(display_ml_integration_status)
    if(DEFINED ML_FRAMEWORKS_ENABLED AND ML_FRAMEWORKS_FOUND GREATER 0)
        message(STATUS "  ML Frameworks (${ML_FRAMEWORKS_FOUND}):")
        foreach(framework IN LISTS ML_FRAMEWORKS_ENABLED)
            message(STATUS "    ${framework}")
        endforeach()
    else()
        message(STATUS "  ML Frameworks: NONE (CPU-only mode)")
    endif()
    
    # Show configuration options
    message(STATUS "  TensorRT: ${ENABLE_TENSORRT}")
    message(STATUS "  ONNX Runtime: ${ENABLE_ONNX_RUNTIME}")
endfunction()

# Helper function to check ML framework requirements for targets
function(require_ml_framework target_name)
    if(ML_FRAMEWORKS_FOUND EQUAL 0)
        message(FATAL_ERROR 
            "Target '${target_name}' requires ML framework support, but no frameworks are enabled.\n"
            "Please install and enable at least one of:\n"
            "  - TensorRT (set ENABLE_TENSORRT=ON)\n"
            "  - ONNX Runtime (set ENABLE_ONNX_RUNTIME=ON)\n"
            "Or build without ML-dependent targets."
        )
    endif()
endfunction()

# Helper function to link ML frameworks to a target conditionally
function(target_link_ml_frameworks target_name)
    if(TARGET ML::Frameworks)
        target_link_libraries(${target_name} PRIVATE ML::Frameworks)
    endif()
endfunction()

# Validate configuration options
function(validate_ml_options)
    # Validate option values
    set(VALID_OPTIONS "ON" "OFF" "AUTO")
    
    if(NOT ENABLE_TENSORRT IN_LIST VALID_OPTIONS)
        message(FATAL_ERROR "ENABLE_TENSORRT must be one of: ${VALID_OPTIONS}")
    endif()
    
    if(NOT ENABLE_ONNX_RUNTIME IN_LIST VALID_OPTIONS)
        message(FATAL_ERROR "ENABLE_ONNX_RUNTIME must be one of: ${VALID_OPTIONS}")
    endif()
    
    # Check for conflicting configurations
    if(ENABLE_TENSORRT STREQUAL "OFF" AND ENABLE_ONNX_RUNTIME STREQUAL "OFF")
        message(STATUS "Note: All ML frameworks explicitly disabled - building in CPU-only mode")
    endif()
endfunction()
