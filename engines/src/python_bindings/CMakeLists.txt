# Python Bindings for Inference Systems Laboratory
# Builds the inference_lab Python module using pybind11

# pybind11 is already configured by cmake/Pybind11.cmake

# Define the Python module
pybind11_add_module(inference_lab
    main.cpp
    result_bindings.cpp
    tensor_bindings.cpp
    inference_bindings.cpp
    logging_bindings.cpp
)

# Link against project libraries
target_link_libraries(inference_lab PRIVATE
    common
    engines  # Added engines library for InferenceEngine access
    ${CAPNP_LIBRARIES}
)

# Include directories
target_include_directories(inference_lab PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CAPNP_INCLUDE_DIRS}
)

# Set C++ standard and compile features
target_compile_features(inference_lab PRIVATE cxx_std_17)

# Compiler-specific optimizations
target_compile_options(inference_lab PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# Module properties
set_target_properties(inference_lab PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Set version info from CMake
target_compile_definitions(inference_lab PRIVATE
    VERSION_INFO=${CMAKE_PROJECT_VERSION}
    CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    CMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
)

# Install the Python module (optional)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    install(TARGETS inference_lab DESTINATION .)
endif()
