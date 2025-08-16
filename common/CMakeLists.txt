# MIT License
# Copyright (c) 2025 dbjwhs
#
# This software is provided "as is" without warranty of any kind, express or implied.
# The authors are not liable for any damages arising from the use of this software.

cmake_minimum_required(VERSION 3.20)
project(common)

# Common utilities and abstractions library

# Create common library
add_library(common STATIC
    src/logging.cpp
    src/utils.cpp
    src/inference_types.cpp
    src/inference_builders.cpp
    src/schema_evolution.cpp
)

target_include_directories(common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>  # For generated Cap'n Proto headers
    $<INSTALL_INTERFACE:include>
)

# Set C++23 standard
target_compile_features(common PUBLIC cxx_std_23)

# Find dependencies
find_package(Threads REQUIRED)
find_package(CapnProto REQUIRED)

# Generate C++ code from Cap'n Proto schemas
capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS schemas/inference_types.capnp)

# Add generated sources to the library
target_sources(common PRIVATE ${CAPNP_SRCS})

target_link_libraries(common
    Threads::Threads
    CapnProto::capnp
)

# Demo executable
add_executable(demo_logging
    examples/demo_logging.cpp
)

target_link_libraries(demo_logging
    common
)

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest REQUIRED)

# Unit tests
add_executable(logging_tests
    tests/test_logging_unit.cpp
)

target_link_libraries(logging_tests
    common
    GTest::gtest_main
)

# Add test to CTest
add_test(NAME LoggingUnitTests COMMAND logging_tests)

# Inference types demo
add_executable(inference_types_demo
    examples/inference_types_demo.cpp
)

target_link_libraries(inference_types_demo
    common
)

# Schema evolution demo
add_executable(schema_evolution_demo
    examples/schema_evolution_demo.cpp
)

target_link_libraries(schema_evolution_demo
    common
)