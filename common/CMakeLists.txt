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
)

target_include_directories(common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Set C++23 standard
target_compile_features(common PUBLIC cxx_std_23)

# Find and link threads
find_package(Threads REQUIRED)
target_link_libraries(common
    Threads::Threads
)

# Demo executable
add_executable(demo_logging
    examples/demo_logging.cpp
)

target_link_libraries(demo_logging
    common
)