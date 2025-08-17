cmake_minimum_required(VERSION 3.16)

project(InferenceSystemsLab
    VERSION 0.1.0
    DESCRIPTION "A modern C++17+ research laboratory for inference engines, distributed systems, and performance engineering"
    LANGUAGES CXX
)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add cmake modules to path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Configure all build components using modules
include(PackageConfig)
configure_packages()

include(CompilerOptions)
configure_compiler_options()

include(Sanitizers)
configure_sanitizers()

include(Testing)
configure_testing()
configure_coverage()

include(Benchmarking)
configure_benchmarking()

include(StaticAnalysis)
configure_static_analysis()

include(Documentation)
configure_documentation()
configure_formatting()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Common library for shared utilities
add_subdirectory(common)

# Core domain libraries
add_subdirectory(engines)
add_subdirectory(distributed)
add_subdirectory(performance)

# Integration and experiments
add_subdirectory(integration)
add_subdirectory(experiments)

# Installation configuration
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/common/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/inference_lab
    FILES_MATCHING PATTERN "*.hpp"
)

# Package configuration
set(CPACK_PACKAGE_NAME "InferenceSystemsLab")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_CONTACT "research@inference-lab.org")
set(CPACK_GENERATOR "TGZ")

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Inference Systems Lab Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
display_package_status()
message(STATUS "")
message(STATUS "Optional features:")
display_static_analysis_status()
display_sanitizer_status()
display_testing_status()
display_benchmarking_status()
display_documentation_status()
message(STATUS "==========================================")
message(STATUS "")
