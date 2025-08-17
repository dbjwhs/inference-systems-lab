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

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wcast-align -Wcast-qual -Wctor-dtor-privacy
        -Wdisabled-optimization -Winit-self
        -Wmissing-declarations -Wmissing-include-dirs
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls
        -Wshadow -Wsign-promo
        -Wstrict-overflow=5
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0 -fno-omit-frame-pointer)
        add_compile_definitions(INFERENCE_LAB_DEBUG)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(/O2)
    endif()
endif()

# Enable testing
enable_testing()

# Find required packages
find_package(Threads REQUIRED)

# Optional packages for enhanced development
find_package(PkgConfig QUIET)

# Google Test and Benchmark - prefer system-installed version for ABI compatibility
include(FetchContent)

# Try to find system-installed GoogleTest first (better ABI compatibility on Apple Silicon)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "System GoogleTest not found, using FetchContent")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "Using system-installed GoogleTest: ${GTEST_VERSION}")
endif()

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing")
FetchContent_MakeAvailable(benchmark)

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

# Development tools
option(ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
option(ENABLE_CPPCHECK "Enable cppcheck analysis" OFF)
option(ENABLE_SANITIZERS "Enable sanitizer builds" OFF)

if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    else()
        message(WARNING "clang-tidy not found")
    endif()
endif()

# Sanitizer support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Sanitizer build configurations
    set(SANITIZER_TYPE "none" CACHE STRING "Sanitizer to use (none, address, thread, memory, undefined, address+undefined)")
    set_property(CACHE SANITIZER_TYPE PROPERTY STRINGS "none;address;thread;memory;undefined;address+undefined")
    
    # Individual sanitizer options for advanced users
    option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
    option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
    option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
    option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
    
    # Apply sanitizers based on SANITIZER_TYPE
    if(SANITIZER_TYPE STREQUAL "address")
        set(ENABLE_ASAN ON)
        message(STATUS "Enabled AddressSanitizer")
    elseif(SANITIZER_TYPE STREQUAL "thread")
        set(ENABLE_TSAN ON)
        message(STATUS "Enabled ThreadSanitizer")
    elseif(SANITIZER_TYPE STREQUAL "memory")
        set(ENABLE_MSAN ON)
        message(STATUS "Enabled MemorySanitizer")
    elseif(SANITIZER_TYPE STREQUAL "undefined")
        set(ENABLE_UBSAN ON)
        message(STATUS "Enabled UndefinedBehaviorSanitizer")
    elseif(SANITIZER_TYPE STREQUAL "address+undefined")
        set(ENABLE_ASAN ON)
        set(ENABLE_UBSAN ON)
        message(STATUS "Enabled AddressSanitizer + UndefinedBehaviorSanitizer")
    endif()
    
    # Apply sanitizer flags
    set(SANITIZER_FLAGS "")
    set(SANITIZER_LINK_FLAGS "")
    
    if(ENABLE_ASAN)
        list(APPEND SANITIZER_FLAGS "-fsanitize=address" "-fno-omit-frame-pointer")
        list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=address")
        # Enable better stack traces
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND SANITIZER_FLAGS "-O1" "-g")
        endif()
    endif()
    
    if(ENABLE_TSAN)
        # ThreadSanitizer is incompatible with AddressSanitizer
        if(ENABLE_ASAN)
            message(FATAL_ERROR "ThreadSanitizer is incompatible with AddressSanitizer")
        endif()
        list(APPEND SANITIZER_FLAGS "-fsanitize=thread")
        list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=thread")
    endif()
    
    if(ENABLE_MSAN)
        # MemorySanitizer is incompatible with AddressSanitizer
        if(ENABLE_ASAN)
            message(FATAL_ERROR "MemorySanitizer is incompatible with AddressSanitizer")
        endif()
        list(APPEND SANITIZER_FLAGS "-fsanitize=memory" "-fno-omit-frame-pointer")
        list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=memory")
    endif()
    
    if(ENABLE_UBSAN)
        list(APPEND SANITIZER_FLAGS "-fsanitize=undefined")
        list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=undefined")
        # Additional UBSan checks
        list(APPEND SANITIZER_FLAGS 
            "-fsanitize=signed-integer-overflow"
            "-fsanitize=null"
            "-fsanitize=bounds"
            "-fsanitize=alignment"
            "-fsanitize=object-size"
            "-fsanitize=vptr"
        )
    endif()
    
    # Apply flags if any sanitizers are enabled
    if(SANITIZER_FLAGS)
        add_compile_options(${SANITIZER_FLAGS})
        add_link_options(${SANITIZER_LINK_FLAGS})
        
        # Set environment variables for better output
        set(ENV{ASAN_OPTIONS} "abort_on_error=1:fast_unwind_on_malloc=0:detect_leaks=1")
        set(ENV{UBSAN_OPTIONS} "abort_on_error=1:print_stacktrace=1")
        
        message(STATUS "Sanitizer flags: ${SANITIZER_FLAGS}")
        message(STATUS "Sanitizer link flags: ${SANITIZER_LINK_FLAGS}")
    endif()
else()
    if(NOT SANITIZER_TYPE STREQUAL "none")
        message(WARNING "Sanitizers are only supported with GCC or Clang compilers")
    endif()
endif()

# Custom targets for development workflow

# Format target (requires clang-format)
find_program(CLANG_FORMAT_EXE NAMES "clang-format")
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/**/src/*.cpp
        ${CMAKE_SOURCE_DIR}/**/src/*.hpp
        ${CMAKE_SOURCE_DIR}/**/tests/*.cpp
        ${CMAKE_SOURCE_DIR}/**/benchmarks/*.cpp
        ${CMAKE_SOURCE_DIR}/**/examples/*.cpp
    )
    
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
        COMMENT "Formatting source code with clang-format"
    )
endif()

# Documentation target (requires Doxygen)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    if(EXISTS ${DOXYGEN_IN})
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
        )
    endif()
endif()

# Benchmarks target
add_custom_target(benchmarks
    DEPENDS 
        common_benchmarks
        engines_benchmarks
        distributed_benchmarks
        performance_benchmarks
        integration_benchmarks
    COMMENT "Building all benchmarks"
)

# Coverage target (requires gcov/lcov)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
    if(ENABLE_COVERAGE)
        add_compile_options(--coverage)
        add_link_options(--coverage)
        
        find_program(LCOV_EXE NAMES "lcov")
        find_program(GENHTML_EXE NAMES "genhtml")
        
        if(LCOV_EXE AND GENHTML_EXE)
            add_custom_target(coverage
                COMMAND ${LCOV_EXE} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV_EXE} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${LCOV_EXE} --list coverage.info
                COMMAND ${GENHTML_EXE} -o coverage coverage.info
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating coverage report"
            )
        endif()
    endif()
endif()

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
message(STATUS "Optional features:")
message(STATUS "  clang-tidy: ${ENABLE_CLANG_TIDY}")
message(STATUS "  Sanitizer: ${SANITIZER_TYPE}")
if(SANITIZER_FLAGS)
    message(STATUS "    Address: ${ENABLE_ASAN}")
    message(STATUS "    Thread: ${ENABLE_TSAN}")
    message(STATUS "    Memory: ${ENABLE_MSAN}")
    message(STATUS "    UndefinedBehavior: ${ENABLE_UBSAN}")
endif()
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Doxygen: ${DOXYGEN_FOUND}")
message(STATUS "==========================================")
message(STATUS "")