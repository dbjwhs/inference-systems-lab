# Inference engines library

# Create engines library
add_library(engines STATIC
    src/placeholder.cpp
    src/inference_engine.cpp
    src/forward_chaining.cpp
)

target_include_directories(engines PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(engines
    common
    Threads::Threads
)

# Engines tests
add_executable(engines_tests
    tests/placeholder.cpp
)

target_link_libraries(engines_tests
    engines
    GTest::gtest_main
)

add_test(NAME EnginesTests COMMAND engines_tests)

# Engines benchmarks (using placeholder for now)
add_executable(engines_benchmarks
    benchmarks/placeholder.cpp
)

target_link_libraries(engines_benchmarks
    engines
    benchmark::benchmark_main
)

# Forward chaining demos have been replaced with simple_forward_chaining_demo

# Simple forward chaining demo
add_executable(simple_forward_chaining_demo
    examples/simple_forward_chaining_demo.cpp
)

target_link_libraries(simple_forward_chaining_demo
    engines
)

# Python bindings (Phase 1: Foundation setup)
# Temporarily disabled - enable in Phase 2 when testing
# add_subdirectory(src/python_bindings)

