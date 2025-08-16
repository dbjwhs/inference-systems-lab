# Inference engines library

# Create engines library (using placeholder for now)
add_library(engines STATIC
    src/placeholder.cpp
)

target_include_directories(engines PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(engines
    common
    Threads::Threads
)

# Engines tests (using placeholder for now)
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
    benchmark::benchmark
)

