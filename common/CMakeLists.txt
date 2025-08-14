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

target_link_libraries(common
    Threads::Threads
)

# Common tests
add_executable(common_tests
    tests/test_logging.cpp
    tests/test_utils.cpp
)

target_link_libraries(common_tests
    common
    gtest_main
    gtest
)

add_test(NAME CommonTests COMMAND common_tests)

# Common benchmarks
add_executable(common_benchmarks
    benchmarks/bench_logging.cpp
    benchmarks/bench_utils.cpp
)

target_link_libraries(common_benchmarks
    common
    benchmark::benchmark
)