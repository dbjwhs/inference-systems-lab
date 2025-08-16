# Performance optimization library

# Create performance library (using placeholder for now)
add_library(performance STATIC
    src/placeholder.cpp
)

target_include_directories(performance PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(performance
    common
    Threads::Threads
)

# Performance tests (using placeholder for now)
add_executable(performance_tests
    tests/placeholder.cpp
)

target_link_libraries(performance_tests
    performance
    GTest::gtest_main
)

add_test(NAME PerformanceTests COMMAND performance_tests)

# Performance benchmarks (using placeholder for now)
add_executable(performance_benchmarks
    benchmarks/placeholder.cpp
)

target_link_libraries(performance_benchmarks
    performance
    benchmark::benchmark_main
)

