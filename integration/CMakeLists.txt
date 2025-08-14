# Integration scenarios library

# Create integration library
add_library(integration STATIC
    src/distributed_inference.cpp
    src/performance_consensus.cpp
    src/end_to_end_scenarios.cpp
)

target_include_directories(integration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(integration
    engines
    distributed
    performance
    common
    Threads::Threads
)

# Integration tests
add_executable(integration_tests
    tests/test_distributed_inference.cpp
    tests/test_performance_consensus.cpp
    tests/test_end_to_end_scenarios.cpp
)

target_link_libraries(integration_tests
    integration
    gtest_main
    gtest
)

add_test(NAME IntegrationTests COMMAND integration_tests)

# Integration benchmarks
add_executable(integration_benchmarks
    benchmarks/bench_distributed_inference.cpp
    benchmarks/bench_consensus_under_load.cpp
    benchmarks/bench_full_system.cpp
)

target_link_libraries(integration_benchmarks
    integration
    benchmark::benchmark
)