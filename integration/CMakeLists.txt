# Integration scenarios library

# Create integration library (using placeholder for now)
add_library(integration STATIC
    src/placeholder.cpp
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

# Integration tests (using placeholder for now)
add_executable(integration_tests
    tests/placeholder.cpp
)

target_link_libraries(integration_tests
    integration
    GTest::gtest_main
)

add_test(NAME IntegrationTests COMMAND integration_tests)

# Integration benchmarks (using placeholder for now)
add_executable(integration_benchmarks
    benchmarks/placeholder.cpp
)

target_link_libraries(integration_benchmarks
    integration
    benchmark::benchmark
)

