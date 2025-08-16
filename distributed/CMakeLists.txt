# Distributed systems library

# Create distributed library (using placeholder for now)
add_library(distributed STATIC
    src/placeholder.cpp
)

target_include_directories(distributed PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(distributed
    common
    Threads::Threads
)

# Distributed tests (using placeholder for now)
add_executable(distributed_tests
    tests/placeholder.cpp
)

target_link_libraries(distributed_tests
    distributed
    GTest::gtest_main
)

add_test(NAME DistributedTests COMMAND distributed_tests)

# Distributed benchmarks (using placeholder for now)
add_executable(distributed_benchmarks
    benchmarks/placeholder.cpp
)

target_link_libraries(distributed_benchmarks
    distributed
    benchmark::benchmark
)

