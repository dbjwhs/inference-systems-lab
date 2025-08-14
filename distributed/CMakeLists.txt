# Distributed systems library

# Create distributed library
add_library(distributed STATIC
    src/raft_consensus.cpp
    src/distributed_state_machine.cpp
    src/node_discovery.cpp
    src/message_passing.cpp
)

target_include_directories(distributed PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(distributed
    common
    Threads::Threads
)

# Distributed tests
add_executable(distributed_tests
    tests/test_raft_consensus.cpp
    tests/test_distributed_state_machine.cpp
    tests/test_node_discovery.cpp
    tests/test_message_passing.cpp
)

target_link_libraries(distributed_tests
    distributed
    gtest_main
    gtest
)

add_test(NAME DistributedTests COMMAND distributed_tests)

# Distributed benchmarks
add_executable(distributed_benchmarks
    benchmarks/bench_consensus_latency.cpp
    benchmarks/bench_message_throughput.cpp
    benchmarks/bench_fault_recovery.cpp
)

target_link_libraries(distributed_benchmarks
    distributed
    benchmark::benchmark
)