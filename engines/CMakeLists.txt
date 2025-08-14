# Inference engines library

# Create engines library
add_library(engines STATIC
    src/forward_chainer.cpp
    src/backward_chainer.cpp
    src/rule_engine.cpp
    src/fact_store.cpp
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
    tests/test_forward_chainer.cpp
    tests/test_backward_chainer.cpp
    tests/test_rule_engine.cpp
    tests/test_fact_store.cpp
)

target_link_libraries(engines_tests
    engines
    gtest_main
    gtest
)

add_test(NAME EnginesTests COMMAND engines_tests)

# Engines benchmarks
add_executable(engines_benchmarks
    benchmarks/bench_rule_evaluation.cpp
    benchmarks/bench_fact_storage.cpp
    benchmarks/bench_pattern_matching.cpp
)

target_link_libraries(engines_benchmarks
    engines
    benchmark::benchmark
)