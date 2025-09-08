# Experimental research scenarios

# Experiments are built as individual executables
# rather than a library since they're standalone research programs

# Experiment 1: Rule evaluation optimization
add_executable(exp_rule_optimization
    src/rule_optimization_experiment.cpp
)

target_link_libraries(exp_rule_optimization
    engines
    performance
)

# Experiment 2: Distributed consensus comparison
add_executable(exp_consensus_comparison
    src/consensus_comparison_experiment.cpp
)

target_link_libraries(exp_consensus_comparison
    distributed
    performance
    common
)

# Experiment 3: Memory-aware inference
add_executable(exp_memory_inference
    src/memory_inference_experiment.cpp
)

target_link_libraries(exp_memory_inference
    engines
    performance
)

# Experiments tests (validation of experimental setups) - using placeholder for now
add_executable(experiments_tests
    tests/placeholder.cpp
)

target_link_libraries(experiments_tests
    GTest::gtest_main
)

add_test(NAME ExperimentsTests COMMAND experiments_tests)

# Create a target to run all experiments
add_custom_target(run_experiments
    COMMAND exp_rule_optimization
    COMMAND exp_consensus_comparison
    COMMAND exp_memory_inference
    DEPENDS 
        exp_rule_optimization
        exp_consensus_comparison
        exp_memory_inference
    COMMENT "Running all research experiments"
)
