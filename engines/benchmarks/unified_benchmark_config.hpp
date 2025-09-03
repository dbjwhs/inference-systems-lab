#pragma once

#include <cstdint>

namespace inference_lab::engines::unified_benchmarks {

struct UnifiedBenchmarkConfig {
    // Momentum-Enhanced BP Configuration
    static constexpr std::uint32_t MOMENTUM_BP_MAX_ITERATIONS = 100;
    static constexpr double MOMENTUM_BP_CONVERGENCE_THRESHOLD = 1e-6;
    static constexpr bool MOMENTUM_BP_ENABLE_MOMENTUM = true;
    static constexpr bool MOMENTUM_BP_ENABLE_ADAGRAD = true;

    // Circular BP Configuration
    static constexpr std::uint32_t CIRCULAR_BP_MAX_ITERATIONS = 100;
    static constexpr double CIRCULAR_BP_CONVERGENCE_THRESHOLD = 1e-6;
    static constexpr double CIRCULAR_BP_CORRELATION_THRESHOLD = 0.8;
    static constexpr std::uint32_t CIRCULAR_BP_MAX_CYCLE_LENGTH = 10;
    static constexpr std::uint32_t CIRCULAR_BP_STRATEGY = 2;

    // Mamba SSM Configuration
    static constexpr std::uint32_t MAMBA_SSM_D_MODEL = 128;
    static constexpr std::uint32_t MAMBA_SSM_D_STATE = 16;
    static constexpr std::uint32_t MAMBA_SSM_D_INNER = 512;
    static constexpr std::uint32_t MAMBA_SSM_SEQUENCE_LENGTH = 40;

    // Memory and Performance Constants
    static constexpr double MIN_MEMORY_MB = 0.1;
    static constexpr double DEFAULT_BASELINE_MEMORY_MB = 5.0;
    static constexpr double DEFAULT_SSM_ACCURACY = 0.95;
    static constexpr std::uint32_t DEFAULT_SSM_CONVERGENCE_ITERATIONS = 1;

    // Dataset Configuration
    static constexpr std::uint32_t RANDOM_SEED = 42;
    static constexpr std::uint32_t SMALL_BINARY_NODES = 4;
    static constexpr std::uint32_t SMALL_BINARY_EDGES = 4;
    static constexpr double SMALL_BINARY_DIFFICULTY = 0.5;
    static constexpr std::uint32_t MEDIUM_CHAIN_NODES = 10;
    static constexpr std::uint32_t MEDIUM_CHAIN_EDGES = 9;
    static constexpr double MEDIUM_CHAIN_DIFFICULTY = 0.18;
    static constexpr std::uint32_t LARGE_GRID_NODES = 25;
    static constexpr std::uint32_t LARGE_GRID_EDGES = 40;
    static constexpr double LARGE_GRID_DIFFICULTY = 0.13;
};

}  // namespace inference_lab::engines::unified_benchmarks
