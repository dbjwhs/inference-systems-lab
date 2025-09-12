// MIT License
// Copyright (c) 2025 dbjwhs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>

namespace inference_lab::engines::unified_benchmarks {

/**
 * @brief Unified configuration constants for benchmarking all POC techniques
 *
 * This struct provides centralized configuration for:
 * - Algorithm-specific parameters (iterations, thresholds)
 * - Memory management constants
 * - Dataset generation parameters
 * - Random seed for reproducibility
 *
 * All constants are compile-time configurable to enable easy tuning
 * for different experimental conditions while maintaining consistency.
 *
 * @example
 * ```cpp
 * // Use configuration in algorithm setup
 * momentum_bp::MomentumBPConfig config;
 * config.max_iterations = UnifiedBenchmarkConfig::MOMENTUM_BP_MAX_ITERATIONS;
 * config.convergence_threshold = UnifiedBenchmarkConfig::MOMENTUM_BP_CONVERGENCE_THRESHOLD;
 * ```
 */
struct UnifiedBenchmarkConfig {
    // Momentum-Enhanced BP Configuration
    /// Maximum iterations for momentum-enhanced belief propagation convergence
    static constexpr std::uint32_t MOMENTUM_BP_MAX_ITERATIONS = 100;
    /// Convergence threshold for momentum-enhanced belief propagation
    static constexpr double MOMENTUM_BP_CONVERGENCE_THRESHOLD = 1e-6;
    /// Enable momentum acceleration in belief propagation
    static constexpr bool MOMENTUM_BP_ENABLE_MOMENTUM = true;
    /// Enable AdaGrad adaptive learning rate in belief propagation
    static constexpr bool MOMENTUM_BP_ENABLE_ADAGRAD = true;

    // Circular BP Configuration
    /// Maximum iterations for circular belief propagation convergence
    static constexpr std::uint32_t CIRCULAR_BP_MAX_ITERATIONS = 100;
    /// Convergence threshold for circular belief propagation
    static constexpr double CIRCULAR_BP_CONVERGENCE_THRESHOLD = 1e-6;
    /// Correlation threshold for detecting spurious correlations in cycles
    static constexpr double CIRCULAR_BP_CORRELATION_THRESHOLD = 0.8;
    /// Maximum cycle length to consider in cycle detection algorithms
    static constexpr std::uint32_t CIRCULAR_BP_MAX_CYCLE_LENGTH = 10;
    /// Strategy identifier for circular BP algorithm variant (0=basic, 1=enhanced, 2=adaptive)
    static constexpr std::uint32_t CIRCULAR_BP_STRATEGY = 2;

    // Mamba SSM Configuration
    /// Model dimension for Mamba State Space Model hidden representations
    static constexpr std::uint32_t MAMBA_SSM_D_MODEL = 128;
    /// State dimension for Mamba SSM internal state vectors
    static constexpr std::uint32_t MAMBA_SSM_D_STATE = 16;
    /// Inner dimension for Mamba SSM feed-forward network expansion
    static constexpr std::uint32_t MAMBA_SSM_D_INNER = 512;
    /// Default sequence length for Mamba SSM input sequences (unused, computed dynamically)
    static constexpr std::uint32_t MAMBA_SSM_SEQUENCE_LENGTH = 40;

    // Memory and Performance Constants
    /// Minimum memory usage threshold in MB for realistic measurements
    static constexpr double MIN_MEMORY_MB = 0.1;
    /// Default baseline memory usage for process overhead estimation
    static constexpr double DEFAULT_BASELINE_MEMORY_MB = 5.0;
    /// Default accuracy value assigned to Mamba SSM for consistent reporting
    static constexpr double DEFAULT_SSM_ACCURACY = 0.95;
    /// Default convergence iterations for single-pass Mamba SSM inference
    static constexpr std::uint32_t DEFAULT_SSM_CONVERGENCE_ITERATIONS = 1;

    // Dataset Configuration
    /// Fixed random seed for reproducible dataset generation across all techniques
    static constexpr std::uint32_t RANDOM_SEED = 42;

    /// Number of nodes in small binary dataset for basic algorithm testing
    static constexpr std::uint32_t SMALL_BINARY_NODES = 4;
    /// Number of edges in small binary dataset (ensures equivalent computational complexity)
    static constexpr std::uint32_t SMALL_BINARY_EDGES = 4;
    /// Difficulty level (convergence challenge) for small binary dataset
    static constexpr double SMALL_BINARY_DIFFICULTY = 0.5;

    /// Number of nodes in medium chain dataset for moderate complexity testing
    static constexpr std::uint32_t MEDIUM_CHAIN_NODES = 10;
    /// Number of edges in medium chain dataset (linear scaling with nodes)
    static constexpr std::uint32_t MEDIUM_CHAIN_EDGES = 9;
    /// Difficulty level for medium chain dataset (lower = easier convergence)
    static constexpr double MEDIUM_CHAIN_DIFFICULTY = 0.18;

    /// Number of nodes in large grid dataset for high complexity testing
    static constexpr std::uint32_t LARGE_GRID_NODES = 25;
    /// Number of edges in large grid dataset (higher connectivity for stress testing)
    static constexpr std::uint32_t LARGE_GRID_EDGES = 40;
    /// Difficulty level for large grid dataset (lowest for challenging convergence)
    static constexpr double LARGE_GRID_DIFFICULTY = 0.13;
};

}  // namespace inference_lab::engines::unified_benchmarks
