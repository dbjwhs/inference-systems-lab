#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engines::mixture_experts {

/**
 * @brief Comprehensive configuration for Mixture of Experts system
 *
 * This configuration combines all component configurations into a unified
 * system configuration with validation and default value management.
 */
struct MoESystemConfig {
    // === Core System Configuration ===
    std::string system_name = "mixture_of_experts";
    std::string version = "1.0.0";
    bool enable_debug_logging = false;
    bool enable_performance_monitoring = true;

    // === Expert Configuration ===
    std::size_t num_experts = 8;               // Total number of expert networks
    std::size_t expert_capacity = 2;           // Number of experts selected per inference
    std::size_t parameters_per_expert = 1024;  // Parameter count per expert
    std::size_t input_dimension = 256;         // Input feature dimension
    std::size_t output_dimension = 256;        // Output dimension per expert

    // === Routing Configuration ===
    std::size_t routing_hidden_dimension = 128;  // Hidden layer size for routing network
    float routing_learning_rate = 0.001f;        // Learning rate for routing parameters
    float entropy_regularization = 0.1f;         // Entropy regularization weight
    std::size_t top_k_experts = 2;               // Top-k expert selection
    bool enable_gradient_computation = true;     // Enable routing parameter updates

    // === Load Balancing Configuration ===
    float load_balance_weight = 0.1f;            // Weight for load balancing in routing
    std::size_t monitoring_window_ms = 1000;     // Performance monitoring window
    std::size_t max_queue_size_per_expert = 50;  // Maximum requests queued per expert
    float overload_threshold = 0.8f;             // Expert utilization threshold
    bool enable_adaptive_routing = true;         // Dynamic routing adjustment
    float expert_capacity_factor = 1.2f;         // Safety factor for capacity planning

    // === Sparse Activation Configuration ===
    float sparsity_threshold = 0.01f;      // Threshold for sparse activation
    bool enable_simd_optimization = true;  // SIMD vectorization
    std::size_t vector_alignment = 32;     // Memory alignment (256-bit AVX2)
    std::size_t block_size = 64;           // Processing block size
    bool enable_dynamic_sparsity = true;   // Adaptive sparsity thresholding
    float target_sparsity_ratio = 0.7f;    // Target 70% sparsity

    // === Memory Management Configuration ===
    std::size_t memory_pool_size_mb = 500;     // Memory pool size per expert
    bool enable_parameter_compression = true;  // Parameter compression
    bool enable_parameter_streaming = false;   // Parameter streaming for large models
    float compression_ratio = 0.8f;            // Target compression ratio
    std::size_t cache_size_experts = 4;        // Experts to keep in fast cache

    // === Performance Configuration ===
    std::size_t max_concurrent_requests = 100;   // Maximum concurrent inference requests
    float target_latency_p95_ms = 150.0f;        // Target P95 latency
    float target_latency_p50_ms = 75.0f;         // Target P50 latency
    float target_throughput_multiplier = 3.0f;   // Target throughput improvement
    float target_efficiency_multiplier = 20.0f;  // Target computational efficiency

    // === Quality and Reliability Configuration ===
    float minimum_accuracy_threshold = 0.98f;        // Minimum accuracy preservation
    float maximum_utilization_variance = 0.2f;       // Maximum expert utilization variance
    std::size_t failure_recovery_timeout_ms = 1000;  // Expert failure recovery timeout
    bool enable_health_monitoring = true;            // System health monitoring
    std::size_t health_check_interval_ms = 5000;     // Health check frequency

    /**
     * @brief Validate configuration parameters
     * @return True if configuration is valid, false otherwise
     */
    auto validate() const -> bool;

    /**
     * @brief Get configuration summary as string
     * @return Human-readable configuration summary
     */
    auto to_string() const -> std::string;

    /**
     * @brief Create optimized configuration for development/testing
     * @return Configuration suitable for development environment
     */
    static auto create_development_config() -> MoESystemConfig;

    /**
     * @brief Create optimized configuration for production deployment
     * @return Configuration suitable for production environment
     */
    static auto create_production_config() -> MoESystemConfig;

    /**
     * @brief Create configuration optimized for memory-constrained environments
     * @return Configuration with reduced memory footprint
     */
    static auto create_lightweight_config() -> MoESystemConfig;

    /**
     * @brief Create configuration optimized for maximum performance
     * @return Configuration prioritizing performance over memory usage
     */
    static auto create_performance_config() -> MoESystemConfig;
};

/**
 * @brief Validation result for configuration parameters
 */
struct ConfigValidationResult {
    bool is_valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<std::string> recommendations;

    /**
     * @brief Check if configuration passed validation
     */
    auto is_configuration_valid() const -> bool { return is_valid && errors.empty(); }

    /**
     * @brief Get formatted validation report
     */
    auto get_validation_report() const -> std::string;
};

/**
 * @brief Comprehensive configuration validator
 */
class MoEConfigValidator {
  public:
    /**
     * @brief Validate MoE system configuration
     * @param config Configuration to validate
     * @return Detailed validation result with errors, warnings, recommendations
     */
    static auto validate_configuration(const MoESystemConfig& config) -> ConfigValidationResult;

    /**
     * @brief Validate configuration against system capabilities
     * @param config Configuration to validate
     * @return Validation result considering current system resources
     */
    static auto validate_against_system_capabilities(const MoESystemConfig& config)
        -> ConfigValidationResult;

    /**
     * @brief Recommend configuration optimizations
     * @param config Current configuration
     * @return Optimized configuration with performance improvements
     */
    static auto recommend_optimizations(const MoESystemConfig& config) -> MoESystemConfig;

    static auto estimate_memory_requirements(const MoESystemConfig& config) -> std::size_t;

    static auto estimate_performance_characteristics(const MoESystemConfig& config)
        -> std::pair<float, float>;  // Returns (expected_latency_ms, expected_throughput)

  private:
    static auto validate_expert_configuration(const MoESystemConfig& config,
                                              ConfigValidationResult& result) -> void;

    static auto validate_routing_configuration(const MoESystemConfig& config,
                                               ConfigValidationResult& result) -> void;

    static auto validate_performance_configuration(const MoESystemConfig& config,
                                                   ConfigValidationResult& result) -> void;

    static auto validate_memory_configuration(const MoESystemConfig& config,
                                              ConfigValidationResult& result) -> void;

    static auto check_system_memory_availability(std::size_t required_memory_mb) -> bool;

    static auto check_simd_capability_availability() -> bool;
};

/**
 * @brief Configuration constants aligned with roadmap success metrics
 */
namespace MoEConstants {
// Performance targets from roadmap
constexpr float TARGET_EFFICIENCY_MIN = 15.0f;  // Minimum 15x efficiency improvement
constexpr float TARGET_EFFICIENCY_MAX = 25.0f;  // Maximum 25x efficiency improvement
constexpr float TARGET_THROUGHPUT_MIN = 2.0f;   // Minimum 2x throughput improvement
constexpr float TARGET_THROUGHPUT_MAX = 5.0f;   // Maximum 5x throughput improvement

// Latency targets (milliseconds)
constexpr float TARGET_P50_LATENCY_MS = 75.0f;      // P50 latency target
constexpr float TARGET_P95_LATENCY_MS = 150.0f;     // P95 latency target
constexpr float TARGET_P99_LATENCY_MS = 300.0f;     // P99 latency target
constexpr float TARGET_EXPERT_SELECTION_MS = 5.0f;  // Expert selection time
constexpr float TARGET_EXPERT_LOADING_MS = 50.0f;   // Expert loading time

// Memory efficiency targets
constexpr float TARGET_MEMORY_OVERHEAD_MAX = 0.3f;        // Maximum 30% memory overhead
constexpr float TARGET_PARAMETER_UTILIZATION_MIN = 0.6f;  // Minimum 60% parameter utilization
constexpr float TARGET_PARAMETER_UTILIZATION_MAX = 0.8f;  // Maximum 80% parameter utilization
constexpr float TARGET_MEMORY_BANDWIDTH_MAX = 2.0f;       // Maximum 2GB/s memory bandwidth
constexpr float TARGET_EXPERT_STORAGE_MB = 500.0f;        // Target expert storage size

// Accuracy and quality targets
constexpr float TARGET_ACCURACY_PRESERVATION = 0.98f;    // Minimum 98% accuracy preservation
constexpr float TARGET_EXPERT_CONSENSUS = 0.9f;          // Minimum 90% expert consensus
constexpr float TARGET_UTILIZATION_VARIANCE_MAX = 0.2f;  // Maximum 20% utilization variance
constexpr float TARGET_ROBUSTNESS_MIN = 0.98f;  // Minimum 98% robustness (2% degradation max)

// Production readiness targets
constexpr float TARGET_TEST_COVERAGE = 0.87f;               // Minimum 87% test coverage
constexpr std::size_t TARGET_CONCURRENT_REQUESTS = 100;     // Support 100+ concurrent requests
constexpr float TARGET_PERFORMANCE_DEGRADATION_MAX = 0.1f;  // Maximum 10% performance degradation
constexpr float TARGET_RECOVERY_TIME_MS = 1000.0f;          // Maximum 1 second recovery time

// Integration targets
constexpr float TARGET_EXISTING_REGRESSION_MAX =
    0.05f;  // Maximum 5% regression in existing techniques
constexpr float TARGET_MEMORY_POOL_OVERHEAD_MAX = 0.1f;  // Maximum 10% memory pool overhead
}  // namespace MoEConstants

}  // namespace engines::mixture_experts
