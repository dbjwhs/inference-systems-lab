#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

#include "../../../common/src/result.hpp"
#include "expert_parameters.hpp"
#include "expert_router.hpp"
#include "load_balancer.hpp"
#include "sparse_activation.hpp"

namespace engines::mixture_experts {

/**
 * @brief Error types for Mixture of Experts operations
 */
enum class MoEError : std::uint8_t {
    EXPERT_INITIALIZATION_FAILED,
    ROUTING_NETWORK_FAILURE,
    LOAD_BALANCING_ERROR,
    PARAMETER_STORAGE_ERROR,
    SPARSE_ACTIVATION_ERROR,
    EXPERT_SELECTION_TIMEOUT,
    MEMORY_ALLOCATION_FAILURE,
    CONCURRENT_ACCESS_VIOLATION
};

/**
 * @brief Configuration for Mixture of Experts system
 */
struct MoEConfig {
    std::size_t num_experts = 8;
    std::size_t expert_capacity = 2;  // Number of experts to select per inference
    float load_balancing_weight = 0.1f;
    bool enable_sparse_activation = true;
    std::size_t max_concurrent_requests = 100;
    std::size_t memory_pool_size_mb = 500;
};

/**
 * @brief Input for MoE inference
 */
struct MoEInput {
    std::vector<float> features;
    std::size_t batch_size = 1;
    bool enable_load_balancing = true;
    std::size_t request_id = 0;
    float priority = 1.0f;
};

/**
 * @brief Response from MoE inference
 */
struct MoEResponse {
    std::vector<float> outputs;
    std::vector<std::size_t> selected_experts;
    std::vector<float> expert_weights;
    float routing_latency_ms;
    float inference_latency_ms;
    std::size_t active_parameters;
};

/**
 * @brief Main Mixture of Experts inference engine
 *
 * This class implements a production-ready Mixture of Experts system with:
 * - Dynamic expert routing with learnable parameters
 * - Load balancing across expert networks
 * - Sparse activation patterns for computational efficiency
 * - Memory-efficient parameter management using existing infrastructure
 * - Integration with existing Result<T,E> error handling patterns
 *
 * Performance targets:
 * - 15-25x computational efficiency improvement over single-expert baselines
 * - <5ms expert selection latency per request
 * - <30% memory overhead while maintaining >98% accuracy
 * - Support for 100+ concurrent inference requests
 */
class MoEEngine {
  public:
    /**
     * @brief Create MoE engine with specified configuration
     * @param config MoE system configuration parameters
     * @return Result containing initialized engine or error
     */
    static auto create(const MoEConfig& config)
        -> inference_lab::common::Result<std::unique_ptr<MoEEngine>, MoEError>;

    // Delete copy and move operations (due to atomic members and complexity)
    MoEEngine(const MoEEngine&) = delete;
    MoEEngine& operator=(const MoEEngine&) = delete;
    MoEEngine(MoEEngine&&) = delete;
    MoEEngine& operator=(MoEEngine&&) = delete;
    // Destructor
    ~MoEEngine() = default;

    /**
     * @brief Run inference using mixture of experts
     * @param input Input features and configuration
     * @return Result containing inference response or error
     */
    auto run_inference(const MoEInput& input)
        -> inference_lab::common::Result<MoEResponse, MoEError>;

    /**
     * @brief Get current expert utilization statistics
     * @return Expert utilization variance and load distribution metrics
     */
    auto get_expert_utilization() const -> std::vector<float>;

    /**
     * @brief Get memory usage statistics for all experts
     * @return Memory usage in MB per expert
     */
    auto get_memory_usage() const -> std::vector<float>;

    /**
     * @brief Validate system health and performance metrics
     * @return Result indicating system health status
     */
    auto validate_system_health() const -> inference_lab::common::Result<std::monostate, MoEError>;

  private:
    MoEEngine(const MoEConfig& config);

    // Core components
    std::unique_ptr<ExpertRouter> expert_router_;
    std::unique_ptr<ExpertParameters> expert_parameters_;
    std::unique_ptr<SparseActivation> sparse_activation_;
    std::unique_ptr<LoadBalancer> load_balancer_;

    // Configuration and state
    MoEConfig config_;
    std::vector<float> expert_utilization_history_;
    std::atomic<std::size_t> total_requests_{0};
    std::atomic<float> average_routing_time_ms_{0.0f};

    // Performance monitoring
    mutable std::mutex metrics_mutex_;
    std::vector<float> recent_latencies_;
    std::vector<std::size_t> recent_expert_selections_;

    // Helper methods
    auto select_experts(const std::vector<float>& features)
        -> inference_lab::common::Result<std::vector<std::size_t>, MoEError>;

    auto compute_expert_weights(const std::vector<float>& features,
                                const std::vector<std::size_t>& selected_experts)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;

    auto execute_expert_inference(const MoEInput& input,
                                  const std::vector<std::size_t>& selected_experts,
                                  const std::vector<float>& expert_weights)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;

    auto update_performance_metrics(float routing_latency,
                                    float inference_latency,
                                    const std::vector<std::size_t>& selected_experts) -> void;
};

}  // namespace engines::mixture_experts
