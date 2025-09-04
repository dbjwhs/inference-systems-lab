#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <variant>

#include "../../../common/src/result.hpp"

namespace engines::mixture_experts {

enum class MoEError : std::uint8_t;

/**
 * @brief Configuration for expert routing network
 */
struct RouterConfig {
    std::size_t num_experts = 8;
    std::size_t input_dimension = 256;
    std::size_t hidden_dimension = 128;
    float learning_rate = 0.001f;
    float entropy_regularization = 0.1f;
    std::size_t top_k_experts = 2;
    bool enable_gradient_computation = true;
};

/**
 * @brief Routing statistics for monitoring and optimization
 */
struct RoutingStats {
    std::vector<float> expert_selection_frequency;
    std::vector<float> expert_weights_history;
    float entropy_score;
    float load_balance_coefficient;
    std::size_t total_routing_decisions;
};

/**
 * @brief Expert routing network with learnable parameters
 * 
 * Implements intelligent expert selection using gradient-based optimization:
 * - Learnable routing parameters with gradient computation
 * - Entropy-based load balancing to prevent expert collapse
 * - Adaptive routing algorithms with real-time performance monitoring
 * - O(log n) routing complexity scaling with number of experts
 * 
 * Performance targets:
 * - <5ms routing decision time per inference request  
 * - Expert utilization variance <20% under balanced workloads
 * - Routing algorithm convergence under various input distributions
 * - Graceful degradation when experts become unavailable
 */
class ExpertRouter {
public:
    /**
     * @brief Create expert router with specified configuration
     * @param config Router network configuration parameters
     * @return Result containing initialized router or error
     */
    static auto create(const RouterConfig& config) 
        -> inference_lab::common::Result<std::unique_ptr<ExpertRouter>, MoEError>;

    /**
     * @brief Select top-k experts for given input features
     * @param features Input feature vector
     * @return Result containing selected expert indices or error
     */
    auto select_experts(const std::vector<float>& features) 
        -> inference_lab::common::Result<std::vector<std::size_t>, MoEError>;

    /**
     * @brief Compute expert selection weights using routing network
     * @param features Input feature vector  
     * @param selected_experts Previously selected expert indices
     * @return Result containing expert weights or error
     */
    auto compute_expert_weights(const std::vector<float>& features,
                               const std::vector<std::size_t>& selected_experts)
        -> inference_lab::common::Result<std::vector<float>, MoEError>;

    /**
     * @brief Update routing parameters based on performance feedback
     * @param features Input that was routed
     * @param selected_experts Experts that were selected
     * @param performance_score Performance feedback (higher is better)
     * @return Result indicating update success or error
     */
    auto update_routing_parameters(const std::vector<float>& features,
                                  const std::vector<std::size_t>& selected_experts,
                                  float performance_score)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Get current routing statistics for monitoring
     * @return Current routing performance and load balancing metrics
     */
    auto get_routing_stats() const -> RoutingStats;

    /**
     * @brief Validate routing network health and convergence
     * @return Result indicating routing network status
     */
    auto validate_routing_health() const -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Reset routing statistics and performance counters
     */
    auto reset_statistics() -> void;

private:
    explicit ExpertRouter(const RouterConfig& config);

    // Routing network parameters (learnable)
    std::vector<std::vector<float>> routing_weights_;      // [hidden_dim x input_dim]
    std::vector<float> routing_biases_;                    // [hidden_dim]
    std::vector<std::vector<float>> expert_weights_;       // [num_experts x hidden_dim]
    std::vector<float> expert_biases_;                     // [num_experts]

    // Configuration and state
    RouterConfig config_;
    std::atomic<std::size_t> total_routing_calls_{0};
    std::atomic<float> average_routing_time_ms_{0.0f};

    // Performance monitoring (thread-safe)
    mutable std::mutex stats_mutex_;
    std::vector<std::atomic<std::size_t>> expert_selection_counts_;
    std::vector<float> recent_routing_times_;
    std::vector<float> recent_entropy_scores_;

    // Load balancing state
    std::vector<float> expert_load_history_;
    float load_balance_weight_;

    // Helper methods
    auto forward_pass(const std::vector<float>& features) 
        -> inference_lab::common::Result<std::vector<float>, MoEError>;
    
    auto apply_top_k_selection(const std::vector<float>& logits) 
        -> inference_lab::common::Result<std::vector<std::size_t>, MoEError>;
    
    auto compute_entropy_regularization(const std::vector<float>& probabilities) 
        -> float;
    
    auto update_load_balancing(const std::vector<std::size_t>& selected_experts) 
        -> void;
    
    auto compute_gradients(const std::vector<float>& features,
                          const std::vector<std::size_t>& selected_experts,
                          float performance_score)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    // Mathematical operations
    auto softmax(const std::vector<float>& logits) -> std::vector<float>;
    auto relu(float x) -> float { return std::max(0.0f, x); }
    auto sigmoid(float x) -> float { return 1.0f / (1.0f + std::exp(-x)); }
};

} // namespace engines::mixture_experts