// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../../common/src/result.hpp"

namespace engines::mixture_experts {

enum class MoEError : std::uint8_t;

/**
 * @brief Configuration for load balancing system
 */
struct LoadBalancerConfig {
    std::size_t num_experts = 8;
    float load_balance_weight = 0.1f;
    std::size_t monitoring_window_ms = 1000;     // Performance monitoring window
    std::size_t max_queue_size_per_expert = 50;  // Maximum requests queued per expert
    float overload_threshold = 0.8f;             // Expert utilization threshold for overload
    bool enable_adaptive_routing = true;         // Dynamically adjust routing based on load
    float expert_capacity_factor = 1.2f;         // Safety factor for expert capacity planning
};

/**
 * @brief Real-time expert load information
 */
struct ExpertLoad {
    std::size_t expert_id;
    std::size_t active_requests{0};          // Protected by load_tracking_mutex_
    std::size_t queued_requests{0};          // Protected by load_tracking_mutex_
    float average_processing_time_ms{0.0f};  // Protected by load_tracking_mutex_
    float current_utilization{0.0f};         // Protected by load_tracking_mutex_
    std::chrono::steady_clock::time_point last_update;
    bool is_overloaded{false};
};

/**
 * @brief Load balancing statistics and metrics
 */
struct LoadBalancingStats {
    std::vector<float> expert_utilization_rates;
    std::vector<std::size_t> expert_request_counts;
    std::vector<float> expert_average_latencies;
    float overall_utilization_variance;
    float load_balance_coefficient;
    std::size_t total_requests_processed;
    std::size_t total_requests_rejected;
    float average_queue_depth;
};

/**
 * @brief Request queued for expert processing
 */
struct QueuedRequest {
    std::size_t request_id;
    std::size_t expert_id;
    std::chrono::steady_clock::time_point enqueue_time;
    std::vector<float> input_data;
    float priority_score{1.0f};
};

/**
 * @brief Dynamic load balancing and dispatch system
 *
 * Implements intelligent work distribution across expert networks:
 * - Real-time load monitoring with performance metrics
 * - Adaptive routing algorithms preventing computational bottlenecks
 * - Queue management with priority scheduling
 * - Automatic expert rebalancing based on utilization patterns
 *
 * Performance targets:
 * - Expert utilization variance <20% under balanced workloads
 * - <1 second recovery from individual expert failures
 * - Support 100+ concurrent inference requests with <10% degradation
 * - Intelligent work distribution preventing expert bottlenecks
 */
class LoadBalancer {
    friend class RequestTracker;  // Allow access to private members for validation
  public:
    /**
     * @brief Create load balancer with specified configuration
     * @param config Load balancing configuration
     * @return Result containing initialized load balancer or error
     */
    static auto create(const LoadBalancerConfig& config)
        -> inference_lab::common::Result<std::unique_ptr<LoadBalancer>, MoEError>;

    /**
     * @brief Select optimal expert for new request based on current load
     * @param candidate_experts List of expert candidates from routing
     * @param expert_weights Routing weights for candidates
     * @return Result containing selected expert ID or error
     */
    auto select_optimal_expert(const std::vector<std::size_t>& candidate_experts,
                               const std::vector<float>& expert_weights)
        -> inference_lab::common::Result<std::size_t, MoEError>;

    /**
     * @brief Register start of request processing for expert
     * @param expert_id Expert handling the request
     * @param request_id Unique request identifier
     * @return Result indicating registration success or error
     */
    auto register_request_start(std::size_t expert_id, std::size_t request_id)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Register completion of request processing for expert
     * @param expert_id Expert that handled the request
     * @param request_id Unique request identifier
     * @param processing_time_ms Time taken to process request
     * @return Result indicating completion success or error
     */
    auto register_request_completion(std::size_t expert_id,
                                     std::size_t request_id,
                                     float processing_time_ms)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Get current load balancing statistics
     * @return Load distribution and performance metrics
     */
    auto get_load_balancing_stats() const -> LoadBalancingStats;

    /**
     * @brief Get current load information for all experts
     * @return Vector of expert load information
     */
    auto get_expert_loads() const -> std::vector<ExpertLoad>;

    /**
     * @brief Trigger rebalancing of expert assignments
     * @return Result indicating rebalancing success or error
     */
    auto trigger_expert_rebalancing() -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Handle expert failure and redistribute load
     * @param failed_expert_id Expert that has failed
     * @return Result indicating failure handling success or error
     */
    auto handle_expert_failure(std::size_t failed_expert_id)
        -> inference_lab::common::Result<std::monostate, MoEError>;

    /**
     * @brief Validate load balancing system health
     * @return Result indicating system health status
     */
    auto validate_load_balancing_health() const
        -> inference_lab::common::Result<std::monostate, MoEError>;

  private:
    explicit LoadBalancer(const LoadBalancerConfig& config);

    // Configuration
    LoadBalancerConfig config_;

    // Expert load tracking
    std::vector<std::unique_ptr<ExpertLoad>> expert_loads_;
    mutable std::mutex load_tracking_mutex_;

    // Request queuing system
    std::unordered_map<std::size_t, std::queue<QueuedRequest>> expert_queues_;
    std::mutex queue_mutex_;

    // Performance monitoring
    std::size_t total_requests_processed_{0};   // Protected by load_tracking_mutex_
    std::size_t total_requests_rejected_{0};    // Protected by load_tracking_mutex_
    float overall_utilization_variance_{0.0f};  // Protected by load_tracking_mutex_

    // Load balancing algorithms
    std::vector<float> expert_load_history_;
    std::vector<float> expert_performance_scores_;
    mutable std::mutex balancing_mutex_;

    // Adaptive routing state
    std::vector<float> routing_adjustment_factors_;
    std::chrono::steady_clock::time_point last_rebalancing_time_;

    // Helper methods
    auto update_expert_load(std::size_t expert_id, float processing_time_ms) -> void;

    auto calculate_utilization_variance() -> float;

    auto compute_expert_score(std::size_t expert_id, const std::vector<float>& expert_weights)
        -> float;

    auto select_least_loaded_expert(const std::vector<std::size_t>& candidates) -> std::size_t;

    auto select_weighted_expert(const std::vector<std::size_t>& candidates,
                                const std::vector<float>& weights) -> std::size_t;

    auto is_expert_overloaded(std::size_t expert_id) const -> bool;

    auto get_expert_capacity(std::size_t expert_id) const -> float;

    auto redistribute_queued_requests(std::size_t failed_expert_id) -> void;

    auto update_routing_adjustments() -> void;

    // Performance monitoring helpers
    auto update_performance_window() -> void;

    auto calculate_load_balance_coefficient() const -> float;

    auto detect_load_imbalance() const -> bool;

    // Queue management
    auto enqueue_request(const QueuedRequest& request) -> bool;

    auto dequeue_next_request(std::size_t expert_id)
        -> inference_lab::common::Result<QueuedRequest, MoEError>;

    auto get_queue_depth(std::size_t expert_id) const -> std::size_t;

    auto cleanup_stale_requests() -> void;

    // Adaptive algorithms
    auto compute_adaptive_weights(const std::vector<std::size_t>& candidates,
                                  const std::vector<float>& routing_weights) -> std::vector<float>;

    auto apply_load_balancing_penalty(std::size_t expert_id, float base_weight) -> float;
};

/**
 * @brief RAII request tracker for automatic load balancing
 */
class RequestTracker {
  public:
    /**
     * @brief Create a RequestTracker with error handling
     * @param load_balancer Reference to the load balancer
     * @param expert_id ID of the expert being tracked
     * @param request_id Unique request identifier
     * @return Result containing RequestTracker or error
     */
    static auto create(LoadBalancer& load_balancer, std::size_t expert_id, std::size_t request_id)
        -> inference_lab::common::Result<RequestTracker, MoEError>;

    ~RequestTracker();

    // Non-copyable, movable
    RequestTracker(const RequestTracker&) = delete;
    RequestTracker& operator=(const RequestTracker&) = delete;

    RequestTracker(RequestTracker&& other) noexcept;
    RequestTracker& operator=(RequestTracker&& other) noexcept;

    /**
     * @brief Mark request as completed (called automatically in destructor)
     */
    auto complete() -> void;

  private:
    // Private constructor - use create() factory method
    RequestTracker(LoadBalancer& load_balancer, std::size_t expert_id, std::size_t request_id);
    LoadBalancer* load_balancer_;
    std::size_t expert_id_;
    std::size_t request_id_;
    std::chrono::steady_clock::time_point start_time_;
    bool completed_{false};
};

}  // namespace engines::mixture_experts
