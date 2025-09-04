#include "moe_engine.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "moe_config.hpp"

using namespace inference_lab::common;

namespace engines::mixture_experts {

MoEEngine::MoEEngine(const MoEConfig& config) : config_(config) {
    // Initialize performance tracking
    recent_latencies_.reserve(1000);  // Keep last 1000 latencies for statistics
    recent_expert_selections_.reserve(1000);
}

auto MoEEngine::create(const MoEConfig& config)
    -> inference_lab::common::Result<std::unique_ptr<MoEEngine>, MoEError> {
    // Validate configuration first
    MoESystemConfig system_config{};
    system_config.num_experts = config.num_experts;
    system_config.expert_capacity = config.expert_capacity;
    system_config.max_concurrent_requests = config.max_concurrent_requests;
    system_config.memory_pool_size_mb = config.memory_pool_size_mb;

    auto validation_result = MoEConfigValidator::validate_configuration(system_config);
    if (!validation_result.is_configuration_valid()) {
        return Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }

    // Create MoE engine instance
    auto engine = std::unique_ptr<MoEEngine>(new MoEEngine(config));

    // Initialize core components
    RouterConfig router_config{};
    router_config.num_experts = config.num_experts;
    router_config.top_k_experts = config.expert_capacity;

    auto router_result = ExpertRouter::create(router_config);
    if (!router_result.is_ok()) {
        return Err(MoEError::ROUTING_NETWORK_FAILURE);
    }
    engine->expert_router_ = std::move(router_result).unwrap();

    // Initialize expert parameters
    ParameterConfig param_config{};
    param_config.num_experts = config.num_experts;
    param_config.memory_pool_size_mb = config.memory_pool_size_mb;

    auto param_result = ExpertParameters::create(param_config);
    if (!param_result.is_ok()) {
        return Err(MoEError::PARAMETER_STORAGE_ERROR);
    }
    engine->expert_parameters_ = std::move(param_result).unwrap();

    // Initialize sparse activation system
    SparseConfig sparse_config{};
    sparse_config.enable_simd_optimization = config.enable_sparse_activation;

    auto sparse_result = SparseActivation::create(sparse_config);
    if (!sparse_result.is_ok()) {
        return Err(MoEError::SPARSE_ACTIVATION_ERROR);
    }
    engine->sparse_activation_ = std::move(sparse_result).unwrap();

    // Initialize load balancer
    LoadBalancerConfig lb_config{};
    lb_config.num_experts = config.num_experts;
    lb_config.max_queue_size_per_expert = config.max_concurrent_requests / config.num_experts;

    auto lb_result = LoadBalancer::create(lb_config);
    if (!lb_result.is_ok()) {
        return Err(MoEError::LOAD_BALANCING_ERROR);
    }
    engine->load_balancer_ = std::move(lb_result).unwrap();

    // Initialize expert utilization tracking
    engine->expert_utilization_history_.resize(config.num_experts, 0.0f);

    return Ok(std::move(engine));
}

auto MoEEngine::run_inference(const MoEInput& input)
    -> inference_lab::common::Result<MoEResponse, MoEError> {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Select experts using routing network
    auto routing_start = std::chrono::high_resolution_clock::now();
    auto expert_selection = select_experts(input.features);
    if (!expert_selection.is_ok()) {
        return Err(expert_selection.unwrap_err());
    }
    auto selected_experts = expert_selection.unwrap();
    auto routing_end = std::chrono::high_resolution_clock::now();

    float routing_latency =
        std::chrono::duration<float, std::milli>(routing_end - routing_start).count();

    // Step 2: Compute expert weights
    auto weights_result = compute_expert_weights(input.features, selected_experts);
    if (!weights_result.is_ok()) {
        return Err(weights_result.unwrap_err());
    }
    auto expert_weights = weights_result.unwrap();

    // Step 3: Execute inference using selected experts
    auto inference_start = std::chrono::high_resolution_clock::now();
    auto inference_result = execute_expert_inference(input, selected_experts, expert_weights);
    if (!inference_result.is_ok()) {
        return Err(inference_result.unwrap_err());
    }
    auto outputs = inference_result.unwrap();
    auto inference_end = std::chrono::high_resolution_clock::now();

    float inference_latency =
        std::chrono::duration<float, std::milli>(inference_end - inference_start).count();

    // Step 4: Update performance metrics
    update_performance_metrics(routing_latency, inference_latency, selected_experts);

    // Calculate active parameters (for sparse activation)
    std::size_t active_parameters = 0;
    for (auto expert_id : selected_experts) {
        // Each expert has parameters_per_expert parameters, but only a fraction are active
        // This would be calculated based on sparse activation patterns
        active_parameters += static_cast<std::size_t>(1024 * 0.7f);  // Assuming 70% sparsity
    }

    // Construct response
    MoEResponse response{};
    response.outputs = std::move(outputs);
    response.selected_experts = std::move(selected_experts);
    response.expert_weights = std::move(expert_weights);
    response.routing_latency_ms = routing_latency;
    response.inference_latency_ms = inference_latency;
    response.active_parameters = active_parameters;

    // Increment request counter
    total_requests_.fetch_add(1, std::memory_order_relaxed);

    return Ok(std::move(response));
}

auto MoEEngine::select_experts(const std::vector<float>& features)
    -> inference_lab::common::Result<std::vector<std::size_t>, MoEError> {
    // Use expert router to select experts
    return expert_router_->select_experts(features);
}

auto MoEEngine::compute_expert_weights(const std::vector<float>& features,
                                       const std::vector<std::size_t>& selected_experts)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    // Use expert router to compute weights
    return expert_router_->compute_expert_weights(features, selected_experts);
}

auto MoEEngine::execute_expert_inference(const MoEInput& input,
                                         const std::vector<std::size_t>& selected_experts,
                                         const std::vector<float>& expert_weights)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    // This is a simplified implementation for the initial version
    // In a full implementation, this would:
    // 1. Load expert parameters from ExpertParameters
    // 2. Apply sparse activation patterns using SparseActivation
    // 3. Execute actual neural network forward pass for each expert
    // 4. Combine expert outputs using weights

    std::vector<float> combined_output(256, 0.0f);  // Assuming 256-dimensional output

    for (std::size_t i = 0; i < selected_experts.size(); ++i) {
        auto expert_id = selected_experts[i];
        auto weight = expert_weights[i];

        // Simulate expert computation (would be actual neural network inference)
        std::vector<float> expert_output(256);
        std::iota(expert_output.begin(), expert_output.end(), static_cast<float>(expert_id));

        // Combine with weight
        for (std::size_t j = 0; j < expert_output.size(); ++j) {
            combined_output[j] += weight * expert_output[j];
        }
    }

    return Ok(std::move(combined_output));
}

auto MoEEngine::update_performance_metrics(float routing_latency,
                                           float inference_latency,
                                           const std::vector<std::size_t>& selected_experts)
    -> void {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    // Update latency tracking
    float total_latency = routing_latency + inference_latency;
    recent_latencies_.push_back(total_latency);
    if (recent_latencies_.size() > 1000) {
        recent_latencies_.erase(recent_latencies_.begin());
    }

    // Update expert selection tracking
    for (auto expert_id : selected_experts) {
        recent_expert_selections_.push_back(expert_id);
    }
    if (recent_expert_selections_.size() > 1000) {
        recent_expert_selections_.erase(recent_expert_selections_.begin());
    }

    // Update running averages
    auto total_requests = total_requests_.load(std::memory_order_relaxed);
    if (total_requests > 0) {
        float current_avg = average_routing_time_ms_.load(std::memory_order_relaxed);
        float new_avg = (current_avg * (total_requests - 1) + routing_latency) / total_requests;
        average_routing_time_ms_.store(new_avg, std::memory_order_relaxed);
    }
}

auto MoEEngine::get_expert_utilization() const -> std::vector<float> {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    if (recent_expert_selections_.empty()) {
        return std::vector<float>(config_.num_experts, 0.0f);
    }

    std::vector<std::size_t> counts(config_.num_experts, 0);
    for (auto expert_id : recent_expert_selections_) {
        if (expert_id < config_.num_experts) {
            counts[expert_id]++;
        }
    }

    std::vector<float> utilization(config_.num_experts);
    float total_selections = static_cast<float>(recent_expert_selections_.size());

    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        utilization[i] = static_cast<float>(counts[i]) / total_selections;
    }

    return utilization;
}

auto MoEEngine::get_memory_usage() const -> std::vector<float> {
    // Get memory usage from expert parameters system
    auto param_stats = expert_parameters_->get_parameter_stats();
    return param_stats.expert_memory_usage;
}

auto MoEEngine::validate_system_health() const
    -> inference_lab::common::Result<std::monostate, MoEError> {
    // Check expert router health
    auto router_health = expert_router_->validate_routing_health();
    if (!router_health.is_ok()) {
        return Err(MoEError::ROUTING_NETWORK_FAILURE);
    }

    // Check parameter storage health
    auto param_health = expert_parameters_->validate_storage_integrity();
    if (!param_health.is_ok()) {
        return Err(MoEError::PARAMETER_STORAGE_ERROR);
    }

    // Check load balancer health
    auto lb_health = load_balancer_->validate_load_balancing_health();
    if (!lb_health.is_ok()) {
        return Err(MoEError::LOAD_BALANCING_ERROR);
    }

    // Check sparse activation health
    auto sparse_health = sparse_activation_->validate_simd_capabilities();
    if (!sparse_health.is_ok()) {
        return Err(MoEError::SPARSE_ACTIVATION_ERROR);
    }

    return Ok(std::monostate{});
}

}  // namespace engines::mixture_experts
