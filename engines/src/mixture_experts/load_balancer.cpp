#include "load_balancer.hpp"
#include <algorithm>
#include <random>
#include <numeric>

namespace engines::mixture_experts {

LoadBalancer::LoadBalancer(const LoadBalancerConfig& config) 
    : config_(config), last_rebalancing_time_(std::chrono::steady_clock::now()) {
    
    // Initialize expert loads
    expert_loads_.reserve(config_.num_experts);
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        auto load = std::make_unique<ExpertLoad>();
        load->expert_id = i;
        expert_loads_.push_back(std::move(load));
    }
    
    // Initialize queues
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        expert_queues_[i] = std::queue<QueuedRequest>();
    }
    
    // Initialize load tracking arrays
    expert_load_history_.resize(config_.num_experts, 0.0f);
    expert_performance_scores_.resize(config_.num_experts, 1.0f);
    routing_adjustment_factors_.resize(config_.num_experts, 1.0f);
}

auto LoadBalancer::create(const LoadBalancerConfig& config) 
    -> inference_lab::common::Result<std::unique_ptr<LoadBalancer>, MoEError> {
    
    if (config.num_experts == 0) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    if (config.max_queue_size_per_expert == 0) {
        return inference_lab::common::Err(MoEError::LOAD_BALANCING_ERROR);
    }
    
    auto balancer = std::unique_ptr<LoadBalancer>(new LoadBalancer(config));
    return inference_lab::common::Ok(std::move(balancer));
}

auto LoadBalancer::select_optimal_expert(const std::vector<std::size_t>& candidate_experts,
                                        const std::vector<float>& expert_weights)
    -> inference_lab::common::Result<std::size_t, MoEError> {
    
    if (candidate_experts.empty() || candidate_experts.size() != expert_weights.size()) {
        return inference_lab::common::Err(MoEError::LOAD_BALANCING_ERROR);
    }
    
    // Check if all candidates are valid
    for (auto expert_id : candidate_experts) {
        if (expert_id >= config_.num_experts) {
            return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
        }
    }
    
    std::size_t selected_expert;
    
    if (config_.enable_adaptive_routing) {
        // Use adaptive algorithm considering both routing weights and current load
        auto adaptive_weights = compute_adaptive_weights(candidate_experts, expert_weights);
        selected_expert = select_weighted_expert(candidate_experts, adaptive_weights);
    } else {
        // Simple weighted selection based on routing network output
        selected_expert = select_weighted_expert(candidate_experts, expert_weights);
    }
    
    return inference_lab::common::Ok(selected_expert);
}

auto LoadBalancer::register_request_start(std::size_t expert_id, std::size_t request_id)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    if (expert_id >= config_.num_experts) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    std::lock_guard<std::mutex> lock(load_tracking_mutex_);
    
    auto& expert_load = expert_loads_[expert_id];
    expert_load->active_requests.fetch_add(1);
    expert_load->last_update = std::chrono::steady_clock::now();
    
    return inference_lab::common::Ok(std::monostate{});
}

auto LoadBalancer::register_request_completion(std::size_t expert_id, 
                                              std::size_t request_id,
                                              float processing_time_ms)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    if (expert_id >= config_.num_experts) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    {
        std::lock_guard<std::mutex> lock(load_tracking_mutex_);
        
        auto& expert_load = expert_loads_[expert_id];
        expert_load->active_requests.fetch_sub(1);
        expert_load->last_update = std::chrono::steady_clock::now();
        
        // Update average processing time with exponential moving average
        float current_avg = expert_load->average_processing_time_ms.load();
        const float alpha = 0.1f;
        float new_avg = alpha * processing_time_ms + (1.0f - alpha) * current_avg;
        expert_load->average_processing_time_ms.store(new_avg);
        
        update_expert_load(expert_id, processing_time_ms);
    }
    
    total_requests_processed_.fetch_add(1);
    
    return inference_lab::common::Ok(std::monostate{});
}

auto LoadBalancer::update_expert_load(std::size_t expert_id, float processing_time_ms) -> void {
    // Update load history with exponential moving average
    const float alpha = 0.05f;
    float current_load = expert_loads_[expert_id]->active_requests.load();
    expert_load_history_[expert_id] = alpha * current_load + (1.0f - alpha) * expert_load_history_[expert_id];
    
    // Update utilization (simplified calculation)
    float utilization = current_load / static_cast<float>(config_.max_queue_size_per_expert);
    expert_loads_[expert_id]->current_utilization.store(std::min(utilization, 1.0f));
    expert_loads_[expert_id]->is_overloaded = (utilization > config_.overload_threshold);
    
    // Update overall utilization variance
    float variance = calculate_utilization_variance();
    overall_utilization_variance_.store(variance);
}

auto LoadBalancer::calculate_utilization_variance() -> float {
    if (expert_load_history_.empty()) {
        return 0.0f;
    }
    
    float mean = 0.0f;
    for (float load : expert_load_history_) {
        mean += load;
    }
    mean /= static_cast<float>(expert_load_history_.size());
    
    float variance = 0.0f;
    for (float load : expert_load_history_) {
        float diff = load - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(expert_load_history_.size());
    
    return variance;
}

auto LoadBalancer::compute_expert_score(std::size_t expert_id, 
                                       const std::vector<float>& expert_weights) -> float {
    // Find the weight for this expert
    float routing_weight = 0.0f;
    // Since expert_weights corresponds to candidate_experts, we need the index
    // This is a simplified version - in practice, we'd need the mapping
    
    float load_penalty = 1.0f;
    if (is_expert_overloaded(expert_id)) {
        load_penalty = 0.1f;  // Heavy penalty for overloaded experts
    } else {
        float current_utilization = expert_loads_[expert_id]->current_utilization.load();
        load_penalty = 1.0f - (current_utilization * config_.load_balance_weight);
    }
    
    float performance_bonus = expert_performance_scores_[expert_id];
    float routing_adjustment = routing_adjustment_factors_[expert_id];
    
    return routing_weight * load_penalty * performance_bonus * routing_adjustment;
}

auto LoadBalancer::select_least_loaded_expert(const std::vector<std::size_t>& candidates) 
    -> std::size_t {
    
    std::size_t best_expert = candidates[0];
    float lowest_utilization = expert_loads_[best_expert]->current_utilization.load();
    
    for (std::size_t i = 1; i < candidates.size(); ++i) {
        std::size_t expert_id = candidates[i];
        float utilization = expert_loads_[expert_id]->current_utilization.load();
        
        if (utilization < lowest_utilization) {
            lowest_utilization = utilization;
            best_expert = expert_id;
        }
    }
    
    return best_expert;
}

auto LoadBalancer::select_weighted_expert(const std::vector<std::size_t>& candidates,
                                         const std::vector<float>& weights) -> std::size_t {
    
    // Weighted random selection
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float total_weight = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (total_weight <= 1e-10f) {
        // Fallback to least loaded expert
        return select_least_loaded_expert(candidates);
    }
    
    std::uniform_real_distribution<float> dist(0.0f, total_weight);
    float random_value = dist(gen);
    
    float cumulative_weight = 0.0f;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        cumulative_weight += weights[i];
        if (random_value <= cumulative_weight) {
            return candidates[i];
        }
    }
    
    // Fallback (shouldn't happen)
    return candidates.back();
}

auto LoadBalancer::is_expert_overloaded(std::size_t expert_id) const -> bool {
    if (expert_id >= expert_loads_.size()) {
        return true;
    }
    
    return expert_loads_[expert_id]->is_overloaded;
}

auto LoadBalancer::get_expert_capacity(std::size_t expert_id) const -> float {
    if (expert_id >= expert_loads_.size()) {
        return 0.0f;
    }
    
    float current_utilization = expert_loads_[expert_id]->current_utilization.load();
    return std::max(0.0f, 1.0f - current_utilization);
}

auto LoadBalancer::compute_adaptive_weights(const std::vector<std::size_t>& candidates,
                                           const std::vector<float>& routing_weights) 
    -> std::vector<float> {
    
    std::vector<float> adaptive_weights;
    adaptive_weights.reserve(candidates.size());
    
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        std::size_t expert_id = candidates[i];
        float base_weight = routing_weights[i];
        
        float adjusted_weight = apply_load_balancing_penalty(expert_id, base_weight);
        adaptive_weights.push_back(adjusted_weight);
    }
    
    return adaptive_weights;
}

auto LoadBalancer::apply_load_balancing_penalty(std::size_t expert_id, float base_weight) 
    -> float {
    
    if (is_expert_overloaded(expert_id)) {
        return base_weight * 0.01f;  // Heavy penalty
    }
    
    float utilization = expert_loads_[expert_id]->current_utilization.load();
    float penalty_factor = 1.0f - (utilization * config_.load_balance_weight);
    
    return base_weight * std::max(0.01f, penalty_factor);
}

auto LoadBalancer::get_load_balancing_stats() const -> LoadBalancingStats {
    LoadBalancingStats stats;
    
    std::lock_guard<std::mutex> lock(load_tracking_mutex_);
    
    stats.expert_utilization_rates.reserve(config_.num_experts);
    stats.expert_request_counts.reserve(config_.num_experts);
    stats.expert_average_latencies.reserve(config_.num_experts);
    
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        const auto& expert_load = expert_loads_[i];
        stats.expert_utilization_rates.push_back(expert_load->current_utilization.load());
        stats.expert_request_counts.push_back(expert_load->active_requests.load());
        stats.expert_average_latencies.push_back(expert_load->average_processing_time_ms.load());
    }
    
    stats.overall_utilization_variance = overall_utilization_variance_.load();
    stats.load_balance_coefficient = calculate_load_balance_coefficient();
    stats.total_requests_processed = total_requests_processed_.load();
    stats.total_requests_rejected = total_requests_rejected_.load();
    
    // Calculate average queue depth
    float total_queue_depth = 0.0f;
    for (const auto& [expert_id, queue] : expert_queues_) {
        total_queue_depth += static_cast<float>(queue.size());
    }
    stats.average_queue_depth = total_queue_depth / static_cast<float>(config_.num_experts);
    
    return stats;
}

auto LoadBalancer::calculate_load_balance_coefficient() const -> float {
    if (expert_load_history_.empty()) {
        return 1.0f;
    }
    
    float mean = std::accumulate(expert_load_history_.begin(), expert_load_history_.end(), 0.0f) 
                 / static_cast<float>(expert_load_history_.size());
    
    if (mean < 1e-6f) {
        return 0.0f;
    }
    
    float variance = 0.0f;
    for (float load : expert_load_history_) {
        float diff = load - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(expert_load_history_.size());
    
    return std::sqrt(variance) / mean;  // Coefficient of variation
}

auto LoadBalancer::get_expert_loads() const -> std::vector<ExpertLoad> {
    std::lock_guard<std::mutex> lock(load_tracking_mutex_);
    
    std::vector<ExpertLoad> loads;
    loads.reserve(expert_loads_.size());
    
    for (const auto& load_ptr : expert_loads_) {
        loads.push_back(*load_ptr);  // Copy the ExpertLoad
    }
    
    return loads;
}

auto LoadBalancer::trigger_expert_rebalancing() -> inference_lab::common::Result<std::monostate, MoEError> {
    std::lock_guard<std::mutex> lock(balancing_mutex_);
    
    update_routing_adjustments();
    last_rebalancing_time_ = std::chrono::steady_clock::now();
    
    return inference_lab::common::Ok(std::monostate{});
}

auto LoadBalancer::update_routing_adjustments() -> void {
    // Adjust routing factors based on current load distribution
    float mean_load = std::accumulate(expert_load_history_.begin(), expert_load_history_.end(), 0.0f) 
                      / static_cast<float>(expert_load_history_.size());
    
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        float load_ratio = (mean_load > 1e-6f) ? (expert_load_history_[i] / mean_load) : 1.0f;
        
        // Adjust routing factor inversely proportional to load
        if (load_ratio > 1.5f) {
            routing_adjustment_factors_[i] *= 0.8f;  // Reduce routing to overloaded experts
        } else if (load_ratio < 0.5f) {
            routing_adjustment_factors_[i] *= 1.2f;  // Increase routing to underutilized experts
        }
        
        // Clamp adjustment factors
        routing_adjustment_factors_[i] = std::clamp(routing_adjustment_factors_[i], 0.1f, 2.0f);
    }
}

auto LoadBalancer::handle_expert_failure(std::size_t failed_expert_id)
    -> inference_lab::common::Result<std::monostate, MoEError> {
    
    if (failed_expert_id >= config_.num_experts) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }
    
    {
        std::lock_guard<std::mutex> lock(load_tracking_mutex_);
        
        // Mark expert as overloaded to prevent new requests
        expert_loads_[failed_expert_id]->is_overloaded = true;
        expert_loads_[failed_expert_id]->current_utilization.store(1.0f);
        
        // Set routing adjustment to minimum to avoid routing to failed expert
        routing_adjustment_factors_[failed_expert_id] = 0.01f;
    }
    
    // Redistribute queued requests
    redistribute_queued_requests(failed_expert_id);
    
    return inference_lab::common::Ok(std::monostate{});
}

auto LoadBalancer::redistribute_queued_requests(std::size_t failed_expert_id) -> void {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    
    auto it = expert_queues_.find(failed_expert_id);
    if (it == expert_queues_.end() || it->second.empty()) {
        return;
    }
    
    // Find alternative experts (simple round-robin redistribution)
    std::vector<std::size_t> available_experts;
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        if (i != failed_expert_id && !is_expert_overloaded(i)) {
            available_experts.push_back(i);
        }
    }
    
    if (available_experts.empty()) {
        // No available experts - reject requests
        std::size_t rejected_count = it->second.size();
        total_requests_rejected_.fetch_add(rejected_count);
        it->second = std::queue<QueuedRequest>();  // Clear the queue
        return;
    }
    
    std::size_t redistribution_index = 0;
    while (!it->second.empty()) {
        auto request = it->second.front();
        it->second.pop();
        
        std::size_t new_expert = available_experts[redistribution_index % available_experts.size()];
        request.expert_id = new_expert;
        
        expert_queues_[new_expert].push(request);
        redistribution_index++;
    }
}

auto LoadBalancer::validate_load_balancing_health() const -> inference_lab::common::Result<std::monostate, MoEError> {
    float variance = overall_utilization_variance_.load();
    
    // Check if utilization variance is reasonable
    if (variance > 0.5f) {  // High variance indicates poor load balancing
        return inference_lab::common::Err(MoEError::LOAD_BALANCING_ERROR);
    }
    
    // Check if too many experts are overloaded
    std::size_t overloaded_count = 0;
    for (const auto& load_ptr : expert_loads_) {
        if (load_ptr->is_overloaded) {
            overloaded_count++;
        }
    }
    
    if (overloaded_count > config_.num_experts / 2) {
        return inference_lab::common::Err(MoEError::LOAD_BALANCING_ERROR);
    }
    
    return inference_lab::common::Ok(std::monostate{});
}

// RequestTracker implementation

RequestTracker::RequestTracker(LoadBalancer& load_balancer, 
                              std::size_t expert_id, 
                              std::size_t request_id)
    : load_balancer_(&load_balancer), expert_id_(expert_id), request_id_(request_id),
      start_time_(std::chrono::steady_clock::now()), completed_(false) {
    
    load_balancer_->register_request_start(expert_id_, request_id_);
}

RequestTracker::~RequestTracker() {
    if (!completed_ && load_balancer_) {
        complete();
    }
}

RequestTracker::RequestTracker(RequestTracker&& other) noexcept
    : load_balancer_(other.load_balancer_), expert_id_(other.expert_id_), 
      request_id_(other.request_id_), start_time_(other.start_time_), 
      completed_(other.completed_) {
    other.load_balancer_ = nullptr;
    other.completed_ = true;
}

RequestTracker& RequestTracker::operator=(RequestTracker&& other) noexcept {
    if (this != &other) {
        if (!completed_ && load_balancer_) {
            complete();
        }
        
        load_balancer_ = other.load_balancer_;
        expert_id_ = other.expert_id_;
        request_id_ = other.request_id_;
        start_time_ = other.start_time_;
        completed_ = other.completed_;
        
        other.load_balancer_ = nullptr;
        other.completed_ = true;
    }
    return *this;
}

auto RequestTracker::complete() -> void {
    if (completed_ || !load_balancer_) {
        return;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
    float processing_time_ms = static_cast<float>(duration.count()) / 1000.0f;
    
    load_balancer_->register_request_completion(expert_id_, request_id_, processing_time_ms);
    completed_ = true;
}

} // namespace engines::mixture_experts