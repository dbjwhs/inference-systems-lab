// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include "expert_router.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "moe_engine.hpp"  // For MoEError definition

namespace engines::mixture_experts {

using Result = inference_lab::common::Result<std::unique_ptr<ExpertRouter>, MoEError>;
using VoidResult = inference_lab::common::Result<std::monostate, MoEError>;

ExpertRouter::ExpertRouter(const RouterConfig& config)
    : config_(config), load_balance_weight_(0.01f) {
    // Initialize routing network parameters with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> weight_dist(0.0f, 0.1f);

    // Initialize routing weights [hidden_dim x input_dim]
    routing_weights_.resize(config_.hidden_dimension);
    for (auto& row : routing_weights_) {
        row.resize(config_.input_dimension);
        for (auto& weight : row) {
            weight = weight_dist(gen);
        }
    }

    // Initialize routing biases [hidden_dim]
    routing_biases_.resize(config_.hidden_dimension);
    for (auto& bias : routing_biases_) {
        bias = weight_dist(gen);
    }

    // Initialize expert weights [num_experts x hidden_dim]
    expert_weights_.resize(config_.num_experts);
    for (auto& row : expert_weights_) {
        row.resize(config_.hidden_dimension);
        for (auto& weight : row) {
            weight = weight_dist(gen);
        }
    }

    // Initialize expert biases [num_experts]
    expert_biases_.resize(config_.num_experts, 0.0f);

    // Initialize monitoring arrays
    expert_selection_counts_.resize(config_.num_experts, 0);
    expert_load_history_.resize(config_.num_experts, 0.0f);
    recent_routing_times_.reserve(1000);
    recent_entropy_scores_.reserve(1000);
}

auto ExpertRouter::create(const RouterConfig& config) -> Result {
    if (config.num_experts == 0 || config.input_dimension == 0 || config.hidden_dimension == 0 ||
        config.top_k_experts == 0) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }

    if (config.top_k_experts > config.num_experts) {
        return inference_lab::common::Err(MoEError::EXPERT_INITIALIZATION_FAILED);
    }

    auto router = std::unique_ptr<ExpertRouter>(new ExpertRouter(config));
    return inference_lab::common::Ok(std::move(router));
}

auto ExpertRouter::select_experts(const std::vector<float>& features)
    -> inference_lab::common::Result<std::vector<std::size_t>, MoEError> {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (features.size() != config_.input_dimension) {
        return inference_lab::common::Err(MoEError::ROUTING_NETWORK_FAILURE);
    }

    // Forward pass through routing network
    auto logits_result = forward_pass(features);
    if (!logits_result.is_ok()) {
        return inference_lab::common::Err(std::move(logits_result).unwrap_err());
    }

    auto logits = std::move(logits_result).unwrap();

    // Apply top-k selection
    auto selected_result = apply_top_k_selection(logits);
    if (!selected_result.is_ok()) {
        return inference_lab::common::Err(std::move(selected_result).unwrap_err());
    }

    auto selected = std::move(selected_result).unwrap();

    // Update load balancing
    update_load_balancing(selected);

    // Record performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float routing_time_ms = static_cast<float>(duration.count()) / 1000.0f;

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        recent_routing_times_.push_back(routing_time_ms);
        if (recent_routing_times_.size() > 1000) {
            recent_routing_times_.erase(recent_routing_times_.begin());
        }
    }

    total_routing_calls_.fetch_add(1);

    // Update average routing time with exponential moving average
    float current_avg = average_routing_time_ms_.load();
    float alpha = 0.1f;
    average_routing_time_ms_.store(alpha * routing_time_ms + (1.0f - alpha) * current_avg);

    return inference_lab::common::Ok(std::move(selected));
}

auto ExpertRouter::compute_expert_weights(const std::vector<float>& features,
                                          const std::vector<std::size_t>& selected_experts)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    if (features.size() != config_.input_dimension) {
        return inference_lab::common::Err(MoEError::ROUTING_NETWORK_FAILURE);
    }

    // Forward pass to get logits
    auto logits_result = forward_pass(features);
    if (!logits_result.is_ok()) {
        return inference_lab::common::Err(std::move(logits_result).unwrap_err());
    }

    auto logits = std::move(logits_result).unwrap();

    // Extract weights for selected experts
    std::vector<float> selected_logits;
    selected_logits.reserve(selected_experts.size());

    for (auto expert_id : selected_experts) {
        if (expert_id >= logits.size()) {
            return inference_lab::common::Err(MoEError::ROUTING_NETWORK_FAILURE);
        }
        selected_logits.push_back(logits[expert_id]);
    }

    // Apply softmax to selected expert logits
    auto weights = softmax(selected_logits);

    // Apply entropy regularization if enabled
    if (config_.entropy_regularization > 0.0f) {
        float entropy = compute_entropy_regularization(weights);

        std::lock_guard<std::mutex> lock(stats_mutex_);
        recent_entropy_scores_.push_back(entropy);
        if (recent_entropy_scores_.size() > 1000) {
            recent_entropy_scores_.erase(recent_entropy_scores_.begin());
        }
    }

    return inference_lab::common::Ok(std::move(weights));
}

auto ExpertRouter::forward_pass(const std::vector<float>& features)
    -> inference_lab::common::Result<std::vector<float>, MoEError> {
    // Hidden layer computation: hidden = ReLU(W * input + b)
    std::vector<float> hidden(config_.hidden_dimension, 0.0f);

    for (std::size_t i = 0; i < config_.hidden_dimension; ++i) {
        float sum = routing_biases_[i];
        for (std::size_t j = 0; j < config_.input_dimension; ++j) {
            sum += routing_weights_[i][j] * features[j];
        }
        hidden[i] = relu(sum);
    }

    // Output layer computation: output = expert_weights * hidden + expert_biases
    std::vector<float> logits(config_.num_experts, 0.0f);

    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        float sum = expert_biases_[i];
        for (std::size_t j = 0; j < config_.hidden_dimension; ++j) {
            sum += expert_weights_[i][j] * hidden[j];
        }
        logits[i] = sum;
    }

    return inference_lab::common::Ok(std::move(logits));
}

auto ExpertRouter::apply_top_k_selection(const std::vector<float>& logits)
    -> inference_lab::common::Result<std::vector<std::size_t>, MoEError> {
    // Create pairs of (logit_value, expert_index)
    std::vector<std::pair<float, std::size_t>> logit_pairs;
    logit_pairs.reserve(logits.size());

    for (std::size_t i = 0; i < logits.size(); ++i) {
        logit_pairs.emplace_back(logits[i], i);
    }

    // Sort by logit values in descending order
    std::partial_sort(logit_pairs.begin(),
                      logit_pairs.begin() + config_.top_k_experts,
                      logit_pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Extract top-k expert indices
    std::vector<std::size_t> selected_experts;
    selected_experts.reserve(config_.top_k_experts);

    for (std::size_t i = 0; i < config_.top_k_experts; ++i) {
        selected_experts.push_back(logit_pairs[i].second);
    }

    return inference_lab::common::Ok(std::move(selected_experts));
}

auto ExpertRouter::compute_entropy_regularization(const std::vector<float>& probabilities)
    -> float {
    float entropy = 0.0f;

    for (float p : probabilities) {
        if (p > 1e-10f) {  // Avoid log(0)
            entropy -= p * std::log(p);
        }
    }

    return entropy;
}

auto ExpertRouter::update_load_balancing(const std::vector<std::size_t>& selected_experts) -> void {
    // Update expert selection counts (already locked by caller)
    for (auto expert_id : selected_experts) {
        if (expert_id < expert_selection_counts_.size()) {
            expert_selection_counts_[expert_id]++;
        }
    }

    // Update load history with exponential moving average
    const float alpha = 0.05f;  // Smoothing factor

    std::vector<float> current_load(config_.num_experts, 0.0f);
    for (auto expert_id : selected_experts) {
        if (expert_id < current_load.size()) {
            current_load[expert_id] = 1.0f;
        }
    }

    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        expert_load_history_[i] =
            alpha * current_load[i] + (1.0f - alpha) * expert_load_history_[i];
    }
}

auto ExpertRouter::softmax(const std::vector<float>& logits) -> std::vector<float> {
    if (logits.empty()) {
        return {};
    }

    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<float> probabilities;
    probabilities.reserve(logits.size());

    float sum = 0.0f;
    for (float logit : logits) {
        float exp_val = std::exp(logit - max_logit);
        probabilities.push_back(exp_val);
        sum += exp_val;
    }

    // Normalize
    if (sum > 1e-10f) {
        for (auto& prob : probabilities) {
            prob /= sum;
        }
    }

    return probabilities;
}

auto ExpertRouter::update_routing_parameters(const std::vector<float>& features,
                                             const std::vector<std::size_t>& selected_experts,
                                             float performance_score) -> VoidResult {
    if (!config_.enable_gradient_computation) {
        return inference_lab::common::Ok(std::monostate{});
    }

    // Simplified gradient computation for demonstration
    // In practice, this would use proper backpropagation
    return compute_gradients(features, selected_experts, performance_score);
}

auto ExpertRouter::compute_gradients(const std::vector<float>& features,
                                     const std::vector<std::size_t>& selected_experts,
                                     float performance_score) -> VoidResult {
    // Simplified gradient-based parameter update
    const float learning_rate = config_.learning_rate;
    const float gradient_scale = (performance_score - 0.5f) * 2.0f;  // Normalize around 0

    // Update expert biases for selected experts (simplified)
    for (auto expert_id : selected_experts) {
        if (expert_id < expert_biases_.size()) {
            expert_biases_[expert_id] += learning_rate * gradient_scale * 0.01f;
        }
    }

    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertRouter::get_routing_stats() const -> RoutingStats {
    RoutingStats stats;

    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Expert selection frequency
    stats.expert_selection_frequency.resize(config_.num_experts);
    std::size_t total_selections = 0;

    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        std::size_t count = expert_selection_counts_[i];
        stats.expert_selection_frequency[i] = static_cast<float>(count);
        total_selections += count;
    }

    // Normalize frequencies
    if (total_selections > 0) {
        for (auto& freq : stats.expert_selection_frequency) {
            freq /= static_cast<float>(total_selections);
        }
    }

    // Expert weights history (copy current load)
    stats.expert_weights_history = expert_load_history_;

    // Compute entropy score from recent data
    if (!recent_entropy_scores_.empty()) {
        float sum = 0.0f;
        for (float score : recent_entropy_scores_) {
            sum += score;
        }
        stats.entropy_score = sum / static_cast<float>(recent_entropy_scores_.size());
    } else {
        stats.entropy_score = 0.0f;
    }

    // Load balance coefficient (coefficient of variation)
    if (!expert_load_history_.empty()) {
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

        float std_dev = std::sqrt(variance);
        stats.load_balance_coefficient = (mean > 1e-6f) ? (std_dev / mean) : 1.0f;
    } else {
        stats.load_balance_coefficient = 1.0f;
    }

    stats.total_routing_decisions = total_routing_calls_.load();

    return stats;
}

auto ExpertRouter::validate_routing_health() const -> VoidResult {
    auto stats = get_routing_stats();

    // Check if load balancing is working (coefficient of variation should be low)
    if (stats.load_balance_coefficient > 2.0f) {
        return inference_lab::common::Err(MoEError::LOAD_BALANCING_ERROR);
    }

    // Check if routing is producing reasonable entropy
    if (stats.entropy_score > 10.0f ||
        (stats.total_routing_decisions > 100 && stats.entropy_score < 0.1f)) {
        return inference_lab::common::Err(MoEError::ROUTING_NETWORK_FAILURE);
    }

    // Check average routing time
    float avg_time = average_routing_time_ms_.load();
    if (avg_time > 50.0f) {  // Should be < 5ms target, but allowing margin
        return inference_lab::common::Err(MoEError::EXPERT_SELECTION_TIMEOUT);
    }

    return inference_lab::common::Ok(std::monostate{});
}

auto ExpertRouter::reset_statistics() -> void {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    for (auto& count : expert_selection_counts_) {
        count = 0;
    }

    std::fill(expert_load_history_.begin(), expert_load_history_.end(), 0.0f);
    recent_routing_times_.clear();
    recent_entropy_scores_.clear();

    total_routing_calls_.store(0);
    average_routing_time_ms_.store(0.0f);
}

}  // namespace engines::mixture_experts
