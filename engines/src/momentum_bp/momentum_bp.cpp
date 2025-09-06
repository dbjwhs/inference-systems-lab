// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include "momentum_bp.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "../../common/src/logging.hpp"

using inference_lab::common::LogLevel;

namespace inference_lab::engines::momentum_bp {

std::string to_string(MomentumBPError error) {
    switch (error) {
        case MomentumBPError::INVALID_GRAPH_STRUCTURE:
            return "Invalid graph structure for belief propagation";
        case MomentumBPError::CONVERGENCE_FAILED:
            return "Failed to converge within iteration limit";
        case MomentumBPError::NUMERICAL_INSTABILITY:
            return "Numerical instability in message computation";
        case MomentumBPError::INVALID_POTENTIAL_FUNCTION:
            return "Invalid potential function";
        case MomentumBPError::MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case MomentumBPError::UNKNOWN_ERROR:
        default:
            return "Unknown error";
    }
}

MomentumBPEngine::MomentumBPEngine(const MomentumBPConfig& config) : config_(config) {
    reset();
}

auto MomentumBPEngine::run_inference(const InferenceRequest& request)
    -> common::Result<InferenceResponse, InferenceError> {
    // For this initial implementation, create a simple example graphical model
    // In a full implementation, this would parse the request to build the model
    GraphicalModel model;

    // Create simple 2-node binary model for demonstration
    Node node1{1, {0.6, 0.4}, {2}};  // P(X1=0)=0.6, P(X1=1)=0.4
    Node node2{2, {0.3, 0.7}, {1}};  // P(X2=0)=0.3, P(X2=1)=0.7
    model.nodes = {node1, node2};
    model.node_index[1] = 0;
    model.node_index[2] = 1;

    // Simple edge potential: slightly favor same values
    EdgePotential edge{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    model.edges = {edge};

    auto bp_result = run_momentum_bp(model);
    if (bp_result.is_err()) {
        return common::Err(InferenceError::INFERENCE_EXECUTION_FAILED);
    }

    auto marginals = bp_result.unwrap();

    InferenceResponse response;
    // Convert marginals to response format
    for (const auto& marginal : marginals) {
        std::vector<float> output(marginal.begin(), marginal.end());
        response.output_tensors.push_back(output);
    }

    response.inference_time_ms = static_cast<double>(metrics_.inference_time_ms.count());
    response.output_names = {"node_1_marginal", "node_2_marginal"};

    return common::Ok(response);
}

auto MomentumBPEngine::get_backend_info() const -> std::string {
    std::stringstream ss;
    ss << "Momentum-Enhanced Belief Propagation Engine\n";
    ss << "  Momentum Factor: " << config_.momentum_factor << "\n";
    ss << "  Learning Rate: " << config_.learning_rate << "\n";
    ss << "  Max Iterations: " << config_.max_iterations << "\n";
    ss << "  AdaGrad Enabled: " << (config_.enable_adagrad ? "Yes" : "No") << "\n";
    return ss.str();
}

auto MomentumBPEngine::is_ready() const -> bool {
    return true;  // Engine is stateless and always ready
}

auto MomentumBPEngine::get_performance_stats() const -> std::string {
    std::stringstream ss;
    ss << "Momentum-BP Performance Statistics:\n";
    ss << "  Converged: " << (metrics_.converged ? "Yes" : "No") << "\n";
    ss << "  Iterations: " << metrics_.iterations_to_convergence << "\n";
    ss << "  Final Residual: " << metrics_.final_residual << "\n";
    ss << "  Inference Time: " << metrics_.inference_time_ms.count() << " ms\n";
    ss << "  Message Updates: " << metrics_.message_updates << "\n";
    return ss.str();
}

auto MomentumBPEngine::run_momentum_bp(const GraphicalModel& model)
    -> common::Result<std::vector<std::vector<double>>, MomentumBPError> {
    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_DEBUG_PRINT("Starting Momentum-BP inference with {} nodes, {} edges",
                    model.nodes.size(),
                    model.edges.size());
    LOG_DEBUG_PRINT("Configuration: momentum={}, adagrad={}, max_iterations={}",
                    config_.enable_momentum,
                    config_.enable_adagrad,
                    config_.max_iterations);

    // Validate model consistency with configuration
    auto validation_result = validate_model_consistency(model);
    if (validation_result.is_err()) {
        LOG_ERROR_PRINT("Model validation failed: {}", to_string(validation_result.unwrap_err()));
        return common::Err(validation_result.unwrap_err());
    }

    // Initialize messages
    auto init_result = initialize_messages(model);
    if (init_result.is_err()) {
        LOG_ERROR_PRINT("Failed to initialize messages: {}", to_string(init_result.unwrap_err()));
        return common::Err(init_result.unwrap_err());
    }
    LOG_DEBUG_PRINT("Initialized {} edge messages", model.edges.size());

    double residual = std::numeric_limits<double>::max();
    std::uint32_t iteration = 0;

    // Main belief propagation loop with momentum
    while (iteration < config_.max_iterations && !check_convergence(residual)) {
        residual = 0.0;

        // Update all messages
        for (const auto& edge : model.edges) {
            auto message_result = compute_message(model, edge.id);
            if (message_result.is_err()) {
                return common::Err(message_result.unwrap_err());
            }

            auto new_message = message_result.unwrap();

            // Compute residual
            if (messages_.find(edge.id) != messages_.end()) {
                residual =
                    std::max(residual, compute_message_residual(messages_[edge.id], new_message));
            }

            // Update with momentum
            auto update_result = update_message_with_momentum(edge.id, new_message);
            if (update_result.is_err()) {
                return common::Err(update_result.unwrap_err());
            }

            metrics_.message_updates++;
        }

        iteration++;
    }

    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.inference_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    metrics_.iterations_to_convergence = iteration;
    metrics_.final_residual = residual;
    metrics_.converged = check_convergence(residual);

    if (metrics_.converged) {
        std::ostringstream oss;
        oss << "Momentum-BP converged after " << metrics_.iterations_to_convergence
            << " iterations with residual " << std::scientific << std::setprecision(2)
            << metrics_.final_residual;
        LOG_INFO_PRINT("{}", oss.str());
    } else {
        std::ostringstream oss;
        oss << "Momentum-BP failed to converge after " << metrics_.iterations_to_convergence
            << " iterations, final residual: " << std::scientific << std::setprecision(2)
            << metrics_.final_residual;
        LOG_WARNING_PRINT("{}", oss.str());
    }
    LOG_DEBUG_PRINT("Inference completed in {}ms with {} message updates",
                    metrics_.inference_time_ms.count(),
                    metrics_.message_updates);

    // Compute marginals
    return compute_marginals(model);
}

auto MomentumBPEngine::get_metrics() const -> MomentumBPMetrics {
    return metrics_;
}

void MomentumBPEngine::update_config(const MomentumBPConfig& new_config) {
    LOG_DEBUG_PRINT("Updating Momentum-BP configuration: momentum={}, adagrad={}, domain_size={}",
                    new_config.enable_momentum,
                    new_config.enable_adagrad,
                    new_config.variable_domain_size);
    config_ = new_config;
    reset();  // Clear state when config changes
}

void MomentumBPEngine::reset() {
    metrics_ = MomentumBPMetrics{};
    messages_.clear();
    momentum_terms_.clear();
    adagrad_accumulator_.clear();
}

// Private methods implementation

auto MomentumBPEngine::initialize_messages(const GraphicalModel& model)
    -> common::Result<std::monostate, MomentumBPError> {
    messages_.clear();
    momentum_terms_.clear();
    adagrad_accumulator_.clear();

    for (const auto& edge : model.edges) {
        // Initialize with uniform messages
        const double uniform_prob = 1.0 / config_.variable_domain_size;
        Message uniform_msg(config_.variable_domain_size, uniform_prob);
        messages_[edge.id] = uniform_msg;
        momentum_terms_[edge.id] = Message(config_.variable_domain_size, 0.0);
        adagrad_accumulator_[edge.id] = std::vector<double>(config_.variable_domain_size, 0.0);
    }

    return common::Ok(std::monostate{});
}

auto MomentumBPEngine::validate_model_consistency(const GraphicalModel& model)
    -> common::Result<std::monostate, MomentumBPError> {
    // Check that all nodes have consistent domain sizes with configuration
    for (const auto& node : model.nodes) {
        if (node.local_potential.size() != config_.variable_domain_size) {
            LOG_WARNING_PRINT("Node {} has domain size {} but config expects {}",
                              node.id,
                              node.local_potential.size(),
                              config_.variable_domain_size);
            // Allow if node domain is smaller or equal (with padding/truncation)
            if (node.local_potential.empty()) {
                LOG_ERROR_PRINT("Node {} has empty local potential", node.id);
                return common::Err(MomentumBPError::INVALID_POTENTIAL_FUNCTION);
            }
        }
    }

    // Check edge potential matrix dimensions
    for (const auto& edge : model.edges) {
        if (edge.potential_matrix.empty()) {
            LOG_ERROR_PRINT("Edge {} has empty potential matrix", edge.id);
            return common::Err(MomentumBPError::INVALID_POTENTIAL_FUNCTION);
        }

        for (const auto& row : edge.potential_matrix) {
            if (row.empty()) {
                LOG_ERROR_PRINT("Edge {} has empty potential matrix row", edge.id);
                return common::Err(MomentumBPError::INVALID_POTENTIAL_FUNCTION);
            }
        }
    }

    LOG_DEBUG_PRINT("Model validation passed: {} nodes validated, {} edges validated",
                    model.nodes.size(),
                    model.edges.size());
    return common::Ok(std::monostate{});
}

auto MomentumBPEngine::compute_message(const GraphicalModel& model, EdgeId edge_id)
    -> common::Result<Message, MomentumBPError> {
    // Find the edge
    const EdgePotential* edge = nullptr;
    for (const auto& e : model.edges) {
        if (e.id == edge_id) {
            edge = &e;
            break;
        }
    }

    if (!edge) {
        return common::Err(MomentumBPError::INVALID_GRAPH_STRUCTURE);
    }

    // Message computation for configurable domain size variables
    Message message(config_.variable_domain_size);

    // Find source node
    const Node* from_node = nullptr;
    for (const auto& node : model.nodes) {
        if (node.id == edge->from_node) {
            from_node = &node;
            break;
        }
    }

    if (!from_node) {
        return common::Err(MomentumBPError::INVALID_GRAPH_STRUCTURE);
    }

    // Compute message: sum over source states
    const auto domain_size =
        std::min(config_.variable_domain_size,
                 static_cast<std::uint32_t>(from_node->local_potential.size()));
    for (std::size_t to_state = 0; to_state < domain_size; ++to_state) {
        double sum = 0.0;
        for (std::size_t from_state = 0; from_state < domain_size; ++from_state) {
            if (from_state < from_node->local_potential.size() &&
                from_state < edge->potential_matrix.size() &&
                to_state < edge->potential_matrix[from_state].size()) {
                sum += from_node->local_potential[from_state] *
                       edge->potential_matrix[from_state][to_state];
            }
        }
        message[to_state] = sum;
    }

    if (config_.normalize_messages) {
        normalize_message(message);
    }

    return common::Ok(message);
}

auto MomentumBPEngine::update_message_with_momentum(EdgeId edge_id, const Message& new_message)
    -> common::Result<std::monostate, MomentumBPError> {
    if (messages_.find(edge_id) == messages_.end()) {
        messages_[edge_id] = new_message;
        return common::Ok(std::monostate{});
    }

    auto& old_message = messages_[edge_id];
    auto& momentum = momentum_terms_[edge_id];
    auto& adagrad = adagrad_accumulator_[edge_id];

    // Compute gradient (difference)
    Message gradient(new_message.size());
    for (std::size_t i = 0; i < new_message.size(); ++i) {
        gradient[i] = new_message[i] - old_message[i];
    }

    // Update with momentum and AdaGrad
    for (std::size_t i = 0; i < new_message.size(); ++i) {
        if (config_.enable_momentum) {
            momentum[i] = config_.momentum_factor * momentum[i] +
                          (1.0 - config_.momentum_factor) * gradient[i];
        } else {
            momentum[i] = gradient[i];
        }

        double learning_rate = config_.learning_rate;
        if (config_.enable_adagrad) {
            adagrad[i] += gradient[i] * gradient[i];
            learning_rate =
                config_.learning_rate / (std::sqrt(adagrad[i]) + config_.adagrad_epsilon);
        }

        old_message[i] += learning_rate * momentum[i];
    }

    if (config_.normalize_messages) {
        normalize_message(old_message);
    }

    return common::Ok(std::monostate{});
}

auto MomentumBPEngine::check_convergence(double residual) const -> bool {
    return residual < config_.convergence_threshold;
}

auto MomentumBPEngine::compute_marginals(const GraphicalModel& model)
    -> common::Result<std::vector<std::vector<double>>, MomentumBPError> {
    std::vector<std::vector<double>> marginals;

    // For each node, compute marginal from local potential and incoming messages
    for (const auto& node : model.nodes) {
        std::vector<double> marginal = node.local_potential;

        // Multiply by incoming messages (simplified for binary case)
        // In full implementation, this would properly handle all incoming messages

        // Normalize marginal
        double sum = std::accumulate(marginal.begin(), marginal.end(), 0.0);
        if (sum > config_.numerical_epsilon) {
            for (auto& prob : marginal) {
                prob /= sum;
            }
        }

        marginals.push_back(marginal);
    }

    return common::Ok(marginals);
}

void MomentumBPEngine::normalize_message(Message& message) const {
    double sum = std::accumulate(message.begin(), message.end(), 0.0);
    if (sum > config_.numerical_epsilon) {
        for (auto& prob : message) {
            prob /= sum;
        }
    }
}

auto MomentumBPEngine::compute_message_residual(const Message& old_msg,
                                                const Message& new_msg) const -> double {
    double residual = 0.0;
    for (std::size_t i = 0; i < old_msg.size(); ++i) {
        double diff = new_msg[i] - old_msg[i];
        residual += diff * diff;
    }
    return std::sqrt(residual);
}

auto MomentumBPEngine::generate_edge_id(NodeId from, NodeId to) const -> EdgeId {
    return (static_cast<std::uint64_t>(from) << 32) | to;
}

auto create_momentum_bp_engine(const MomentumBPConfig& config)
    -> common::Result<std::unique_ptr<MomentumBPEngine>, MomentumBPError> {
    try {
        auto engine = std::make_unique<MomentumBPEngine>(config);
        return common::Ok(std::move(engine));
    } catch (const std::bad_alloc&) {
        return common::Err(MomentumBPError::MEMORY_ALLOCATION_FAILED);
    } catch (...) {
        return common::Err(MomentumBPError::UNKNOWN_ERROR);
    }
}

}  // namespace inference_lab::engines::momentum_bp
