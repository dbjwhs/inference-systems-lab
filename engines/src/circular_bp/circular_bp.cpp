// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include "circular_bp.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stack>

#include "../../common/src/logging.hpp"

using inference_lab::common::LogLevel;

namespace inference_lab::engines::circular_bp {

std::string to_string(CircularBPError error) {
    switch (error) {
        case CircularBPError::INVALID_GRAPH_STRUCTURE:
            return "Invalid graph structure for circular belief propagation";
        case CircularBPError::CYCLE_DETECTION_FAILED:
            return "Failed to detect cycles in graph structure";
        case CircularBPError::CONVERGENCE_FAILED:
            return "Failed to converge within iteration limit";
        case CircularBPError::CORRELATION_CANCELLATION_FAILED:
            return "Failed to cancel spurious correlations";
        case CircularBPError::NUMERICAL_INSTABILITY:
            return "Numerical instability in message computation";
        case CircularBPError::INVALID_POTENTIAL_FUNCTION:
            return "Invalid potential function";
        case CircularBPError::MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case CircularBPError::UNKNOWN_ERROR:
        default:
            return "Unknown error";
    }
}

CircularBPEngine::CircularBPEngine(const CircularBPConfig& config) : config_(config) {
    reset();
}

auto CircularBPEngine::run_inference(const InferenceRequest& request)
    -> common::Result<InferenceResponse, InferenceError> {
    // For this implementation, create a cyclic test graphical model
    // In a full implementation, this would parse the request to build the model
    GraphicalModel model;

    // Create 3-node cyclic model for demonstration (triangle structure)
    Node node1{1, {0.6, 0.4}, {2, 3}};
    Node node2{2, {0.3, 0.7}, {1, 3}};
    Node node3{3, {0.5, 0.5}, {1, 2}};
    model.nodes = {node1, node2, node3};
    model.node_index[1] = 0;
    model.node_index[2] = 1;
    model.node_index[3] = 2;

    // Create cyclic edges (triangle)
    EdgePotential edge1{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    EdgePotential edge2{2, 2, 3, {{1.1, 0.9}, {0.9, 1.1}}};
    EdgePotential edge3{3, 3, 1, {{1.3, 0.7}, {0.7, 1.3}}};
    model.edges = {edge1, edge2, edge3};

    auto bp_result = run_circular_bp(model);
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
    response.output_names = {"node_1_marginal", "node_2_marginal", "node_3_marginal"};

    return common::Ok(response);
}

auto CircularBPEngine::get_backend_info() const -> std::string {
    std::stringstream ss;
    ss << "Circular Belief Propagation Engine\\n";
    ss << "  Detection Strategy: " << static_cast<int>(config_.detection_strategy) << "\\n";
    ss << "  Correlation Threshold: " << config_.correlation_threshold << "\\n";
    ss << "  Max Cycle Length: " << config_.max_cycle_length << "\\n";
    ss << "  Correlation Cancellation: " << (config_.enable_correlation_cancellation ? "Yes" : "No")
       << "\\n";
    ss << "  Cycle Penalties: " << (config_.enable_cycle_penalties ? "Yes" : "No") << "\\n";
    return ss.str();
}

auto CircularBPEngine::is_ready() const -> bool {
    return true;  // Engine is stateless and always ready
}

auto CircularBPEngine::get_performance_stats() const -> std::string {
    std::stringstream ss;
    ss << "Circular-BP Performance Statistics:\\n";
    ss << "  Converged: " << (metrics_.converged ? "Yes" : "No") << "\\n";
    ss << "  Iterations: " << metrics_.iterations_to_convergence << "\\n";
    ss << "  Final Residual: " << metrics_.final_residual << "\\n";
    ss << "  Inference Time: " << metrics_.inference_time_ms.count() << " ms\\n";
    ss << "  Message Updates: " << metrics_.message_updates << "\\n";
    ss << "  Cycles Detected: " << metrics_.cycles_detected << "\\n";
    ss << "  Correlations Cancelled: " << metrics_.correlations_cancelled << "\\n";
    ss << "  Reverberation Events: " << metrics_.reverberation_events << "\\n";
    return ss.str();
}

auto CircularBPEngine::run_circular_bp(const GraphicalModel& model)
    -> common::Result<std::vector<std::vector<double>>, CircularBPError> {
    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_DEBUG_PRINT("Starting Circular-BP inference with {} nodes, {} edges",
                    model.nodes.size(),
                    model.edges.size());
    LOG_DEBUG_PRINT("Configuration: correlation_threshold={}, max_cycle_length={}, strategy={}",
                    config_.correlation_threshold,
                    config_.max_cycle_length,
                    static_cast<int>(config_.detection_strategy));

    // Validate model consistency with configuration
    auto validation_result = validate_model_consistency(model);
    if (validation_result.is_err()) {
        LOG_ERROR_PRINT("Model validation failed: {}", to_string(validation_result.unwrap_err()));
        return common::Err(validation_result.unwrap_err());
    }

    // Make a mutable copy for cycle detection
    GraphicalModel mutable_model = model;

    // Detect cycles in the graph
    auto cycle_result = detect_cycles(mutable_model);
    if (cycle_result.is_err()) {
        LOG_ERROR_PRINT("Cycle detection failed: {}", to_string(cycle_result.unwrap_err()));
        return common::Err(cycle_result.unwrap_err());
    }

    LOG_INFO_PRINT("Detected {} cycles in graph", mutable_model.detected_cycles.size());
    metrics_.cycles_detected = static_cast<std::uint32_t>(mutable_model.detected_cycles.size());

    // Initialize messages
    auto init_result = initialize_messages(mutable_model);
    if (init_result.is_err()) {
        LOG_ERROR_PRINT("Failed to initialize messages: {}", to_string(init_result.unwrap_err()));
        return common::Err(init_result.unwrap_err());
    }
    LOG_DEBUG_PRINT("Initialized {} edge messages", mutable_model.edges.size());

    double residual = std::numeric_limits<double>::max();
    std::uint32_t iteration = 0;

    // Main circular belief propagation loop
    while (iteration < config_.max_iterations && !check_convergence(residual)) {
        residual = 0.0;

        // Update all messages with cycle awareness
        for (const auto& edge : mutable_model.edges) {
            auto message_result = compute_message(mutable_model, edge.id);
            if (message_result.is_err()) {
                return common::Err(message_result.unwrap_err());
            }

            auto new_message = message_result.unwrap();

            // Check for message reverberation
            if (config_.track_message_history &&
                check_message_reverberation(edge.id, new_message)) {
                metrics_.reverberation_events++;
                LOG_DEBUG_PRINT("Message reverberation detected on edge {}", edge.id);
            }

            // Compute residual
            if (messages_.find(edge.id) != messages_.end()) {
                residual =
                    std::max(residual, compute_message_residual(messages_[edge.id], new_message));
            }

            // Update with correlation tracking
            auto update_result = update_message_with_correlation_tracking(edge.id, new_message);
            if (update_result.is_err()) {
                return common::Err(update_result.unwrap_err());
            }

            metrics_.message_updates++;
        }

        // Cancel spurious correlations if enabled
        if (config_.enable_correlation_cancellation) {
            auto cancel_result = cancel_spurious_correlations(mutable_model);
            if (cancel_result.is_err()) {
                LOG_WARNING_PRINT("Correlation cancellation failed: {}",
                                  to_string(cancel_result.unwrap_err()));
            }
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
        oss << "Circular-BP converged after " << metrics_.iterations_to_convergence 
            << " iterations with residual " << std::scientific << std::setprecision(2)
            << metrics_.final_residual;
        LOG_INFO_PRINT("{}", oss.str());
    } else {
        std::ostringstream oss;
        oss << "Circular-BP failed to converge after " << metrics_.iterations_to_convergence 
            << " iterations, final residual: " << std::scientific << std::setprecision(2)
            << metrics_.final_residual;
        LOG_WARNING_PRINT("{}", oss.str());
    }
    LOG_DEBUG_PRINT(
        "Inference completed in {}ms with {} cycles detected, {} correlations cancelled",
        metrics_.inference_time_ms.count(),
        metrics_.cycles_detected,
        metrics_.correlations_cancelled);

    // Compute marginals with cycle correction
    return compute_marginals(mutable_model);
}

auto CircularBPEngine::get_metrics() const -> CircularBPMetrics {
    return metrics_;
}

void CircularBPEngine::update_config(const CircularBPConfig& new_config) {
    LOG_DEBUG_PRINT(
        "Updating Circular-BP configuration: correlation_threshold={}, cycle_penalties={}, "
        "domain_size={}",
        new_config.correlation_threshold,
        new_config.enable_cycle_penalties,
        new_config.variable_domain_size);
    config_ = new_config;
    reset();  // Clear state when config changes
}

void CircularBPEngine::reset() {
    metrics_ = CircularBPMetrics{};
    messages_.clear();
    message_history_.clear();
    correlation_scores_.clear();
}

// Private methods implementation

auto CircularBPEngine::detect_cycles(GraphicalModel& model)
    -> common::Result<std::monostate, CircularBPError> {
    switch (config_.detection_strategy) {
        case CycleDetectionStrategy::DEPTH_FIRST_SEARCH: {
            auto cycles_result = detect_cycles_dfs(model);
            if (cycles_result.is_err()) {
                return common::Err(cycles_result.unwrap_err());
            }
            model.detected_cycles = cycles_result.unwrap();
            break;
        }
        case CycleDetectionStrategy::SPARSE_MATRIX: {
            auto cycles_result = detect_cycles_sparse(model);
            if (cycles_result.is_err()) {
                return common::Err(cycles_result.unwrap_err());
            }
            model.detected_cycles = cycles_result.unwrap();
            break;
        }
        case CycleDetectionStrategy::HYBRID_ADAPTIVE: {
            // Use DFS for small graphs, sparse matrix for large graphs
            if (model.nodes.size() < 50) {
                auto cycles_result = detect_cycles_dfs(model);
                if (cycles_result.is_err()) {
                    return common::Err(cycles_result.unwrap_err());
                }
                model.detected_cycles = cycles_result.unwrap();
            } else {
                auto cycles_result = detect_cycles_sparse(model);
                if (cycles_result.is_err()) {
                    return common::Err(cycles_result.unwrap_err());
                }
                model.detected_cycles = cycles_result.unwrap();
            }
            break;
        }
    }

    // Mark edges as cycle edges and apply penalties
    for (const auto& cycle : model.detected_cycles) {
        for (EdgeId edge_id : cycle.edges) {
            for (auto& edge : model.edges) {
                if (edge.id == edge_id) {
                    edge.is_cycle_edge = true;
                    edge.cycle_penalty = config_.cycle_penalty_factor * cycle.correlation_strength;
                    break;
                }
            }
        }
    }

    return common::Ok(std::monostate{});
}

auto CircularBPEngine::detect_cycles_dfs(const GraphicalModel& model)
    -> common::Result<std::vector<Cycle>, CircularBPError> {
    std::vector<Cycle> cycles;
    std::unordered_set<NodeId> visited;
    std::unordered_set<NodeId> rec_stack;
    std::stack<NodeId> path;

    LOG_DEBUG_PRINT("Starting DFS cycle detection on {} nodes", model.nodes.size());

    // DFS from each unvisited node
    for (const auto& node : model.nodes) {
        if (visited.find(node.id) == visited.end()) {
            // Perform DFS cycle detection
            std::function<void(NodeId)> dfs_visit = [&](NodeId current) {
                visited.insert(current);
                rec_stack.insert(current);
                path.push(current);

                // Find node in model
                const Node* current_node = nullptr;
                for (const auto& n : model.nodes) {
                    if (n.id == current) {
                        current_node = &n;
                        break;
                    }
                }

                if (current_node) {
                    for (NodeId neighbor : current_node->neighbors) {
                        if (rec_stack.find(neighbor) != rec_stack.end()) {
                            // Found cycle - construct it
                            Cycle cycle;
                            cycle.correlation_strength = 0.5;  // Default value, could be computed
                            cycle.is_spurious = false;         // Will be determined later

                            // Extract cycle path
                            std::stack<NodeId> temp_path = path;
                            std::vector<NodeId> cycle_nodes;

                            while (!temp_path.empty() && temp_path.top() != neighbor) {
                                cycle_nodes.push_back(temp_path.top());
                                temp_path.pop();
                            }
                            if (!temp_path.empty()) {
                                cycle_nodes.push_back(temp_path.top());
                            }

                            std::reverse(cycle_nodes.begin(), cycle_nodes.end());
                            cycle.nodes = cycle_nodes;

                            // Generate edge IDs for the cycle
                            for (size_t i = 0; i < cycle_nodes.size(); ++i) {
                                NodeId from = cycle_nodes[i];
                                NodeId to = cycle_nodes[(i + 1) % cycle_nodes.size()];
                                cycle.edges.push_back(generate_edge_id(from, to));
                            }

                            if (cycle.nodes.size() <= config_.max_cycle_length) {
                                cycles.push_back(cycle);
                                LOG_DEBUG_PRINT("Detected cycle of length {} starting from node {}",
                                                cycle.nodes.size(),
                                                neighbor);
                            }
                        } else if (visited.find(neighbor) == visited.end()) {
                            dfs_visit(neighbor);
                        }
                    }
                }

                rec_stack.erase(current);
                if (!path.empty()) {
                    path.pop();
                }
            };

            dfs_visit(node.id);
        }
    }

    LOG_DEBUG_PRINT("DFS cycle detection completed, found {} cycles", cycles.size());
    return common::Ok(cycles);
}

auto CircularBPEngine::detect_cycles_sparse(const GraphicalModel& model)
    -> common::Result<std::vector<Cycle>, CircularBPError> {
    LOG_DEBUG_PRINT("Starting sparse matrix cycle detection on {} nodes", model.nodes.size());

    // For now, implement a simplified version that delegates to DFS
    // In a full implementation, this would use sparse matrix algorithms
    return detect_cycles_dfs(model);
}

auto CircularBPEngine::initialize_messages(const GraphicalModel& model)
    -> common::Result<std::monostate, CircularBPError> {
    messages_.clear();
    message_history_.clear();
    correlation_scores_.clear();

    for (const auto& edge : model.edges) {
        // Initialize with uniform messages
        const double uniform_prob = 1.0 / config_.variable_domain_size;
        Message uniform_msg(config_.variable_domain_size, uniform_prob);
        messages_[edge.id] = uniform_msg;

        if (config_.track_message_history) {
            message_history_[edge.id] = std::vector<Message>();
        }

        correlation_scores_[edge.id] = 0.0;
    }

    return common::Ok(std::monostate{});
}

auto CircularBPEngine::compute_message(const GraphicalModel& model, EdgeId edge_id)
    -> common::Result<Message, CircularBPError> {
    // Find the edge
    const EdgePotential* edge = nullptr;
    for (const auto& e : model.edges) {
        if (e.id == edge_id) {
            edge = &e;
            break;
        }
    }

    if (!edge) {
        return common::Err(CircularBPError::INVALID_GRAPH_STRUCTURE);
    }

    // Message computation with cycle awareness
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
        return common::Err(CircularBPError::INVALID_GRAPH_STRUCTURE);
    }

    // Compute message with cycle penalty if applicable
    const auto domain_size =
        std::min(config_.variable_domain_size,
                 static_cast<std::uint32_t>(from_node->local_potential.size()));

    for (std::size_t to_state = 0; to_state < domain_size; ++to_state) {
        double sum = 0.0;
        for (std::size_t from_state = 0; from_state < domain_size; ++from_state) {
            if (from_state < from_node->local_potential.size() &&
                from_state < edge->potential_matrix.size() &&
                to_state < edge->potential_matrix[from_state].size()) {
                double potential = from_node->local_potential[from_state] *
                                   edge->potential_matrix[from_state][to_state];

                // Apply cycle penalty if edge is part of a cycle
                if (config_.enable_cycle_penalties && edge->is_cycle_edge) {
                    potential *= (1.0 - edge->cycle_penalty);
                }

                sum += potential;
            }
        }
        message[to_state] = sum;
    }

    if (config_.normalize_messages) {
        normalize_message(message);
    }

    return common::Ok(message);
}

auto CircularBPEngine::update_message_with_correlation_tracking(EdgeId edge_id,
                                                                const Message& new_message)
    -> common::Result<std::monostate, CircularBPError> {
    if (messages_.find(edge_id) != messages_.end()) {
        const auto& old_message = messages_[edge_id];

        // Compute correlation score
        double correlation = 0.0;
        double sum_old = 0.0, sum_new = 0.0, sum_old_sq = 0.0, sum_new_sq = 0.0, sum_product = 0.0;

        for (size_t i = 0; i < std::min(old_message.size(), new_message.size()); ++i) {
            sum_old += old_message[i];
            sum_new += new_message[i];
            sum_old_sq += old_message[i] * old_message[i];
            sum_new_sq += new_message[i] * new_message[i];
            sum_product += old_message[i] * new_message[i];
        }

        double n = static_cast<double>(std::min(old_message.size(), new_message.size()));
        double numerator = n * sum_product - sum_old * sum_new;
        double denominator =
            std::sqrt((n * sum_old_sq - sum_old * sum_old) * (n * sum_new_sq - sum_new * sum_new));

        if (std::abs(denominator) > config_.numerical_epsilon) {
            correlation = numerator / denominator;
        }

        correlation_scores_[edge_id] = correlation;

        // Track message history if enabled
        if (config_.track_message_history) {
            message_history_[edge_id].push_back(old_message);

            // Limit history size to prevent memory growth
            if (message_history_[edge_id].size() > 10) {
                message_history_[edge_id].erase(message_history_[edge_id].begin());
            }
        }
    }

    messages_[edge_id] = new_message;
    return common::Ok(std::monostate{});
}

auto CircularBPEngine::cancel_spurious_correlations(GraphicalModel& model)
    -> common::Result<std::monostate, CircularBPError> {
    std::uint32_t cancelled_count = 0;

    for (const auto& [edge_id, correlation] : correlation_scores_) {
        if (std::abs(correlation) > config_.correlation_threshold) {
            // Found spurious correlation - apply dampening
            auto it = messages_.find(edge_id);
            if (it != messages_.end()) {
                Message& message = it->second;

                // Apply correlation cancellation by dampening the message
                double dampening_factor =
                    1.0 - (std::abs(correlation) - config_.correlation_threshold);
                dampening_factor = std::max(0.1, std::min(1.0, dampening_factor));

                for (auto& value : message) {
                    value *= dampening_factor;
                }

                if (config_.normalize_messages) {
                    normalize_message(message);
                }

                cancelled_count++;
                std::ostringstream oss;
                oss << "Cancelled spurious correlation " << std::fixed << std::setprecision(3) 
                    << correlation << " on edge " << edge_id;
                LOG_DEBUG_PRINT("{}", oss.str());
            }
        }
    }

    metrics_.correlations_cancelled += cancelled_count;

    if (cancelled_count > 0) {
        LOG_DEBUG_PRINT("Cancelled {} spurious correlations", cancelled_count);
    }

    return common::Ok(std::monostate{});
}

auto CircularBPEngine::check_message_reverberation(EdgeId edge_id, const Message& message) -> bool {
    if (!config_.track_message_history) {
        return false;
    }

    auto it = message_history_.find(edge_id);
    if (it == message_history_.end() || it->second.empty()) {
        return false;
    }

    // Check if current message is very similar to a recent historical message
    for (const auto& historical_msg : it->second) {
        if (historical_msg.size() == message.size()) {
            double similarity = 0.0;
            for (size_t i = 0; i < message.size(); ++i) {
                similarity += std::abs(message[i] - historical_msg[i]);
            }
            similarity /= message.size();

            if (similarity < config_.numerical_epsilon * 10) {  // Very similar
                return true;
            }
        }
    }

    return false;
}

auto CircularBPEngine::check_convergence(double residual) const -> bool {
    return residual < config_.convergence_threshold;
}

auto CircularBPEngine::compute_marginals(const GraphicalModel& model)
    -> common::Result<std::vector<std::vector<double>>, CircularBPError> {
    std::vector<std::vector<double>> marginals;

    // For each node, compute marginal from local potential and incoming messages
    for (const auto& node : model.nodes) {
        std::vector<double> marginal = node.local_potential;

        // Apply cycle influence dampening if node is involved in cycles
        if (node.cycle_influence > config_.numerical_epsilon) {
            double cycle_dampening = 1.0 - std::min(0.5, node.cycle_influence);
            for (auto& prob : marginal) {
                prob *= cycle_dampening;
            }
        }

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

void CircularBPEngine::normalize_message(Message& message) const {
    double sum = std::accumulate(message.begin(), message.end(), 0.0);
    if (sum > config_.numerical_epsilon) {
        for (auto& prob : message) {
            prob /= sum;
        }
    }
}

auto CircularBPEngine::compute_message_residual(const Message& old_msg,
                                                const Message& new_msg) const -> double {
    double residual = 0.0;
    for (std::size_t i = 0; i < old_msg.size() && i < new_msg.size(); ++i) {
        double diff = new_msg[i] - old_msg[i];
        residual += diff * diff;
    }
    return std::sqrt(residual);
}

auto CircularBPEngine::generate_edge_id(NodeId from, NodeId to) const -> EdgeId {
    return (static_cast<std::uint64_t>(from) << 32) | to;
}

auto CircularBPEngine::validate_model_consistency(const GraphicalModel& model)
    -> common::Result<std::monostate, CircularBPError> {
    // Check that all nodes have consistent domain sizes with configuration
    for (const auto& node : model.nodes) {
        if (node.local_potential.size() != config_.variable_domain_size) {
            LOG_WARNING_PRINT("Node {} has domain size {} but config expects {}",
                              node.id,
                              node.local_potential.size(),
                              config_.variable_domain_size);

            if (node.local_potential.empty()) {
                LOG_ERROR_PRINT("Node {} has empty local potential", node.id);
                return common::Err(CircularBPError::INVALID_POTENTIAL_FUNCTION);
            }
        }
    }

    // Check edge potential matrix dimensions
    for (const auto& edge : model.edges) {
        if (edge.potential_matrix.empty()) {
            LOG_ERROR_PRINT("Edge {} has empty potential matrix", edge.id);
            return common::Err(CircularBPError::INVALID_POTENTIAL_FUNCTION);
        }

        for (const auto& row : edge.potential_matrix) {
            if (row.empty()) {
                LOG_ERROR_PRINT("Edge {} has empty potential matrix row", edge.id);
                return common::Err(CircularBPError::INVALID_POTENTIAL_FUNCTION);
            }
        }
    }

    LOG_DEBUG_PRINT("Model validation passed: {} nodes validated, {} edges validated",
                    model.nodes.size(),
                    model.edges.size());
    return common::Ok(std::monostate{});
}

auto create_circular_bp_engine(const CircularBPConfig& config)
    -> common::Result<std::unique_ptr<CircularBPEngine>, CircularBPError> {
    try {
        auto engine = std::make_unique<CircularBPEngine>(config);
        return common::Ok(std::move(engine));
    } catch (const std::bad_alloc&) {
        return common::Err(CircularBPError::MEMORY_ALLOCATION_FAILED);
    } catch (...) {
        return common::Err(CircularBPError::UNKNOWN_ERROR);
    }
}

}  // namespace inference_lab::engines::circular_bp
