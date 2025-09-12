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

/**
 * @file circular_bp.hpp
 * @brief Circular Belief Propagation inference engine with cycle detection and correlation
 * cancellation
 *
 * Implements belief propagation with advanced cycle detection and spurious correlation
 * cancellation to improve accuracy on cyclic graphical models. Addresses the fundamental
 * limitation of standard BP in graphs with cycles by detecting message reverberation
 * and implementing correlation cancellation mechanisms.
 *
 * Key innovations over standard BP:
 * - Cycle detection in factor graphs using almost-linear algorithms
 * - Spurious correlation identification and mitigation
 * - Message reverberation prevention through intelligent gating
 * - Support for both directed and undirected graph topologies
 * - Integration with existing Result<T,E> error handling patterns
 *
 * Research Foundation:
 * Based on "Circular Belief Propagation for Approximate Probabilistic Inference" (2024)
 * and modern cycle detection algorithms for dynamic graphs.
 *
 * Usage Example:
 * @code
 * auto engine = create_circular_bp_engine(config);
 * if (engine.is_ok()) {
 *     auto result = engine.unwrap()->run_inference(cyclic_graph_model);
 *     // Process marginal probabilities with improved cyclic accuracy...
 * }
 * @endcode
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../common/src/result.hpp"
#include "../inference_engine.hpp"

namespace inference_lab::engines::circular_bp {

/**
 * @brief Circular-BP specific error types
 */
enum class CircularBPError : std::uint8_t {
    INVALID_GRAPH_STRUCTURE,          ///< Graph has invalid topology for Circular BP
    CYCLE_DETECTION_FAILED,           ///< Failed to detect cycles in graph structure
    CONVERGENCE_FAILED,               ///< Failed to converge within iteration limit
    CORRELATION_CANCELLATION_FAILED,  ///< Failed to cancel spurious correlations
    NUMERICAL_INSTABILITY,            ///< Numerical issues in message computation
    INVALID_POTENTIAL_FUNCTION,       ///< Malformed potential function
    MEMORY_ALLOCATION_FAILED,         ///< Failed to allocate message storage
    UNKNOWN_ERROR                     ///< Unexpected error condition
};

/**
 * @brief Convert CircularBPError to human-readable string
 */
std::string to_string(CircularBPError error);

/**
 * @brief Node identifier in the graphical model
 */
using NodeId = std::uint64_t;

/**
 * @brief Edge identifier for message passing
 */
using EdgeId = std::uint64_t;

/**
 * @brief Message vector for probability distributions
 */
using Message = std::vector<double>;

/**
 * @brief Cycle representation in the graph
 */
struct Cycle {
    std::vector<NodeId> nodes;    ///< Nodes forming the cycle
    std::vector<EdgeId> edges;    ///< Edges forming the cycle
    double correlation_strength;  ///< Measured correlation strength
    bool is_spurious;             ///< Whether correlation is spurious
};

/**
 * @brief Cycle detection strategy interface
 */
enum class CycleDetectionStrategy : std::uint8_t {
    DEPTH_FIRST_SEARCH,  ///< DFS-based cycle detection
    SPARSE_MATRIX,       ///< Sparse matrix approach for large graphs
    HYBRID_ADAPTIVE      ///< Adaptive strategy based on graph characteristics
};

/**
 * @brief Configuration for Circular-BP engine
 */
struct CircularBPConfig {
    std::uint32_t max_iterations{100};      ///< Maximum BP iterations
    double convergence_threshold{1e-6};     ///< Convergence tolerance
    double correlation_threshold{0.8};      ///< Spurious correlation detection threshold
    double cycle_penalty_factor{0.1};       ///< Penalty for cyclic message paths
    double numerical_epsilon{1e-10};        ///< Epsilon for numerical stability checks
    std::uint32_t variable_domain_size{2};  ///< Domain size for variables (default binary)
    std::uint32_t max_cycle_length{10};     ///< Maximum cycle length to detect
    CycleDetectionStrategy detection_strategy{CycleDetectionStrategy::HYBRID_ADAPTIVE};
    bool enable_correlation_cancellation{true};  ///< Enable spurious correlation cancellation
    bool enable_cycle_penalties{true};           ///< Enable cycle penalty mechanisms
    bool normalize_messages{true};               ///< Normalize messages to valid probabilities
    bool track_message_history{true};  ///< Track message history for reverberation detection
};

/**
 * @brief Performance metrics for Circular-BP
 */
struct CircularBPMetrics {
    std::uint32_t iterations_to_convergence{0};      ///< Iterations until convergence
    double final_residual{0.0};                      ///< Final message residual
    std::chrono::milliseconds inference_time_ms{0};  ///< Total inference time
    std::uint64_t message_updates{0};                ///< Total message updates performed
    std::uint32_t cycles_detected{0};                ///< Number of cycles detected
    std::uint32_t correlations_cancelled{0};         ///< Spurious correlations cancelled
    std::uint32_t reverberation_events{0};           ///< Message reverberation events detected
    bool converged{false};                           ///< Whether algorithm converged
};

/**
 * @brief Enhanced graphical model node for cycle detection
 */
struct Node {
    NodeId id;
    std::vector<double> local_potential;         ///< Local potential function
    std::vector<NodeId> neighbors;               ///< Connected node IDs
    std::unordered_set<NodeId> cycle_neighbors;  ///< Neighbors involved in cycles
    double cycle_influence{0.0};                 ///< Accumulated cycle influence
};

/**
 * @brief Enhanced edge potential with cycle tracking
 */
struct EdgePotential {
    EdgeId id;
    NodeId from_node;
    NodeId to_node;
    std::vector<std::vector<double>> potential_matrix;  ///< Pairwise potential
    bool is_cycle_edge{false};  ///< Whether edge is part of a detected cycle
    double cycle_penalty{0.0};  ///< Accumulated cycle penalty
};

/**
 * @brief Enhanced graphical model for circular belief propagation
 */
struct GraphicalModel {
    std::vector<Node> nodes;
    std::vector<EdgePotential> edges;
    std::unordered_map<NodeId, std::uint32_t> node_index;  ///< Node ID to vector index
    std::vector<Cycle> detected_cycles;                    ///< Detected cycles in graph
};

/**
 * @brief Circular Belief Propagation inference engine
 *
 * Implements belief propagation with cycle detection and correlation cancellation
 * to improve accuracy on cyclic graphical models while maintaining efficiency.
 */
class CircularBPEngine : public InferenceEngine {
  public:
    /**
     * @brief Construct Circular-BP engine with configuration
     */
    explicit CircularBPEngine(const CircularBPConfig& config = CircularBPConfig{});

    /**
     * @brief Virtual destructor for proper cleanup
     */
    ~CircularBPEngine() override = default;

    // InferenceEngine interface implementation

    /**
     * @brief Execute circular belief propagation with cycle detection
     * @param request Inference request containing graphical model
     * @return Result containing marginal probabilities or error
     */
    auto run_inference(const InferenceRequest& request)
        -> common::Result<InferenceResponse, InferenceError> override;

    /**
     * @brief Get backend information
     */
    auto get_backend_info() const -> std::string override;

    /**
     * @brief Check if engine is ready for inference
     */
    auto is_ready() const -> bool override;

    /**
     * @brief Get performance statistics
     */
    auto get_performance_stats() const -> std::string override;

    // Circular-BP specific interface

    /**
     * @brief Run inference on a graphical model directly
     * @param model The graphical model to perform inference on
     * @return Result containing marginal probabilities or error
     */
    auto run_circular_bp(const GraphicalModel& model)
        -> common::Result<std::vector<std::vector<double>>, CircularBPError>;

    /**
     * @brief Get detailed performance metrics
     */
    auto get_metrics() const -> CircularBPMetrics;

    /**
     * @brief Update configuration parameters
     */
    void update_config(const CircularBPConfig& new_config);

    /**
     * @brief Reset metrics and internal state
     */
    void reset();

  private:
    CircularBPConfig config_;
    mutable CircularBPMetrics metrics_;

    // Message storage with history tracking
    std::unordered_map<EdgeId, Message> messages_;
    std::unordered_map<EdgeId, std::vector<Message>> message_history_;
    std::unordered_map<EdgeId, double> correlation_scores_;

    /**
     * @brief Detect cycles in the graphical model
     */
    auto detect_cycles(GraphicalModel& model) -> common::Result<std::monostate, CircularBPError>;

    /**
     * @brief Depth-first search cycle detection
     */
    auto detect_cycles_dfs(const GraphicalModel& model)
        -> common::Result<std::vector<Cycle>, CircularBPError>;

    /**
     * @brief Sparse matrix cycle detection for large graphs
     */
    auto detect_cycles_sparse(const GraphicalModel& model)
        -> common::Result<std::vector<Cycle>, CircularBPError>;

    /**
     * @brief Initialize message storage for a graphical model
     */
    auto initialize_messages(const GraphicalModel& model)
        -> common::Result<std::monostate, CircularBPError>;

    /**
     * @brief Compute message with cycle awareness
     */
    auto compute_message(const GraphicalModel& model, EdgeId edge_id)
        -> common::Result<Message, CircularBPError>;

    /**
     * @brief Update message with correlation tracking
     */
    auto update_message_with_correlation_tracking(EdgeId edge_id, const Message& new_message)
        -> common::Result<std::monostate, CircularBPError>;

    /**
     * @brief Detect and cancel spurious correlations
     */
    auto cancel_spurious_correlations(GraphicalModel& model)
        -> common::Result<std::monostate, CircularBPError>;

    /**
     * @brief Check for message reverberation
     */
    auto check_message_reverberation(EdgeId edge_id, const Message& message) -> bool;

    /**
     * @brief Check convergence with cycle-aware criteria
     */
    auto check_convergence(double residual) const -> bool;

    /**
     * @brief Compute marginal probabilities with cycle correction
     */
    auto compute_marginals(const GraphicalModel& model)
        -> common::Result<std::vector<std::vector<double>>, CircularBPError>;

    /**
     * @brief Normalize message to valid probability distribution
     */
    void normalize_message(Message& message) const;

    /**
     * @brief Compute L2 residual between old and new messages
     */
    auto compute_message_residual(const Message& old_msg, const Message& new_msg) const -> double;

    /**
     * @brief Generate unique edge ID from node pair
     */
    auto generate_edge_id(NodeId from, NodeId to) const -> EdgeId;

    /**
     * @brief Validate model consistency with configuration
     */
    auto validate_model_consistency(const GraphicalModel& model)
        -> common::Result<std::monostate, CircularBPError>;
};

/**
 * @brief Factory function to create Circular-BP engine
 */
auto create_circular_bp_engine(const CircularBPConfig& config = CircularBPConfig{})
    -> common::Result<std::unique_ptr<CircularBPEngine>, CircularBPError>;

}  // namespace inference_lab::engines::circular_bp
