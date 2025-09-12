// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file momentum_bp.hpp
 * @brief Momentum-Enhanced Belief Propagation inference engine
 *
 * Implements belief propagation with ML-inspired optimization techniques including
 * momentum and adaptive learning rates (AdaGrad) to improve convergence and reduce
 * oscillations in message passing on graphical models.
 *
 * Key innovations over standard BP:
 * - Momentum updates for message passing to reduce oscillations
 * - Adaptive learning rates per edge to handle heterogeneous convergence
 * - Improved convergence detection with momentum-aware criteria
 * - Integration with existing Result<T,E> error handling patterns
 *
 * Research Foundation:
 * Based on "Improved Belief Propagation Decoding Algorithms for Surface Codes" (2024)
 * and ML optimization techniques applied to probabilistic inference.
 *
 * Usage Example:
 * @code
 * auto engine = create_momentum_bp_engine(config);
 * if (engine.is_ok()) {
 *     auto result = engine.unwrap()->run_inference(graph_model);
 *     // Process marginal probabilities...
 * }
 * @endcode
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/src/result.hpp"
#include "../inference_engine.hpp"

namespace inference_lab::engines::momentum_bp {

/**
 * @brief Momentum-BP specific error types
 */
enum class MomentumBPError : std::uint8_t {
    INVALID_GRAPH_STRUCTURE,     ///< Graph has invalid topology for BP
    CONVERGENCE_FAILED,          ///< Failed to converge within iteration limit
    NUMERICAL_INSTABILITY,       ///< Numerical issues in message computation
    INVALID_POTENTIAL_FUNCTION,  ///< Malformed potential function
    MEMORY_ALLOCATION_FAILED,    ///< Failed to allocate message storage
    UNKNOWN_ERROR                ///< Unexpected error condition
};

/**
 * @brief Convert MomentumBPError to human-readable string
 */
std::string to_string(MomentumBPError error);

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
 * @brief Configuration for Momentum-BP engine
 */
struct MomentumBPConfig {
    std::uint32_t max_iterations{100};      ///< Maximum BP iterations
    double convergence_threshold{1e-6};     ///< Convergence tolerance
    double momentum_factor{0.9};            ///< Momentum coefficient (β₁)
    double learning_rate{0.1};              ///< Base learning rate
    double adagrad_epsilon{1e-8};           ///< AdaGrad epsilon for numerical stability
    double numerical_epsilon{1e-10};        ///< Epsilon for numerical stability checks
    std::uint32_t variable_domain_size{2};  ///< Domain size for variables (default binary)
    bool enable_momentum{true};             ///< Enable momentum updates
    bool enable_adagrad{true};              ///< Enable adaptive learning rates
    bool normalize_messages{true};          ///< Normalize messages to valid probabilities
};

/**
 * @brief Performance metrics for Momentum-BP
 */
struct MomentumBPMetrics {
    std::uint32_t iterations_to_convergence{0};      ///< Iterations until convergence
    double final_residual{0.0};                      ///< Final message residual
    std::chrono::milliseconds inference_time_ms{0};  ///< Total inference time
    std::uint64_t message_updates{0};                ///< Total message updates performed
    std::uint32_t oscillation_cycles{0};             ///< Detected oscillation cycles
    bool converged{false};                           ///< Whether algorithm converged
};

/**
 * @brief Simple graphical model node
 */
struct Node {
    NodeId id;
    std::vector<double> local_potential;  ///< Local potential function
    std::vector<NodeId> neighbors;        ///< Connected node IDs
};

/**
 * @brief Edge potential function between nodes
 */
struct EdgePotential {
    EdgeId id;
    NodeId from_node;
    NodeId to_node;
    std::vector<std::vector<double>> potential_matrix;  ///< Pairwise potential
};

/**
 * @brief Simple graphical model for belief propagation
 */
struct GraphicalModel {
    std::vector<Node> nodes;
    std::vector<EdgePotential> edges;
    std::unordered_map<NodeId, std::uint32_t> node_index;  ///< Node ID to vector index
};

/**
 * @brief Momentum-Enhanced Belief Propagation inference engine
 *
 * Implements belief propagation with momentum and adaptive learning rates
 * to improve convergence characteristics on graphical models.
 */
class MomentumBPEngine : public InferenceEngine {
  public:
    /**
     * @brief Construct Momentum-BP engine with configuration
     */
    explicit MomentumBPEngine(const MomentumBPConfig& config = MomentumBPConfig{});

    /**
     * @brief Virtual destructor for proper cleanup
     */
    ~MomentumBPEngine() override = default;

    // InferenceEngine interface implementation

    /**
     * @brief Execute momentum-enhanced belief propagation
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

    // Momentum-BP specific interface

    /**
     * @brief Run inference on a graphical model directly
     * @param model The graphical model to perform inference on
     * @return Result containing marginal probabilities or error
     */
    auto run_momentum_bp(const GraphicalModel& model)
        -> common::Result<std::vector<std::vector<double>>, MomentumBPError>;

    /**
     * @brief Get detailed performance metrics
     */
    auto get_metrics() const -> MomentumBPMetrics;

    /**
     * @brief Update configuration parameters
     */
    void update_config(const MomentumBPConfig& new_config);

    /**
     * @brief Reset metrics and internal state
     */
    void reset();

  private:
    MomentumBPConfig config_;
    mutable MomentumBPMetrics metrics_;

    // Message storage and momentum terms
    std::unordered_map<EdgeId, Message> messages_;
    std::unordered_map<EdgeId, Message> momentum_terms_;
    std::unordered_map<EdgeId, std::vector<double>> adagrad_accumulator_;

    /**
     * @brief Initialize message storage for a graphical model
     */
    auto initialize_messages(const GraphicalModel& model)
        -> common::Result<std::monostate, MomentumBPError>;

    /**
     * @brief Validate model consistency with configuration
     */
    auto validate_model_consistency(const GraphicalModel& model)
        -> common::Result<std::monostate, MomentumBPError>;

    /**
     * @brief Compute message from one node to another
     */
    auto compute_message(const GraphicalModel& model, EdgeId edge_id)
        -> common::Result<Message, MomentumBPError>;

    /**
     * @brief Update message with momentum and adaptive learning rate
     */
    auto update_message_with_momentum(EdgeId edge_id, const Message& new_message)
        -> common::Result<std::monostate, MomentumBPError>;

    /**
     * @brief Check convergence based on message residuals
     */
    auto check_convergence(double residual) const -> bool;

    /**
     * @brief Compute marginal probabilities from final messages
     */
    auto compute_marginals(const GraphicalModel& model)
        -> common::Result<std::vector<std::vector<double>>, MomentumBPError>;

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
};

/**
 * @brief Factory function to create Momentum-BP engine
 */
auto create_momentum_bp_engine(const MomentumBPConfig& config = MomentumBPConfig{})
    -> common::Result<std::unique_ptr<MomentumBPEngine>, MomentumBPError>;

}  // namespace inference_lab::engines::momentum_bp
