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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../../../common/src/result.hpp"
#include "../../../common/src/type_system.hpp"
#include "differentiable_ops.hpp"
#include "fuzzy_logic.hpp"
#include "tensor_logic_bridge.hpp"

namespace inference_lab::engines::neuro_symbolic {

// ================================================================================================
// THREAD-SAFE UTILITIES
// ================================================================================================

/**
 * @brief Thread-safe random number generator
 *
 * Provides a thread-safe way to get a random number generator instance.
 * Each thread gets its own generator, avoiding race conditions during
 * static initialization.
 */
inline std::mt19937& get_thread_generator() {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    return gen;
}

// ================================================================================================
// LTN CONFIGURATION AND TYPES
// ================================================================================================

/**
 * @brief Configuration parameters for Logic Tensor Networks
 */
struct LTNConfig {
    std::size_t embedding_dim = 64;         ///< Dimensionality of symbol embeddings
    float learning_rate = 0.001f;           ///< Learning rate for gradient descent
    float regularization_weight = 0.01f;    ///< L2 regularization strength
    std::size_t max_iterations = 1000;      ///< Maximum training iterations
    float convergence_threshold = 1e-6f;    ///< Convergence tolerance
    float quantifier_temperature = 10.0f;   ///< Temperature for smooth quantifiers
    bool use_batch_learning = true;         ///< Enable batch gradient updates
    std::size_t batch_size = 32;            ///< Batch size for training
    bool enable_symbolic_reasoning = true;  ///< Enable pure symbolic inference
    std::string optimizer_type = "adam";    ///< Optimizer: "sgd", "adam", "rmsprop"
};

/**
 * @brief Error types specific to Logic Tensor Networks
 */
enum class LTNError : std::uint8_t {
    INVALID_FORMULA_SYNTAX,   ///< Malformed logical formula
    UNDEFINED_PREDICATE,      ///< Reference to non-existent predicate
    UNDEFINED_INDIVIDUAL,     ///< Reference to non-existent individual
    DIMENSION_MISMATCH,       ///< Incompatible tensor dimensions
    OPTIMIZATION_FAILURE,     ///< Gradient descent failed to converge
    MEMORY_ALLOCATION_ERROR,  ///< Insufficient memory for operations
    NUMERICAL_INSTABILITY,    ///< Gradient explosion or vanishing
    INVALID_CONFIGURATION     ///< Invalid configuration parameters
};

/**
 * @brief Individual entity in the LTN domain
 *
 * Represents a grounded individual with a unique identifier and
 * learned vector embedding in the semantic space.
 */
struct Individual {
    std::string name;              ///< Human-readable identifier
    std::size_t id;                ///< Unique numerical identifier
    std::vector<float> embedding;  ///< Learned vector representation
    bool is_trainable = true;      ///< Whether embedding can be updated

    Individual() = default;  // Default constructor for containers

    Individual(std::string name, std::size_t id, std::size_t embedding_dim)
        : name(std::move(name)), id(id), embedding(embedding_dim, 0.0f) {
        // Initialize embedding with small random values using thread-safe generator
        initialize_embedding_random();
    }

  private:
    void initialize_embedding_random() {
        // Use thread-safe random generator
        auto& gen = get_thread_generator();
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto& val : embedding) {
            val = dist(gen);
        }
    }
};

/**
 * @brief Predicate definition with learnable parameters
 *
 * Represents a logical predicate (e.g., "Human", "Mortal") with
 * associated neural network parameters for evaluation.
 */
struct Predicate {
    std::string name;            ///< Predicate name
    std::size_t arity;           ///< Number of arguments
    std::size_t id;              ///< Unique identifier
    std::vector<float> weights;  ///< Learnable weight matrix
    std::vector<float> bias;     ///< Learnable bias vector
    bool is_trainable = true;    ///< Whether parameters can be updated

    Predicate() = default;  // Default constructor for containers

    Predicate(std::string name, std::size_t arity, std::size_t id, std::size_t embedding_dim)
        : name(std::move(name)), arity(arity), id(id) {
        // Initialize weight matrix: (arity * embedding_dim) → 1
        std::size_t weight_size = arity * embedding_dim;
        weights.resize(weight_size);
        bias.resize(1);

        // Xavier initialization with thread-safe random generation
        initialize_weights_xavier(weight_size);
        bias[0] = 0.0f;  // Initialize bias to zero
    }

  private:
    void initialize_weights_xavier(std::size_t weight_size) {
        // Use thread-safe random generator
        auto& gen = get_thread_generator();
        float std_dev = std::sqrt(2.0f / static_cast<float>(weight_size));
        std::normal_distribution<float> weight_dist(0.0f, std_dev);

        for (auto& w : weights) {
            w = weight_dist(gen);
        }
    }
};

// ================================================================================================
// LOGIC TENSOR NETWORK CORE CLASS
// ================================================================================================

/**
 * @brief Main Logic Tensor Network implementation
 *
 * This class orchestrates the entire LTN framework, managing:
 * - Symbol embeddings and predicate parameters
 * - Logical formula compilation and evaluation
 * - Gradient-based learning and optimization
 * - Integration between symbolic and neural components
 */
class LogicTensorNetwork {
  public:
    // ============================================================================================
    // CONSTRUCTION AND CONFIGURATION
    // ============================================================================================

    /**
     * @brief Create new Logic Tensor Network with given configuration
     * @param config LTN configuration parameters
     * @return Result containing LTN instance or error
     */
    static auto create(const LTNConfig& config)
        -> common::Result<std::unique_ptr<LogicTensorNetwork>, LTNError>;

    /**
     * @brief Destructor
     */
    ~LogicTensorNetwork() = default;

    // Non-copyable but movable
    LogicTensorNetwork(const LogicTensorNetwork&) = delete;
    LogicTensorNetwork& operator=(const LogicTensorNetwork&) = delete;
    LogicTensorNetwork(LogicTensorNetwork&&) = default;
    LogicTensorNetwork& operator=(LogicTensorNetwork&&) = default;

    // ============================================================================================
    // SYMBOL AND PREDICATE MANAGEMENT
    // ============================================================================================

    /**
     * @brief Add individual to the domain
     * @param name Individual name/identifier
     * @return Result containing individual ID or error
     */
    auto add_individual(const std::string& name) -> common::Result<std::size_t, LTNError>;

    /**
     * @brief Add predicate definition
     * @param name Predicate name
     * @param arity Number of arguments (1 for unary, 2 for binary, etc.)
     * @return Result containing predicate ID or error
     */
    auto add_predicate(const std::string& name, std::size_t arity)
        -> common::Result<std::size_t, LTNError>;

    /**
     * @brief Get individual by name
     * @param name Individual name
     * @return Result containing individual reference or error
     */
    auto get_individual(const std::string& name) -> common::Result<Individual*, LTNError>;
    auto get_individual(const std::string& name) const
        -> common::Result<const Individual*, LTNError>;

    /**
     * @brief Get predicate by name
     * @param name Predicate name
     * @return Result containing predicate reference or error
     */
    auto get_predicate(const std::string& name) -> common::Result<Predicate*, LTNError>;
    auto get_predicate(const std::string& name) const -> common::Result<const Predicate*, LTNError>;

    // ============================================================================================
    // PREDICATE EVALUATION
    // ============================================================================================

    /**
     * @brief Evaluate unary predicate on individual
     * @param predicate_name Name of predicate to evaluate
     * @param individual_name Name of individual
     * @return Result containing fuzzy truth value or error
     */
    auto evaluate_predicate(const std::string& predicate_name, const std::string& individual_name)
        -> common::Result<FuzzyValue, LTNError>;

    /**
     * @brief Evaluate binary predicate on pair of individuals
     * @param predicate_name Name of binary predicate/relation
     * @param individual1 First individual
     * @param individual2 Second individual
     * @return Result containing fuzzy truth value or error
     */
    auto evaluate_relation(const std::string& predicate_name,
                           const std::string& individual1,
                           const std::string& individual2) -> common::Result<FuzzyValue, LTNError>;

    /**
     * @brief Batch evaluate predicate on multiple individuals
     * @param predicate_name Name of predicate
     * @param individual_names Vector of individual names
     * @return Result containing vector of truth values or error
     */
    auto batch_evaluate_predicate(const std::string& predicate_name,
                                  const std::vector<std::string>& individual_names)
        -> common::Result<std::vector<FuzzyValue>, LTNError>;

    // ============================================================================================
    // LOGICAL FORMULA CONSTRUCTION AND EVALUATION
    // ============================================================================================

    /**
     * @brief Logical formula representation for evaluation
     */
    class Formula {
      public:
        virtual ~Formula() = default;

        /**
         * @brief Evaluate formula in given context
         * @param ltn Reference to parent LTN instance
         * @return Fuzzy truth value of formula
         */
        virtual auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> = 0;

        /**
         * @brief Get human-readable representation
         * @return String representation of formula
         */
        virtual auto to_string() const -> std::string = 0;

        /**
         * @brief Get list of free variables in formula
         * @return Vector of variable names
         */
        virtual auto free_variables() const -> std::vector<std::string> = 0;
    };

    /**
     * @brief Atomic formula: predicate applied to individuals/variables
     */
    class AtomicFormula : public Formula {
      public:
        AtomicFormula(std::string predicate_name, std::vector<std::string> arguments)
            : predicate_name_(std::move(predicate_name)), arguments_(std::move(arguments)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::string predicate_name_;
        std::vector<std::string> arguments_;
    };

    /**
     * @brief Conjunction formula: A ∧ B
     */
    class ConjunctionFormula : public Formula {
      public:
        ConjunctionFormula(std::unique_ptr<Formula> left, std::unique_ptr<Formula> right)
            : left_(std::move(left)), right_(std::move(right)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::unique_ptr<Formula> left_;
        std::unique_ptr<Formula> right_;
    };

    /**
     * @brief Disjunction formula: A ∨ B
     */
    class DisjunctionFormula : public Formula {
      public:
        DisjunctionFormula(std::unique_ptr<Formula> left, std::unique_ptr<Formula> right)
            : left_(std::move(left)), right_(std::move(right)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::unique_ptr<Formula> left_;
        std::unique_ptr<Formula> right_;
    };

    /**
     * @brief Implication formula: A → B
     */
    class ImplicationFormula : public Formula {
      public:
        ImplicationFormula(std::unique_ptr<Formula> antecedent, std::unique_ptr<Formula> consequent)
            : antecedent_(std::move(antecedent)), consequent_(std::move(consequent)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::unique_ptr<Formula> antecedent_;
        std::unique_ptr<Formula> consequent_;
    };

    /**
     * @brief Negation formula: ¬A
     */
    class NegationFormula : public Formula {
      public:
        explicit NegationFormula(std::unique_ptr<Formula> operand) : operand_(std::move(operand)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::unique_ptr<Formula> operand_;
    };

    /**
     * @brief Universal quantification: ∀x. P(x)
     */
    class ForallFormula : public Formula {
      public:
        ForallFormula(std::string variable, std::unique_ptr<Formula> body)
            : variable_(std::move(variable)), body_(std::move(body)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::string variable_;
        std::unique_ptr<Formula> body_;
    };

    /**
     * @brief Existential quantification: ∃x. P(x)
     */
    class ExistsFormula : public Formula {
      public:
        ExistsFormula(std::string variable, std::unique_ptr<Formula> body)
            : variable_(std::move(variable)), body_(std::move(body)) {}

        auto evaluate(LogicTensorNetwork& ltn) -> common::Result<FuzzyValue, LTNError> override;
        auto to_string() const -> std::string override;
        auto free_variables() const -> std::vector<std::string> override;

      private:
        std::string variable_;
        std::unique_ptr<Formula> body_;
    };

    // ============================================================================================
    // FORMULA CONSTRUCTION HELPERS
    // ============================================================================================

    /**
     * @brief Create atomic formula (predicate application)
     * @param predicate_name Name of predicate
     * @param arguments List of argument names
     * @return Unique pointer to atomic formula
     */
    auto atomic(const std::string& predicate_name, const std::vector<std::string>& arguments)
        -> std::unique_ptr<Formula>;

    /**
     * @brief Create conjunction of two formulas
     * @param left Left operand
     * @param right Right operand
     * @return Unique pointer to conjunction formula
     */
    auto conjunction(std::unique_ptr<Formula> left, std::unique_ptr<Formula> right)
        -> std::unique_ptr<Formula>;

    /**
     * @brief Create disjunction of two formulas
     * @param left Left operand
     * @param right Right operand
     * @return Unique pointer to disjunction formula
     */
    auto disjunction(std::unique_ptr<Formula> left, std::unique_ptr<Formula> right)
        -> std::unique_ptr<Formula>;

    /**
     * @brief Create implication formula
     * @param antecedent Antecedent formula
     * @param consequent Consequent formula
     * @return Unique pointer to implication formula
     */
    auto implication(std::unique_ptr<Formula> antecedent, std::unique_ptr<Formula> consequent)
        -> std::unique_ptr<Formula>;

    /**
     * @brief Create negation formula
     * @param operand Formula to negate
     * @return Unique pointer to negation formula
     */
    auto negation(std::unique_ptr<Formula> operand) -> std::unique_ptr<Formula>;

    /**
     * @brief Create universal quantification
     * @param variable Variable name to quantify over
     * @param body Formula body
     * @return Unique pointer to forall formula
     */
    auto forall(const std::string& variable, std::unique_ptr<Formula> body)
        -> std::unique_ptr<Formula>;

    /**
     * @brief Create existential quantification
     * @param variable Variable name to quantify over
     * @param body Formula body
     * @return Unique pointer to exists formula
     */
    auto exists(const std::string& variable, std::unique_ptr<Formula> body)
        -> std::unique_ptr<Formula>;

    // ============================================================================================
    // KNOWLEDGE BASE AND TRAINING
    // ============================================================================================

    /**
     * @brief Add logical formula as constraint to knowledge base
     * @param name Formula name/identifier
     * @param formula Formula to add
     * @param weight Importance weight for training (default: 1.0)
     * @return Result indicating success or error
     */
    auto add_formula(const std::string& name, std::unique_ptr<Formula> formula, float weight = 1.0f)
        -> common::Result<std::monostate, LTNError>;

    /**
     * @brief Training example for supervised learning
     */
    struct Example {
        std::string predicate_name;          ///< Predicate being supervised
        std::vector<std::string> arguments;  ///< Arguments to predicate
        FuzzyValue target_truth;             ///< Target truth value
        float weight = 1.0f;                 ///< Example importance weight
    };

    /**
     * @brief Train LTN on examples and knowledge base constraints
     * @param examples Training examples
     * @param epochs Number of training epochs
     * @return Result containing final loss or error
     */
    auto train(const std::vector<Example>& examples, std::size_t epochs = 100)
        -> common::Result<float, LTNError>;

    /**
     * @brief Evaluate current knowledge base consistency
     * @return Result containing average satisfaction of all formulas
     */
    auto evaluate_knowledge_base() -> common::Result<float, LTNError>;

    // ============================================================================================
    // INFERENCE AND QUERYING
    // ============================================================================================

    /**
     * @brief Query truth value of formula
     * @param formula Formula to evaluate
     * @return Result containing fuzzy truth value or error
     */
    auto query(Formula& formula) -> common::Result<FuzzyValue, LTNError>;

    /**
     * @brief Find individuals satisfying given predicate above threshold
     * @param predicate_name Predicate to check
     * @param threshold Truth value threshold (default: 0.5)
     * @return Result containing list of individual names or error
     */
    auto find_individuals(const std::string& predicate_name, float threshold = 0.5f)
        -> common::Result<std::vector<std::string>, LTNError>;

    /**
     * @brief Get embedding vector for individual
     * @param individual_name Name of individual
     * @return Result containing embedding vector or error
     */
    auto get_embedding(const std::string& individual_name)
        -> common::Result<std::vector<float>, LTNError>;

    // ============================================================================================
    // STATISTICS AND DEBUGGING
    // ============================================================================================

    /**
     * @brief Get statistics about current LTN state
     */
    struct Statistics {
        std::size_t num_individuals;
        std::size_t num_predicates;
        std::size_t num_formulas;
        float average_formula_satisfaction;
        float total_parameters;
        std::vector<float> embedding_norms;  // L2 norms of individual embeddings
        std::vector<float> predicate_norms;  // L2 norms of predicate weights
    };

    auto get_statistics() const -> Statistics;

    /**
     * @brief Export current model parameters
     * @return Serializable model state
     */
    auto export_model() const -> std::unordered_map<std::string, std::vector<float>>;

    /**
     * @brief Import model parameters
     * @param model_data Previously exported model state
     * @return Result indicating success or error
     */
    auto import_model(const std::unordered_map<std::string, std::vector<float>>& model_data)
        -> common::Result<std::monostate, LTNError>;

  private:
    // ============================================================================================
    // PRIVATE IMPLEMENTATION
    // ============================================================================================

    /**
     * @brief Private constructor - use create() factory method
     */
    explicit LogicTensorNetwork(LTNConfig config);

    /**
     * @brief Compute gradients for all parameters
     * @param loss Total loss value
     * @return Result indicating success or error
     */
    auto compute_gradients(float loss) -> common::Result<std::monostate, LTNError>;

    /**
     * @brief Update parameters using computed gradients
     * @return Result indicating success or error
     */
    auto update_parameters() -> common::Result<std::monostate, LTNError>;

    /**
     * @brief Evaluate neural network for predicate
     * @param predicate Predicate to evaluate
     * @param embeddings Input embeddings
     * @return Fuzzy truth value
     */
    auto evaluate_neural_predicate(const Predicate& predicate,
                                   const std::vector<std::vector<float>>& embeddings) -> FuzzyValue;

    // Configuration and state
    LTNConfig config_;

    // Symbol management
    std::unordered_map<std::string, Individual> individuals_;
    std::unordered_map<std::string, Predicate> predicates_;
    std::size_t next_individual_id_ = 0;
    std::size_t next_predicate_id_ = 0;

    // Knowledge base
    std::unordered_map<std::string, std::unique_ptr<Formula>> formulas_;
    std::unordered_map<std::string, float> formula_weights_;

    // Training state
    std::unordered_map<std::string, std::vector<float>> parameter_gradients_;
    std::unordered_map<std::string, std::vector<float>> optimizer_state_;  // For Adam, etc.
    float current_loss_ = 0.0f;
    std::size_t training_step_ = 0;
};

}  // namespace inference_lab::engines::neuro_symbolic
