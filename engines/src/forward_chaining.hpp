// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file forward_chaining.hpp
 * @brief Forward chaining inference engine implementation
 *
 * This header provides a complete forward chaining inference engine that derives new facts
 * from existing facts using production rules. The engine follows the classical AI approach
 * of data-driven inference, where the presence of facts triggers applicable rules to derive
 * new conclusions.
 *
 * Key Features:
 * - Pattern matching with variable binding and unification
 * - Conflict resolution strategies for rule priority handling
 * - Fact database with efficient indexing by predicate name
 * - Integration with existing Result<T,E> error handling patterns
 * - Performance monitoring and execution tracing
 * - Memory-efficient rule evaluation with cycle detection
 *
 * Architecture Overview:
 * @code
 *   ┌─────────────┐   add_facts()    ┌─────────────────┐
 *   │ Knowledge   │ ────────────────▶│ Fact Database   │
 *   │ Base Input  │                  │ (indexed by     │
 *   │             │                  │  predicate)     │
 *   └─────────────┘                  └─────────────────┘
 *          │                                   │
 *          │ add_rules()                       │ pattern_match()
 *          ▼                                   ▼
 *   ┌─────────────┐   run_inference() ┌─────────────────┐
 *   │ Rule Base   │ ────────────────▶ │ Inference       │
 *   │             │                   │ Engine          │
 *   │ Priority    │◄──────────────────│ • Pattern Match │
 *   │ Ordered     │  conflict_resolve │ • Variable Bind │
 *   └─────────────┘                   │ • Rule Fire     │
 *          │                          │ • Cycle Detect  │
 *          │ new_facts_derived         └─────────────────┘
 *          ▼                                   │
 *   ┌─────────────┐                          │ derived_facts
 *   │ Inference   │◄─────────────────────────┘
 *   │ Results     │
 *   └─────────────┘
 * @endcode
 *
 * Example Usage:
 * @code
 * // Create forward chaining engine
 * ForwardChainingEngine engine;
 *
 * // Add facts to knowledge base
 * engine.add_fact(Fact(1, "isHuman", {"socrates"}));
 * engine.add_fact(Fact(2, "isHuman", {"plato"}));
 *
 * // Add inference rule: "All humans are mortal"
 * Rule mortality_rule(1, "mortality_rule");
 * mortality_rule.add_condition("isHuman", {"X"});
 * mortality_rule.add_conclusion("isMortal", {"X"});
 * engine.add_rule(mortality_rule);
 *
 * // Run inference to derive new facts
 * auto result = engine.run_inference();
 * if (result.is_ok()) {
 *     auto derived = result.unwrap();
 *     // derived contains: isMortal(socrates), isMortal(plato)
 * }
 * @endcode
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../common/src/inference_types.hpp"
#include "../../common/src/result.hpp"
#include "inference_engine.hpp"

namespace inference_lab::engines {

/**
 * @brief Forward chaining specific error types
 *
 * Extends the general InferenceError enum with forward chaining specific
 * error conditions for more precise error handling and debugging.
 */
enum class ForwardChainingError : std::uint8_t {
    INVALID_RULE_FORMAT,      ///< Rule has malformed conditions or conclusions
    VARIABLE_BINDING_FAILED,  ///< Failed to unify variables across conditions
    CYCLE_DETECTED,           ///< Infinite loop detected in rule firing
    FACT_DATABASE_CORRUPT,    ///< Inconsistent state in fact storage
    RULE_EVALUATION_TIMEOUT,  ///< Inference took too long (safety limit)
    UNKNOWN_ERROR             ///< Unexpected error condition
};

/**
 * @brief Convert ForwardChainingError to human-readable string
 * @param error The error code to convert
 * @return String description of the error
 */
std::string to_string(ForwardChainingError error);

/**
 * @brief Variable binding map for pattern matching
 *
 * Maps variable names (e.g., "X", "Y") to their bound values during
 * pattern matching and unification. Variables are identified by starting
 * with uppercase letters or being explicitly marked as variables.
 */
using VariableBindings = std::unordered_map<std::string, common::Value>;

/**
 * @brief Rule firing record for debugging and tracing
 *
 * Tracks when a rule was fired, what facts triggered it, and what
 * new facts were derived. Useful for explanation generation and debugging.
 */
struct RuleFiring {
    std::uint64_t rule_id{};                        ///< ID of the rule that fired
    std::string rule_name{};                        ///< Human-readable rule name
    std::vector<std::uint64_t> triggering_facts{};  ///< Facts that matched conditions
    std::vector<common::Fact> derived_facts{};      ///< New facts derived from rule
    VariableBindings bindings{};                    ///< Variable bindings used in firing
    std::chrono::milliseconds firing_time_ms{};     ///< Time when rule fired
};

/**
 * @brief Conflict resolution strategies for rule firing order
 *
 * When multiple rules can fire simultaneously, these strategies determine
 * the order of execution. Different strategies optimize for different goals
 * like performance, determinism, or domain-specific priorities.
 */
enum class ConflictResolutionStrategy : std::uint8_t {
    PRIORITY_ORDER,     ///< Fire highest priority rules first
    RECENCY_FIRST,      ///< Fire rules triggered by most recent facts first
    SPECIFICITY_FIRST,  ///< Fire more specific rules (more conditions) first
    RANDOM_ORDER        ///< Random order (for testing non-deterministic behavior)
};

/**
 * @brief Performance metrics for inference engine monitoring
 *
 * Tracks detailed performance and execution statistics for optimization
 * and monitoring purposes. Essential for production deployment analysis.
 */
struct InferenceMetrics {
    std::uint64_t facts_processed{0};               ///< Total facts considered
    std::uint64_t rules_evaluated{0};               ///< Total rule evaluations
    std::uint64_t rules_fired{0};                   ///< Successfully fired rules
    std::uint64_t pattern_matches{0};               ///< Successful pattern matches
    std::uint64_t variable_unifications{0};         ///< Variable binding operations
    std::uint64_t facts_derived{0};                 ///< New facts derived
    std::chrono::milliseconds total_time_ms{0};     ///< Total inference time
    std::chrono::milliseconds indexing_time_ms{0};  ///< Time spent on indexing
    std::chrono::milliseconds matching_time_ms{0};  ///< Time spent pattern matching
};

/**
 * @brief Forward chaining inference engine implementation
 *
 * Implements the classical forward chaining algorithm for rule-based inference.
 * The engine maintains a fact database and rule base, continuously applying
 * applicable rules to derive new facts until no more rules can fire.
 *
 * Key algorithmic features:
 * - Efficient fact indexing by predicate for O(1) lookup
 * - Pattern matching with variable unification
 * - Cycle detection to prevent infinite loops
 * - Configurable conflict resolution strategies
 * - Comprehensive performance monitoring
 * - Memory-efficient implementation with lazy evaluation
 */
class ForwardChainingEngine : public InferenceEngine {
  public:
    /**
     * @brief Construct a new Forward Chaining Engine
     * @param strategy Conflict resolution strategy to use
     * @param max_iterations Maximum inference iterations (safety limit)
     * @param enable_tracing Whether to record rule firing trace
     */
    explicit ForwardChainingEngine(
        ConflictResolutionStrategy strategy = ConflictResolutionStrategy::PRIORITY_ORDER,
        std::uint32_t max_iterations = 1000,
        bool enable_tracing = false);

    /**
     * @brief Virtual destructor for proper cleanup
     */
    ~ForwardChainingEngine() override = default;

    // InferenceEngine interface implementation

    /**
     * @brief Execute forward chaining inference
     * @param request Inference request (facts and rules will be extracted)
     * @return Result containing derived facts or error
     */
    auto run_inference(const InferenceRequest& request)
        -> common::Result<InferenceResponse, InferenceError> override;

    /**
     * @brief Get backend information
     * @return String describing the forward chaining backend
     */
    auto get_backend_info() const -> std::string override;

    /**
     * @brief Check if engine is ready for inference
     * @return True if facts and rules are loaded
     */
    auto is_ready() const -> bool override;

    /**
     * @brief Get performance statistics
     * @return String containing detailed performance metrics
     */
    auto get_performance_stats() const -> std::string override;

    // Forward chaining specific interface

    /**
     * @brief Add a single fact to the knowledge base
     * @param fact The fact to add
     * @return Result indicating success or error
     */
    auto add_fact(const common::Fact& fact) -> common::Result<std::monostate, ForwardChainingError>;

    /**
     * @brief Add multiple facts to the knowledge base
     * @param facts Vector of facts to add
     * @return Result indicating success or error
     */
    auto add_facts(const std::vector<common::Fact>& facts)
        -> common::Result<std::monostate, ForwardChainingError>;

    /**
     * @brief Add a single rule to the rule base
     * @param rule The rule to add
     * @return Result indicating success or error
     */
    auto add_rule(const common::Rule& rule) -> common::Result<std::monostate, ForwardChainingError>;

    /**
     * @brief Add multiple rules to the rule base
     * @param rules Vector of rules to add
     * @return Result indicating success or error
     */
    auto add_rules(const std::vector<common::Rule>& rules)
        -> common::Result<std::monostate, ForwardChainingError>;

    /**
     * @brief Run forward chaining inference with current facts and rules
     * @return Result containing derived facts or error
     */
    auto run_forward_chaining() -> common::Result<std::vector<common::Fact>, ForwardChainingError>;

    /**
     * @brief Clear all facts from the knowledge base
     */
    void clear_facts();

    /**
     * @brief Clear all rules from the rule base
     */
    void clear_rules();

    /**
     * @brief Get all facts currently in the knowledge base
     * @return Vector of all facts
     */
    auto get_all_facts() const -> std::vector<common::Fact>;

    /**
     * @brief Get all rules currently in the rule base
     * @return Vector of all rules
     */
    auto get_all_rules() const -> std::vector<common::Rule>;

    /**
     * @brief Get facts matching a specific predicate
     * @param predicate The predicate name to search for
     * @return Vector of matching facts
     */
    auto get_facts_by_predicate(const std::string& predicate) const -> std::vector<common::Fact>;

    /**
     * @brief Get rule firing trace from last inference run
     * @return Vector of rule firing records (empty if tracing disabled)
     */
    auto get_firing_trace() const -> std::vector<RuleFiring>;

    /**
     * @brief Get current inference metrics
     * @return Current performance metrics
     */
    auto get_metrics() const -> InferenceMetrics;

    /**
     * @brief Reset all performance metrics
     */
    void reset_metrics();

    /**
     * @brief Set conflict resolution strategy
     * @param strategy New strategy to use
     */
    void set_conflict_resolution_strategy(ConflictResolutionStrategy strategy);

    /**
     * @brief Enable or disable rule firing tracing
     * @param enable Whether to enable tracing
     */
    void set_tracing_enabled(bool enable);

  private:
    // Internal data structures

    /**
     * @brief Index of facts organized by predicate name for efficient lookup
     */
    std::unordered_map<std::string, std::vector<common::Fact>> fact_index_;

    /**
     * @brief Complete list of all facts in insertion order
     */
    std::vector<common::Fact> all_facts_;

    /**
     * @brief All rules sorted by priority for conflict resolution
     */
    std::vector<common::Rule> rules_;

    /**
     * @brief Configuration and state
     */
    ConflictResolutionStrategy conflict_strategy_;
    std::uint32_t max_iterations_;
    bool tracing_enabled_;

    /**
     * @brief Performance monitoring
     */
    mutable InferenceMetrics metrics_;

    /**
     * @brief Rule firing trace (if tracing enabled)
     */
    std::vector<RuleFiring> firing_trace_;

    // Internal algorithms

    /**
     * @brief Find all rules that can fire with current facts
     * @return Vector of applicable rules with their variable bindings
     */
    auto find_applicable_rules() const -> std::vector<std::pair<common::Rule, VariableBindings>>;

    /**
     * @brief Check if a single rule can fire with current facts
     * @param rule The rule to check
     * @return Optional variable bindings if rule can fire, nullopt otherwise
     */
    auto can_rule_fire(const common::Rule& rule) const -> std::optional<VariableBindings>;

    /**
     * @brief Match a rule condition against available facts
     * @param condition The condition to match
     * @param bindings Current variable bindings (input/output)
     * @return True if condition can be satisfied with bindings
     */
    auto match_condition(const common::Rule::Condition& condition, VariableBindings& bindings) const
        -> bool;

    /**
     * @brief Unify two values, updating variable bindings
     * @param pattern Pattern value (may contain variables)
     * @param instance Concrete value to match against
     * @param bindings Variable bindings to update
     * @return True if unification succeeds
     */
    auto unify_values(const common::Value& pattern,
                      const common::Value& instance,
                      VariableBindings& bindings) const -> bool;

    /**
     * @brief Check if a value is a variable (starts with uppercase)
     * @param value The value to check
     * @return True if value represents a variable
     */
    auto is_variable(const common::Value& value) const -> bool;

    /**
     * @brief Apply variable bindings to substitute variables in conclusions
     * @param conclusion Rule conclusion template
     * @param bindings Variable bindings to apply
     * @return Instantiated fact with variables replaced
     */
    auto instantiate_conclusion(const common::Rule::Conclusion& conclusion,
                                const VariableBindings& bindings) const -> common::Fact;

    /**
     * @brief Fire a rule with given bindings, deriving new facts
     * @param rule The rule to fire
     * @param bindings Variable bindings to use
     * @return Vector of newly derived facts
     */
    auto fire_rule(const common::Rule& rule, const VariableBindings& bindings)
        -> std::vector<common::Fact>;

    /**
     * @brief Apply conflict resolution strategy to order applicable rules
     * @param applicable_rules Rules that can fire with their bindings
     * @return Ordered vector based on conflict resolution strategy
     */
    auto resolve_conflicts(std::vector<std::pair<common::Rule, VariableBindings>>& applicable_rules)
        const -> std::vector<std::pair<common::Rule, VariableBindings>>;

    /**
     * @brief Check if a fact already exists in the knowledge base
     * @param fact The fact to check
     * @return True if fact already exists
     */
    auto fact_exists(const common::Fact& fact) const -> bool;

    /**
     * @brief Update fact index when adding new facts
     * @param fact The fact to index
     */
    void update_fact_index(const common::Fact& fact);

    /**
     * @brief Rebuild complete fact index (used after clearing facts)
     */
    void rebuild_fact_index();

    /**
     * @brief Generate next unique fact ID
     * @return Unique fact identifier
     */
    auto generate_fact_id() const -> std::uint64_t;

    /**
     * @brief Record rule firing for tracing (if enabled)
     * @param rule The rule that fired
     * @param bindings Variable bindings used
     * @param triggering_facts Facts that triggered the rule
     * @param derived_facts New facts derived
     */
    void record_rule_firing(const common::Rule& rule,
                            const VariableBindings& bindings,
                            const std::vector<std::uint64_t>& triggering_facts,
                            const std::vector<common::Fact>& derived_facts);
};

/**
 * @brief Factory function to create a forward chaining engine
 * @param strategy Conflict resolution strategy
 * @param max_iterations Maximum inference iterations
 * @param enable_tracing Whether to enable rule firing trace
 * @return Result containing engine instance or error
 */
auto create_forward_chaining_engine(
    ConflictResolutionStrategy strategy = ConflictResolutionStrategy::PRIORITY_ORDER,
    std::uint32_t max_iterations = 1000,
    bool enable_tracing = false)
    -> common::Result<std::unique_ptr<ForwardChainingEngine>, ForwardChainingError>;

}  // namespace inference_lab::engines
