// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_builders.hpp
 * @brief Fluent builder interfaces for inference engine types
 *
 * This file provides builder classes that use the fluent interface pattern to make
 * creating complex inference types (Facts, Rules, Queries) much easier and more readable.
 * Instead of constructing objects with many parameters, builders allow step-by-step
 * construction with method chaining.
 *
 * Example usage:
 * ```cpp
 * auto fact = FactBuilder("isHuman")
 *     .withArg("socrates")
 *     .withConfidence(0.95)
 *     .build();
 *
 * auto rule = RuleBuilder("mortality_rule")
 *     .when("isHuman").withVariable("X")
 *     .then("isMortal").withVariable("X")
 *     .build();
 * ```
 *
 * All builders are designed to be safe - they will throw exceptions if you try to
 * build incomplete or invalid objects.
 */

#pragma once

#include <cstdint>

#include "inference_types.hpp"

namespace inference_lab::common {

/**
 * @class FactBuilder
 * @brief Fluent builder for creating Fact objects with method chaining
 *
 * This builder simplifies the creation of Fact objects by providing a fluent interface
 * that allows adding arguments and setting properties step by step. The builder ensures
 * that all required fields are set and provides sensible defaults for optional fields.
 *
 * Example:
 * ```cpp
 * auto fact = FactBuilder("livesIn")
 *     .withArg("socrates")
 *     .withArg("athens")
 *     .withConfidence(0.95)
 *     .with_metadata("source", Value::from_text("historical_records"))
 *     .build();
 * ```
 */
class FactBuilder {
  public:
    /**
     * @brief Construct a new FactBuilder with the given predicate
     * @param predicate The predicate name for the fact (e.g., "isHuman", "livesIn")
     */
    explicit FactBuilder(const std::string& predicate);

    // Methods for adding arguments to the fact
    // These can be chained together: .withArg(arg1).withArg(arg2).withArg(arg3)

    /** @brief Add a Value argument to the fact */
    auto with_arg(const Value& arg) -> FactBuilder&;

    /** @brief Add an int64 argument to the fact */
    auto with_arg(int64_t value) -> FactBuilder&;

    /** @brief Add a double argument to the fact */
    auto with_arg(double value) -> FactBuilder&;

    /** @brief Add a string argument to the fact */
    auto with_arg(const std::string& value) -> FactBuilder&;

    /** @brief Add a C-string argument to the fact (avoids bool conversion) */
    auto with_arg(const char* value) -> FactBuilder&;

    /** @brief Add a boolean argument to the fact */
    auto with_arg(bool value) -> FactBuilder&;

    // Methods for setting optional properties of the fact

    /** @brief Set the unique ID (if not set, auto-generated) */
    auto with_id(uint64_t id) -> FactBuilder&;

    /** @brief Set the confidence level (0.0 to 1.0, default: 1.0) */
    auto with_confidence(double confidence) -> FactBuilder&;

    /** @brief Set the timestamp (if not set, uses current time) */
    auto with_timestamp(uint64_t timestamp) -> FactBuilder&;

    /** @brief Add a metadata key-value pair */
    auto with_metadata(const std::string& key, const Value& value) -> FactBuilder&;

    /**
     * @brief Build the final Fact object
     * @return Completed Fact with all specified properties
     * @throws std::runtime_error if required fields are missing
     */
    auto build() -> Fact;

  private:
    std::string predicate_;                            ///< Predicate name for the fact
    std::vector<Value> args_;                          ///< Arguments being accumulated
    uint64_t id_ = 0;                                  ///< Fact ID (0 = auto-generate)
    double confidence_ = 1.0;                          ///< Confidence level
    uint64_t timestamp_ = 0;                           ///< Timestamp (0 = current time)
    std::unordered_map<std::string, Value> metadata_;  ///< Metadata key-value pairs

    /** @brief Generate next unique ID for facts */
    static auto next_id() -> uint64_t;
};

/**
 * @class RuleBuilder
 * @brief Fluent builder for creating Rule objects with complex condition/conclusion chains
 *
 * This builder provides an intuitive way to construct rules by chaining method calls.
 * It supports building complex rules with multiple conditions and conclusions.
 * The builder maintains state about the current condition or conclusion being built,
 * allowing arguments to be added incrementally.
 *
 * Example:
 * ```cpp
 * auto rule = RuleBuilder("mortality_rule")
 *     .when("isHuman").withVariable("X")           // First condition
 *     .when("age").withVariable("X").withArg(60)    // Second condition
 *     .then("isMortal").withVariable("X")          // First conclusion
 *     .withPriority(10)
 *     .build();
 * ```
 */
class RuleBuilder {
  public:
    /**
     * @brief Construct a new RuleBuilder with the given name
     * @param name Human-readable name for the rule (for debugging)
     */
    explicit RuleBuilder(const std::string& name);

    // Methods for adding complete conditions (all arguments at once)

    /**
     * @brief Add a complete condition to the rule
     * @param predicate Predicate name to match
     * @param args Arguments for the condition
     * @param negated Whether this is a NOT condition
     */
    auto when_condition(const std::string& predicate,
                        const std::vector<Value>& args,
                        bool negated = false) -> RuleBuilder&;

    /**
     * @brief Start building a new positive condition
     * @param predicate Predicate name for the condition
     * @return Reference to this builder for chaining withArg() calls
     */
    auto when(const std::string& predicate) -> RuleBuilder&;

    /**
     * @brief Start building a new negative (NOT) condition
     * @param predicate Predicate name for the negated condition
     * @return Reference to this builder for chaining withArg() calls
     */
    auto when_not(const std::string& predicate) -> RuleBuilder&;

    // Methods for adding arguments to the current condition/conclusion being built

    /** @brief Add a Value argument to current condition/conclusion */
    auto with_arg(const Value& arg) -> RuleBuilder&;

    /** @brief Add an int64 argument to current condition/conclusion */
    auto with_arg(int64_t value) -> RuleBuilder&;

    /** @brief Add a double argument to current condition/conclusion */
    auto with_arg(double value) -> RuleBuilder&;

    /** @brief Add a string argument to current condition/conclusion */
    auto with_arg(const std::string& value) -> RuleBuilder&;

    /** @brief Add a C-string argument to current condition/conclusion */
    auto with_arg(const char* value) -> RuleBuilder&;

    /** @brief Add a boolean argument to current condition/conclusion */
    auto with_arg(bool value) -> RuleBuilder&;

    /**
     * @brief Add a logic variable to current condition/conclusion
     * @param varName Variable name (will be converted to uppercase if needed)
     *
     * Variables are used to bind values across conditions and conclusions.
     * For example, variable "X" in condition isHuman(X) will be bound to the same
     * value as "X" in conclusion isMortal(X).
     */
    auto with_variable(const std::string& var_name) -> RuleBuilder&;

    // Methods for adding complete conclusions

    /**
     * @brief Add a complete conclusion to the rule
     * @param predicate Predicate name to assert
     * @param args Arguments for the conclusion
     * @param confidence Confidence level for this conclusion
     */
    auto then_conclusion(const std::string& predicate,
                         const std::vector<Value>& args,
                         double confidence = 1.0) -> RuleBuilder&;

    /**
     * @brief Start building a new conclusion
     * @param predicate Predicate name for the conclusion
     * @return Reference to this builder for chaining withArg() calls
     */
    auto then(const std::string& predicate) -> RuleBuilder&;

    // Methods for setting rule properties

    /** @brief Set the unique ID (if not set, auto-generated) */
    auto with_id(uint64_t id) -> RuleBuilder&;

    /** @brief Set the priority for conflict resolution (higher = more important) */
    auto with_priority(int32_t priority) -> RuleBuilder&;

    /** @brief Set the overall confidence level for the rule */
    auto with_confidence(double confidence) -> RuleBuilder&;

    /**
     * @brief Build the final Rule object
     * @return Completed Rule with all conditions and conclusions
     * @throws std::runtime_error if rule has no conditions or no conclusions
     */
    auto build() -> Rule;

  private:
    std::string name_;                           ///< Human-readable rule name
    std::vector<Rule::Condition> conditions_;    ///< Completed conditions
    std::vector<Rule::Conclusion> conclusions_;  ///< Completed conclusions
    uint64_t id_ = 0;                            ///< Rule ID (0 = auto-generate)
    int32_t priority_ = 0;                       ///< Priority for conflict resolution
    double confidence_ = 1.0;                    ///< Overall rule confidence

    // State for building the current condition or conclusion
    std::string current_predicate_;    ///< Predicate being built
    std::vector<Value> current_args_;  ///< Arguments being accumulated
    bool current_negated_ = false;     ///< Whether current condition is negated
    double current_confidence_ = 1.0;  ///< Confidence for current conclusion

    /** @brief Tracks whether we're currently building a condition or conclusion */
    enum class BuildingState : std::uint8_t {  // NOLINT(performance-enum-size) - false positive,
                                               // uint8_t is correct
        CONDITION,
        CONCLUSION,
        NONE
    } building_state_ = BuildingState::NONE;

    /** @brief Finish building current condition and add it to the conditions list */
    void finish_current_condition();

    /** @brief Finish building current conclusion and add it to the conclusions list */
    void finish_current_conclusion();

    /** @brief Generate next unique ID for rules */
    static auto next_id() -> uint64_t;
};

/**
 * @class QueryBuilder
 * @brief Fluent builder for creating Query objects with goal patterns
 *
 * This builder simplifies creating queries for the inference engine. It provides
 * static factory methods for common query types and allows setting the goal pattern
 * and query parameters through method chaining.
 *
 * Example:
 * ```cpp
 * auto query = QueryBuilder::findAll()
 *     .goal("isHuman").withVariable("X")
 *     .maxResults(50)
 *     .timeout(10000)
 *     .build();
 * ```
 */
class QueryBuilder {
  public:
    /**
     * @brief Construct a QueryBuilder with the specified query type
     * @param type The type of query to create (FindAll, Prove, FindFirst, Explain)
     */
    explicit QueryBuilder(Query::Type type);

    // Methods for setting the goal pattern

    /**
     * @brief Set a complete goal pattern for the query
     * @param predicate Predicate name to search for
     * @param args Arguments for the goal pattern
     * @param negated Whether this is a negated goal (NOT pattern)
     */
    auto goal(const std::string& predicate, const std::vector<Value>& args, bool negated = false)
        -> QueryBuilder&;

    /**
     * @brief Start building a goal pattern
     * @param predicate Predicate name for the goal
     * @return Reference to this builder for chaining withArg() calls
     */
    auto goal(const std::string& predicate) -> QueryBuilder&;

    // Methods for adding arguments to the goal pattern

    /** @brief Add a Value argument to the goal */
    auto with_arg(const Value& arg) -> QueryBuilder&;

    /** @brief Add an int64 argument to the goal */
    auto with_arg(int64_t value) -> QueryBuilder&;

    /** @brief Add a double argument to the goal */
    auto with_arg(double value) -> QueryBuilder&;

    /** @brief Add a string argument to the goal */
    auto with_arg(const std::string& value) -> QueryBuilder&;

    /** @brief Add a C-string argument to the goal */
    auto with_arg(const char* value) -> QueryBuilder&;

    /** @brief Add a boolean argument to the goal */
    auto with_arg(bool value) -> QueryBuilder&;

    /**
     * @brief Add a logic variable to the goal
     * @param varName Variable name (will be converted to uppercase if needed)
     */
    auto with_variable(const std::string& var_name) -> QueryBuilder&;

    // Methods for setting query parameters

    /** @brief Set the unique ID (if not set, auto-generated) */
    auto with_id(uint64_t id) -> QueryBuilder&;

    /** @brief Set the maximum number of results to return */
    auto max_results(uint32_t max) -> QueryBuilder&;

    /** @brief Set the query timeout in milliseconds */
    auto timeout(uint32_t timeout_ms) -> QueryBuilder&;

    /** @brief Add a metadata key-value pair */
    auto with_metadata(const std::string& key, const Value& value) -> QueryBuilder&;

    /**
     * @brief Build the final Query object
     * @return Completed Query with goal and parameters
     * @throws std::runtime_error if no goal is set
     */
    auto build() -> Query;

    // Static factory methods for common query types - these provide convenient shortcuts

    /** @brief Create a FindAll query builder */
    static auto find_all() -> QueryBuilder;

    /** @brief Create a Prove query builder */
    static auto prove() -> QueryBuilder;

    /** @brief Create a FindFirst query builder with specified limit */
    static auto find_first(uint32_t limit = 1) -> QueryBuilder;

    /** @brief Create an Explain query builder */
    static auto explain() -> QueryBuilder;

  private:
    Query::Type type_;                                 ///< Type of query to create
    uint64_t id_ = 0;                                  ///< Query ID (0 = auto-generate)
    uint32_t max_results_ = 100;                       ///< Maximum results to return
    uint32_t timeout_ms_ = 5000;                       ///< Timeout in milliseconds
    std::unordered_map<std::string, Value> metadata_;  ///< Metadata key-value pairs

    // State for building the goal pattern
    std::string goal_predicate_;    ///< Goal predicate being built
    std::vector<Value> goal_args_;  ///< Goal arguments being accumulated
    bool goal_negated_ = false;     ///< Whether goal is negated
    bool goal_set_ = false;         ///< Whether goal has been set

    /** @brief Generate next unique ID for queries */
    static auto next_id() -> uint64_t;
};

/**
 * @namespace builders
 * @brief Convenience factory functions for common inference type creation patterns
 *
 * This namespace provides simple function templates that make creating common
 * inference types even easier than using the builders directly. These are
 * particularly useful for simple cases where you don't need the full builder
 * functionality.
 *
 * Examples:
 * ```cpp
 * auto fact = builders::fact("isHuman", "socrates");
 * auto rule = builders::rule("my_rule").when("A").then("B").build();
 * auto query = builders::findAll("isHuman", builders::var("X")).build();
 * ```
 */
namespace builders {

// Factory functions for creating simple facts

/**
 * @brief Create a simple fact with no arguments
 * @param predicate Predicate name for the fact
 * @return Completed Fact object
 */
inline auto fact(const std::string& predicate) -> Fact {
    return FactBuilder(predicate).build();
}

/**
 * @brief Create a fact with multiple arguments using variadic templates
 * @tparam Args Types of the arguments (automatically deduced)
 * @param predicate Predicate name for the fact
 * @param args Arguments to add to the fact
 * @return Completed Fact object
 *
 * This template function allows creating facts with any number of arguments
 * in a single call, e.g., fact("livesIn", "socrates", "athens")
 */
template <typename... ArgumentTypes>  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
inline auto fact(const std::string& predicate, ArgumentTypes&&... args) -> Fact {
    FactBuilder builder(predicate);
    (builder.with_arg(std::forward<ArgumentTypes>(args)), ...);
    return builder.build();
}

// Factory functions for creating rules

/**
 * @brief Create a rule builder with the given name
 * @param name Human-readable rule name
 * @return RuleBuilder for further configuration
 *
 * This is just a convenience wrapper around RuleBuilder constructor.
 */
inline auto rule(const std::string& name) -> RuleBuilder {
    return RuleBuilder(name);
}

// Factory functions for creating queries

/**
 * @brief Create a FindAll query builder with a simple goal
 * @param predicate Goal predicate name
 * @return QueryBuilder for further configuration
 */
inline auto find_all(const std::string& predicate) -> QueryBuilder {
    return QueryBuilder::find_all().goal(predicate);
}

/**
 * @brief Create a FindAll query builder with goal and arguments
 * @tparam Args Types of the arguments (automatically deduced)
 * @param predicate Goal predicate name
 * @param args Arguments for the goal
 * @return QueryBuilder for further configuration
 */
template <typename... ArgumentTypes>  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
inline auto find_all(const std::string& predicate, ArgumentTypes&&... args) -> QueryBuilder {
    auto builder = QueryBuilder::find_all().goal(predicate);
    (builder.with_arg(std::forward<ArgumentTypes>(args)), ...);
    return builder;
}

/**
 * @brief Create a Prove query builder with a simple goal
 * @param predicate Goal predicate name
 * @return QueryBuilder for further configuration
 */
inline auto prove(const std::string& predicate) -> QueryBuilder {
    return QueryBuilder::prove().goal(predicate);
}

/**
 * @brief Create a Prove query builder with goal and arguments
 * @tparam Args Types of the arguments (automatically deduced)
 * @param predicate Goal predicate name
 * @param args Arguments for the goal
 * @return QueryBuilder for further configuration
 */
template <typename... ArgumentTypes>  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
inline auto prove(const std::string& predicate, ArgumentTypes&&... args) -> QueryBuilder {
    auto builder = QueryBuilder::prove().goal(predicate);
    (builder.with_arg(std::forward<ArgumentTypes>(args)), ...);
    return builder;
}

// Utility functions

/**
 * @brief Create a logic variable Value with proper naming convention
 * @param name Variable name (will be converted to uppercase if needed)
 * @return Value representing a logic variable
 *
 * Logic variables are used in rules and queries to bind values across
 * conditions and conclusions. By convention, variables start with uppercase
 * letters to distinguish them from constants.
 */
inline auto var(const std::string& name) -> Value {
    // Variables start with uppercase by convention
    std::string var_name = name;
    if (!var_name.empty() && std::islower(var_name[0])) {
        var_name[0] = std::toupper(var_name[0]);
    }
    return Value::from_text(var_name);
}

}  // namespace builders

}  // namespace inference_lab::common
