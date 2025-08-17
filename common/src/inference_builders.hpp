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
 *     .withMetadata("source", Value::fromText("historical_records"))
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
    FactBuilder& withArg(const Value& arg);

    /** @brief Add an int64 argument to the fact */
    FactBuilder& withArg(int64_t value);

    /** @brief Add a double argument to the fact */
    FactBuilder& withArg(double value);

    /** @brief Add a string argument to the fact */
    FactBuilder& withArg(const std::string& value);

    /** @brief Add a C-string argument to the fact (avoids bool conversion) */
    FactBuilder& withArg(const char* value);

    /** @brief Add a boolean argument to the fact */
    FactBuilder& withArg(bool value);

    // Methods for setting optional properties of the fact

    /** @brief Set the unique ID (if not set, auto-generated) */
    FactBuilder& withId(uint64_t id);

    /** @brief Set the confidence level (0.0 to 1.0, default: 1.0) */
    FactBuilder& withConfidence(double confidence);

    /** @brief Set the timestamp (if not set, uses current time) */
    FactBuilder& withTimestamp(uint64_t timestamp);

    /** @brief Add a metadata key-value pair */
    FactBuilder& withMetadata(const std::string& key, const Value& value);

    /**
     * @brief Build the final Fact object
     * @return Completed Fact with all specified properties
     * @throws std::runtime_error if required fields are missing
     */
    Fact build();

  private:
    std::string predicate_;                            ///< Predicate name for the fact
    std::vector<Value> args_;                          ///< Arguments being accumulated
    uint64_t id_ = 0;                                  ///< Fact ID (0 = auto-generate)
    double confidence_ = 1.0;                          ///< Confidence level
    uint64_t timestamp_ = 0;                           ///< Timestamp (0 = current time)
    std::unordered_map<std::string, Value> metadata_;  ///< Metadata key-value pairs

    /** @brief Generate next unique ID for facts */
    static uint64_t nextId();
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
    RuleBuilder& whenCondition(const std::string& predicate,
                               const std::vector<Value>& args,
                               bool negated = false);

    /**
     * @brief Start building a new positive condition
     * @param predicate Predicate name for the condition
     * @return Reference to this builder for chaining withArg() calls
     */
    RuleBuilder& when(const std::string& predicate);

    /**
     * @brief Start building a new negative (NOT) condition
     * @param predicate Predicate name for the negated condition
     * @return Reference to this builder for chaining withArg() calls
     */
    RuleBuilder& whenNot(const std::string& predicate);

    // Methods for adding arguments to the current condition/conclusion being built

    /** @brief Add a Value argument to current condition/conclusion */
    RuleBuilder& withArg(const Value& arg);

    /** @brief Add an int64 argument to current condition/conclusion */
    RuleBuilder& withArg(int64_t value);

    /** @brief Add a double argument to current condition/conclusion */
    RuleBuilder& withArg(double value);

    /** @brief Add a string argument to current condition/conclusion */
    RuleBuilder& withArg(const std::string& value);

    /** @brief Add a C-string argument to current condition/conclusion */
    RuleBuilder& withArg(const char* value);

    /** @brief Add a boolean argument to current condition/conclusion */
    RuleBuilder& withArg(bool value);

    /**
     * @brief Add a logic variable to current condition/conclusion
     * @param varName Variable name (will be converted to uppercase if needed)
     *
     * Variables are used to bind values across conditions and conclusions.
     * For example, variable "X" in condition isHuman(X) will be bound to the same
     * value as "X" in conclusion isMortal(X).
     */
    RuleBuilder& withVariable(const std::string& varName);

    // Methods for adding complete conclusions

    /**
     * @brief Add a complete conclusion to the rule
     * @param predicate Predicate name to assert
     * @param args Arguments for the conclusion
     * @param confidence Confidence level for this conclusion
     */
    RuleBuilder& thenConclusion(const std::string& predicate,
                                const std::vector<Value>& args,
                                double confidence = 1.0);

    /**
     * @brief Start building a new conclusion
     * @param predicate Predicate name for the conclusion
     * @return Reference to this builder for chaining withArg() calls
     */
    RuleBuilder& then(const std::string& predicate);

    // Methods for setting rule properties

    /** @brief Set the unique ID (if not set, auto-generated) */
    RuleBuilder& withId(uint64_t id);

    /** @brief Set the priority for conflict resolution (higher = more important) */
    RuleBuilder& withPriority(int32_t priority);

    /** @brief Set the overall confidence level for the rule */
    RuleBuilder& withConfidence(double confidence);

    /**
     * @brief Build the final Rule object
     * @return Completed Rule with all conditions and conclusions
     * @throws std::runtime_error if rule has no conditions or no conclusions
     */
    Rule build();

  private:
    std::string name_;                           ///< Human-readable rule name
    std::vector<Rule::Condition> conditions_;    ///< Completed conditions
    std::vector<Rule::Conclusion> conclusions_;  ///< Completed conclusions
    uint64_t id_ = 0;                            ///< Rule ID (0 = auto-generate)
    int32_t priority_ = 0;                       ///< Priority for conflict resolution
    double confidence_ = 1.0;                    ///< Overall rule confidence

    // State for building the current condition or conclusion
    std::string currentPredicate_;    ///< Predicate being built
    std::vector<Value> currentArgs_;  ///< Arguments being accumulated
    bool currentNegated_ = false;     ///< Whether current condition is negated
    double currentConfidence_ = 1.0;  ///< Confidence for current conclusion

    /** @brief Tracks whether we're currently building a condition or conclusion */
    enum class BuildingState { Condition, Conclusion, None } buildingState_ = BuildingState::None;

    /** @brief Finish building current condition and add it to the conditions list */
    void finishCurrentCondition();

    /** @brief Finish building current conclusion and add it to the conclusions list */
    void finishCurrentConclusion();

    /** @brief Generate next unique ID for rules */
    static uint64_t nextId();
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
    QueryBuilder& goal(const std::string& predicate,
                       const std::vector<Value>& args,
                       bool negated = false);

    /**
     * @brief Start building a goal pattern
     * @param predicate Predicate name for the goal
     * @return Reference to this builder for chaining withArg() calls
     */
    QueryBuilder& goal(const std::string& predicate);

    // Methods for adding arguments to the goal pattern

    /** @brief Add a Value argument to the goal */
    QueryBuilder& withArg(const Value& arg);

    /** @brief Add an int64 argument to the goal */
    QueryBuilder& withArg(int64_t value);

    /** @brief Add a double argument to the goal */
    QueryBuilder& withArg(double value);

    /** @brief Add a string argument to the goal */
    QueryBuilder& withArg(const std::string& value);

    /** @brief Add a C-string argument to the goal */
    QueryBuilder& withArg(const char* value);

    /** @brief Add a boolean argument to the goal */
    QueryBuilder& withArg(bool value);

    /**
     * @brief Add a logic variable to the goal
     * @param varName Variable name (will be converted to uppercase if needed)
     */
    QueryBuilder& withVariable(const std::string& varName);

    // Methods for setting query parameters

    /** @brief Set the unique ID (if not set, auto-generated) */
    QueryBuilder& withId(uint64_t id);

    /** @brief Set the maximum number of results to return */
    QueryBuilder& maxResults(uint32_t max);

    /** @brief Set the query timeout in milliseconds */
    QueryBuilder& timeout(uint32_t timeoutMs);

    /** @brief Add a metadata key-value pair */
    QueryBuilder& withMetadata(const std::string& key, const Value& value);

    /**
     * @brief Build the final Query object
     * @return Completed Query with goal and parameters
     * @throws std::runtime_error if no goal is set
     */
    Query build();

    // Static factory methods for common query types - these provide convenient shortcuts

    /** @brief Create a FindAll query builder */
    static QueryBuilder findAll();

    /** @brief Create a Prove query builder */
    static QueryBuilder prove();

    /** @brief Create a FindFirst query builder with specified limit */
    static QueryBuilder findFirst(uint32_t limit = 1);

    /** @brief Create an Explain query builder */
    static QueryBuilder explain();

  private:
    Query::Type type_;                                 ///< Type of query to create
    uint64_t id_ = 0;                                  ///< Query ID (0 = auto-generate)
    uint32_t maxResults_ = 100;                        ///< Maximum results to return
    uint32_t timeoutMs_ = 5000;                        ///< Timeout in milliseconds
    std::unordered_map<std::string, Value> metadata_;  ///< Metadata key-value pairs

    // State for building the goal pattern
    std::string goalPredicate_;    ///< Goal predicate being built
    std::vector<Value> goalArgs_;  ///< Goal arguments being accumulated
    bool goalNegated_ = false;     ///< Whether goal is negated
    bool goalSet_ = false;         ///< Whether goal has been set

    /** @brief Generate next unique ID for queries */
    static uint64_t nextId();
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
inline Fact fact(const std::string& predicate) {
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
template <typename... ArgumentTypes>
inline Fact fact(const std::string& predicate, ArgumentTypes&&... args) {
    FactBuilder builder(predicate);
    (builder.withArg(std::forward<ArgumentTypes>(args)), ...);
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
inline RuleBuilder rule(const std::string& name) {
    return RuleBuilder(name);
}

// Factory functions for creating queries

/**
 * @brief Create a FindAll query builder with a simple goal
 * @param predicate Goal predicate name
 * @return QueryBuilder for further configuration
 */
inline QueryBuilder findAll(const std::string& predicate) {
    return QueryBuilder::findAll().goal(predicate);
}

/**
 * @brief Create a FindAll query builder with goal and arguments
 * @tparam Args Types of the arguments (automatically deduced)
 * @param predicate Goal predicate name
 * @param args Arguments for the goal
 * @return QueryBuilder for further configuration
 */
template <typename... ArgumentTypes>
inline QueryBuilder findAll(const std::string& predicate, ArgumentTypes&&... args) {
    auto builder = QueryBuilder::findAll().goal(predicate);
    (builder.withArg(std::forward<ArgumentTypes>(args)), ...);
    return builder;
}

/**
 * @brief Create a Prove query builder with a simple goal
 * @param predicate Goal predicate name
 * @return QueryBuilder for further configuration
 */
inline QueryBuilder prove(const std::string& predicate) {
    return QueryBuilder::prove().goal(predicate);
}

/**
 * @brief Create a Prove query builder with goal and arguments
 * @tparam Args Types of the arguments (automatically deduced)
 * @param predicate Goal predicate name
 * @param args Arguments for the goal
 * @return QueryBuilder for further configuration
 */
template <typename... ArgumentTypes>
inline QueryBuilder prove(const std::string& predicate, ArgumentTypes&&... args) {
    auto builder = QueryBuilder::prove().goal(predicate);
    (builder.withArg(std::forward<ArgumentTypes>(args)), ...);
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
inline Value var(const std::string& name) {
    // Variables start with uppercase by convention
    std::string varName = name;
    if (!varName.empty() && std::islower(varName[0])) {
        varName[0] = std::toupper(varName[0]);
    }
    return Value::fromText(varName);
}

}  // namespace builders

}  // namespace inference_lab::common
