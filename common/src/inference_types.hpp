// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file inference_types.hpp
 * @brief C++ wrapper interface for Cap'n Proto inference engine types
 *
 * This file provides a clean, type-safe C++ interface over the generated Cap'n Proto
 * schema for inference engine data types. It abstracts away the low-level Cap'n Proto
 * API and provides intuitive C++ classes for working with facts, rules, queries, and values.
 *
 * Key features:
 * - Type-safe value handling with automatic conversions
 * - RAII-compliant resource management
 * - Fluent builder interfaces for easy object construction
 * - Seamless serialization/deserialization with Cap'n Proto
 * - Human-readable string representations for debugging
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Include generated Cap'n Proto headers

#include "schemas/inference_types.capnp.h"

namespace inference_lab::common {

// Forward declarations for builder classes
class ValueBuilder;
class FactBuilder;
class RuleBuilder;
class QueryBuilder;

/**
 * @class Value
 * @brief C++ wrapper for Cap'n Proto Value type - represents a polymorphic value in the inference
 * engine
 *
 * This class provides a type-safe wrapper around Cap'n Proto's Value union type, which can hold
 * different primitive and complex data types used throughout the inference engine. It supports:
 * - Primitive types: int64, float64, text, bool
 * - Complex types: lists of values, structured objects (key-value maps)
 *
 * The class uses a discriminated union approach with type checking and safe extraction methods
 * to prevent runtime type errors common with raw Cap'n Proto usage.
 */
class Value {
  public:
    /**
     * @brief Default constructor - creates an int64 value initialized to 0
     *
     * This allows Value objects to be used in STL containers that require
     * default-constructible types (like std::unordered_map, std::vector).
     */
    Value() : type_(Type::INT64), int64_value_(0), text_value_{}, list_value_{}, struct_value_{} {}

    // Factory methods for creating typed values - these are preferred over constructors
    // to make the type creation explicit and prevent accidental conversions

    /** @brief Create a 64-bit signed integer value */
    static auto from_int64(int64_t value) -> Value;

    /** @brief Create a 64-bit floating point value */
    static auto from_float64(double value) -> Value;

    /** @brief Create a text/string value */
    static auto from_text(const std::string& value) -> Value;

    /** @brief Create a boolean value */
    static auto from_bool(bool value) -> Value;

    /** @brief Create a list/array value containing other Values */
    static auto from_list(const std::vector<Value>& values) -> Value;

    /** @brief Create a structured object value (key-value map) */
    static auto from_struct(const std::unordered_map<std::string, Value>& fields) -> Value;

    // Type checking methods - these are const and do not throw

    /** @brief Check if this value contains a 64-bit integer */
    auto is_int64() const -> bool;

    /** @brief Check if this value contains a 64-bit float */
    auto is_float64() const -> bool;

    /** @brief Check if this value contains text/string data */
    auto is_text() const -> bool;

    /** @brief Check if this value contains a boolean */
    auto is_bool() const -> bool;

    /** @brief Check if this value contains a list of other values */
    auto is_list() const -> bool;

    /** @brief Check if this value contains a structured object */
    auto is_struct() const -> bool;

    // Unsafe value extraction methods - these throw std::runtime_error if type doesn't match
    // Use these when you're certain of the type or want to fail fast on type mismatches

    /** @brief Extract int64 value - throws if not an int64 */
    auto as_int64() const -> int64_t;

    /** @brief Extract float64 value - throws if not a float64 */
    auto as_float64() const -> double;

    /** @brief Extract text value - throws if not text */
    auto as_text() const -> std::string;

    /** @brief Extract bool value - throws if not a bool */
    auto as_bool() const -> bool;

    /** @brief Extract list value - throws if not a list */
    auto as_list() const -> std::vector<Value>;

    /** @brief Extract struct value - throws if not a struct */
    auto as_struct() const -> std::unordered_map<std::string, Value>;

    // Safe value extraction methods - these return std::nullopt if type doesn't match
    // Use these when you want to handle type mismatches gracefully

    /** @brief Safely extract int64 value - returns nullopt if wrong type */
    auto try_as_int64() const -> std::optional<int64_t>;

    /** @brief Safely extract float64 value - returns nullopt if wrong type */
    auto try_as_float64() const -> std::optional<double>;

    /** @brief Safely extract text value - returns nullopt if wrong type */
    auto try_as_text() const -> std::optional<std::string>;

    /** @brief Safely extract bool value - returns nullopt if wrong type */
    auto try_as_bool() const -> std::optional<bool>;

    /** @brief Safely extract list value - returns nullopt if wrong type */
    auto try_as_list() const -> std::optional<std::vector<Value>>;

    /** @brief Safely extract struct value - returns nullopt if wrong type */
    auto try_as_struct() const -> std::optional<std::unordered_map<std::string, Value>>;

    /**
     * @brief Generate human-readable string representation for debugging
     * @return String representation that shows both type and value
     */
    auto to_string() const -> std::string;

    // Cap'n Proto interoperability methods

    /**
     * @brief Construct Value from Cap'n Proto reader
     * @param reader Cap'n Proto Value reader to read from
     *
     * This constructor allows seamless conversion from Cap'n Proto's
     * native types to our C++ wrapper objects.
     */
    explicit Value(schemas::Value::Reader reader);

    /**
     * @brief Write this Value to a Cap'n Proto builder
     * @param builder Cap'n Proto Value builder to write to
     *
     * This method allows seamless conversion from our C++ wrapper
     * back to Cap'n Proto's native format for serialization.
     */
    auto write_to(schemas::Value::Builder builder) const -> void;

  private:
    /**
     * @brief Internal type discriminator enum
     *
     * This enum tracks which type of value is currently stored in the Value object.
     * It's used internally for type checking and safe casting operations.
     */
    enum class Type : std::uint8_t { INT64, FLOAT64, TEXT, BOOL, LIST, STRUCT };

    /** @brief Current type of the stored value */
    Type type_;

    /**
     * @brief Anonymous union for storing primitive types efficiently
     *
     * Only one of these will be active at a time, based on the type_ field.
     * This saves memory compared to storing all possible values separately.
     */
    union {
        int64_t int64_value_{};  ///< Storage for 64-bit integers
        double float64_value_;   ///< Storage for 64-bit floats
        bool bool_value_;        ///< Storage for boolean values
    };

    // Complex types are stored separately since they can't be in unions
    std::string text_value_{};                               ///< Storage for text/string values
    std::vector<Value> list_value_{};                        ///< Storage for list values
    std::unordered_map<std::string, Value> struct_value_{};  ///< Storage for structured object
                                                             ///< values

    /**
     * @brief Private constructor for internal use
     * @param type The type of value to create
     *
     * This constructor is used internally by the factory methods to create
     * appropriately typed Value objects.
     */
    explicit Value(Type type)
        : type_(type), int64_value_{}, text_value_{}, list_value_{}, struct_value_{} {}
};

/**
 * @class Fact
 * @brief C++ wrapper for Cap'n Proto Fact type - represents a single fact in the knowledge base
 *
 * A Fact represents a piece of knowledge in the inference engine, consisting of:
 * - A predicate name (e.g., "isHuman", "livesIn", "hasProperty")
 * - A list of arguments/parameters (e.g., for "isHuman(socrates)", args = ["socrates"])
 * - Optional confidence level for probabilistic reasoning
 * - Timestamp indicating when the fact was asserted
 * - Optional metadata for additional information
 *
 * Facts are the basic building blocks of the knowledge base and are used by rules
 * for pattern matching and inference.
 */
class Fact {
  public:
    /**
     * @brief Construct a new Fact
     * @param id Unique identifier for this fact
     * @param predicate Predicate name (e.g., "isHuman", "livesIn")
     * @param args Arguments for the predicate
     * @param confidence Confidence level (0.0 to 1.0, default: 1.0)
     * @param timestamp Unix timestamp in milliseconds (0 = current time)
     */
    Fact(uint64_t id,
         const std::string& predicate,
         const std::vector<Value>& args,
         double confidence = 1.0,
         uint64_t timestamp = 0);

    // Accessor methods - all const to ensure facts are immutable after creation

    /** @brief Get the unique ID of this fact */
    auto get_id() const -> uint64_t { return id_; }

    /** @brief Get the predicate name */
    auto get_predicate() const -> const std::string& { return predicate_; }

    /** @brief Get the arguments list */
    auto get_args() const -> const std::vector<Value>& { return args_; }

    /** @brief Get the confidence level (0.0 to 1.0) */
    auto get_confidence() const -> double { return confidence_; }

    /** @brief Get the timestamp when this fact was created */
    auto get_timestamp() const -> uint64_t { return timestamp_; }

    /** @brief Get all metadata as a key-value map */
    auto get_metadata() const -> const std::unordered_map<std::string, Value>& { return metadata_; }

    /** @brief Get the schema version this fact was created with (returns empty string if not set)
     */
    auto get_schema_version_string() const -> std::string { return schema_version_string_; }

    // Metadata management methods

    /**
     * @brief Set a metadata key-value pair
     * @param key Metadata key name
     * @param value Metadata value
     *
     * Metadata can store additional information about facts, such as source,
     * creation context, or other properties not part of the core fact structure.
     */
    auto set_metadata(const std::string& key, const Value& value) -> void;

    /**
     * @brief Get a specific metadata value by key
     * @param key Metadata key to look up
     * @return The metadata value if found, nullopt otherwise
     */
    auto get_metadata(const std::string& key) const -> std::optional<Value>;

    /**
     * @brief Generate human-readable string representation
     * @return String in the format "predicate(arg1, arg2, ...) [confidence: X]"
     *
     * This is primarily used for debugging and logging purposes.
     */
    auto to_string() const -> std::string;

    // Cap'n Proto interoperability methods

    /**
     * @brief Construct Fact from Cap'n Proto reader
     * @param reader Cap'n Proto Fact reader to deserialize from
     */
    explicit Fact(schemas::Fact::Reader reader);

    /**
     * @brief Write this Fact to a Cap'n Proto builder
     * @param builder Cap'n Proto Fact builder to serialize to
     */
    auto write_to(schemas::Fact::Builder builder) const -> void;

  private:
    uint64_t id_;                                      ///< Unique identifier for this fact
    std::string predicate_;                            ///< Predicate name (e.g., "isHuman")
    std::vector<Value> args_;                          ///< Arguments for the predicate
    double confidence_;                                ///< Confidence level (0.0 to 1.0)
    uint64_t timestamp_;                               ///< Creation timestamp in milliseconds
    std::unordered_map<std::string, Value> metadata_;  ///< Additional metadata key-value pairs
    std::string schema_version_string_;  ///< Schema version this fact was created with (as string)
};

/**
 * @class Rule
 * @brief C++ wrapper for Cap'n Proto Rule type - represents an inference rule
 *
 * A Rule defines logical inference patterns in the form "IF conditions THEN conclusions".
 * Rules are used by the inference engine to derive new facts from existing ones.
 * Each rule consists of:
 * - One or more conditions that must be satisfied (AND logic)
 * - One or more conclusions to assert if conditions are met
 * - Optional priority for conflict resolution
 * - Optional confidence level for the rule
 *
 * Example: "IF isHuman(X) AND livesIn(X, athens) THEN isPhilosopher(X)"
 */
class Rule {
  public:
    /**
     * @struct Condition
     * @brief Represents a condition within a rule that must be satisfied
     *
     * Conditions are patterns that the inference engine tries to match against
     * existing facts in the knowledge base. Variables (uppercase names) can be
     * used to bind values across multiple conditions and conclusions.
     */
    struct Condition {
        std::string predicate_{};    ///< Predicate to match (e.g., "isHuman")
        std::vector<Value> args_{};  ///< Arguments with variables and constants
        bool negated_{false};        ///< Whether this is a NOT condition

        /** @brief Generate string representation of this condition */
        auto to_string() const -> std::string;
    };

    /**
     * @struct Conclusion
     * @brief Represents a conclusion that will be asserted if rule conditions are met
     *
     * Conclusions define what new facts should be created when a rule fires.
     * They can reference variables bound in the conditions.
     */
    struct Conclusion {
        std::string predicate_{};    ///< Predicate to assert (e.g., "isMortal")
        std::vector<Value> args_{};  ///< Arguments (may contain variables from conditions)
        double confidence_{1.0};     ///< Confidence for this conclusion

        /** @brief Generate string representation of this conclusion */
        auto to_string() const -> std::string;
    };

    /**
     * @brief Construct a new Rule
     * @param id Unique identifier for this rule
     * @param name Human-readable name for debugging
     * @param conditions List of conditions that must be satisfied (AND logic)
     * @param conclusions List of conclusions to assert if conditions are met
     * @param priority Rule priority for conflict resolution (higher = more important)
     * @param confidence Overall confidence for this rule (0.0 to 1.0)
     */
    Rule(uint64_t id,
         const std::string& name,
         const std::vector<Condition>& conditions,
         const std::vector<Conclusion>& conclusions,
         int32_t priority = 0,
         double confidence = 1.0);

    // Accessor methods - all const to ensure rules are immutable after creation

    /** @brief Get the unique ID of this rule */
    auto get_id() const -> uint64_t { return id_; }

    /** @brief Get the human-readable name */
    auto get_name() const -> const std::string& { return name_; }

    /** @brief Get all conditions that must be satisfied */
    auto get_conditions() const -> const std::vector<Condition>& { return conditions_; }

    /** @brief Get all conclusions that will be asserted */
    auto get_conclusions() const -> const std::vector<Conclusion>& { return conclusions_; }

    /** @brief Get the priority level for conflict resolution */
    auto get_priority() const -> int32_t { return priority_; }

    /** @brief Get the overall confidence level */
    auto get_confidence() const -> double { return confidence_; }

    /** @brief Get the schema version this rule was created with (returns empty string if not set)
     */
    auto get_schema_version_string() const -> std::string { return schema_version_string_; }

    /**
     * @brief Generate human-readable string representation
     * @return String in the format "name: IF condition1 AND condition2 THEN conclusion1"
     */
    auto to_string() const -> std::string;

    // Cap'n Proto interoperability methods

    /** @brief Construct Rule from Cap'n Proto reader */
    explicit Rule(schemas::Rule::Reader reader);

    /** @brief Write this Rule to a Cap'n Proto builder */
    auto write_to(schemas::Rule::Builder builder) const -> void;

  private:
    uint64_t id_;                                      ///< Unique identifier for this rule
    std::string name_;                                 ///< Human-readable name for debugging
    std::vector<Condition> conditions_;                ///< Conditions that must be satisfied
    std::vector<Conclusion> conclusions_;              ///< Conclusions to assert when rule fires
    int32_t priority_;                                 ///< Priority for conflict resolution
    double confidence_;                                ///< Overall confidence level
    std::unordered_map<std::string, Value> metadata_;  ///< Additional metadata (currently unused)
    std::string schema_version_string_;  ///< Schema version this rule was created with (as string)
};

/**
 * @class Query
 * @brief C++ wrapper for Cap'n Proto Query type - represents a query to the inference engine
 *
 * A Query specifies what the inference engine should search for or prove.
 * Different query types provide different behaviors:
 * - FindAll: Find all facts matching the goal pattern
 * - Prove: Check if the goal can be proven true/false
 * - FindFirst: Find the first N solutions that match
 * - Explain: Provide a proof trace showing how the goal can be derived
 *
 * Queries can contain variables (uppercase names) that will be bound to values
 * in the results.
 */
class Query {
  public:
    /**
     * @enum Type
     * @brief Enumeration of different query types supported by the inference engine
     */
    enum class Type : std::uint8_t {
        FIND_ALL,    ///< Find all facts that match the goal
        PROVE,       ///< Check if goal can be proven (true/false)
        FIND_FIRST,  ///< Find first N solutions
        EXPLAIN      ///< Explain how a goal can be proven
    };

    /**
     * @brief Construct a new Query
     * @param id Unique identifier for this query
     * @param type Type of query to perform
     * @param goal The goal pattern to search for or prove
     * @param maxResults Maximum number of results to return
     * @param timeoutMs Query timeout in milliseconds
     */
    Query(uint64_t id,
          Type type,
          const Rule::Condition& goal,
          uint32_t max_results = 100,
          uint32_t timeout_ms = 5000);

    // Accessor methods - all const since queries are immutable after creation

    /** @brief Get the unique ID of this query */
    auto get_id() const -> uint64_t { return id_; }

    /** @brief Get the type of query */
    auto get_type() const -> Type { return type_; }

    /** @brief Get the goal pattern to search for */
    auto get_goal() const -> const Rule::Condition& { return goal_; }

    /** @brief Get the maximum number of results to return */
    auto get_max_results() const -> uint32_t { return max_results_; }

    /** @brief Get the timeout in milliseconds */
    auto get_timeout_ms() const -> uint32_t { return timeout_ms_; }

    /**
     * @brief Generate human-readable string representation
     * @return String in the format "Query[id]: TYPE goal_pattern"
     */
    auto to_string() const -> std::string;

    // Cap'n Proto interoperability methods

    /** @brief Construct Query from Cap'n Proto reader */
    explicit Query(schemas::Query::Reader reader);

    /** @brief Write this Query to a Cap'n Proto builder */
    auto write_to(schemas::Query::Builder builder) const -> void;

  private:
    uint64_t id_;                                      ///< Unique identifier for this query
    Type type_;                                        ///< Type of query to perform
    Rule::Condition goal_;                             ///< Goal pattern to search for
    uint32_t max_results_;                             ///< Maximum number of results
    uint32_t timeout_ms_;                              ///< Timeout in milliseconds
    std::unordered_map<std::string, Value> metadata_;  ///< Additional metadata (currently unused)
};

/**
 * @class Serializer
 * @brief Utility class for serializing/deserializing inference types to Cap'n Proto format
 *
 * This class provides static methods for converting between our C++ wrapper objects
 * and binary Cap'n Proto format for storage, transmission, or persistence.
 * It also provides JSON-like text serialization for debugging and human-readable output.
 *
 * All serialization methods are thread-safe since they only use local state.
 */
class Serializer {
  public:
    // Binary serialization methods - produce compact binary format suitable for
    // storage/transmission

    /** @brief Serialize a Fact to binary Cap'n Proto format */
    static auto serialize(const Fact& fact) -> std::vector<uint8_t>;

    /** @brief Serialize a Rule to binary Cap'n Proto format */
    static auto serialize(const Rule& rule) -> std::vector<uint8_t>;

    /** @brief Serialize a Query to binary Cap'n Proto format */
    static auto serialize(const Query& query) -> std::vector<uint8_t>;

    // Binary deserialization methods - convert binary data back to C++ objects

    /**
     * @brief Deserialize a Fact from binary Cap'n Proto data
     * @param data Binary data produced by serialize(Fact)
     * @return Fact object if successful, nullopt if data is invalid/corrupted
     */
    static auto deserialize_fact(const std::vector<uint8_t>& data) -> std::optional<Fact>;

    /**
     * @brief Deserialize a Rule from binary Cap'n Proto data
     * @param data Binary data produced by serialize(Rule)
     * @return Rule object if successful, nullopt if data is invalid/corrupted
     */
    static auto deserialize_rule(const std::vector<uint8_t>& data) -> std::optional<Rule>;

    /**
     * @brief Deserialize a Query from binary Cap'n Proto data
     * @param data Binary data produced by serialize(Query)
     * @return Query object if successful, nullopt if data is invalid/corrupted
     */
    static auto deserialize_query(const std::vector<uint8_t>& data) -> std::optional<Query>;

    // JSON-like text serialization methods - for debugging and human-readable output

    /** @brief Convert a Fact to JSON-like string representation */
    static auto to_json(const Fact& fact) -> std::string;

    /** @brief Convert a Rule to JSON-like string representation */
    static auto to_json(const Rule& rule) -> std::string;

    /** @brief Convert a Query to JSON-like string representation */
    static auto to_json(const Query& query) -> std::string;
};

/**
 * @class VersionedSerializer
 * @brief Enhanced serializer with schema versioning and migration support
 *
 * This class extends the basic Serializer with awareness of schema versions
 * and automatic migration capabilities. It can read data created with older
 * schema versions and automatically migrate it to the current format.
 */
class VersionedSerializer {
  public:
    /**
     * @brief Serialize data with current schema version metadata
     * @param fact Fact to serialize
     * @return Binary data with embedded schema version information
     */
    static auto serialize_with_version(const Fact& fact) -> std::vector<uint8_t>;

    /**
     * @brief Serialize data with current schema version metadata
     * @param rule Rule to serialize
     * @return Binary data with embedded schema version information
     */
    static auto serialize_with_version(const Rule& rule) -> std::vector<uint8_t>;

    /**
     * @brief Serialize a complete knowledge base with schema evolution metadata
     * @param facts All facts in the knowledge base
     * @param rules All rules in the knowledge base
     * @param metadata Additional metadata
     * @return Binary data with complete versioning information
     */
    static auto serialize_knowledge_base(
        const std::vector<Fact>& facts,
        const std::vector<Rule>& rules,
        const std::unordered_map<std::string, Value>& metadata = {}) -> std::vector<uint8_t>;

    /**
     * @brief Deserialize with automatic migration support
     * @param data Binary data that may be from an older schema version
     * @return Fact migrated to current schema version, or nullopt if failed
     */
    static auto deserialize_fact_with_migration(const std::vector<uint8_t>& data)
        -> std::optional<Fact>;

    /**
     * @brief Deserialize with automatic migration support
     * @param data Binary data that may be from an older schema version
     * @return Rule migrated to current schema version, or nullopt if failed
     */
    static auto deserialize_rule_with_migration(const std::vector<uint8_t>& data)
        -> std::optional<Rule>;

    /**
     * @brief Deserialize a complete knowledge base with migration
     * @param data Binary data from any supported schema version
     * @return Tuple of (facts, rules, metadata) migrated to current version
     */
    static auto deserialize_knowledge_base(const std::vector<uint8_t>& data) -> std::optional<
        std::tuple<std::vector<Fact>, std::vector<Rule>, std::unordered_map<std::string, Value>>>;

    /**
     * @brief Check if data can be read by examining its schema version
     * @param data Binary data to check
     * @return Schema version string of the data, or empty string if unreadable
     */
    static auto detect_schema_version_string(const std::vector<uint8_t>& data) -> std::string;

    /**
     * @brief Validate data integrity including schema version compatibility
     * @param data Binary data to validate
     * @return Vector of validation errors (empty if valid)
     */
    static auto validate_data(const std::vector<uint8_t>& data) -> std::vector<std::string>;
};

}  // namespace inference_lab::common
