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

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>

// Include generated Cap'n Proto headers
#include "schemas/inference_types.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize.h>

namespace inference_lab::common {

// Forward declarations for builder classes
class ValueBuilder;
class FactBuilder; 
class RuleBuilder;
class QueryBuilder;

/**
 * @class Value
 * @brief C++ wrapper for Cap'n Proto Value type - represents a polymorphic value in the inference engine
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
    Value() : type_(Type::Int64), int64_value_(0) {}
    
    // Factory methods for creating typed values - these are preferred over constructors
    // to make the type creation explicit and prevent accidental conversions
    
    /** @brief Create a 64-bit signed integer value */
    static Value fromInt64(int64_t value);
    
    /** @brief Create a 64-bit floating point value */
    static Value fromFloat64(double value);
    
    /** @brief Create a text/string value */
    static Value fromText(const std::string& value);
    
    /** @brief Create a boolean value */
    static Value fromBool(bool value);
    
    /** @brief Create a list/array value containing other Values */
    static Value fromList(const std::vector<Value>& values);
    
    /** @brief Create a structured object value (key-value map) */
    static Value fromStruct(const std::unordered_map<std::string, Value>& fields);
    
    // Type checking methods - these are const and do not throw
    
    /** @brief Check if this value contains a 64-bit integer */
    bool isInt64() const;
    
    /** @brief Check if this value contains a 64-bit float */
    bool isFloat64() const;
    
    /** @brief Check if this value contains text/string data */
    bool isText() const;
    
    /** @brief Check if this value contains a boolean */
    bool isBool() const;
    
    /** @brief Check if this value contains a list of other values */
    bool isList() const;
    
    /** @brief Check if this value contains a structured object */
    bool isStruct() const;
    
    // Unsafe value extraction methods - these throw std::runtime_error if type doesn't match
    // Use these when you're certain of the type or want to fail fast on type mismatches
    
    /** @brief Extract int64 value - throws if not an int64 */
    int64_t asInt64() const;
    
    /** @brief Extract float64 value - throws if not a float64 */
    double asFloat64() const;
    
    /** @brief Extract text value - throws if not text */
    std::string asText() const;
    
    /** @brief Extract bool value - throws if not a bool */
    bool asBool() const;
    
    /** @brief Extract list value - throws if not a list */
    std::vector<Value> asList() const;
    
    /** @brief Extract struct value - throws if not a struct */
    std::unordered_map<std::string, Value> asStruct() const;
    
    // Safe value extraction methods - these return std::nullopt if type doesn't match
    // Use these when you want to handle type mismatches gracefully
    
    /** @brief Safely extract int64 value - returns nullopt if wrong type */
    std::optional<int64_t> tryAsInt64() const;
    
    /** @brief Safely extract float64 value - returns nullopt if wrong type */
    std::optional<double> tryAsFloat64() const;
    
    /** @brief Safely extract text value - returns nullopt if wrong type */
    std::optional<std::string> tryAsText() const;
    
    /** @brief Safely extract bool value - returns nullopt if wrong type */
    std::optional<bool> tryAsBool() const;
    
    /** @brief Safely extract list value - returns nullopt if wrong type */
    std::optional<std::vector<Value>> tryAsList() const;
    
    /** @brief Safely extract struct value - returns nullopt if wrong type */
    std::optional<std::unordered_map<std::string, Value>> tryAsStruct() const;
    
    /**
     * @brief Generate human-readable string representation for debugging
     * @return String representation that shows both type and value
     */
    std::string toString() const;
    
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
    void writeTo(schemas::Value::Builder builder) const;
    
private:
    /**
     * @brief Internal type discriminator enum
     * 
     * This enum tracks which type of value is currently stored in the Value object.
     * It's used internally for type checking and safe casting operations.
     */
    enum class Type { Int64, Float64, Text, Bool, List, Struct };
    
    /** @brief Current type of the stored value */
    Type type_;
    
    /**
     * @brief Anonymous union for storing primitive types efficiently
     * 
     * Only one of these will be active at a time, based on the type_ field.
     * This saves memory compared to storing all possible values separately.
     */
    union {
        int64_t int64_value_;    ///< Storage for 64-bit integers
        double float64_value_;   ///< Storage for 64-bit floats
        bool bool_value_;        ///< Storage for boolean values
    };
    
    // Complex types are stored separately since they can't be in unions
    std::string text_value_;                                ///< Storage for text/string values
    std::vector<Value> list_value_;                         ///< Storage for list values
    std::unordered_map<std::string, Value> struct_value_;   ///< Storage for structured object values
    
    /**
     * @brief Private constructor for internal use
     * @param type The type of value to create
     * 
     * This constructor is used internally by the factory methods to create
     * appropriately typed Value objects.
     */
    Value(Type type) : type_(type) {}
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
    Fact(uint64_t id, const std::string& predicate, const std::vector<Value>& args,
         double confidence = 1.0, uint64_t timestamp = 0);
    
    // Accessor methods - all const to ensure facts are immutable after creation
    
    /** @brief Get the unique ID of this fact */
    uint64_t getId() const { return id_; }
    
    /** @brief Get the predicate name */
    const std::string& getPredicate() const { return predicate_; }
    
    /** @brief Get the arguments list */
    const std::vector<Value>& getArgs() const { return args_; }
    
    /** @brief Get the confidence level (0.0 to 1.0) */
    double getConfidence() const { return confidence_; }
    
    /** @brief Get the timestamp when this fact was created */
    uint64_t getTimestamp() const { return timestamp_; }
    
    /** @brief Get all metadata as a key-value map */
    const std::unordered_map<std::string, Value>& getMetadata() const { return metadata_; }
    
    // Metadata management methods
    
    /**
     * @brief Set a metadata key-value pair
     * @param key Metadata key name
     * @param value Metadata value
     * 
     * Metadata can store additional information about facts, such as source,
     * creation context, or other properties not part of the core fact structure.
     */
    void setMetadata(const std::string& key, const Value& value);
    
    /**
     * @brief Get a specific metadata value by key
     * @param key Metadata key to look up
     * @return The metadata value if found, nullopt otherwise
     */
    std::optional<Value> getMetadata(const std::string& key) const;
    
    /**
     * @brief Generate human-readable string representation
     * @return String in the format "predicate(arg1, arg2, ...) [confidence: X]"
     * 
     * This is primarily used for debugging and logging purposes.
     */
    std::string toString() const;
    
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
    void writeTo(schemas::Fact::Builder builder) const;
    
private:
    uint64_t id_;                                           ///< Unique identifier for this fact
    std::string predicate_;                                 ///< Predicate name (e.g., "isHuman")
    std::vector<Value> args_;                               ///< Arguments for the predicate
    double confidence_;                                     ///< Confidence level (0.0 to 1.0)
    uint64_t timestamp_;                                    ///< Creation timestamp in milliseconds
    std::unordered_map<std::string, Value> metadata_;      ///< Additional metadata key-value pairs
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
        std::string predicate;          ///< Predicate to match (e.g., "isHuman")
        std::vector<Value> args;        ///< Arguments with variables and constants
        bool negated = false;           ///< Whether this is a NOT condition
        
        /** @brief Generate string representation of this condition */
        std::string toString() const;
    };
    
    /**
     * @struct Conclusion
     * @brief Represents a conclusion that will be asserted if rule conditions are met
     * 
     * Conclusions define what new facts should be created when a rule fires.
     * They can reference variables bound in the conditions.
     */
    struct Conclusion {
        std::string predicate;          ///< Predicate to assert (e.g., "isMortal")
        std::vector<Value> args;        ///< Arguments (may contain variables from conditions)
        double confidence = 1.0;        ///< Confidence for this conclusion
        
        /** @brief Generate string representation of this conclusion */
        std::string toString() const;
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
    Rule(uint64_t id, const std::string& name, 
         const std::vector<Condition>& conditions,
         const std::vector<Conclusion>& conclusions,
         int32_t priority = 0, double confidence = 1.0);
    
    // Accessor methods - all const to ensure rules are immutable after creation
    
    /** @brief Get the unique ID of this rule */
    uint64_t getId() const { return id_; }
    
    /** @brief Get the human-readable name */
    const std::string& getName() const { return name_; }
    
    /** @brief Get all conditions that must be satisfied */
    const std::vector<Condition>& getConditions() const { return conditions_; }
    
    /** @brief Get all conclusions that will be asserted */
    const std::vector<Conclusion>& getConclusions() const { return conclusions_; }
    
    /** @brief Get the priority level for conflict resolution */
    int32_t getPriority() const { return priority_; }
    
    /** @brief Get the overall confidence level */
    double getConfidence() const { return confidence_; }
    
    /**
     * @brief Generate human-readable string representation
     * @return String in the format "name: IF condition1 AND condition2 THEN conclusion1"
     */
    std::string toString() const;
    
    // Cap'n Proto interoperability methods
    
    /** @brief Construct Rule from Cap'n Proto reader */
    explicit Rule(schemas::Rule::Reader reader);
    
    /** @brief Write this Rule to a Cap'n Proto builder */
    void writeTo(schemas::Rule::Builder builder) const;
    
private:
    uint64_t id_;                                           ///< Unique identifier for this rule
    std::string name_;                                      ///< Human-readable name for debugging
    std::vector<Condition> conditions_;                     ///< Conditions that must be satisfied
    std::vector<Conclusion> conclusions_;                   ///< Conclusions to assert when rule fires
    int32_t priority_;                                      ///< Priority for conflict resolution
    double confidence_;                                     ///< Overall confidence level
    std::unordered_map<std::string, Value> metadata_;      ///< Additional metadata (currently unused)
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
    enum class Type { 
        FindAll,    ///< Find all facts that match the goal
        Prove,      ///< Check if goal can be proven (true/false)
        FindFirst,  ///< Find first N solutions
        Explain     ///< Explain how a goal can be proven
    };
    
    /**
     * @brief Construct a new Query
     * @param id Unique identifier for this query
     * @param type Type of query to perform
     * @param goal The goal pattern to search for or prove
     * @param maxResults Maximum number of results to return
     * @param timeoutMs Query timeout in milliseconds
     */
    Query(uint64_t id, Type type, const Rule::Condition& goal,
          uint32_t maxResults = 100, uint32_t timeoutMs = 5000);
    
    // Accessor methods - all const since queries are immutable after creation
    
    /** @brief Get the unique ID of this query */
    uint64_t getId() const { return id_; }
    
    /** @brief Get the type of query */
    Type getType() const { return type_; }
    
    /** @brief Get the goal pattern to search for */
    const Rule::Condition& getGoal() const { return goal_; }
    
    /** @brief Get the maximum number of results to return */
    uint32_t getMaxResults() const { return maxResults_; }
    
    /** @brief Get the timeout in milliseconds */
    uint32_t getTimeoutMs() const { return timeoutMs_; }
    
    /**
     * @brief Generate human-readable string representation
     * @return String in the format "Query[id]: TYPE goal_pattern"
     */
    std::string toString() const;
    
    // Cap'n Proto interoperability methods
    
    /** @brief Construct Query from Cap'n Proto reader */
    explicit Query(schemas::Query::Reader reader);
    
    /** @brief Write this Query to a Cap'n Proto builder */
    void writeTo(schemas::Query::Builder builder) const;
    
private:
    uint64_t id_;                                           ///< Unique identifier for this query
    Type type_;                                             ///< Type of query to perform
    Rule::Condition goal_;                                  ///< Goal pattern to search for
    uint32_t maxResults_;                                   ///< Maximum number of results
    uint32_t timeoutMs_;                                    ///< Timeout in milliseconds
    std::unordered_map<std::string, Value> metadata_;      ///< Additional metadata (currently unused)
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
    // Binary serialization methods - produce compact binary format suitable for storage/transmission
    
    /** @brief Serialize a Fact to binary Cap'n Proto format */
    static std::vector<uint8_t> serialize(const Fact& fact);
    
    /** @brief Serialize a Rule to binary Cap'n Proto format */
    static std::vector<uint8_t> serialize(const Rule& rule);
    
    /** @brief Serialize a Query to binary Cap'n Proto format */
    static std::vector<uint8_t> serialize(const Query& query);
    
    // Binary deserialization methods - convert binary data back to C++ objects
    
    /**
     * @brief Deserialize a Fact from binary Cap'n Proto data
     * @param data Binary data produced by serialize(Fact)
     * @return Fact object if successful, nullopt if data is invalid/corrupted
     */
    static std::optional<Fact> deserializeFact(const std::vector<uint8_t>& data);
    
    /**
     * @brief Deserialize a Rule from binary Cap'n Proto data
     * @param data Binary data produced by serialize(Rule)
     * @return Rule object if successful, nullopt if data is invalid/corrupted
     */
    static std::optional<Rule> deserializeRule(const std::vector<uint8_t>& data);
    
    /**
     * @brief Deserialize a Query from binary Cap'n Proto data
     * @param data Binary data produced by serialize(Query)
     * @return Query object if successful, nullopt if data is invalid/corrupted
     */
    static std::optional<Query> deserializeQuery(const std::vector<uint8_t>& data);
    
    // JSON-like text serialization methods - for debugging and human-readable output
    
    /** @brief Convert a Fact to JSON-like string representation */
    static std::string toJson(const Fact& fact);
    
    /** @brief Convert a Rule to JSON-like string representation */
    static std::string toJson(const Rule& rule);
    
    /** @brief Convert a Query to JSON-like string representation */
    static std::string toJson(const Query& query);
};

} // namespace inference_lab::common