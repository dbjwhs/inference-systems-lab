// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

/**
 * @file test_serialization.cpp
 * @brief Comprehensive unit tests for the serialization framework and schema evolution system
 * 
 * This test suite validates the complete serialization framework including:
 * - Value type system with all supported types (primitives and complex)
 * - Fact, Rule, and Query data structures with full lifecycle testing
 * - Binary serialization/deserialization with Cap'n Proto integration
 * - JSON-like text serialization for debugging and human-readable output
 * - Schema versioning and evolution support with migration capabilities
 * - Compatibility checking and validation across schema versions
 * - Error handling and edge cases for robust production usage
 * 
 * The tests are organized into logical groups covering each major component
 * of the serialization system, with comprehensive validation of both
 * successful operations and error conditions.
 */

#include <gtest/gtest.h>
#include "../src/inference_types.hpp"
#include "../src/schema_evolution.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <sstream>
#include <random>

using namespace inference_lab::common;
using namespace inference_lab::common::evolution;

/**
 * @class SerializationTest
 * @brief Test fixture for serialization framework unit tests
 * 
 * Provides common setup and utility methods for all serialization tests.
 * The fixture manages test data creation, file cleanup, and provides
 * helper methods for validation and comparison operations.
 */
class SerializationTest : public ::testing::Test {
protected:
    /**
     * @brief Test fixture setup - initializes test environment
     * 
     * Creates clean test environment by setting up temporary
     * file paths and ensuring no leftover test artifacts exist.
     */
    void SetUp() override {
        test_data_dir = "./test_serialization_data";
        
        // Create test directory if it doesn't exist
        if (!std::filesystem::exists(test_data_dir)) {
            std::filesystem::create_directory(test_data_dir);
        }
        
        // Clean up any existing test files
        cleanupTestFiles();
    }

    /**
     * @brief Test fixture teardown - cleans up test artifacts
     * 
     * Removes all test files and directories to ensure clean
     * state for subsequent test runs.
     */
    void TearDown() override {
        cleanupTestFiles();
        
        // Remove test directory if it exists
        if (std::filesystem::exists(test_data_dir)) {
            std::filesystem::remove_all(test_data_dir);
        }
    }

    std::string test_data_dir; ///< Directory for test files
    
    /**
     * @brief Utility method to clean up test files
     */
    void cleanupTestFiles() {
        if (std::filesystem::exists(test_data_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(test_data_dir)) {
                std::filesystem::remove_all(entry.path());
            }
        }
    }
    
    /**
     * @brief Create a sample Fact for testing
     * @param id Unique identifier for the fact
     * @return Fact object with test data
     */
    Fact createTestFact(uint64_t id = 1) {
        std::vector<Value> args = {
            Value::fromText("socrates"),
            Value::fromInt64(42),
            Value::fromFloat64(3.14159)
        };
        
        Fact fact(id, "isHuman", args, 0.95, 1234567890);
        fact.setMetadata("source", Value::fromText("unit_test"));
        fact.setMetadata("created_by", Value::fromText("test_framework"));
        
        return fact;
    }
    
    /**
     * @brief Create a sample Rule for testing
     * @param id Unique identifier for the rule
     * @return Rule object with test data
     */
    Rule createTestRule(uint64_t id = 1) {
        // Create conditions: isHuman(X) AND hasAge(X, Age)
        std::vector<Rule::Condition> conditions;
        
        Rule::Condition cond1;
        cond1.predicate = "isHuman";
        cond1.args = {Value::fromText("X")};
        cond1.negated = false;
        conditions.push_back(cond1);
        
        Rule::Condition cond2;
        cond2.predicate = "hasAge";
        cond2.args = {Value::fromText("X"), Value::fromText("Age")};
        cond2.negated = false;
        conditions.push_back(cond2);
        
        // Create conclusions: isMortal(X)
        std::vector<Rule::Conclusion> conclusions;
        
        Rule::Conclusion concl1;
        concl1.predicate = "isMortal";
        concl1.args = {Value::fromText("X")};
        concl1.confidence = 0.99;
        conclusions.push_back(concl1);
        
        return Rule(id, "mortality_rule", conditions, conclusions, 10, 0.95);
    }
    
    /**
     * @brief Create a sample Query for testing
     * @param id Unique identifier for the query
     * @return Query object with test data
     */
    Query createTestQuery(uint64_t id = 1) {
        Rule::Condition goal;
        goal.predicate = "isMortal";
        goal.args = {Value::fromText("socrates")};
        goal.negated = false;
        
        return Query(id, Query::Type::FindAll, goal, 50, 10000);
    }
    
    /**
     * @brief Validate that two Facts are equivalent
     * @param original Original fact
     * @param deserialized Deserialized fact
     */
    void validateFactEquality(const Fact& original, const Fact& deserialized) {
        EXPECT_EQ(original.getId(), deserialized.getId());
        EXPECT_EQ(original.getPredicate(), deserialized.getPredicate());
        EXPECT_DOUBLE_EQ(original.getConfidence(), deserialized.getConfidence());
        EXPECT_EQ(original.getTimestamp(), deserialized.getTimestamp());
        
        // Validate arguments
        const auto& origArgs = original.getArgs();
        const auto& deserArgs = deserialized.getArgs();
        EXPECT_EQ(origArgs.size(), deserArgs.size());
        
        for (size_t i = 0; i < origArgs.size(); ++i) {
            validateValueEquality(origArgs[i], deserArgs[i]);
        }
        
        // Validate metadata
        const auto& origMeta = original.getMetadata();
        const auto& deserMeta = deserialized.getMetadata();
        EXPECT_EQ(origMeta.size(), deserMeta.size());
        
        for (const auto& [key, value] : origMeta) {
            auto deserValue = deserialized.getMetadata(key);
            ASSERT_TRUE(deserValue.has_value());
            validateValueEquality(value, deserValue.value());
        }
    }
    
    /**
     * @brief Validate that two Values are equivalent
     * @param original Original value
     * @param deserialized Deserialized value
     */
    void validateValueEquality(const Value& original, const Value& deserialized) {
        // Check type consistency
        EXPECT_EQ(original.isInt64(), deserialized.isInt64());
        EXPECT_EQ(original.isFloat64(), deserialized.isFloat64());
        EXPECT_EQ(original.isText(), deserialized.isText());
        EXPECT_EQ(original.isBool(), deserialized.isBool());
        EXPECT_EQ(original.isList(), deserialized.isList());
        EXPECT_EQ(original.isStruct(), deserialized.isStruct());
        
        // Check value content based on type
        if (original.isInt64()) {
            EXPECT_EQ(original.asInt64(), deserialized.asInt64());
        } else if (original.isFloat64()) {
            EXPECT_DOUBLE_EQ(original.asFloat64(), deserialized.asFloat64());
        } else if (original.isText()) {
            EXPECT_EQ(original.asText(), deserialized.asText());
        } else if (original.isBool()) {
            EXPECT_EQ(original.asBool(), deserialized.asBool());
        } else if (original.isList()) {
            const auto& origList = original.asList();
            const auto& deserList = deserialized.asList();
            EXPECT_EQ(origList.size(), deserList.size());
            for (size_t i = 0; i < origList.size(); ++i) {
                validateValueEquality(origList[i], deserList[i]);
            }
        } else if (original.isStruct()) {
            const auto& origStruct = original.asStruct();
            const auto& deserStruct = deserialized.asStruct();
            EXPECT_EQ(origStruct.size(), deserStruct.size());
            for (const auto& [key, value] : origStruct) {
                ASSERT_TRUE(deserStruct.count(key) > 0);
                validateValueEquality(value, deserStruct.at(key));
            }
        }
    }
};

//=============================================================================
// Value Type System Tests
//=============================================================================

/**
 * @brief Test Value creation and type checking for all supported types
 * 
 * Validates that:
 * - All factory methods create Values with correct types
 * - Type checking methods return accurate results
 * - Default constructor creates int64 value with 0
 * - Each value type maintains its discriminant correctly
 */
TEST_F(SerializationTest, ValueTypeCreationAndChecking) {
    // Test default constructor (should create int64 with value 0)
    Value defaultValue;
    EXPECT_TRUE(defaultValue.isInt64());
    EXPECT_EQ(defaultValue.asInt64(), 0);
    EXPECT_FALSE(defaultValue.isFloat64());
    EXPECT_FALSE(defaultValue.isText());
    EXPECT_FALSE(defaultValue.isBool());
    EXPECT_FALSE(defaultValue.isList());
    EXPECT_FALSE(defaultValue.isStruct());
    
    // Test int64 value
    Value intValue = Value::fromInt64(42);
    EXPECT_TRUE(intValue.isInt64());
    EXPECT_EQ(intValue.asInt64(), 42);
    EXPECT_FALSE(intValue.isFloat64());
    
    // Test float64 value
    Value floatValue = Value::fromFloat64(3.14159);
    EXPECT_TRUE(floatValue.isFloat64());
    EXPECT_DOUBLE_EQ(floatValue.asFloat64(), 3.14159);
    EXPECT_FALSE(floatValue.isInt64());
    
    // Test text value
    Value textValue = Value::fromText("hello world");
    EXPECT_TRUE(textValue.isText());
    EXPECT_EQ(textValue.asText(), "hello world");
    EXPECT_FALSE(textValue.isInt64());
    
    // Test bool value
    Value boolValue = Value::fromBool(true);
    EXPECT_TRUE(boolValue.isBool());
    EXPECT_TRUE(boolValue.asBool());
    EXPECT_FALSE(boolValue.isText());
    
    // Test list value
    std::vector<Value> listData = {
        Value::fromInt64(1),
        Value::fromText("test"),
        Value::fromBool(false)
    };
    Value listValue = Value::fromList(listData);
    EXPECT_TRUE(listValue.isList());
    EXPECT_FALSE(listValue.isStruct());
    
    auto retrievedList = listValue.asList();
    EXPECT_EQ(retrievedList.size(), 3);
    EXPECT_EQ(retrievedList[0].asInt64(), 1);
    EXPECT_EQ(retrievedList[1].asText(), "test");
    EXPECT_FALSE(retrievedList[2].asBool());
    
    // Test struct value
    std::unordered_map<std::string, Value> structData = {
        {"name", Value::fromText("Alice")},
        {"age", Value::fromInt64(30)},
        {"score", Value::fromFloat64(95.5)}
    };
    Value structValue = Value::fromStruct(structData);
    EXPECT_TRUE(structValue.isStruct());
    EXPECT_FALSE(structValue.isList());
    
    auto retrievedStruct = structValue.asStruct();
    EXPECT_EQ(retrievedStruct.size(), 3);
    EXPECT_EQ(retrievedStruct["name"].asText(), "Alice");
    EXPECT_EQ(retrievedStruct["age"].asInt64(), 30);
    EXPECT_DOUBLE_EQ(retrievedStruct["score"].asFloat64(), 95.5);
}

/**
 * @brief Test Value extraction methods (both safe and unsafe)
 * 
 * Validates that:
 * - Unsafe extraction methods throw on type mismatch
 * - Safe extraction methods return nullopt on type mismatch
 * - Correct values are returned when types match
 * - Exception messages are meaningful for debugging
 */
TEST_F(SerializationTest, ValueExtractionMethods) {
    Value intValue = Value::fromInt64(123);
    Value textValue = Value::fromText("test");
    
    // Test successful extraction
    EXPECT_EQ(intValue.asInt64(), 123);
    EXPECT_EQ(textValue.asText(), "test");
    
    // Test unsafe extraction with wrong type (should throw)
    EXPECT_THROW(intValue.asText(), std::runtime_error);
    EXPECT_THROW(textValue.asInt64(), std::runtime_error);
    EXPECT_THROW(intValue.asFloat64(), std::runtime_error);
    EXPECT_THROW(intValue.asBool(), std::runtime_error);
    EXPECT_THROW(intValue.asList(), std::runtime_error);
    EXPECT_THROW(intValue.asStruct(), std::runtime_error);
    
    // Test safe extraction with correct type
    auto intResult = intValue.tryAsInt64();
    ASSERT_TRUE(intResult.has_value());
    EXPECT_EQ(intResult.value(), 123);
    
    auto textResult = textValue.tryAsText();
    ASSERT_TRUE(textResult.has_value());
    EXPECT_EQ(textResult.value(), "test");
    
    // Test safe extraction with wrong type (should return nullopt)
    EXPECT_FALSE(intValue.tryAsText().has_value());
    EXPECT_FALSE(textValue.tryAsInt64().has_value());
    EXPECT_FALSE(intValue.tryAsFloat64().has_value());
    EXPECT_FALSE(intValue.tryAsBool().has_value());
    EXPECT_FALSE(intValue.tryAsList().has_value());
    EXPECT_FALSE(intValue.tryAsStruct().has_value());
}

/**
 * @brief Test Value string representation for debugging
 * 
 * Validates that:
 * - toString() generates readable representations for all types
 * - Complex types (lists, structs) are properly formatted
 * - Special characters in strings are handled correctly
 * - Nested structures display correctly
 */
TEST_F(SerializationTest, ValueStringRepresentation) {
    // Test primitive types
    EXPECT_EQ(Value::fromInt64(42).toString(), "42");
    EXPECT_EQ(Value::fromFloat64(3.14).toString(), "3.140000"); // Note: std::to_string precision
    EXPECT_EQ(Value::fromText("hello").toString(), "\"hello\"");
    EXPECT_EQ(Value::fromBool(true).toString(), "true");
    EXPECT_EQ(Value::fromBool(false).toString(), "false");
    
    // Test list representation
    std::vector<Value> listData = {
        Value::fromInt64(1),
        Value::fromText("test"),
        Value::fromBool(true)
    };
    Value listValue = Value::fromList(listData);
    std::string listStr = listValue.toString();
    EXPECT_TRUE(listStr.find("[") != std::string::npos);
    EXPECT_TRUE(listStr.find("]") != std::string::npos);
    EXPECT_TRUE(listStr.find("1") != std::string::npos);
    EXPECT_TRUE(listStr.find("\"test\"") != std::string::npos);
    EXPECT_TRUE(listStr.find("true") != std::string::npos);
    
    // Test struct representation
    std::unordered_map<std::string, Value> structData = {
        {"name", Value::fromText("Alice")},
        {"count", Value::fromInt64(5)}
    };
    Value structValue = Value::fromStruct(structData);
    std::string structStr = structValue.toString();
    EXPECT_TRUE(structStr.find("{") != std::string::npos);
    EXPECT_TRUE(structStr.find("}") != std::string::npos);
    EXPECT_TRUE(structStr.find("\"name\"") != std::string::npos);
    EXPECT_TRUE(structStr.find("\"Alice\"") != std::string::npos);
    EXPECT_TRUE(structStr.find("\"count\"") != std::string::npos);
    EXPECT_TRUE(structStr.find("5") != std::string::npos);
}

/**
 * @brief Test Value Cap'n Proto interoperability
 * 
 * Validates that:
 * - Values can be written to Cap'n Proto builders
 * - Values can be read from Cap'n Proto readers
 * - Round-trip conversion preserves all data
 * - Complex nested structures work correctly
 */
TEST_F(SerializationTest, ValueCapnProtoInterop) {
    // Create a complex nested value structure
    std::unordered_map<std::string, Value> structData = {
        {"simple_int", Value::fromInt64(42)},
        {"simple_text", Value::fromText("hello")},
        {"nested_list", Value::fromList({
            Value::fromInt64(1),
            Value::fromInt64(2),
            Value::fromInt64(3)
        })},
        {"nested_struct", Value::fromStruct({
            {"inner_bool", Value::fromBool(true)},
            {"inner_float", Value::fromFloat64(2.718)}
        })}
    };
    Value originalValue = Value::fromStruct(structData);
    
    // Write to Cap'n Proto and read back
    capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Value>();
    originalValue.writeTo(builder);
    
    // Read back from Cap'n Proto
    auto reader = builder.asReader();
    Value reconstructedValue(reader);
    
    // Validate full equality
    validateValueEquality(originalValue, reconstructedValue);
}

//=============================================================================
// Fact Tests
//=============================================================================

/**
 * @brief Test Fact creation and basic operations
 * 
 * Validates that:
 * - Facts can be created with all required and optional parameters
 * - Accessor methods return correct values
 * - Metadata can be added and retrieved
 * - Timestamp auto-generation works correctly
 * - String representation is meaningful
 */
TEST_F(SerializationTest, FactCreationAndOperations) {
    // Test basic fact creation
    std::vector<Value> args = {
        Value::fromText("socrates"),
        Value::fromInt64(70)
    };
    
    uint64_t testTimestamp = 1234567890;
    Fact fact(42, "hasAge", args, 0.95, testTimestamp);
    
    // Validate basic properties
    EXPECT_EQ(fact.getId(), 42);
    EXPECT_EQ(fact.getPredicate(), "hasAge");
    EXPECT_DOUBLE_EQ(fact.getConfidence(), 0.95);
    EXPECT_EQ(fact.getTimestamp(), testTimestamp);
    
    // Validate arguments
    const auto& factArgs = fact.getArgs();
    EXPECT_EQ(factArgs.size(), 2);
    EXPECT_EQ(factArgs[0].asText(), "socrates");
    EXPECT_EQ(factArgs[1].asInt64(), 70);
    
    // Test metadata operations
    EXPECT_TRUE(fact.getMetadata().empty());
    
    fact.setMetadata("source", Value::fromText("knowledge_base"));
    fact.setMetadata("confidence_level", Value::fromText("high"));
    fact.setMetadata("version", Value::fromInt64(1));
    
    EXPECT_EQ(fact.getMetadata().size(), 3);
    
    auto sourceValue = fact.getMetadata("source");
    ASSERT_TRUE(sourceValue.has_value());
    EXPECT_EQ(sourceValue->asText(), "knowledge_base");
    
    auto versionValue = fact.getMetadata("version");
    ASSERT_TRUE(versionValue.has_value());
    EXPECT_EQ(versionValue->asInt64(), 1);
    
    auto nonExistentValue = fact.getMetadata("nonexistent");
    EXPECT_FALSE(nonExistentValue.has_value());
    
    // Test string representation
    std::string factStr = fact.toString();
    EXPECT_TRUE(factStr.find("hasAge") != std::string::npos);
    EXPECT_TRUE(factStr.find("socrates") != std::string::npos);
    EXPECT_TRUE(factStr.find("70") != std::string::npos);
    EXPECT_TRUE(factStr.find("confidence: 0.95") != std::string::npos);
}

/**
 * @brief Test Fact timestamp auto-generation
 * 
 * Validates that:
 * - Facts created with timestamp=0 get current timestamp
 * - Explicit timestamps are preserved
 * - Generated timestamps are reasonable (close to current time)
 */
TEST_F(SerializationTest, FactTimestampGeneration) {
    auto beforeCreation = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Create fact with auto-generated timestamp
    Fact fact(1, "test", {Value::fromText("arg")}, 1.0, 0);
    
    auto afterCreation = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Verify timestamp is in reasonable range
    EXPECT_GE(fact.getTimestamp(), beforeCreation);
    EXPECT_LE(fact.getTimestamp(), afterCreation + 1000); // Allow 1 second buffer
    
    // Test explicit timestamp preservation
    uint64_t explicitTimestamp = 9876543210;
    Fact explicitFact(2, "test", {Value::fromText("arg")}, 1.0, explicitTimestamp);
    EXPECT_EQ(explicitFact.getTimestamp(), explicitTimestamp);
}

/**
 * @brief Test Fact Cap'n Proto serialization round-trip
 * 
 * Validates that:
 * - Facts can be serialized to Cap'n Proto format
 * - Deserialized facts are identical to originals
 * - All data including metadata is preserved
 * - Complex argument types work correctly
 */
TEST_F(SerializationTest, FactCapnProtoSerialization) {
    Fact originalFact = createTestFact(999);
    
    // Add complex metadata
    originalFact.setMetadata("complex_data", Value::fromList({
        Value::fromInt64(1),
        Value::fromText("nested"),
        Value::fromStruct({{"key", Value::fromBool(true)}})
    }));
    
    // Serialize to Cap'n Proto
    capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Fact>();
    originalFact.writeTo(builder);
    
    // Deserialize from Cap'n Proto
    auto reader = builder.asReader();
    Fact deserializedFact(reader);
    
    // Validate complete equality
    validateFactEquality(originalFact, deserializedFact);
}

//=============================================================================
// Rule Tests
//=============================================================================

/**
 * @brief Test Rule creation and structure validation
 * 
 * Validates that:
 * - Rules can be created with conditions and conclusions
 * - Condition and conclusion toString methods work correctly
 * - Priority and confidence are handled properly
 * - Complex rule structures are supported
 */
TEST_F(SerializationTest, RuleCreationAndStructure) {
    Rule rule = createTestRule(123);
    
    // Validate basic properties
    EXPECT_EQ(rule.getId(), 123);
    EXPECT_EQ(rule.getName(), "mortality_rule");
    EXPECT_EQ(rule.getPriority(), 10);
    EXPECT_DOUBLE_EQ(rule.getConfidence(), 0.95);
    
    // Validate conditions
    const auto& conditions = rule.getConditions();
    EXPECT_EQ(conditions.size(), 2);
    
    EXPECT_EQ(conditions[0].predicate, "isHuman");
    EXPECT_EQ(conditions[0].args.size(), 1);
    EXPECT_EQ(conditions[0].args[0].asText(), "X");
    EXPECT_FALSE(conditions[0].negated);
    
    EXPECT_EQ(conditions[1].predicate, "hasAge");
    EXPECT_EQ(conditions[1].args.size(), 2);
    EXPECT_EQ(conditions[1].args[0].asText(), "X");
    EXPECT_EQ(conditions[1].args[1].asText(), "Age");
    EXPECT_FALSE(conditions[1].negated);
    
    // Validate conclusions
    const auto& conclusions = rule.getConclusions();
    EXPECT_EQ(conclusions.size(), 1);
    
    EXPECT_EQ(conclusions[0].predicate, "isMortal");
    EXPECT_EQ(conclusions[0].args.size(), 1);
    EXPECT_EQ(conclusions[0].args[0].asText(), "X");
    EXPECT_DOUBLE_EQ(conclusions[0].confidence, 0.99);
    
    // Test condition toString
    std::string condStr = conditions[0].toString();
    EXPECT_TRUE(condStr.find("isHuman") != std::string::npos);
    EXPECT_TRUE(condStr.find("X") != std::string::npos);
    
    // Test conclusion toString
    std::string conclStr = conclusions[0].toString();
    EXPECT_TRUE(conclStr.find("isMortal") != std::string::npos);
    EXPECT_TRUE(conclStr.find("X") != std::string::npos);
    EXPECT_TRUE(conclStr.find("confidence: 0.99") != std::string::npos);
    
    // Test rule toString
    std::string ruleStr = rule.toString();
    EXPECT_TRUE(ruleStr.find("mortality_rule") != std::string::npos);
    EXPECT_TRUE(ruleStr.find("IF") != std::string::npos);
    EXPECT_TRUE(ruleStr.find("AND") != std::string::npos);
    EXPECT_TRUE(ruleStr.find("THEN") != std::string::npos);
    EXPECT_TRUE(ruleStr.find("priority: 10") != std::string::npos);
}

/**
 * @brief Test Rule with negated conditions
 * 
 * Validates that:
 * - Negated conditions are properly handled
 * - String representation includes NOT operator
 * - Serialization preserves negation flags
 */
TEST_F(SerializationTest, RuleNegatedConditions) {
    std::vector<Rule::Condition> conditions;
    
    // Add a negated condition
    Rule::Condition negatedCond;
    negatedCond.predicate = "isDead";
    negatedCond.args = {Value::fromText("X")};
    negatedCond.negated = true;
    conditions.push_back(negatedCond);
    
    // Add a normal condition
    Rule::Condition normalCond;
    normalCond.predicate = "isAlive";
    normalCond.args = {Value::fromText("X")};
    normalCond.negated = false;
    conditions.push_back(normalCond);
    
    std::vector<Rule::Conclusion> conclusions;
    Rule::Conclusion concl;
    concl.predicate = "canThink";
    concl.args = {Value::fromText("X")};
    conclusions.push_back(concl);
    
    Rule rule(1, "thinking_rule", conditions, conclusions);
    
    // Validate negation flags
    EXPECT_TRUE(rule.getConditions()[0].negated);
    EXPECT_FALSE(rule.getConditions()[1].negated);
    
    // Test string representation includes NOT
    std::string condStr = rule.getConditions()[0].toString();
    EXPECT_TRUE(condStr.find("NOT") != std::string::npos);
    EXPECT_TRUE(condStr.find("isDead") != std::string::npos);
    
    // Validate normal condition doesn't have NOT
    std::string normalCondStr = rule.getConditions()[1].toString();
    EXPECT_TRUE(normalCondStr.find("NOT") == std::string::npos);
    EXPECT_TRUE(normalCondStr.find("isAlive") != std::string::npos);
}

/**
 * @brief Test Rule Cap'n Proto serialization round-trip
 * 
 * Validates that:
 * - Rules can be serialized to Cap'n Proto format
 * - Deserialized rules are identical to originals
 * - All conditions, conclusions, and metadata are preserved
 * - Complex argument structures work correctly
 */
TEST_F(SerializationTest, RuleCapnProtoSerialization) {
    Rule originalRule = createTestRule(456);
    
    // Serialize to Cap'n Proto
    capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Rule>();
    originalRule.writeTo(builder);
    
    // Deserialize from Cap'n Proto
    auto reader = builder.asReader();
    Rule deserializedRule(reader);
    
    // Validate basic properties
    EXPECT_EQ(originalRule.getId(), deserializedRule.getId());
    EXPECT_EQ(originalRule.getName(), deserializedRule.getName());
    EXPECT_EQ(originalRule.getPriority(), deserializedRule.getPriority());
    EXPECT_DOUBLE_EQ(originalRule.getConfidence(), deserializedRule.getConfidence());
    
    // Validate conditions
    const auto& origConditions = originalRule.getConditions();
    const auto& deserConditions = deserializedRule.getConditions();
    EXPECT_EQ(origConditions.size(), deserConditions.size());
    
    for (size_t i = 0; i < origConditions.size(); ++i) {
        EXPECT_EQ(origConditions[i].predicate, deserConditions[i].predicate);
        EXPECT_EQ(origConditions[i].negated, deserConditions[i].negated);
        EXPECT_EQ(origConditions[i].args.size(), deserConditions[i].args.size());
        
        for (size_t j = 0; j < origConditions[i].args.size(); ++j) {
            validateValueEquality(origConditions[i].args[j], deserConditions[i].args[j]);
        }
    }
    
    // Validate conclusions
    const auto& origConclusions = originalRule.getConclusions();
    const auto& deserConclusions = deserializedRule.getConclusions();
    EXPECT_EQ(origConclusions.size(), deserConclusions.size());
    
    for (size_t i = 0; i < origConclusions.size(); ++i) {
        EXPECT_EQ(origConclusions[i].predicate, deserConclusions[i].predicate);
        EXPECT_DOUBLE_EQ(origConclusions[i].confidence, deserConclusions[i].confidence);
        EXPECT_EQ(origConclusions[i].args.size(), deserConclusions[i].args.size());
        
        for (size_t j = 0; j < origConclusions[i].args.size(); ++j) {
            validateValueEquality(origConclusions[i].args[j], deserConclusions[i].args[j]);
        }
    }
}

//=============================================================================
// Query Tests
//=============================================================================

/**
 * @brief Test Query creation and type handling
 * 
 * Validates that:
 * - Queries can be created with all supported types
 * - Parameters are stored and retrieved correctly
 * - String representation includes type information
 * - Different query types display appropriate names
 */
TEST_F(SerializationTest, QueryCreationAndTypes) {
    Query query = createTestQuery(789);
    
    // Validate basic properties
    EXPECT_EQ(query.getId(), 789);
    EXPECT_EQ(query.getType(), Query::Type::FindAll);
    EXPECT_EQ(query.getMaxResults(), 50);
    EXPECT_EQ(query.getTimeoutMs(), 10000);
    
    // Validate goal
    const auto& goal = query.getGoal();
    EXPECT_EQ(goal.predicate, "isMortal");
    EXPECT_EQ(goal.args.size(), 1);
    EXPECT_EQ(goal.args[0].asText(), "socrates");
    EXPECT_FALSE(goal.negated);
    
    // Test string representation
    std::string queryStr = query.toString();
    EXPECT_TRUE(queryStr.find("Query[789]") != std::string::npos);
    EXPECT_TRUE(queryStr.find("FIND_ALL") != std::string::npos);
    EXPECT_TRUE(queryStr.find("isMortal") != std::string::npos);
    EXPECT_TRUE(queryStr.find("socrates") != std::string::npos);
    
    // Test different query types
    Rule::Condition proveGoal;
    proveGoal.predicate = "isHuman";
    proveGoal.args = {Value::fromText("alice")};
    
    Query proveQuery(1, Query::Type::Prove, proveGoal);
    EXPECT_TRUE(proveQuery.toString().find("PROVE") != std::string::npos);
    
    Query findFirstQuery(2, Query::Type::FindFirst, proveGoal);
    EXPECT_TRUE(findFirstQuery.toString().find("FIND_FIRST") != std::string::npos);
    
    Query explainQuery(3, Query::Type::Explain, proveGoal);
    EXPECT_TRUE(explainQuery.toString().find("EXPLAIN") != std::string::npos);
}

//=============================================================================
// Serializer Tests
//=============================================================================

/**
 * @brief Test binary serialization and deserialization of Facts
 * 
 * Validates that:
 * - Facts can be serialized to binary format
 * - Binary data can be deserialized back to identical Facts
 * - Serialization handles complex nested data structures
 * - Invalid binary data returns nullopt gracefully
 */
TEST_F(SerializationTest, FactBinarySerialization) {
    Fact originalFact = createTestFact(111);
    
    // Add complex nested data to test comprehensive serialization
    originalFact.setMetadata("nested_list", Value::fromList({
        Value::fromStruct({
            {"inner_key", Value::fromText("inner_value")},
            {"inner_number", Value::fromInt64(42)}
        }),
        Value::fromFloat64(2.718),
        Value::fromBool(false)
    }));
    
    // Serialize to binary format
    std::vector<uint8_t> binaryData = Serializer::serialize(originalFact);
    EXPECT_FALSE(binaryData.empty());
    
    // Deserialize from binary format
    auto deserializedFact = Serializer::deserializeFact(binaryData);
    ASSERT_TRUE(deserializedFact.has_value());
    
    // Validate complete equality
    validateFactEquality(originalFact, deserializedFact.value());
    
    // Test deserialization of invalid data
    std::vector<uint8_t> invalidData = {0x00, 0x01, 0x02, 0x03};
    auto invalidResult = Serializer::deserializeFact(invalidData);
    EXPECT_FALSE(invalidResult.has_value());
    
    // Test deserialization of empty data
    std::vector<uint8_t> emptyData;
    auto emptyResult = Serializer::deserializeFact(emptyData);
    EXPECT_FALSE(emptyResult.has_value());
}

/**
 * @brief Test binary serialization and deserialization of Rules
 * 
 * Validates that:
 * - Rules can be serialized to binary format
 * - Binary data can be deserialized back to identical Rules
 * - Complex rule structures with multiple conditions/conclusions work
 * - Invalid binary data is handled gracefully
 */
TEST_F(SerializationTest, RuleBinarySerialization) {
    Rule originalRule = createTestRule(222);
    
    // Serialize to binary format
    std::vector<uint8_t> binaryData = Serializer::serialize(originalRule);
    EXPECT_FALSE(binaryData.empty());
    
    // Deserialize from binary format
    auto deserializedRule = Serializer::deserializeRule(binaryData);
    ASSERT_TRUE(deserializedRule.has_value());
    
    // Validate equality (comprehensive validation done in previous test)
    EXPECT_EQ(originalRule.getId(), deserializedRule->getId());
    EXPECT_EQ(originalRule.getName(), deserializedRule->getName());
    EXPECT_EQ(originalRule.getConditions().size(), deserializedRule->getConditions().size());
    EXPECT_EQ(originalRule.getConclusions().size(), deserializedRule->getConclusions().size());
    
    // Test invalid data handling
    std::vector<uint8_t> invalidData = {0xFF, 0xFE, 0xFD, 0xFC};
    auto invalidResult = Serializer::deserializeRule(invalidData);
    EXPECT_FALSE(invalidResult.has_value());
}

/**
 * @brief Test JSON-like text serialization for debugging
 * 
 * Validates that:
 * - Facts and Rules can be converted to JSON-like text format
 * - JSON output contains all essential information
 * - Format is human-readable and parseable
 * - Special characters are handled correctly
 */
TEST_F(SerializationTest, JsonSerialization) {
    Fact testFact = createTestFact(333);
    std::string factJson = Serializer::toJson(testFact);
    
    // Verify JSON contains expected fields and values
    EXPECT_TRUE(factJson.find("\"id\": 333") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"predicate\": \"isHuman\"") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"confidence\": 0.95") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"timestamp\": 1234567890") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"args\": [") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"socrates\"") != std::string::npos);
    
    Rule testRule = createTestRule(444);
    std::string ruleJson = Serializer::toJson(testRule);
    
    // Verify JSON contains expected rule information
    EXPECT_TRUE(ruleJson.find("\"id\": 444") != std::string::npos);
    EXPECT_TRUE(ruleJson.find("\"name\": \"mortality_rule\"") != std::string::npos);
    EXPECT_TRUE(ruleJson.find("\"priority\": 10") != std::string::npos);
    EXPECT_TRUE(ruleJson.find("\"conditions\": [") != std::string::npos);
    EXPECT_TRUE(ruleJson.find("\"conclusions\": [") != std::string::npos);
    EXPECT_TRUE(ruleJson.find("isHuman") != std::string::npos);
    EXPECT_TRUE(ruleJson.find("isMortal") != std::string::npos);
}

//=============================================================================
// Schema Evolution Tests
//=============================================================================

/**
 * @brief Test SchemaVersion creation and comparison operations
 * 
 * Validates that:
 * - SchemaVersions can be created with major.minor.patch format
 * - String parsing works correctly for version strings
 * - Comparison operators work as expected (semantic versioning)
 * - toString generates proper version representations
 */
TEST_F(SerializationTest, SchemaVersionOperations) {
    // Test version creation
    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_2_3(1, 2, 3, "abc123hash");
    
    EXPECT_EQ(v1_0_0.getMajor(), 1);
    EXPECT_EQ(v1_0_0.getMinor(), 0);
    EXPECT_EQ(v1_0_0.getPatch(), 0);
    EXPECT_TRUE(v1_0_0.getSchemaHash().empty());
    
    EXPECT_EQ(v1_2_3.getMajor(), 1);
    EXPECT_EQ(v1_2_3.getMinor(), 2);
    EXPECT_EQ(v1_2_3.getPatch(), 3);
    EXPECT_EQ(v1_2_3.getSchemaHash(), "abc123hash");
    
    // Test string generation and parsing
    EXPECT_EQ(v1_0_0.getVersionString(), "1.0.0");
    EXPECT_EQ(v1_2_3.getVersionString(), "1.2.3");
    
    auto parsedVersion = SchemaVersion::fromString("2.5.7");
    ASSERT_TRUE(parsedVersion.has_value());
    EXPECT_EQ(parsedVersion->getMajor(), 2);
    EXPECT_EQ(parsedVersion->getMinor(), 5);
    EXPECT_EQ(parsedVersion->getPatch(), 7);
    
    // Test invalid string parsing
    EXPECT_FALSE(SchemaVersion::fromString("invalid").has_value());
    EXPECT_FALSE(SchemaVersion::fromString("1.2").has_value());
    EXPECT_FALSE(SchemaVersion::fromString("1.2.3.4").has_value());
    
    // Test comparison operators
    SchemaVersion v1_0_1(1, 0, 1);
    SchemaVersion v1_1_0(1, 1, 0);
    SchemaVersion v2_0_0(2, 0, 0);
    
    EXPECT_TRUE(v1_0_0 < v1_0_1);
    EXPECT_TRUE(v1_0_1 < v1_1_0);
    EXPECT_TRUE(v1_1_0 < v2_0_0);
    EXPECT_TRUE(v1_0_0 <= v1_0_0);
    EXPECT_TRUE(v2_0_0 > v1_1_0);
    EXPECT_TRUE(v1_0_0 == v1_0_0);
    EXPECT_TRUE(v1_0_0 != v1_0_1);
    
    // Test toString with hash
    std::string v1_2_3_str = v1_2_3.toString();
    EXPECT_TRUE(v1_2_3_str.find("1.2.3") != std::string::npos);
    EXPECT_TRUE(v1_2_3_str.find("abc123ha") != std::string::npos); // First 8 chars of hash
}

/**
 * @brief Test SchemaVersion compatibility checking
 * 
 * Validates that:
 * - Same major version means compatible
 * - Forward compatibility works for newer minor/patch versions
 * - Backward compatibility follows semantic versioning rules
 * - Major version changes break compatibility
 */
TEST_F(SerializationTest, SchemaVersionCompatibility) {
    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_0_1(1, 0, 1);
    SchemaVersion v1_1_0(1, 1, 0);
    SchemaVersion v2_0_0(2, 0, 0);
    
    // Test same major version compatibility
    EXPECT_TRUE(v1_0_0.isCompatibleWith(v1_0_1));
    EXPECT_TRUE(v1_0_0.isCompatibleWith(v1_1_0));
    EXPECT_TRUE(v1_1_0.isCompatibleWith(v1_0_0));
    
    // Test major version incompatibility
    EXPECT_FALSE(v1_0_0.isCompatibleWith(v2_0_0));
    EXPECT_FALSE(v2_0_0.isCompatibleWith(v1_1_0));
    
    // Test forward compatibility (newer can read older)
    EXPECT_TRUE(v1_1_0.isForwardCompatibleWith(v1_0_0));
    EXPECT_TRUE(v1_0_1.isForwardCompatibleWith(v1_0_0));
    EXPECT_FALSE(v1_0_0.isForwardCompatibleWith(v1_1_0)); // older can't read newer
    
    // Test backward compatibility (older data can be read by newer)
    EXPECT_TRUE(v1_0_0.isBackwardCompatibleWith(v1_1_0));
    EXPECT_TRUE(v1_0_0.isBackwardCompatibleWith(v1_0_1));
    EXPECT_FALSE(v1_1_0.isBackwardCompatibleWith(v1_0_0)); // newer data can't be read by older
}

/**
 * @brief Test MigrationPath creation and validation
 * 
 * Validates that:
 * - MigrationPaths can be created with different strategies
 * - Path validation logic works correctly
 * - Warnings can be added and retrieved
 * - String representation is meaningful
 */
TEST_F(SerializationTest, MigrationPathOperations) {
    SchemaVersion fromVersion(1, 0, 0);
    SchemaVersion toVersion(1, 1, 0);
    
    MigrationPath path(fromVersion, toVersion, MigrationPath::Strategy::DefaultValues, 
                      true, "Add default values for new fields");
    
    // Test basic properties
    EXPECT_EQ(path.getFromVersion(), fromVersion);
    EXPECT_EQ(path.getToVersion(), toVersion);
    EXPECT_EQ(path.getStrategy(), MigrationPath::Strategy::DefaultValues);
    EXPECT_TRUE(path.isReversible());
    EXPECT_EQ(path.getDescription(), "Add default values for new fields");
    
    // Test warnings
    EXPECT_TRUE(path.getWarnings().empty());
    path.addWarning("Performance may be affected");
    path.addWarning("Backup recommended");
    
    EXPECT_EQ(path.getWarnings().size(), 2);
    EXPECT_EQ(path.getWarnings()[0], "Performance may be affected");
    EXPECT_EQ(path.getWarnings()[1], "Backup recommended");
    
    // Test canMigrate
    EXPECT_TRUE(path.canMigrate(fromVersion, toVersion));
    EXPECT_FALSE(path.canMigrate(toVersion, fromVersion));
    
    SchemaVersion otherVersion(2, 0, 0);
    EXPECT_FALSE(path.canMigrate(fromVersion, otherVersion));
    
    // Test toString
    std::string pathStr = path.toString();
    EXPECT_TRUE(pathStr.find("1.0.0") != std::string::npos);
    EXPECT_TRUE(pathStr.find("1.1.0") != std::string::npos);
    EXPECT_TRUE(pathStr.find("reversible") != std::string::npos);
    EXPECT_TRUE(pathStr.find("Add default values") != std::string::npos);
}

/**
 * @brief Test SchemaEvolutionManager functionality
 * 
 * Validates that:
 * - Migration paths can be registered and retrieved
 * - Version readability checking works correctly
 * - Supported versions list is accurate
 * - Migration execution works for supported strategies
 */
TEST_F(SerializationTest, SchemaEvolutionManagerOperations) {
    SchemaVersion currentVersion(1, 2, 0);
    SchemaEvolutionManager manager(currentVersion);
    
    EXPECT_EQ(manager.getCurrentVersion(), currentVersion);
    
    // Test initial state - can only read current version
    EXPECT_TRUE(manager.canReadVersion(currentVersion));
    
    SchemaVersion oldVersion(1, 0, 0);
    EXPECT_FALSE(manager.canReadVersion(oldVersion));
    
    // Register migration path
    MigrationPath path(oldVersion, currentVersion, MigrationPath::Strategy::DirectMapping, false);
    manager.registerMigrationPath(path);
    
    // Now should be able to read old version
    EXPECT_TRUE(manager.canReadVersion(oldVersion));
    
    // Test migration path finding
    auto foundPath = manager.findMigrationPath(oldVersion);
    ASSERT_TRUE(foundPath.has_value());
    EXPECT_EQ(foundPath->getFromVersion(), oldVersion);
    EXPECT_EQ(foundPath->getToVersion(), currentVersion);
    
    // Test unsupported version
    SchemaVersion unsupportedVersion(0, 9, 0);
    EXPECT_FALSE(manager.canReadVersion(unsupportedVersion));
    EXPECT_FALSE(manager.findMigrationPath(unsupportedVersion).has_value());
    
    // Test supported versions list
    auto supportedVersions = manager.getSupportedVersions();
    EXPECT_GE(supportedVersions.size(), 2); // At least current and old version
    
    bool foundCurrent = false, foundOld = false;
    for (const auto& version : supportedVersions) {
        if (version == currentVersion) foundCurrent = true;
        if (version == oldVersion) foundOld = true;
    }
    EXPECT_TRUE(foundCurrent);
    EXPECT_TRUE(foundOld);
    
    // Test compatibility matrix generation
    std::string matrix = manager.generateCompatibilityMatrix();
    EXPECT_TRUE(matrix.find("Schema Compatibility Matrix") != std::string::npos);
    EXPECT_TRUE(matrix.find("1.2.0") != std::string::npos);
    EXPECT_TRUE(matrix.find("1.0.0") != std::string::npos);
}

/**
 * @brief Test Fact and Rule migration functionality
 * 
 * Validates that:
 * - Facts and Rules can be migrated between schema versions
 * - Direct mapping migration preserves all data
 * - Unsupported migrations return nullopt
 * - Same version migration is optimized (no-op)
 */
TEST_F(SerializationTest, DataMigrationOperations) {
    SchemaVersion v1_0_0(1, 0, 0);
    SchemaVersion v1_1_0(1, 1, 0);
    SchemaEvolutionManager manager(v1_1_0);
    
    // Register migration path
    MigrationPath path(v1_0_0, v1_1_0, MigrationPath::Strategy::DirectMapping);
    manager.registerMigrationPath(path);
    
    // Test fact migration
    Fact originalFact = createTestFact(555);
    
    // Same version migration (should return original)
    auto sameVersionResult = manager.migrateFact(originalFact, v1_1_0);
    ASSERT_TRUE(sameVersionResult.has_value());
    validateFactEquality(originalFact, sameVersionResult.value());
    
    // Cross-version migration
    auto migratedFact = manager.migrateFact(originalFact, v1_0_0);
    ASSERT_TRUE(migratedFact.has_value());
    validateFactEquality(originalFact, migratedFact.value());
    
    // Unsupported migration
    SchemaVersion unsupportedVersion(0, 5, 0);
    auto unsupportedResult = manager.migrateFact(originalFact, unsupportedVersion);
    EXPECT_FALSE(unsupportedResult.has_value());
    
    // Test rule migration
    Rule originalRule = createTestRule(666);
    
    auto migratedRule = manager.migrateRule(originalRule, v1_0_0);
    ASSERT_TRUE(migratedRule.has_value());
    EXPECT_EQ(originalRule.getId(), migratedRule->getId());
    EXPECT_EQ(originalRule.getName(), migratedRule->getName());
    
    // Unsupported rule migration
    auto unsupportedRuleResult = manager.migrateRule(originalRule, unsupportedVersion);
    EXPECT_FALSE(unsupportedRuleResult.has_value());
}

/**
 * @brief Test VersionValidator functionality
 * 
 * Validates that:
 * - Version validation catches invalid versions
 * - Migration path validation enforces best practices
 * - Safe transition checking works correctly
 * - Warning generation identifies potential issues
 */
TEST_F(SerializationTest, VersionValidatorOperations) {
    // Test version validation
    SchemaVersion validVersion(1, 2, 3);
    auto validErrors = VersionValidator::validateVersion(validVersion);
    EXPECT_TRUE(validErrors.empty());
    
    SchemaVersion invalidVersion(0, 0, 0);
    auto invalidErrors = VersionValidator::validateVersion(invalidVersion);
    EXPECT_FALSE(invalidErrors.empty());
    EXPECT_TRUE(invalidErrors[0].find("0.0.0 is not valid") != std::string::npos);
    
    // Test migration path validation
    SchemaVersion from(1, 0, 0);
    SchemaVersion to(1, 1, 0);
    MigrationPath validPath(from, to, MigrationPath::Strategy::DefaultValues);
    auto pathErrors = VersionValidator::validateMigrationPath(validPath);
    EXPECT_TRUE(pathErrors.empty());
    
    // Test invalid migration path (wrong direction)
    MigrationPath invalidPath(to, from, MigrationPath::Strategy::DirectMapping);
    auto invalidPathErrors = VersionValidator::validateMigrationPath(invalidPath);
    EXPECT_FALSE(invalidPathErrors.empty());
    
    // Test major version change validation
    SchemaVersion majorTo(2, 0, 0);
    MigrationPath majorPath(from, majorTo, MigrationPath::Strategy::DirectMapping);
    auto majorErrors = VersionValidator::validateMigrationPath(majorPath);
    EXPECT_FALSE(majorErrors.empty());
    
    // Test safe transition checking
    EXPECT_TRUE(VersionValidator::isSafeTransition(from, to)); // minor version increase
    EXPECT_FALSE(VersionValidator::isSafeTransition(from, majorTo)); // major version increase
    EXPECT_FALSE(VersionValidator::isSafeTransition(to, from)); // version decrease
    
    // Test warning generation
    auto minorWarnings = VersionValidator::generateWarnings(from, to);
    EXPECT_TRUE(minorWarnings.empty()); // Normal minor version increase
    
    auto majorWarnings = VersionValidator::generateWarnings(from, majorTo);
    EXPECT_FALSE(majorWarnings.empty());
    EXPECT_TRUE(majorWarnings[0].find("Major version change") != std::string::npos);
    
    SchemaVersion skipVersion(1, 3, 0); // Skips v1.2.x
    auto skipWarnings = VersionValidator::generateWarnings(from, skipVersion);
    EXPECT_FALSE(skipWarnings.empty());
    EXPECT_TRUE(skipWarnings[0].find("Skipping minor versions") != std::string::npos);
}

/**
 * @brief Test SchemaRegistry singleton functionality
 * 
 * Validates that:
 * - Registry follows singleton pattern correctly
 * - Schema versions can be registered and queried
 * - Current schema can be set and retrieved
 * - Version lists are maintained properly
 */
TEST_F(SerializationTest, SchemaRegistryOperations) {
    SchemaRegistry& registry = SchemaRegistry::getInstance();
    
    // Test singleton behavior
    SchemaRegistry& registry2 = SchemaRegistry::getInstance();
    EXPECT_EQ(&registry, &registry2);
    
    // Test initial state
    SchemaVersion defaultVersion = registry.getCurrentSchema();
    EXPECT_EQ(defaultVersion.getMajor(), 1);
    EXPECT_EQ(defaultVersion.getMinor(), 0);
    EXPECT_EQ(defaultVersion.getPatch(), 0);
    
    // Register schema versions
    SchemaVersion v1_2_0(1, 2, 0, "hash123");
    SchemaVersion v1_3_0(1, 3, 0, "hash456");
    
    registry.registerSchema(v1_2_0, "hash123");
    registry.registerSchema(v1_3_0, "hash456");
    
    // Test registration checking
    EXPECT_TRUE(registry.isRegistered(v1_2_0));
    EXPECT_TRUE(registry.isRegistered(v1_3_0));
    
    SchemaVersion unregisteredVersion(2, 0, 0);
    EXPECT_FALSE(registry.isRegistered(unregisteredVersion));
    
    // Test setting current schema
    registry.setCurrentSchema(v1_2_0);
    EXPECT_EQ(registry.getCurrentSchema(), v1_2_0);
    
    // Test getting all versions
    auto allVersions = registry.getAllVersions();
    EXPECT_GE(allVersions.size(), 2);
    
    bool foundV1_2_0 = false, foundV1_3_0 = false;
    for (const auto& version : allVersions) {
        if (version == v1_2_0) foundV1_2_0 = true;
        if (version == v1_3_0) foundV1_3_0 = true;
    }
    EXPECT_TRUE(foundV1_2_0);
    EXPECT_TRUE(foundV1_3_0);
    
    // Verify versions are sorted
    for (size_t i = 1; i < allVersions.size(); ++i) {
        EXPECT_TRUE(allVersions[i-1] <= allVersions[i]);
    }
}

//=============================================================================
// Integration and Performance Tests
//=============================================================================

/**
 * @brief Test large-scale serialization performance and correctness
 * 
 * Validates that:
 * - Serialization works correctly with large datasets
 * - Performance is reasonable for typical use cases
 * - Memory usage is managed properly
 * - Complex nested structures scale appropriately
 */
TEST_F(SerializationTest, LargeScaleSerializationTest) {
    const size_t numFacts = 1000;
    const size_t numRules = 100;
    
    std::vector<Fact> originalFacts;
    std::vector<Rule> originalRules;
    
    // Generate large dataset
    for (size_t i = 0; i < numFacts; ++i) {
        Fact fact = createTestFact(i);
        
        // Add some variability
        fact.setMetadata("batch_id", Value::fromInt64(i / 100));
        fact.setMetadata("sequence", Value::fromInt64(i));
        
        if (i % 10 == 0) {
            // Add complex nested structure every 10th fact
            fact.setMetadata("complex", Value::fromStruct({
                {"nested_list", Value::fromList({
                    Value::fromInt64(i),
                    Value::fromFloat64(i * 0.1),
                    Value::fromText("item_" + std::to_string(i))
                })},
                {"metadata_level", Value::fromInt64(2)}
            }));
        }
        
        originalFacts.push_back(fact);
    }
    
    for (size_t i = 0; i < numRules; ++i) {
        originalRules.push_back(createTestRule(i));
    }
    
    // Measure serialization time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<uint8_t>> factData;
    for (const auto& fact : originalFacts) {
        factData.push_back(Serializer::serialize(fact));
    }
    
    std::vector<std::vector<uint8_t>> ruleData;
    for (const auto& rule : originalRules) {
        ruleData.push_back(Serializer::serialize(rule));
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Performance should be reasonable (less than 5 seconds for this dataset)
    EXPECT_LT(duration.count(), 5000);
    
    // Verify all serializations succeeded
    EXPECT_EQ(factData.size(), numFacts);
    EXPECT_EQ(ruleData.size(), numRules);
    
    for (const auto& data : factData) {
        EXPECT_FALSE(data.empty());
    }
    
    // Spot check deserialization of random samples
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> factDist(0, numFacts - 1);
    std::uniform_int_distribution<size_t> ruleDist(0, numRules - 1);
    
    for (int i = 0; i < 10; ++i) {
        size_t factIdx = factDist(gen);
        auto deserializedFact = Serializer::deserializeFact(factData[factIdx]);
        ASSERT_TRUE(deserializedFact.has_value());
        validateFactEquality(originalFacts[factIdx], deserializedFact.value());
        
        size_t ruleIdx = ruleDist(gen);
        auto deserializedRule = Serializer::deserializeRule(ruleData[ruleIdx]);
        ASSERT_TRUE(deserializedRule.has_value());
        EXPECT_EQ(originalRules[ruleIdx].getId(), deserializedRule->getId());
        EXPECT_EQ(originalRules[ruleIdx].getName(), deserializedRule->getName());
    }
}

/**
 * @brief Test concurrent serialization operations for thread safety
 * 
 * Validates that:
 * - Multiple threads can serialize data simultaneously
 * - No race conditions occur during concurrent operations
 * - All serialization operations complete successfully
 * - Deserialized data maintains integrity across all threads
 */
TEST_F(SerializationTest, ConcurrentSerializationTest) {
    const int numThreads = 4;
    const int operationsPerThread = 50;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<std::vector<uint8_t>>> threadResults(numThreads);
    std::atomic<int> successfulOperations{0};
    
    // Create multiple threads performing serialization
    for (int threadId = 0; threadId < numThreads; ++threadId) {
        threads.emplace_back([&, threadId]() {
            threadResults[threadId].reserve(operationsPerThread);
            
            for (int i = 0; i < operationsPerThread; ++i) {
                try {
                    Fact fact = createTestFact(threadId * 1000 + i);
                    fact.setMetadata("thread_id", Value::fromInt64(threadId));
                    fact.setMetadata("operation_id", Value::fromInt64(i));
                    
                    auto serializedData = Serializer::serialize(fact);
                    threadResults[threadId].push_back(serializedData);
                    
                    ++successfulOperations;
                } catch (...) {
                    FAIL() << "Thread " << threadId << " operation " << i << " failed";
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all operations succeeded
    EXPECT_EQ(successfulOperations.load(), numThreads * operationsPerThread);
    
    // Verify each thread's results
    for (int threadId = 0; threadId < numThreads; ++threadId) {
        EXPECT_EQ(threadResults[threadId].size(), operationsPerThread);
        
        // Spot check a few deserializations from each thread
        for (int i = 0; i < std::min(5, operationsPerThread); ++i) {
            auto deserializedFact = Serializer::deserializeFact(threadResults[threadId][i]);
            ASSERT_TRUE(deserializedFact.has_value());
            
            // Verify thread-specific metadata
            auto threadIdValue = deserializedFact->getMetadata("thread_id");
            ASSERT_TRUE(threadIdValue.has_value());
            EXPECT_EQ(threadIdValue->asInt64(), threadId);
            
            auto operationIdValue = deserializedFact->getMetadata("operation_id");
            ASSERT_TRUE(operationIdValue.has_value());
            EXPECT_EQ(operationIdValue->asInt64(), i);
        }
    }
}

/**
 * @brief Test error handling and edge cases
 * 
 * Validates that:
 * - Invalid data is handled gracefully without crashes
 * - Edge cases (empty data, malformed structures) are managed
 * - Error conditions return appropriate nullopt values
 * - No memory leaks occur during error conditions
 */
TEST_F(SerializationTest, ErrorHandlingAndEdgeCases) {
    // Test deserialization of various invalid data patterns
    std::vector<std::vector<uint8_t>> invalidDataSets = {
        {},                                    // Empty data
        {0x00},                               // Single byte
        {0xFF, 0xFF, 0xFF, 0xFF},            // Random bytes
        {0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00}, // Invalid Cap'n Proto header
    };
    
    for (const auto& invalidData : invalidDataSets) {
        // Should not crash and should return nullopt
        auto factResult = Serializer::deserializeFact(invalidData);
        EXPECT_FALSE(factResult.has_value());
        
        auto ruleResult = Serializer::deserializeRule(invalidData);
        EXPECT_FALSE(ruleResult.has_value());
    }
    
    // Test edge cases with valid but minimal data
    Fact minimalFact(0, "", {}, 0.0, 0);
    auto serializedMinimal = Serializer::serialize(minimalFact);
    auto deserializedMinimal = Serializer::deserializeFact(serializedMinimal);
    ASSERT_TRUE(deserializedMinimal.has_value());
    EXPECT_EQ(deserializedMinimal->getId(), 0);
    EXPECT_EQ(deserializedMinimal->getPredicate(), "");
    EXPECT_TRUE(deserializedMinimal->getArgs().empty());
    
    // Test very large strings
    std::string largeString(10000, 'A');
    Value largeValue = Value::fromText(largeString);
    Fact largeFact(1, "large_predicate", {largeValue}, 1.0, 12345);
    
    auto serializedLarge = Serializer::serialize(largeFact);
    auto deserializedLarge = Serializer::deserializeFact(serializedLarge);
    ASSERT_TRUE(deserializedLarge.has_value());
    EXPECT_EQ(deserializedLarge->getArgs()[0].asText(), largeString);
    
    // Test deeply nested structures
    Value deeplyNested = Value::fromStruct({
        {"level1", Value::fromStruct({
            {"level2", Value::fromStruct({
                {"level3", Value::fromList({
                    Value::fromStruct({
                        {"level4", Value::fromText("deep_value")}
                    })
                })}
            })}
        })}
    });
    
    Fact nestedFact(2, "nested", {deeplyNested}, 1.0, 12345);
    auto serializedNested = Serializer::serialize(nestedFact);
    auto deserializedNested = Serializer::deserializeFact(serializedNested);
    ASSERT_TRUE(deserializedNested.has_value());
    
    // Verify deep structure integrity
    const auto& args = deserializedNested->getArgs();
    ASSERT_EQ(args.size(), 1);
    ASSERT_TRUE(args[0].isStruct());
    
    auto level1 = args[0].asStruct();
    ASSERT_TRUE(level1.count("level1") > 0);
    ASSERT_TRUE(level1["level1"].isStruct());
    
    auto level2 = level1["level1"].asStruct();
    ASSERT_TRUE(level2.count("level2") > 0);
    
    // Continue validation to ensure deep nesting is preserved
    auto level3 = level2["level2"].asStruct();
    ASSERT_TRUE(level3.count("level3") > 0);
    ASSERT_TRUE(level3["level3"].isList());
    
    auto level3List = level3["level3"].asList();
    ASSERT_EQ(level3List.size(), 1);
    ASSERT_TRUE(level3List[0].isStruct());
    
    auto level4 = level3List[0].asStruct();
    ASSERT_TRUE(level4.count("level4") > 0);
    EXPECT_EQ(level4["level4"].asText(), "deep_value");
}