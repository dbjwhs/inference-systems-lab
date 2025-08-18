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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/inference_types.hpp"
#include "../src/schema_evolution.hpp"

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

    std::string test_data_dir;  ///< Directory for test files

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
            Value::from_text("socrates"), Value::from_int64(42), Value::from_float64(3.14159)};

        Fact fact(id, "isHuman", args, 0.95, 1234567890);
        fact.set_metadata("source", Value::from_text("unit_test"));
        fact.set_metadata("created_by", Value::from_text("test_framework"));

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
        cond1.predicate_ = "isHuman";
        cond1.args_ = {Value::from_text("X")};
        cond1.negated_ = false;
        conditions.push_back(cond1);

        Rule::Condition cond2;
        cond2.predicate_ = "hasAge";
        cond2.args_ = {Value::from_text("X"), Value::from_text("Age")};
        cond2.negated_ = false;
        conditions.push_back(cond2);

        // Create conclusions: isMortal(X)
        std::vector<Rule::Conclusion> conclusions;

        Rule::Conclusion concl1;
        concl1.predicate_ = "isMortal";
        concl1.args_ = {Value::from_text("X")};
        concl1.confidence_ = 0.99;
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
        goal.predicate_ = "isMortal";
        goal.args_ = {Value::from_text("socrates")};
        goal.negated_ = false;

        return Query(id, Query::Type::FIND_ALL, goal, 50, 10000);
    }

    /**
     * @brief Validate that two Facts are equivalent
     * @param original Original fact
     * @param deserialized Deserialized fact
     */
    void validateFactEquality(const Fact& original, const Fact& deserialized) {
        EXPECT_EQ(original.get_id(), deserialized.get_id());
        EXPECT_EQ(original.get_predicate(), deserialized.get_predicate());
        EXPECT_DOUBLE_EQ(original.get_confidence(), deserialized.get_confidence());
        EXPECT_EQ(original.get_timestamp(), deserialized.get_timestamp());

        // Validate arguments
        const auto& origArgs = original.get_args();
        const auto& deserArgs = deserialized.get_args();
        EXPECT_EQ(origArgs.size(), deserArgs.size());

        for (size_t i = 0; i < origArgs.size(); ++i) {
            validateValueEquality(origArgs[i], deserArgs[i]);
        }

        // Validate metadata
        const auto& origMeta = original.get_metadata();
        const auto& deserMeta = deserialized.get_metadata();
        EXPECT_EQ(origMeta.size(), deserMeta.size());

        for (const auto& [key, value] : origMeta) {
            auto deserValue = deserialized.get_metadata(key);
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
        EXPECT_EQ(original.is_int64(), deserialized.is_int64());
        EXPECT_EQ(original.is_float64(), deserialized.is_float64());
        EXPECT_EQ(original.is_text(), deserialized.is_text());
        EXPECT_EQ(original.is_bool(), deserialized.is_bool());
        EXPECT_EQ(original.is_list(), deserialized.is_list());
        EXPECT_EQ(original.is_struct(), deserialized.is_struct());

        // Check value content based on type
        if (original.is_int64()) {
            EXPECT_EQ(original.as_int64(), deserialized.as_int64());
        } else if (original.is_float64()) {
            EXPECT_DOUBLE_EQ(original.as_float64(), deserialized.as_float64());
        } else if (original.is_text()) {
            EXPECT_EQ(original.as_text(), deserialized.as_text());
        } else if (original.is_bool()) {
            EXPECT_EQ(original.as_bool(), deserialized.as_bool());
        } else if (original.is_list()) {
            const auto& origList = original.as_list();
            const auto& deserList = deserialized.as_list();
            EXPECT_EQ(origList.size(), deserList.size());
            for (size_t i = 0; i < origList.size(); ++i) {
                validateValueEquality(origList[i], deserList[i]);
            }
        } else if (original.is_struct()) {
            const auto& origStruct = original.as_struct();
            const auto& deserStruct = deserialized.as_struct();
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
    EXPECT_TRUE(defaultValue.is_int64());
    EXPECT_EQ(defaultValue.as_int64(), 0);
    EXPECT_FALSE(defaultValue.is_float64());
    EXPECT_FALSE(defaultValue.is_text());
    EXPECT_FALSE(defaultValue.is_bool());
    EXPECT_FALSE(defaultValue.is_list());
    EXPECT_FALSE(defaultValue.is_struct());

    // Test int64 value
    Value intValue = Value::from_int64(42);
    EXPECT_TRUE(intValue.is_int64());
    EXPECT_EQ(intValue.as_int64(), 42);
    EXPECT_FALSE(intValue.is_float64());

    // Test float64 value
    Value floatValue = Value::from_float64(3.14159);
    EXPECT_TRUE(floatValue.is_float64());
    EXPECT_DOUBLE_EQ(floatValue.as_float64(), 3.14159);
    EXPECT_FALSE(floatValue.is_int64());

    // Test text value
    Value textValue = Value::from_text("hello world");
    EXPECT_TRUE(textValue.is_text());
    EXPECT_EQ(textValue.as_text(), "hello world");
    EXPECT_FALSE(textValue.is_int64());

    // Test bool value
    Value boolValue = Value::from_bool(true);
    EXPECT_TRUE(boolValue.is_bool());
    EXPECT_TRUE(boolValue.as_bool());
    EXPECT_FALSE(boolValue.is_text());

    // Test list value
    std::vector<Value> listData = {
        Value::from_int64(1), Value::from_text("test"), Value::from_bool(false)};
    Value listValue = Value::from_list(listData);
    EXPECT_TRUE(listValue.is_list());
    EXPECT_FALSE(listValue.is_struct());

    auto retrievedList = listValue.as_list();
    EXPECT_EQ(retrievedList.size(), 3);
    EXPECT_EQ(retrievedList[0].as_int64(), 1);
    EXPECT_EQ(retrievedList[1].as_text(), "test");
    EXPECT_FALSE(retrievedList[2].as_bool());

    // Test struct value
    std::unordered_map<std::string, Value> structData = {{"name", Value::from_text("Alice")},
                                                         {"age", Value::from_int64(30)},
                                                         {"score", Value::from_float64(95.5)}};
    Value structValue = Value::from_struct(structData);
    EXPECT_TRUE(structValue.is_struct());
    EXPECT_FALSE(structValue.is_list());

    auto retrievedStruct = structValue.as_struct();
    EXPECT_EQ(retrievedStruct.size(), 3);
    EXPECT_EQ(retrievedStruct["name"].as_text(), "Alice");
    EXPECT_EQ(retrievedStruct["age"].as_int64(), 30);
    EXPECT_DOUBLE_EQ(retrievedStruct["score"].as_float64(), 95.5);
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
    Value intValue = Value::from_int64(123);
    Value textValue = Value::from_text("test");

    // Test successful extraction
    EXPECT_EQ(intValue.as_int64(), 123);
    EXPECT_EQ(textValue.as_text(), "test");

    // Test unsafe extraction with wrong type (should throw)
    EXPECT_THROW(intValue.as_text(), std::runtime_error);
    EXPECT_THROW(textValue.as_int64(), std::runtime_error);
    EXPECT_THROW(intValue.as_float64(), std::runtime_error);
    EXPECT_THROW(intValue.as_bool(), std::runtime_error);
    EXPECT_THROW(intValue.as_list(), std::runtime_error);
    EXPECT_THROW(intValue.as_struct(), std::runtime_error);

    // Test safe extraction with correct type
    auto intResult = intValue.try_as_int64();
    ASSERT_TRUE(intResult.has_value());
    EXPECT_EQ(intResult.value(), 123);

    auto textResult = textValue.try_as_text();
    ASSERT_TRUE(textResult.has_value());
    EXPECT_EQ(textResult.value(), "test");

    // Test safe extraction with wrong type (should return nullopt)
    EXPECT_FALSE(intValue.try_as_text().has_value());
    EXPECT_FALSE(textValue.try_as_int64().has_value());
    EXPECT_FALSE(intValue.try_as_float64().has_value());
    EXPECT_FALSE(intValue.try_as_bool().has_value());
    EXPECT_FALSE(intValue.try_as_list().has_value());
    EXPECT_FALSE(intValue.try_as_struct().has_value());
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
    EXPECT_EQ(Value::from_int64(42).to_string(), "42");
    EXPECT_EQ(Value::from_float64(3.14).to_string(), "3.140000");  // Note: std::to_string precision
    EXPECT_EQ(Value::from_text("hello").to_string(), "\"hello\"");
    EXPECT_EQ(Value::from_bool(true).to_string(), "true");
    EXPECT_EQ(Value::from_bool(false).to_string(), "false");

    // Test list representation
    std::vector<Value> listData = {
        Value::from_int64(1), Value::from_text("test"), Value::from_bool(true)};
    Value listValue = Value::from_list(listData);
    std::string listStr = listValue.to_string();
    EXPECT_TRUE(listStr.find("[") != std::string::npos);
    EXPECT_TRUE(listStr.find("]") != std::string::npos);
    EXPECT_TRUE(listStr.find("1") != std::string::npos);
    EXPECT_TRUE(listStr.find("\"test\"") != std::string::npos);
    EXPECT_TRUE(listStr.find("true") != std::string::npos);

    // Test struct representation
    std::unordered_map<std::string, Value> structData = {{"name", Value::from_text("Alice")},
                                                         {"count", Value::from_int64(5)}};
    Value structValue = Value::from_struct(structData);
    std::string structStr = structValue.to_string();
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
        {"simple_int", Value::from_int64(42)},
        {"simple_text", Value::from_text("hello")},
        {"nested_list",
         Value::from_list({Value::from_int64(1), Value::from_int64(2), Value::from_int64(3)})},
        {"nested_struct",
         Value::from_struct({{"inner_bool", Value::from_bool(true)},
                             {"inner_float", Value::from_float64(2.718)}})}};
    Value originalValue = Value::from_struct(structData);

    // Write to Cap'n Proto and read back
    capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Value>();
    originalValue.write_to(builder);

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
    std::vector<Value> args = {Value::from_text("socrates"), Value::from_int64(70)};

    uint64_t testTimestamp = 1234567890;
    Fact fact(42, "hasAge", args, 0.95, testTimestamp);

    // Validate basic properties
    EXPECT_EQ(fact.get_id(), 42);
    EXPECT_EQ(fact.get_predicate(), "hasAge");
    EXPECT_DOUBLE_EQ(fact.get_confidence(), 0.95);
    EXPECT_EQ(fact.get_timestamp(), testTimestamp);

    // Validate arguments
    const auto& factArgs = fact.get_args();
    EXPECT_EQ(factArgs.size(), 2);
    EXPECT_EQ(factArgs[0].as_text(), "socrates");
    EXPECT_EQ(factArgs[1].as_int64(), 70);

    // Test metadata operations
    EXPECT_TRUE(fact.get_metadata().empty());

    fact.set_metadata("source", Value::from_text("knowledge_base"));
    fact.set_metadata("confidence_level", Value::from_text("high"));
    fact.set_metadata("version", Value::from_int64(1));

    EXPECT_EQ(fact.get_metadata().size(), 3);

    auto sourceValue = fact.get_metadata("source");
    ASSERT_TRUE(sourceValue.has_value());
    EXPECT_EQ(sourceValue->as_text(), "knowledge_base");

    auto versionValue = fact.get_metadata("version");
    ASSERT_TRUE(versionValue.has_value());
    EXPECT_EQ(versionValue->as_int64(), 1);

    auto nonExistentValue = fact.get_metadata("nonexistent");
    EXPECT_FALSE(nonExistentValue.has_value());

    // Test string representation
    std::string factStr = fact.to_string();
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
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();

    // Create fact with auto-generated timestamp
    Fact fact(1, "test", {Value::from_text("arg")}, 1.0, 0);

    auto afterCreation = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();

    // Verify timestamp is in reasonable range
    EXPECT_GE(fact.get_timestamp(), beforeCreation);
    EXPECT_LE(fact.get_timestamp(), afterCreation + 1000);  // Allow 1 second buffer

    // Test explicit timestamp preservation
    uint64_t explicitTimestamp = 9876543210;
    Fact explicitFact(2, "test", {Value::from_text("arg")}, 1.0, explicitTimestamp);
    EXPECT_EQ(explicitFact.get_timestamp(), explicitTimestamp);
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
    originalFact.set_metadata(
        "complex_data",
        Value::from_list({Value::from_int64(1),
                          Value::from_text("nested"),
                          Value::from_struct({{"key", Value::from_bool(true)}})}));

    // Serialize to Cap'n Proto
    capnp::MallocMessageBuilder message;
    auto builder = message.initRoot<schemas::Fact>();
    originalFact.write_to(builder);

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
    EXPECT_EQ(rule.get_id(), 123);
    EXPECT_EQ(rule.get_name(), "mortality_rule");
    EXPECT_EQ(rule.get_priority(), 10);
    EXPECT_DOUBLE_EQ(rule.get_confidence(), 0.95);

    // Validate conditions
    const auto& conditions = rule.get_conditions();
    EXPECT_EQ(conditions.size(), 2);

    EXPECT_EQ(conditions[0].predicate_, "isHuman");
    EXPECT_EQ(conditions[0].args_.size(), 1);
    EXPECT_EQ(conditions[0].args_[0].as_text(), "X");
    EXPECT_FALSE(conditions[0].negated_);

    EXPECT_EQ(conditions[1].predicate_, "hasAge");
    EXPECT_EQ(conditions[1].args_.size(), 2);
    EXPECT_EQ(conditions[1].args_[0].as_text(), "X");
    EXPECT_EQ(conditions[1].args_[1].as_text(), "Age");
    EXPECT_FALSE(conditions[1].negated_);

    // Validate conclusions
    const auto& conclusions = rule.get_conclusions();
    EXPECT_EQ(conclusions.size(), 1);

    EXPECT_EQ(conclusions[0].predicate_, "isMortal");
    EXPECT_EQ(conclusions[0].args_.size(), 1);
    EXPECT_EQ(conclusions[0].args_[0].as_text(), "X");
    EXPECT_DOUBLE_EQ(conclusions[0].confidence_, 0.99);

    // Test condition toString
    std::string condStr = conditions[0].to_string();
    EXPECT_TRUE(condStr.find("isHuman") != std::string::npos);
    EXPECT_TRUE(condStr.find("X") != std::string::npos);

    // Test conclusion toString
    std::string conclStr = conclusions[0].to_string();
    EXPECT_TRUE(conclStr.find("isMortal") != std::string::npos);
    EXPECT_TRUE(conclStr.find("X") != std::string::npos);
    EXPECT_TRUE(conclStr.find("confidence: 0.99") != std::string::npos);

    // Test rule toString
    std::string ruleStr = rule.to_string();
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
    negatedCond.predicate_ = "isDead";
    negatedCond.args_ = {Value::from_text("X")};
    negatedCond.negated_ = true;
    conditions.push_back(negatedCond);

    // Add a normal condition
    Rule::Condition normalCond;
    normalCond.predicate_ = "isAlive";
    normalCond.args_ = {Value::from_text("X")};
    normalCond.negated_ = false;
    conditions.push_back(normalCond);

    std::vector<Rule::Conclusion> conclusions;
    Rule::Conclusion concl;
    concl.predicate_ = "canThink";
    concl.args_ = {Value::from_text("X")};
    conclusions.push_back(concl);

    Rule rule(1, "thinking_rule", conditions, conclusions);

    // Validate negation flags
    EXPECT_TRUE(rule.get_conditions()[0].negated_);
    EXPECT_FALSE(rule.get_conditions()[1].negated_);

    // Test string representation includes NOT
    std::string condStr = rule.get_conditions()[0].to_string();
    EXPECT_TRUE(condStr.find("NOT") != std::string::npos);
    EXPECT_TRUE(condStr.find("isDead") != std::string::npos);

    // Validate normal condition doesn't have NOT
    std::string normalCondStr = rule.get_conditions()[1].to_string();
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
    originalRule.write_to(builder);

    // Deserialize from Cap'n Proto
    auto reader = builder.asReader();
    Rule deserializedRule(reader);

    // Validate basic properties
    EXPECT_EQ(originalRule.get_id(), deserializedRule.get_id());
    EXPECT_EQ(originalRule.get_name(), deserializedRule.get_name());
    EXPECT_EQ(originalRule.get_priority(), deserializedRule.get_priority());
    EXPECT_DOUBLE_EQ(originalRule.get_confidence(), deserializedRule.get_confidence());

    // Validate conditions
    const auto& origConditions = originalRule.get_conditions();
    const auto& deserConditions = deserializedRule.get_conditions();
    EXPECT_EQ(origConditions.size(), deserConditions.size());

    for (size_t i = 0; i < origConditions.size(); ++i) {
        EXPECT_EQ(origConditions[i].predicate_, deserConditions[i].predicate_);
        EXPECT_EQ(origConditions[i].negated_, deserConditions[i].negated_);
        EXPECT_EQ(origConditions[i].args_.size(), deserConditions[i].args_.size());

        for (size_t j = 0; j < origConditions[i].args_.size(); ++j) {
            validateValueEquality(origConditions[i].args_[j], deserConditions[i].args_[j]);
        }
    }

    // Validate conclusions
    const auto& origConclusions = originalRule.get_conclusions();
    const auto& deserConclusions = deserializedRule.get_conclusions();
    EXPECT_EQ(origConclusions.size(), deserConclusions.size());

    for (size_t i = 0; i < origConclusions.size(); ++i) {
        EXPECT_EQ(origConclusions[i].predicate_, deserConclusions[i].predicate_);
        EXPECT_DOUBLE_EQ(origConclusions[i].confidence_, deserConclusions[i].confidence_);
        EXPECT_EQ(origConclusions[i].args_.size(), deserConclusions[i].args_.size());

        for (size_t j = 0; j < origConclusions[i].args_.size(); ++j) {
            validateValueEquality(origConclusions[i].args_[j], deserConclusions[i].args_[j]);
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
    EXPECT_EQ(query.get_id(), 789);
    EXPECT_EQ(query.get_type(), Query::Type::FIND_ALL);
    EXPECT_EQ(query.get_max_results(), 50);
    EXPECT_EQ(query.get_timeout_ms(), 10000);

    // Validate goal
    const auto& goal = query.get_goal();
    EXPECT_EQ(goal.predicate_, "isMortal");
    EXPECT_EQ(goal.args_.size(), 1);
    EXPECT_EQ(goal.args_[0].as_text(), "socrates");
    EXPECT_FALSE(goal.negated_);

    // Test string representation
    std::string queryStr = query.to_string();
    EXPECT_TRUE(queryStr.find("Query[789]") != std::string::npos);
    EXPECT_TRUE(queryStr.find("FIND_ALL") != std::string::npos);
    EXPECT_TRUE(queryStr.find("isMortal") != std::string::npos);
    EXPECT_TRUE(queryStr.find("socrates") != std::string::npos);

    // Test different query types
    Rule::Condition proveGoal;
    proveGoal.predicate_ = "isHuman";
    proveGoal.args_ = {Value::from_text("alice")};

    Query proveQuery(1, Query::Type::PROVE, proveGoal);
    EXPECT_TRUE(proveQuery.to_string().find("PROVE") != std::string::npos);

    Query findFirstQuery(2, Query::Type::FIND_FIRST, proveGoal);
    EXPECT_TRUE(findFirstQuery.to_string().find("FIND_FIRST") != std::string::npos);

    Query explainQuery(3, Query::Type::EXPLAIN, proveGoal);
    EXPECT_TRUE(explainQuery.to_string().find("EXPLAIN") != std::string::npos);
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
    originalFact.set_metadata(
        "nested_list",
        Value::from_list({Value::from_struct({{"inner_key", Value::from_text("inner_value")},
                                              {"inner_number", Value::from_int64(42)}}),
                          Value::from_float64(2.718),
                          Value::from_bool(false)}));

    // Serialize to binary format
    std::vector<uint8_t> binaryData = Serializer::serialize(originalFact);
    EXPECT_FALSE(binaryData.empty());

    // Deserialize from binary format
    auto deserializedFact = Serializer::deserialize_fact(binaryData);
    ASSERT_TRUE(deserializedFact.has_value());

    // Validate complete equality
    validateFactEquality(originalFact, deserializedFact.value());

    // Test deserialization of invalid data
    std::vector<uint8_t> invalidData = {0x00, 0x01, 0x02, 0x03};
    auto invalidResult = Serializer::deserialize_fact(invalidData);
    EXPECT_FALSE(invalidResult.has_value());

    // Test deserialization of empty data
    std::vector<uint8_t> emptyData;
    auto emptyResult = Serializer::deserialize_fact(emptyData);
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
    auto deserializedRule = Serializer::deserialize_rule(binaryData);
    ASSERT_TRUE(deserializedRule.has_value());

    // Validate equality (comprehensive validation done in previous test)
    EXPECT_EQ(originalRule.get_id(), deserializedRule->get_id());
    EXPECT_EQ(originalRule.get_name(), deserializedRule->get_name());
    EXPECT_EQ(originalRule.get_conditions().size(), deserializedRule->get_conditions().size());
    EXPECT_EQ(originalRule.get_conclusions().size(), deserializedRule->get_conclusions().size());

    // Test invalid data handling
    std::vector<uint8_t> invalidData = {0xFF, 0xFE, 0xFD, 0xFC};
    auto invalidResult = Serializer::deserialize_rule(invalidData);
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
    std::string factJson = Serializer::to_json(testFact);

    // Verify JSON contains expected fields and values
    EXPECT_TRUE(factJson.find("\"id\": 333") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"predicate\": \"isHuman\"") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"confidence\": 0.95") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"timestamp\": 1234567890") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"args\": [") != std::string::npos);
    EXPECT_TRUE(factJson.find("\"socrates\"") != std::string::npos);

    Rule testRule = createTestRule(444);
    std::string ruleJson = Serializer::to_json(testRule);

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

    EXPECT_EQ(v1_0_0.get_major(), 1);
    EXPECT_EQ(v1_0_0.get_minor(), 0);
    EXPECT_EQ(v1_0_0.get_patch(), 0);
    EXPECT_TRUE(v1_0_0.get_schema_hash().empty());

    EXPECT_EQ(v1_2_3.get_major(), 1);
    EXPECT_EQ(v1_2_3.get_minor(), 2);
    EXPECT_EQ(v1_2_3.get_patch(), 3);
    EXPECT_EQ(v1_2_3.get_schema_hash(), "abc123hash");

    // Test string generation and parsing
    EXPECT_EQ(v1_0_0.get_version_string(), "1.0.0");
    EXPECT_EQ(v1_2_3.get_version_string(), "1.2.3");

    auto parsedVersion = SchemaVersion::from_string("2.5.7");
    ASSERT_TRUE(parsedVersion.has_value());
    EXPECT_EQ(parsedVersion->get_major(), 2);
    EXPECT_EQ(parsedVersion->get_minor(), 5);
    EXPECT_EQ(parsedVersion->get_patch(), 7);

    // Test invalid string parsing
    EXPECT_FALSE(SchemaVersion::from_string("invalid").has_value());
    EXPECT_FALSE(SchemaVersion::from_string("1.2").has_value());
    EXPECT_FALSE(SchemaVersion::from_string("1.2.3.4").has_value());

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
    std::string v1_2_3_str = v1_2_3.to_string();
    EXPECT_TRUE(v1_2_3_str.find("1.2.3") != std::string::npos);
    EXPECT_TRUE(v1_2_3_str.find("abc123ha") != std::string::npos);  // First 8 chars of hash
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
    EXPECT_TRUE(v1_0_0.is_compatible_with(v1_0_1));
    EXPECT_TRUE(v1_0_0.is_compatible_with(v1_1_0));
    EXPECT_TRUE(v1_1_0.is_compatible_with(v1_0_0));

    // Test major version incompatibility
    EXPECT_FALSE(v1_0_0.is_compatible_with(v2_0_0));
    EXPECT_FALSE(v2_0_0.is_compatible_with(v1_1_0));

    // Test forward compatibility (newer can read older)
    EXPECT_TRUE(v1_1_0.is_forward_compatible_with(v1_0_0));
    EXPECT_TRUE(v1_0_1.is_forward_compatible_with(v1_0_0));
    EXPECT_FALSE(v1_0_0.is_forward_compatible_with(v1_1_0));  // older can't read newer

    // Test backward compatibility (older data can be read by newer)
    EXPECT_TRUE(v1_0_0.is_backward_compatible_with(v1_1_0));
    EXPECT_TRUE(v1_0_0.is_backward_compatible_with(v1_0_1));
    EXPECT_FALSE(v1_1_0.is_backward_compatible_with(v1_0_0));  // newer data can't be read by older
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

    MigrationPath path(fromVersion,
                       toVersion,
                       MigrationPath::Strategy::DEFAULT_VALUES,
                       true,
                       "Add default values for new fields");

    // Test basic properties
    EXPECT_EQ(path.get_from_version(), fromVersion);
    EXPECT_EQ(path.get_to_version(), toVersion);
    EXPECT_EQ(path.get_strategy(), MigrationPath::Strategy::DEFAULT_VALUES);
    EXPECT_TRUE(path.is_reversible());
    EXPECT_EQ(path.get_description(), "Add default values for new fields");

    // Test warnings
    EXPECT_TRUE(path.get_warnings().empty());
    path.add_warning("Performance may be affected");
    path.add_warning("Backup recommended");

    EXPECT_EQ(path.get_warnings().size(), 2);
    EXPECT_EQ(path.get_warnings()[0], "Performance may be affected");
    EXPECT_EQ(path.get_warnings()[1], "Backup recommended");

    // Test canMigrate
    EXPECT_TRUE(path.can_migrate(fromVersion, toVersion));
    EXPECT_FALSE(path.can_migrate(toVersion, fromVersion));

    SchemaVersion otherVersion(2, 0, 0);
    EXPECT_FALSE(path.can_migrate(fromVersion, otherVersion));

    // Test toString
    std::string pathStr = path.to_string();
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

    EXPECT_EQ(manager.get_current_version(), currentVersion);

    // Test initial state - can only read current version
    EXPECT_TRUE(manager.can_read_version(currentVersion));

    SchemaVersion oldVersion(1, 0, 0);
    EXPECT_FALSE(manager.can_read_version(oldVersion));

    // Register migration path
    MigrationPath path(oldVersion, currentVersion, MigrationPath::Strategy::DIRECT_MAPPING, false);
    manager.register_migration_path(path);

    // Now should be able to read old version
    EXPECT_TRUE(manager.can_read_version(oldVersion));

    // Test migration path finding
    auto foundPath = manager.find_migration_path(oldVersion);
    ASSERT_TRUE(foundPath.has_value());
    EXPECT_EQ(foundPath->get_from_version(), oldVersion);
    EXPECT_EQ(foundPath->get_to_version(), currentVersion);

    // Test unsupported version
    SchemaVersion unsupportedVersion(0, 9, 0);
    EXPECT_FALSE(manager.can_read_version(unsupportedVersion));
    EXPECT_FALSE(manager.find_migration_path(unsupportedVersion).has_value());

    // Test supported versions list
    auto supportedVersions = manager.get_supported_versions();
    EXPECT_GE(supportedVersions.size(), 2);  // At least current and old version

    bool foundCurrent = false, foundOld = false;
    for (const auto& version : supportedVersions) {
        if (version == currentVersion)
            foundCurrent = true;
        if (version == oldVersion)
            foundOld = true;
    }
    EXPECT_TRUE(foundCurrent);
    EXPECT_TRUE(foundOld);

    // Test compatibility matrix generation
    std::string matrix = manager.generate_compatibility_matrix();
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
    MigrationPath path(v1_0_0, v1_1_0, MigrationPath::Strategy::DIRECT_MAPPING);
    manager.register_migration_path(path);

    // Test fact migration
    Fact originalFact = createTestFact(555);

    // Same version migration (should return original)
    auto sameVersionResult = manager.migrate_fact(originalFact, v1_1_0);
    ASSERT_TRUE(sameVersionResult.has_value());
    validateFactEquality(originalFact, sameVersionResult.value());

    // Cross-version migration
    auto migratedFact = manager.migrate_fact(originalFact, v1_0_0);
    ASSERT_TRUE(migratedFact.has_value());
    validateFactEquality(originalFact, migratedFact.value());

    // Unsupported migration
    SchemaVersion unsupportedVersion(0, 5, 0);
    auto unsupportedResult = manager.migrate_fact(originalFact, unsupportedVersion);
    EXPECT_FALSE(unsupportedResult.has_value());

    // Test rule migration
    Rule originalRule = createTestRule(666);

    auto migratedRule = manager.migrate_rule(originalRule, v1_0_0);
    ASSERT_TRUE(migratedRule.has_value());
    EXPECT_EQ(originalRule.get_id(), migratedRule->get_id());
    EXPECT_EQ(originalRule.get_name(), migratedRule->get_name());

    // Unsupported rule migration
    auto unsupportedRuleResult = manager.migrate_rule(originalRule, unsupportedVersion);
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
    auto validErrors = VersionValidator::validate_version(validVersion);
    EXPECT_TRUE(validErrors.empty());

    SchemaVersion invalidVersion(0, 0, 0);
    auto invalidErrors = VersionValidator::validate_version(invalidVersion);
    EXPECT_FALSE(invalidErrors.empty());
    EXPECT_TRUE(invalidErrors[0].find("0.0.0 is not valid") != std::string::npos);

    // Test migration path validation
    SchemaVersion from(1, 0, 0);
    SchemaVersion to(1, 1, 0);
    MigrationPath validPath(from, to, MigrationPath::Strategy::DEFAULT_VALUES);
    auto pathErrors = VersionValidator::validate_migration_path(validPath);
    EXPECT_TRUE(pathErrors.empty());

    // Test invalid migration path (wrong direction)
    MigrationPath invalidPath(to, from, MigrationPath::Strategy::DIRECT_MAPPING);
    auto invalidPathErrors = VersionValidator::validate_migration_path(invalidPath);
    EXPECT_FALSE(invalidPathErrors.empty());

    // Test major version change validation
    SchemaVersion majorTo(2, 0, 0);
    MigrationPath majorPath(from, majorTo, MigrationPath::Strategy::DIRECT_MAPPING);
    auto majorErrors = VersionValidator::validate_migration_path(majorPath);
    EXPECT_FALSE(majorErrors.empty());

    // Test safe transition checking
    EXPECT_TRUE(VersionValidator::is_safe_transition(from, to));        // minor version increase
    EXPECT_FALSE(VersionValidator::is_safe_transition(from, majorTo));  // major version increase
    EXPECT_FALSE(VersionValidator::is_safe_transition(to, from));       // version decrease

    // Test warning generation
    auto minorWarnings = VersionValidator::generate_warnings(from, to);
    EXPECT_TRUE(minorWarnings.empty());  // Normal minor version increase

    auto majorWarnings = VersionValidator::generate_warnings(from, majorTo);
    EXPECT_FALSE(majorWarnings.empty());
    EXPECT_TRUE(majorWarnings[0].find("Major version change") != std::string::npos);

    SchemaVersion skipVersion(1, 3, 0);  // Skips v1.2.x
    auto skipWarnings = VersionValidator::generate_warnings(from, skipVersion);
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
    SchemaRegistry& registry = SchemaRegistry::get_instance();

    // Test singleton behavior
    SchemaRegistry& registry2 = SchemaRegistry::get_instance();
    EXPECT_EQ(&registry, &registry2);

    // Test initial state
    SchemaVersion defaultVersion = registry.get_current_schema();
    EXPECT_EQ(defaultVersion.get_major(), 1);
    EXPECT_EQ(defaultVersion.get_minor(), 0);
    EXPECT_EQ(defaultVersion.get_patch(), 0);

    // Register schema versions
    SchemaVersion v1_2_0(1, 2, 0, "hash123");
    SchemaVersion v1_3_0(1, 3, 0, "hash456");

    registry.register_schema(v1_2_0, "hash123");
    registry.register_schema(v1_3_0, "hash456");

    // Test registration checking
    EXPECT_TRUE(registry.is_registered(v1_2_0));
    EXPECT_TRUE(registry.is_registered(v1_3_0));

    SchemaVersion unregisteredVersion(2, 0, 0);
    EXPECT_FALSE(registry.is_registered(unregisteredVersion));

    // Test setting current schema
    registry.set_current_schema(v1_2_0);
    EXPECT_EQ(registry.get_current_schema(), v1_2_0);

    // Test getting all versions
    auto allVersions = registry.get_all_versions();
    EXPECT_GE(allVersions.size(), 2);

    bool foundV1_2_0 = false, foundV1_3_0 = false;
    for (const auto& version : allVersions) {
        if (version == v1_2_0)
            foundV1_2_0 = true;
        if (version == v1_3_0)
            foundV1_3_0 = true;
    }
    EXPECT_TRUE(foundV1_2_0);
    EXPECT_TRUE(foundV1_3_0);

    // Verify versions are sorted
    for (size_t i = 1; i < allVersions.size(); ++i) {
        EXPECT_TRUE(allVersions[i - 1] <= allVersions[i]);
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
        fact.set_metadata("batch_id", Value::from_int64(i / 100));
        fact.set_metadata("sequence", Value::from_int64(i));

        if (i % 10 == 0) {
            // Add complex nested structure every 10th fact
            fact.set_metadata(
                "complex",
                Value::from_struct(
                    {{"nested_list",
                      Value::from_list({Value::from_int64(i),
                                        Value::from_float64(i * 0.1),
                                        Value::from_text("item_" + std::to_string(i))})},
                     {"metadata_level", Value::from_int64(2)}}));
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
        auto deserializedFact = Serializer::deserialize_fact(factData[factIdx]);
        ASSERT_TRUE(deserializedFact.has_value());
        validateFactEquality(originalFacts[factIdx], deserializedFact.value());

        size_t ruleIdx = ruleDist(gen);
        auto deserializedRule = Serializer::deserialize_rule(ruleData[ruleIdx]);
        ASSERT_TRUE(deserializedRule.has_value());
        EXPECT_EQ(originalRules[ruleIdx].get_id(), deserializedRule->get_id());
        EXPECT_EQ(originalRules[ruleIdx].get_name(), deserializedRule->get_name());
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
                    fact.set_metadata("thread_id", Value::from_int64(threadId));
                    fact.set_metadata("operation_id", Value::from_int64(i));

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
            auto deserializedFact = Serializer::deserialize_fact(threadResults[threadId][i]);
            ASSERT_TRUE(deserializedFact.has_value());

            // Verify thread-specific metadata
            auto threadIdValue = deserializedFact->get_metadata("thread_id");
            ASSERT_TRUE(threadIdValue.has_value());
            EXPECT_EQ(threadIdValue->as_int64(), threadId);

            auto operationIdValue = deserializedFact->get_metadata("operation_id");
            ASSERT_TRUE(operationIdValue.has_value());
            EXPECT_EQ(operationIdValue->as_int64(), i);
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
        {},                                                // Empty data
        {0x00},                                            // Single byte
        {0xFF, 0xFF, 0xFF, 0xFF},                          // Random bytes
        {0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00},  // Invalid Cap'n Proto header
    };

    for (const auto& invalidData : invalidDataSets) {
        // Should not crash and should return nullopt
        auto factResult = Serializer::deserialize_fact(invalidData);
        EXPECT_FALSE(factResult.has_value());

        auto ruleResult = Serializer::deserialize_rule(invalidData);
        EXPECT_FALSE(ruleResult.has_value());
    }

    // Test edge cases with valid but minimal data
    Fact minimalFact(0, "", {}, 0.0, 0);
    auto serializedMinimal = Serializer::serialize(minimalFact);
    auto deserializedMinimal = Serializer::deserialize_fact(serializedMinimal);
    ASSERT_TRUE(deserializedMinimal.has_value());
    EXPECT_EQ(deserializedMinimal->get_id(), 0);
    EXPECT_EQ(deserializedMinimal->get_predicate(), "");
    EXPECT_TRUE(deserializedMinimal->get_args().empty());

    // Test very large strings
    std::string largeString(10000, 'A');
    Value largeValue = Value::from_text(largeString);
    Fact largeFact(1, "large_predicate", {largeValue}, 1.0, 12345);

    auto serializedLarge = Serializer::serialize(largeFact);
    auto deserializedLarge = Serializer::deserialize_fact(serializedLarge);
    ASSERT_TRUE(deserializedLarge.has_value());
    EXPECT_EQ(deserializedLarge->get_args()[0].as_text(), largeString);

    // Test deeply nested structures
    Value deeplyNested = Value::from_struct(
        {{"level1",
          Value::from_struct(
              {{"level2",
                Value::from_struct({{"level3",
                                     Value::from_list({Value::from_struct(
                                         {{"level4", Value::from_text("deep_value")}})})}})}})}});

    Fact nestedFact(2, "nested", {deeplyNested}, 1.0, 12345);
    auto serializedNested = Serializer::serialize(nestedFact);
    auto deserializedNested = Serializer::deserialize_fact(serializedNested);
    ASSERT_TRUE(deserializedNested.has_value());

    // Verify deep structure integrity
    const auto& args = deserializedNested->get_args();
    ASSERT_EQ(args.size(), 1);
    ASSERT_TRUE(args[0].is_struct());

    auto level1 = args[0].as_struct();
    ASSERT_TRUE(level1.count("level1") > 0);
    ASSERT_TRUE(level1["level1"].is_struct());

    auto level2 = level1["level1"].as_struct();
    ASSERT_TRUE(level2.count("level2") > 0);

    // Continue validation to ensure deep nesting is preserved
    auto level3 = level2["level2"].as_struct();
    ASSERT_TRUE(level3.count("level3") > 0);
    ASSERT_TRUE(level3["level3"].is_list());

    auto level3List = level3["level3"].as_list();
    ASSERT_EQ(level3List.size(), 1);
    ASSERT_TRUE(level3List[0].is_struct());

    auto level4 = level3List[0].as_struct();
    ASSERT_TRUE(level4.count("level4") > 0);
    EXPECT_EQ(level4["level4"].as_text(), "deep_value");
}
