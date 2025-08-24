/**
 * @file test_inference_builders.cpp
 * @brief Essential tests for inference builder classes
 *
 * Basic coverage tests to boost coverage from 0% to target levels.
 */

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../src/inference_builders.hpp"
#include "../src/inference_types.hpp"

using namespace inference_lab::common;

class InferenceBuilderTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Fresh setup for each test
    }
};

//=============================================================================
// FactBuilder Tests - Core functionality coverage
//=============================================================================

TEST_F(InferenceBuilderTest, FactBuilderBasicConstruction) {
    auto fact = FactBuilder("isHuman").with_arg("Socrates").build();

    EXPECT_EQ(fact.get_predicate(), "isHuman");
    EXPECT_EQ(fact.get_args().size(), 1);
    EXPECT_EQ(fact.get_args()[0].as_text(), "Socrates");
    EXPECT_GT(fact.get_id(), 0);  // Auto-generated ID
}

TEST_F(InferenceBuilderTest, FactBuilderValueTypes) {
    auto fact = FactBuilder("multitype")
                    .with_arg(static_cast<int64_t>(42))
                    .with_arg(3.14)
                    .with_arg("text")
                    .with_arg(true)
                    .build();

    EXPECT_EQ(fact.get_args().size(), 4);
    EXPECT_EQ(fact.get_args()[0].as_int64(), 42);
    EXPECT_DOUBLE_EQ(fact.get_args()[1].as_float64(), 3.14);
    EXPECT_EQ(fact.get_args()[2].as_text(), "text");
    EXPECT_EQ(fact.get_args()[3].as_bool(), true);
}

TEST_F(InferenceBuilderTest, FactBuilderWithMetadata) {
    auto fact = FactBuilder("testFact")
                    .with_id(12345)
                    .with_confidence(0.85)
                    .with_timestamp(1000000)
                    .with_metadata("source", Value::from_text("database"))
                    .build();

    EXPECT_EQ(fact.get_id(), 12345);
    EXPECT_DOUBLE_EQ(fact.get_confidence(), 0.85);
    EXPECT_EQ(fact.get_timestamp(), 1000000);

    auto source = fact.get_metadata("source");
    EXPECT_TRUE(source.has_value());
    EXPECT_EQ(source->as_text(), "database");
}

TEST_F(InferenceBuilderTest, FactBuilderAutoIdGeneration) {
    auto fact1 = FactBuilder("fact1").build();
    auto fact2 = FactBuilder("fact2").build();

    EXPECT_NE(fact1.get_id(), fact2.get_id());
    EXPECT_GT(fact1.get_id(), 0);
    EXPECT_GT(fact2.get_id(), 0);
}

TEST_F(InferenceBuilderTest, FactBuilderCStringHandling) {
    const char* cstr = "C-style string";
    auto fact = FactBuilder("test").with_arg(cstr).build();

    EXPECT_EQ(fact.get_args()[0].as_text(), "C-style string");
}

//=============================================================================
// RuleBuilder Tests - Core functionality coverage
//=============================================================================

TEST_F(InferenceBuilderTest, RuleBuilderBasicConstruction) {
    auto rule = RuleBuilder("mortality")
                    .when("isHuman")
                    .with_variable("X")
                    .then("isMortal")
                    .with_variable("X")
                    .build();

    EXPECT_EQ(rule.get_name(), "mortality");
    EXPECT_EQ(rule.get_conditions().size(), 1);
    EXPECT_EQ(rule.get_conclusions().size(), 1);
    EXPECT_GT(rule.get_id(), 0);
}

TEST_F(InferenceBuilderTest, RuleBuilderWithConditions) {
    auto rule = RuleBuilder("complex_rule")
                    .when("predicate1")
                    .with_variable("X")
                    .when_not("predicate2")
                    .with_variable("Y")
                    .then("conclusion")
                    .with_variable("X")
                    .build();

    EXPECT_EQ(rule.get_conditions().size(), 2);
    EXPECT_FALSE(rule.get_conditions()[0].negated_);
    EXPECT_TRUE(rule.get_conditions()[1].negated_);
}

TEST_F(InferenceBuilderTest, RuleBuilderWithProperties) {
    auto rule = RuleBuilder("prioritized")
                    .with_id(5000)
                    .with_priority(10)
                    .when("condition")
                    .with_variable("X")
                    .then("result")
                    .with_variable("X")
                    .build();

    EXPECT_EQ(rule.get_id(), 5000);
    EXPECT_EQ(rule.get_priority(), 10);
}

TEST_F(InferenceBuilderTest, RuleBuilderDirectCondition) {
    std::vector<Value> args = {Value::from_text("$X"), Value::from_int64(100)};

    auto rule = RuleBuilder("direct")
                    .when_condition("predicate", args, false)
                    .then("conclusion")
                    .with_variable("X")
                    .build();

    EXPECT_EQ(rule.get_conditions().size(), 1);
    EXPECT_EQ(rule.get_conditions()[0].predicate_, "predicate");
    EXPECT_EQ(rule.get_conditions()[0].args_.size(), 2);
}

//=============================================================================
// QueryBuilder Tests - Core functionality coverage
//=============================================================================

TEST_F(InferenceBuilderTest, QueryBuilderBasicConstruction) {
    auto query = QueryBuilder(Query::Type::FIND_ALL).goal("person").with_variable("X").build();

    EXPECT_EQ(query.get_type(), Query::Type::FIND_ALL);
    EXPECT_EQ(query.get_goal().predicate_, "person");
    EXPECT_GT(query.get_id(), 0);
}

TEST_F(InferenceBuilderTest, QueryBuilderQueryTypes) {
    auto find_query = QueryBuilder(Query::Type::FIND_ALL).goal("predicate").build();

    auto prove_query = QueryBuilder(Query::Type::PROVE).goal("fact").build();

    EXPECT_EQ(find_query.get_type(), Query::Type::FIND_ALL);
    EXPECT_EQ(prove_query.get_type(), Query::Type::PROVE);
}

TEST_F(InferenceBuilderTest, QueryBuilderWithArguments) {
    auto query = QueryBuilder(Query::Type::FIND_ALL)
                     .goal("relationship")
                     .with_arg("parent")
                     .with_variable("X")
                     .build();

    EXPECT_EQ(query.get_goal().args_.size(), 2);
    EXPECT_EQ(query.get_goal().args_[0].as_text(), "parent");
}

//=============================================================================
// Edge Cases and Integration Tests
//=============================================================================

TEST_F(InferenceBuilderTest, EmptyBuilders) {
    auto fact = FactBuilder("empty").build();
    EXPECT_EQ(fact.get_args().size(), 0);
    EXPECT_GT(fact.get_id(), 0);

    // Rules require at least one condition and conclusion, test minimal rule
    auto rule = RuleBuilder("minimal_rule")
                    .when("condition")
                    .with_variable("X")
                    .then("result")
                    .with_variable("X")
                    .build();
    EXPECT_EQ(rule.get_conditions().size(), 1);
    EXPECT_EQ(rule.get_conclusions().size(), 1);
}

TEST_F(InferenceBuilderTest, LargeDataHandling) {
    FactBuilder builder("large");
    for (int i = 0; i < 50; ++i) {
        builder.with_arg(static_cast<int64_t>(i));
    }
    auto fact = builder.build();

    EXPECT_EQ(fact.get_args().size(), 50);
    EXPECT_EQ(fact.get_args()[0].as_int64(), 0);
    EXPECT_EQ(fact.get_args()[49].as_int64(), 49);
}

TEST_F(InferenceBuilderTest, SpecialStringHandling) {
    auto fact = FactBuilder("special")
                    .with_arg("")  // Empty string
                    .with_arg("with spaces")
                    .with_arg("unicode: 测试")
                    .build();

    EXPECT_EQ(fact.get_args()[0].as_text(), "");
    EXPECT_EQ(fact.get_args()[1].as_text(), "with spaces");
    EXPECT_EQ(fact.get_args()[2].as_text(), "unicode: 测试");
}

TEST_F(InferenceBuilderTest, ThreadSafetyIdGeneration) {
    std::vector<std::thread> threads;
    std::vector<uint64_t> ids;
    std::mutex id_mutex;

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&id_mutex, &ids]() {
            auto fact = FactBuilder("concurrent").build();
            std::lock_guard<std::mutex> lock(id_mutex);
            ids.push_back(fact.get_id());
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Check all IDs are unique
    std::sort(ids.begin(), ids.end());
    auto last = std::unique(ids.begin(), ids.end());
    EXPECT_EQ(last - ids.begin(), 5);
}
