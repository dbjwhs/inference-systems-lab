/**
 * @file test_forward_chaining.cpp
 * @brief Comprehensive tests for forward chaining inference engine
 *
 * Tests the forward chaining engine with various scenarios including:
 * - Classic "Socrates is mortal" reasoning
 * - Complex rule chains and variable binding
 * - Conflict resolution strategies
 * - Performance and correctness validation
 * - Error handling and edge cases
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "../../common/src/inference_builders.hpp"
#include "../../common/src/logging.hpp"
#include "../src/forward_chaining.hpp"
#include "../src/inference_engine.hpp"

using namespace inference_lab::engines;
using namespace inference_lab::common;

/**
 * @brief Test fixture for forward chaining engine tests
 */
class ForwardChainingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create engine with tracing enabled for testing
        auto result = create_forward_chaining_engine(ConflictResolutionStrategy::PRIORITY_ORDER,
                                                     100,  // reasonable limit for tests
                                                     true  // enable tracing for debugging
        );

        ASSERT_TRUE(result.is_ok()) << "Failed to create forward chaining engine";
        engine_ = std::move(result.unwrap());

        // Create some common facts for testing
        socrates_human_ = Fact(1, "isHuman", {Value::from_text("socrates")});
        plato_human_ = Fact(2, "isHuman", {Value::from_text("plato")});
        aristotle_human_ = Fact(3, "isHuman", {Value::from_text("aristotle")});

        // Create mortality rule: isHuman(X) -> isMortal(X)
        mortality_rule_ = Rule(1, "mortality_rule");
        mortality_rule_.add_condition("isHuman", {Value::from_text("X")});
        mortality_rule_.add_conclusion("isMortal", {Value::from_text("X")});
        mortality_rule_.set_priority(10);
    }

    void TearDown() override {
        // Reset metrics for clean test separation
        engine_->reset_metrics();
    }

    std::unique_ptr<ForwardChainingEngine> engine_;
    Fact socrates_human_;
    Fact plato_human_;
    Fact aristotle_human_;
    Rule mortality_rule_;
};

// ================================================================================================
// BASIC FUNCTIONALITY TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, BasicFactAddition) {
    // Test adding single fact
    auto result = engine_->add_fact(socrates_human_);
    EXPECT_TRUE(result.is_ok());

    auto facts = engine_->get_all_facts();
    EXPECT_EQ(facts.size(), 1);
    EXPECT_EQ(facts[0].predicate(), "isHuman");
    EXPECT_EQ(facts[0].args().size(), 1);
    EXPECT_EQ(facts[0].args()[0].as_text(), "socrates");
}

TEST_F(ForwardChainingTest, BasicRuleAddition) {
    // Test adding single rule
    auto result = engine_->add_rule(mortality_rule_);
    EXPECT_TRUE(result.is_ok());

    auto rules = engine_->get_all_rules();
    EXPECT_EQ(rules.size(), 1);
    EXPECT_EQ(rules[0].name(), "mortality_rule");
    EXPECT_EQ(rules[0].conditions().size(), 1);
    EXPECT_EQ(rules[0].conclusions().size(), 1);
}

TEST_F(ForwardChainingTest, FactIndexing) {
    // Add facts and test indexing
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());
    EXPECT_TRUE(engine_->add_fact(plato_human_).is_ok());

    auto human_facts = engine_->get_facts_by_predicate("isHuman");
    EXPECT_EQ(human_facts.size(), 2);

    auto nonexistent_facts = engine_->get_facts_by_predicate("nonexistent");
    EXPECT_EQ(nonexistent_facts.size(), 0);
}

// ================================================================================================
// CLASSICAL REASONING TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, SocratesIsMortalReasoning) {
    // Classic AI reasoning example

    // Add fact: Socrates is human
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());

    // Add rule: All humans are mortal
    EXPECT_TRUE(engine_->add_rule(mortality_rule_).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok()) << "Forward chaining failed";

    auto derived_facts = result.unwrap();
    EXPECT_EQ(derived_facts.size(), 1);

    // Check that we derived "isMortal(socrates)"
    EXPECT_EQ(derived_facts[0].predicate(), "isMortal");
    EXPECT_EQ(derived_facts[0].args().size(), 1);
    EXPECT_EQ(derived_facts[0].args()[0].as_text(), "socrates");

    // Verify it was added to knowledge base
    auto mortal_facts = engine_->get_facts_by_predicate("isMortal");
    EXPECT_EQ(mortal_facts.size(), 1);
}

TEST_F(ForwardChainingTest, MultipleHumansMortalityReasoning) {
    // Add multiple human facts
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());
    EXPECT_TRUE(engine_->add_fact(plato_human_).is_ok());
    EXPECT_TRUE(engine_->add_fact(aristotle_human_).is_ok());

    // Add mortality rule
    EXPECT_TRUE(engine_->add_rule(mortality_rule_).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    auto derived_facts = result.unwrap();
    EXPECT_EQ(derived_facts.size(), 3);  // All three should be derived as mortal

    // Check all derived facts
    auto mortal_facts = engine_->get_facts_by_predicate("isMortal");
    EXPECT_EQ(mortal_facts.size(), 3);

    // Verify each philosopher is mortal
    std::vector<std::string> names;
    for (const auto& fact : mortal_facts) {
        names.push_back(fact.args()[0].as_text());
    }

    EXPECT_TRUE(std::find(names.begin(), names.end(), "socrates") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "plato") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "aristotle") != names.end());
}

// ================================================================================================
// COMPLEX RULE CHAINING TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, MultiStepRuleChaining) {
    // Test rule chaining: isHuman(X) -> isAnimal(X) -> isLiving(X)

    // Add facts
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());

    // Rule 1: isHuman(X) -> isAnimal(X)
    Rule human_to_animal(1, "human_to_animal");
    human_to_animal.add_condition("isHuman", {Value::from_text("X")});
    human_to_animal.add_conclusion("isAnimal", {Value::from_text("X")});
    human_to_animal.set_priority(10);
    EXPECT_TRUE(engine_->add_rule(human_to_animal).is_ok());

    // Rule 2: isAnimal(X) -> isLiving(X)
    Rule animal_to_living(2, "animal_to_living");
    animal_to_living.add_condition("isAnimal", {Value::from_text("X")});
    animal_to_living.add_conclusion("isLiving", {Value::from_text("X")});
    animal_to_living.set_priority(9);
    EXPECT_TRUE(engine_->add_rule(animal_to_living).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    auto derived_facts = result.unwrap();
    EXPECT_EQ(derived_facts.size(), 2);  // isAnimal(socrates) and isLiving(socrates)

    // Verify both facts were derived
    auto animal_facts = engine_->get_facts_by_predicate("isAnimal");
    auto living_facts = engine_->get_facts_by_predicate("isLiving");

    EXPECT_EQ(animal_facts.size(), 1);
    EXPECT_EQ(living_facts.size(), 1);
    EXPECT_EQ(animal_facts[0].args()[0].as_text(), "socrates");
    EXPECT_EQ(living_facts[0].args()[0].as_text(), "socrates");
}

TEST_F(ForwardChainingTest, MultiConditionRule) {
    // Test rule with multiple conditions: isHuman(X) AND livesIn(X, Y) -> inhabitant(X, Y)

    // Add facts
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());

    Fact socrates_lives_athens(
        2, "livesIn", {Value::from_text("socrates"), Value::from_text("athens")});
    EXPECT_TRUE(engine_->add_fact(socrates_lives_athens).is_ok());

    // Create rule with two conditions
    Rule inhabitant_rule(1, "inhabitant_rule");
    inhabitant_rule.add_condition("isHuman", {Value::from_text("X")});
    inhabitant_rule.add_condition("livesIn", {Value::from_text("X"), Value::from_text("Y")});
    inhabitant_rule.add_conclusion("inhabitant", {Value::from_text("X"), Value::from_text("Y")});
    EXPECT_TRUE(engine_->add_rule(inhabitant_rule).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    auto derived_facts = result.unwrap();
    EXPECT_EQ(derived_facts.size(), 1);

    // Verify derived fact
    EXPECT_EQ(derived_facts[0].predicate(), "inhabitant");
    EXPECT_EQ(derived_facts[0].args().size(), 2);
    EXPECT_EQ(derived_facts[0].args()[0].as_text(), "socrates");
    EXPECT_EQ(derived_facts[0].args()[1].as_text(), "athens");
}

// ================================================================================================
// CONFLICT RESOLUTION TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, PriorityBasedConflictResolution) {
    // Test that higher priority rules fire first

    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());

    // High priority rule
    Rule high_priority_rule(1, "high_priority");
    high_priority_rule.add_condition("isHuman", {Value::from_text("X")});
    high_priority_rule.add_conclusion("isWise", {Value::from_text("X")});
    high_priority_rule.set_priority(20);
    EXPECT_TRUE(engine_->add_rule(high_priority_rule).is_ok());

    // Low priority rule
    Rule low_priority_rule(2, "low_priority");
    low_priority_rule.add_condition("isHuman", {Value::from_text("X")});
    low_priority_rule.add_conclusion("isMortal", {Value::from_text("X")});
    low_priority_rule.set_priority(5);
    EXPECT_TRUE(engine_->add_rule(low_priority_rule).is_ok());

    // Run inference with tracing
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    // Check firing trace to verify order
    auto trace = engine_->get_firing_trace();
    EXPECT_EQ(trace.size(), 2);

    // High priority rule should fire first
    EXPECT_EQ(trace[0].rule_name, "high_priority");
    EXPECT_EQ(trace[1].rule_name, "low_priority");
}

// ================================================================================================
// VARIABLE BINDING TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, ComplexVariableBinding) {
    // Test complex variable binding across multiple arguments

    // Add facts: parent(john, mary), parent(mary, susan)
    Fact john_parent_mary(1, "parent", {Value::from_text("john"), Value::from_text("mary")});
    Fact mary_parent_susan(2, "parent", {Value::from_text("mary"), Value::from_text("susan")});

    EXPECT_TRUE(engine_->add_fact(john_parent_mary).is_ok());
    EXPECT_TRUE(engine_->add_fact(mary_parent_susan).is_ok());

    // Rule: parent(X, Y) AND parent(Y, Z) -> grandparent(X, Z)
    Rule grandparent_rule(1, "grandparent_rule");
    grandparent_rule.add_condition("parent", {Value::from_text("X"), Value::from_text("Y")});
    grandparent_rule.add_condition("parent", {Value::from_text("Y"), Value::from_text("Z")});
    grandparent_rule.add_conclusion("grandparent", {Value::from_text("X"), Value::from_text("Z")});
    EXPECT_TRUE(engine_->add_rule(grandparent_rule).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    auto derived_facts = result.unwrap();
    EXPECT_EQ(derived_facts.size(), 1);

    // Should derive grandparent(john, susan)
    EXPECT_EQ(derived_facts[0].predicate(), "grandparent");
    EXPECT_EQ(derived_facts[0].args()[0].as_text(), "john");
    EXPECT_EQ(derived_facts[0].args()[1].as_text(), "susan");
}

// ================================================================================================
// ERROR HANDLING TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, InvalidRuleFormat) {
    // Test rule with no conditions
    Rule invalid_rule(1, "invalid_rule");
    // Don't add any conditions or conclusions

    auto result = engine_->add_rule(invalid_rule);
    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.err(), ForwardChainingError::INVALID_RULE_FORMAT);
}

TEST_F(ForwardChainingTest, MaxIterationsLimit) {
    // Create engine with very low iteration limit
    auto result = create_forward_chaining_engine(ConflictResolutionStrategy::PRIORITY_ORDER,
                                                 1,  // Very low limit
                                                 false);
    ASSERT_TRUE(result.is_ok());
    auto limited_engine = std::move(result.unwrap());

    // Add facts and rules that could cause many iterations
    EXPECT_TRUE(limited_engine->add_fact(socrates_human_).is_ok());
    EXPECT_TRUE(limited_engine->add_rule(mortality_rule_).is_ok());

    // Should complete within limit for this simple case
    auto inference_result = limited_engine->run_forward_chaining();
    EXPECT_TRUE(inference_result.is_ok());
}

TEST_F(ForwardChainingTest, DuplicateFactHandling) {
    // Test that duplicate facts are not added
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());  // Same fact again

    auto facts = engine_->get_all_facts();
    EXPECT_EQ(facts.size(), 1);  // Should only have one fact
}

// ================================================================================================
// PERFORMANCE AND METRICS TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, PerformanceMetrics) {
    // Add facts and rules
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());
    EXPECT_TRUE(engine_->add_fact(plato_human_).is_ok());
    EXPECT_TRUE(engine_->add_rule(mortality_rule_).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    // Check metrics
    auto metrics = engine_->get_metrics();
    EXPECT_GT(metrics.facts_processed, 0);
    EXPECT_GT(metrics.rules_evaluated, 0);
    EXPECT_GT(metrics.rules_fired, 0);
    EXPECT_EQ(metrics.facts_derived, 2);  // Two mortality facts derived
    EXPECT_GT(metrics.total_time_ms.count(), 0);

    // Performance stats string should contain useful information
    auto stats = engine_->get_performance_stats();
    EXPECT_FALSE(stats.empty());
    EXPECT_TRUE(stats.find("Facts processed:") != std::string::npos);
    EXPECT_TRUE(stats.find("Rules fired:") != std::string::npos);
}

TEST_F(ForwardChainingTest, RuleFiringTrace) {
    // Add facts and rules
    EXPECT_TRUE(engine_->add_fact(socrates_human_).is_ok());
    EXPECT_TRUE(engine_->add_rule(mortality_rule_).is_ok());

    // Run inference
    auto result = engine_->run_forward_chaining();
    ASSERT_TRUE(result.is_ok());

    // Check firing trace
    auto trace = engine_->get_firing_trace();
    EXPECT_EQ(trace.size(), 1);

    const auto& firing = trace[0];
    EXPECT_EQ(firing.rule_name, "mortality_rule");
    EXPECT_EQ(firing.derived_facts.size(), 1);
    EXPECT_EQ(firing.derived_facts[0].predicate(), "isMortal");
    EXPECT_FALSE(firing.bindings.empty());
    EXPECT_TRUE(firing.bindings.find("X") != firing.bindings.end());
    EXPECT_EQ(firing.bindings.at("X").as_text(), "socrates");
}

// ================================================================================================
// INTEGRATION WITH INFERENCE ENGINE INTERFACE TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, InferenceEngineInterface) {
    // Test the base InferenceEngine interface
    EXPECT_TRUE(engine_->is_ready());  // Should be ready even without facts/rules

    auto info = engine_->get_backend_info();
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(info.find("ForwardChainingEngine") != std::string::npos);

    auto stats = engine_->get_performance_stats();
    EXPECT_FALSE(stats.empty());
}

TEST_F(ForwardChainingTest, FactoryFunction) {
    // Test different configuration options
    auto engine1 =
        create_forward_chaining_engine(ConflictResolutionStrategy::RECENCY_FIRST, 500, true);
    EXPECT_TRUE(engine1.is_ok());

    auto engine2 =
        create_forward_chaining_engine(ConflictResolutionStrategy::SPECIFICITY_FIRST, 2000, false);
    EXPECT_TRUE(engine2.is_ok());
}

// ================================================================================================
// UNIFIED INTERFACE TESTS
// ================================================================================================

TEST_F(ForwardChainingTest, UnifiedInferenceInterface) {
    // Test creation through unified interface
    ModelConfig config;
    config.model_path = "rule_based";  // Not used for rule-based, but required
    config.max_batch_size = 1;

    auto result = create_inference_engine(InferenceBackend::RULE_BASED, config);
    ASSERT_TRUE(result.is_ok()) << "Failed to create rule-based engine through unified interface";

    auto engine = std::move(result.unwrap());
    EXPECT_TRUE(engine->is_ready());
    EXPECT_FALSE(engine->get_backend_info().empty());
}

/**
 * @brief Main test entry point
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Set up logging for tests
    // Note: In a real test environment, you might want to reduce log level
    // to avoid cluttering test output

    return RUN_ALL_TESTS();
}
