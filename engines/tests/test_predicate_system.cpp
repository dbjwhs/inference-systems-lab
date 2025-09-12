// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "../src/neuro_symbolic/predicate_system.hpp"
#include "../src/neuro_symbolic/logic_types.hpp"

using namespace inference_lab::engines::neuro_symbolic;

//=============================================================================
// PredicateSignature Tests
//=============================================================================

class PredicateSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test predicates and formulas
        std::vector<std::unique_ptr<Term>> human_args;
        human_args.push_back(std::make_unique<Variable>("X"));
        pred_human_x = std::make_unique<Predicate>("human", std::move(human_args));
        
        std::vector<std::unique_ptr<Term>> mortal_args;
        mortal_args.push_back(std::make_unique<Variable>("X"));
        pred_mortal_x = std::make_unique<Predicate>("mortal", std::move(mortal_args));
        
        // Create atomic formulas
        atomic_human_x = std::make_unique<LogicFormula>(pred_human_x->clone());
        atomic_mortal_x = std::make_unique<LogicFormula>(pred_mortal_x->clone());
        
        // Create facts
        std::vector<std::unique_ptr<Term>> john_args;
        john_args.push_back(std::make_unique<Constant>("john"));
        fact_human_john = std::make_unique<LogicFormula>(
            std::make_unique<Predicate>("human", std::move(john_args))
        );
    }

    std::unique_ptr<Predicate> pred_human_x;
    std::unique_ptr<Predicate> pred_mortal_x;
    std::unique_ptr<LogicFormula> atomic_human_x;
    std::unique_ptr<LogicFormula> atomic_mortal_x;
    std::unique_ptr<LogicFormula> fact_human_john;
};

TEST_F(PredicateSystemTest, PredicateSignatureBasicProperties) {
    PredicateSignature sig("likes", 2);
    EXPECT_EQ(sig.name, "likes");
    EXPECT_EQ(sig.arity, 2);
    EXPECT_TRUE(sig.argument_types.empty());
    EXPECT_EQ(sig.to_string(), "likes/2");
}

TEST_F(PredicateSystemTest, PredicateSignatureWithTypes) {
    std::vector<std::string> types = {"person", "person"};
    PredicateSignature sig("likes", std::move(types));
    EXPECT_EQ(sig.name, "likes");
    EXPECT_EQ(sig.arity, 2);
    EXPECT_EQ(sig.argument_types.size(), 2);
    EXPECT_EQ(sig.argument_types[0], "person");
    EXPECT_EQ(sig.argument_types[1], "person");
}

TEST_F(PredicateSystemTest, PredicateSignatureEquality) {
    PredicateSignature sig1("likes", 2);
    PredicateSignature sig2("likes", 2);
    PredicateSignature sig3("hates", 2);
    PredicateSignature sig4("likes", 3);
    
    EXPECT_TRUE(sig1.equals(sig2));
    EXPECT_FALSE(sig1.equals(sig3)); // Different name
    EXPECT_FALSE(sig1.equals(sig4)); // Different arity
}

//=============================================================================
// LogicRule Tests
//=============================================================================

TEST_F(PredicateSystemTest, LogicRuleBasicProperties) {
    // Create rule: mortal(X) :- human(X)
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    
    LogicRule rule(atomic_mortal_x->clone(), std::move(body));
    
    EXPECT_EQ(rule.get_head().to_string(), "mortal(X)");
    EXPECT_EQ(rule.get_body_size(), 1);
    EXPECT_FALSE(rule.is_fact());
    EXPECT_EQ(rule.to_string(), "mortal(X) :- human(X)");
}

TEST_F(PredicateSystemTest, LogicRuleFact) {
    // Create fact: human(john)
    std::vector<std::unique_ptr<LogicFormula>> empty_body;
    LogicRule fact(fact_human_john->clone(), std::move(empty_body));
    
    EXPECT_EQ(fact.get_head().to_string(), "human(john)");
    EXPECT_EQ(fact.get_body_size(), 0);
    EXPECT_TRUE(fact.is_fact());
    EXPECT_EQ(fact.to_string(), "human(john)");
}

TEST_F(PredicateSystemTest, LogicRuleVariableCollection) {
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    
    LogicRule rule(atomic_mortal_x->clone(), std::move(body));
    
    auto variables = rule.collect_variables();
    EXPECT_EQ(variables.size(), 1); // Both head and body use variable X
}

TEST_F(PredicateSystemTest, LogicRuleCloning) {
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    
    LogicRule original(atomic_mortal_x->clone(), std::move(body));
    auto cloned = original.clone();
    
    EXPECT_EQ(cloned->get_head().to_string(), "mortal(X)");
    EXPECT_EQ(cloned->get_body_size(), 1);
    EXPECT_EQ(cloned->to_string(), "mortal(X) :- human(X)");
    
    // Should be different objects
    EXPECT_NE(&original, cloned.get());
}

TEST_F(PredicateSystemTest, LogicRuleComplexBody) {
    // Create rule with multiple body conditions: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
    std::vector<std::unique_ptr<Term>> parent_xy_args;
    parent_xy_args.push_back(std::make_unique<Variable>("X"));
    parent_xy_args.push_back(std::make_unique<Variable>("Y"));
    auto parent_xy = std::make_unique<LogicFormula>(
        std::make_unique<Predicate>("parent", std::move(parent_xy_args))
    );
    
    std::vector<std::unique_ptr<Term>> parent_yz_args;
    parent_yz_args.push_back(std::make_unique<Variable>("Y"));
    parent_yz_args.push_back(std::make_unique<Variable>("Z"));
    auto parent_yz = std::make_unique<LogicFormula>(
        std::make_unique<Predicate>("parent", std::move(parent_yz_args))
    );
    
    std::vector<std::unique_ptr<Term>> grandparent_args;
    grandparent_args.push_back(std::make_unique<Variable>("X"));
    grandparent_args.push_back(std::make_unique<Variable>("Z"));
    auto grandparent_head = std::make_unique<LogicFormula>(
        std::make_unique<Predicate>("grandparent", std::move(grandparent_args))
    );
    
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(std::move(parent_xy));
    body.push_back(std::move(parent_yz));
    
    LogicRule grandparent_rule(std::move(grandparent_head), std::move(body));
    
    EXPECT_EQ(grandparent_rule.get_body_size(), 2);
    EXPECT_FALSE(grandparent_rule.is_fact());
    EXPECT_EQ(grandparent_rule.to_string(), "grandparent(X, Z) :- parent(X, Y), parent(Y, Z)");
    
    // Check variable collection
    auto vars = grandparent_rule.collect_variables();
    EXPECT_EQ(vars.size(), 3); // X, Y, Z
}

//=============================================================================
// PredicateRegistry Tests
//=============================================================================

TEST_F(PredicateSystemTest, PredicateRegistryBasicOperations) {
    PredicateRegistry registry;
    
    // Register predicate
    PredicateSignature likes_sig("likes", 2);
    registry.register_predicate(likes_sig);
    
    // Check registration
    EXPECT_TRUE(registry.is_registered("likes", 2));
    EXPECT_FALSE(registry.is_registered("likes", 3));
    EXPECT_FALSE(registry.is_registered("hates", 2));
    
    // Get signature
    auto result = registry.get_signature("likes", 2);
    EXPECT_TRUE(result.is_ok());
    if (result.is_ok()) {
        auto sig = std::move(result).unwrap();
        EXPECT_EQ(sig.name, "likes");
        EXPECT_EQ(sig.arity, 2);
    }
    
    // Get non-existent signature
    auto not_found = registry.get_signature("loves", 2);
    EXPECT_TRUE(not_found.is_err());
}

TEST_F(PredicateSystemTest, PredicateRegistryValidation) {
    PredicateRegistry registry;
    
    // Register with types
    std::vector<std::string> types = {"person", "person"};
    PredicateSignature likes_sig("likes", std::move(types));
    registry.register_predicate(likes_sig);
    
    // Create predicate that matches
    std::vector<std::unique_ptr<Term>> valid_args;
    valid_args.push_back(std::make_unique<Constant>("john"));
    valid_args.push_back(std::make_unique<Constant>("mary"));
    Predicate valid_pred("likes", std::move(valid_args));
    
    auto validation = registry.validate_predicate(valid_pred);
    EXPECT_TRUE(validation.is_ok());
    
    // Create predicate with wrong arity
    std::vector<std::unique_ptr<Term>> invalid_args;
    invalid_args.push_back(std::make_unique<Constant>("john"));
    Predicate invalid_pred("likes", std::move(invalid_args));
    
    auto invalid_validation = registry.validate_predicate(invalid_pred);
    EXPECT_TRUE(invalid_validation.is_err());
}

TEST_F(PredicateSystemTest, PredicateRegistryMultiplePredicates) {
    PredicateRegistry registry;
    
    registry.register_predicate(PredicateSignature("likes", 2));
    registry.register_predicate(PredicateSignature("human", 1));
    registry.register_predicate(PredicateSignature("mortal", 1));
    
    const auto& signatures = registry.get_all_signatures();
    EXPECT_EQ(signatures.size(), 3);
    
    // Clear registry
    registry.clear();
    EXPECT_EQ(registry.get_all_signatures().size(), 0);
}

//=============================================================================
// RuleBasedKnowledgeBase Tests
//=============================================================================

TEST_F(PredicateSystemTest, RuleBasedKnowledgeBaseBasicOperations) {
    RuleBasedKnowledgeBase kb;
    EXPECT_TRUE(kb.empty());
    EXPECT_EQ(kb.size(), 0);
    
    // Add fact
    kb.add_fact(fact_human_john->clone());
    EXPECT_FALSE(kb.empty());
    EXPECT_EQ(kb.size(), 1);
    EXPECT_EQ(kb.get_facts().size(), 1);
    EXPECT_EQ(kb.get_rules().size(), 0);
    
    // Add rule
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    auto rule = std::make_unique<LogicRule>(atomic_mortal_x->clone(), std::move(body));
    
    kb.add_rule(std::move(rule));
    EXPECT_EQ(kb.size(), 2);
    EXPECT_EQ(kb.get_facts().size(), 1);
    EXPECT_EQ(kb.get_rules().size(), 1);
}

TEST_F(PredicateSystemTest, RuleBasedKnowledgeBaseQueryFacts) {
    RuleBasedKnowledgeBase kb;
    kb.add_fact(fact_human_john->clone());
    
    // Query for exact match
    auto exact_results = kb.query_facts(*fact_human_john);
    EXPECT_EQ(exact_results.size(), 1);
    
    // Query with variable - should find substitution
    auto var_results = kb.query_facts(*atomic_human_x);
    EXPECT_EQ(var_results.size(), 1);
    EXPECT_FALSE(var_results[0].second.empty()); // Should have substitution
    
    // Query for non-existent fact
    std::vector<std::unique_ptr<Term>> mary_args;
    mary_args.push_back(std::make_unique<Constant>("mary"));
    LogicFormula human_mary(std::make_unique<Predicate>("human", std::move(mary_args)));
    
    auto no_results = kb.query_facts(human_mary);
    EXPECT_TRUE(no_results.empty());
}

TEST_F(PredicateSystemTest, RuleBasedKnowledgeBaseMatchingRules) {
    RuleBasedKnowledgeBase kb;
    
    // Add rule: mortal(X) :- human(X)
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    auto rule = std::make_unique<LogicRule>(atomic_mortal_x->clone(), std::move(body));
    kb.add_rule(std::move(rule));
    
    // Query for rules with head matching mortal(john)
    std::vector<std::unique_ptr<Term>> john_args;
    john_args.push_back(std::make_unique<Constant>("john"));
    LogicFormula mortal_john(std::make_unique<Predicate>("mortal", std::move(john_args)));
    
    auto matching_rules = kb.get_matching_rules(mortal_john);
    EXPECT_EQ(matching_rules.size(), 1);
    EXPECT_FALSE(matching_rules[0].second.empty()); // Should have substitution X->john
}

TEST_F(PredicateSystemTest, RuleBasedKnowledgeBaseClear) {
    RuleBasedKnowledgeBase kb;
    
    kb.add_fact(fact_human_john->clone());
    
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    auto rule = std::make_unique<LogicRule>(atomic_mortal_x->clone(), std::move(body));
    kb.add_rule(std::move(rule));
    
    EXPECT_EQ(kb.size(), 2);
    kb.clear();
    EXPECT_EQ(kb.size(), 0);
    EXPECT_TRUE(kb.empty());
}

//=============================================================================
// SLDReasoner Tests
//=============================================================================

TEST_F(PredicateSystemTest, SLDReasonerBasicQuery) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    kb->add_fact(fact_human_john->clone());
    
    SLDReasoner reasoner(kb);
    
    // Query for existing fact - should succeed
    auto result = reasoner.query(*fact_human_john);
    EXPECT_FALSE(result.timeout);
    EXPECT_EQ(result.solutions.size(), 1);
    EXPECT_TRUE(result.solutions[0].empty()); // No substitution needed
}

TEST_F(PredicateSystemTest, SLDReasonerVariableQuery) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    kb->add_fact(fact_human_john->clone());
    
    SLDReasoner reasoner(kb);
    
    // Query human(X) - should return X=john
    auto result = reasoner.query(*atomic_human_x);
    EXPECT_FALSE(result.timeout);
    EXPECT_EQ(result.solutions.size(), 1);
    EXPECT_FALSE(result.solutions[0].empty()); // Should have substitution
}

TEST_F(PredicateSystemTest, SLDReasonerRuleResolution) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    
    // Add fact: human(john)
    kb->add_fact(fact_human_john->clone());
    
    // Add rule: mortal(X) :- human(X)
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    auto rule = std::make_unique<LogicRule>(atomic_mortal_x->clone(), std::move(body));
    kb->add_rule(std::move(rule));
    
    SLDReasoner reasoner(kb);
    
    // Query mortal(john) - should succeed through resolution
    std::vector<std::unique_ptr<Term>> john_args;
    john_args.push_back(std::make_unique<Constant>("john"));
    LogicFormula mortal_john(std::make_unique<Predicate>("mortal", std::move(john_args)));
    
    auto result = reasoner.query(mortal_john);
    EXPECT_FALSE(result.timeout);
    EXPECT_GE(result.solutions.size(), 1); // Should find at least one solution
}

TEST_F(PredicateSystemTest, SLDReasonerProvabilityCheck) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    kb->add_fact(fact_human_john->clone());
    
    SLDReasoner reasoner(kb);
    
    // Fact should be provable
    EXPECT_TRUE(reasoner.is_provable(*fact_human_john));
    
    // Non-existent fact should not be provable
    std::vector<std::unique_ptr<Term>> mary_args;
    mary_args.push_back(std::make_unique<Constant>("mary"));
    LogicFormula human_mary(std::make_unique<Predicate>("human", std::move(mary_args)));
    
    EXPECT_FALSE(reasoner.is_provable(human_mary));
}

TEST_F(PredicateSystemTest, SLDReasonerConfiguration) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    kb->add_fact(fact_human_john->clone());
    
    SLDReasoner reasoner(kb);
    SLDReasoner::Config config;
    config.max_depth = 5;
    config.max_solutions = 2;
    config.trace_execution = true;
    
    auto result = reasoner.query(*fact_human_john, config);
    EXPECT_FALSE(result.timeout);
    EXPECT_LE(result.solutions.size(), config.max_solutions);
}

//=============================================================================
// Predicate Utility Functions Tests
//=============================================================================

TEST_F(PredicateSystemTest, PredicateUtilityFunctions) {
    // Test make_predicate
    auto pred = predicates::make_predicate("likes", {"john", "mary"});
    EXPECT_EQ(pred->get_name(), "likes");
    EXPECT_EQ(pred->get_arity(), 2);
    EXPECT_EQ(pred->to_string(), "likes(john, mary)");
    
    // Test make_variable
    auto var = predicates::make_variable("X");
    EXPECT_EQ(var->get_type(), TermType::VARIABLE);
    EXPECT_EQ(var->get_name(), "X");
    
    // Test make_constant
    auto constant = predicates::make_constant("john");
    EXPECT_EQ(constant->get_type(), TermType::CONSTANT);
    EXPECT_EQ(constant->get_name(), "john");
    
    // Test make_compound
    auto compound = predicates::make_compound("f", {"john", "mary"});
    EXPECT_EQ(compound->get_type(), TermType::COMPOUND);
    EXPECT_EQ(compound->get_name(), "f");
    EXPECT_EQ(compound->to_string(), "f(john, mary)");
}

TEST_F(PredicateSystemTest, FormulaUtilityFunctions) {
    auto pred = predicates::make_predicate("human", {"john"});
    
    // Test make_atomic_formula
    auto atomic = predicates::make_atomic_formula(std::move(pred));
    EXPECT_EQ(atomic->get_type(), LogicFormula::Type::ATOMIC);
    EXPECT_EQ(atomic->to_string(), "human(john)");
    
    // Test make_conjunction
    std::vector<std::unique_ptr<LogicFormula>> formulas;
    formulas.push_back(atomic_human_x->clone());
    formulas.push_back(atomic_mortal_x->clone());
    
    auto conjunction = predicates::make_conjunction(std::move(formulas));
    EXPECT_EQ(conjunction->get_type(), LogicFormula::Type::COMPOUND);
    EXPECT_EQ(conjunction->get_operator(), LogicOperator::AND);
    
    // Test make_implication
    auto implication = predicates::make_implication(
        atomic_human_x->clone(),
        atomic_mortal_x->clone()
    );
    EXPECT_EQ(implication->get_type(), LogicFormula::Type::COMPOUND);
    EXPECT_EQ(implication->get_operator(), LogicOperator::IMPLIES);
    
    // Test make_universal
    auto universal = predicates::make_universal(
        predicates::make_variable("X"),
        atomic_human_x->clone()
    );
    EXPECT_EQ(universal->get_type(), LogicFormula::Type::QUANTIFIED);
    EXPECT_EQ(universal->get_operator(), LogicOperator::FORALL);
}

//=============================================================================
// Parsing Tests
//=============================================================================

TEST_F(PredicateSystemTest, PredicateParsing) {
    // Test simple predicate parsing
    auto result = predicates::parse_predicate("likes(john, mary)");
    EXPECT_TRUE(result.is_ok());
    
    if (result.is_ok()) {
        auto pred = std::move(result).unwrap();
        EXPECT_EQ(pred->get_name(), "likes");
        EXPECT_EQ(pred->get_arity(), 2);
        EXPECT_EQ(pred->to_string(), "likes(john, mary)");
    }
    
    // Test nullary predicate
    auto nullary_result = predicates::parse_predicate("sunny");
    EXPECT_TRUE(nullary_result.is_ok());
    
    if (nullary_result.is_ok()) {
        auto pred = std::move(nullary_result).unwrap();
        EXPECT_EQ(pred->get_name(), "sunny");
        EXPECT_EQ(pred->get_arity(), 0);
    }
    
    // Test invalid predicate
    auto invalid_result = predicates::parse_predicate("invalid(");
    EXPECT_TRUE(invalid_result.is_err());
}

TEST_F(PredicateSystemTest, RuleParsing) {
    // Test rule parsing
    auto result = predicates::parse_rule("mortal(X) :- human(X)");
    EXPECT_TRUE(result.is_ok());
    
    if (result.is_ok()) {
        auto rule = std::move(result).unwrap();
        EXPECT_EQ(rule->get_head().to_string(), "mortal(X)");
        EXPECT_EQ(rule->get_body_size(), 1);
        EXPECT_FALSE(rule->is_fact());
    }
    
    // Test fact parsing
    auto fact_result = predicates::parse_rule("human(john)");
    EXPECT_TRUE(fact_result.is_ok());
    
    if (fact_result.is_ok()) {
        auto fact = std::move(fact_result).unwrap();
        EXPECT_EQ(fact->get_head().to_string(), "human(john)");
        EXPECT_EQ(fact->get_body_size(), 0);
        EXPECT_TRUE(fact->is_fact());
    }
    
    // Test invalid rule
    auto invalid_result = predicates::parse_rule("invalid :- )");
    EXPECT_TRUE(invalid_result.is_err());
}

//=============================================================================
// Performance and Memory Tests
//=============================================================================

TEST_F(PredicateSystemTest, LargeKnowledgeBasePerformance) {
    RuleBasedKnowledgeBase kb;
    
    // Add many facts
    for (int i = 0; i < 100; ++i) {
        std::vector<std::unique_ptr<Term>> args;
        args.push_back(std::make_unique<Constant>("person" + std::to_string(i)));
        auto fact = std::make_unique<LogicFormula>(
            std::make_unique<Predicate>("human", std::move(args))
        );
        kb.add_fact(std::move(fact));
    }
    
    EXPECT_EQ(kb.get_facts().size(), 100);
    
    // Query should still work efficiently
    auto results = kb.query_facts(*atomic_human_x);
    EXPECT_EQ(results.size(), 100); // Should match all facts
}

TEST_F(PredicateSystemTest, DeepRecursionHandling) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    
    // Add facts and recursive rules that could cause deep recursion
    kb->add_fact(fact_human_john->clone());
    
    // Add rule: mortal(X) :- human(X)
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    auto rule = std::make_unique<LogicRule>(atomic_mortal_x->clone(), std::move(body));
    kb->add_rule(std::move(rule));
    
    SLDReasoner reasoner(kb);
    SLDReasoner::Config config;
    config.max_depth = 10; // Limit depth to prevent infinite recursion
    
    // Should handle depth limit gracefully
    std::vector<std::unique_ptr<Term>> john_args;
    john_args.push_back(std::make_unique<Constant>("john"));
    LogicFormula mortal_john(std::make_unique<Predicate>("mortal", std::move(john_args)));
    
    auto result = reasoner.query(mortal_john, config);
    EXPECT_LE(result.depth_reached, config.max_depth);
}

//=============================================================================
// Edge Cases and Error Conditions
//=============================================================================

TEST_F(PredicateSystemTest, EmptyRuleHandling) {
    RuleBasedKnowledgeBase kb;
    
    // Empty knowledge base queries
    auto empty_results = kb.query_facts(*atomic_human_x);
    EXPECT_TRUE(empty_results.empty());
    
    auto empty_rules = kb.get_matching_rules(*atomic_mortal_x);
    EXPECT_TRUE(empty_rules.empty());
}

TEST_F(PredicateSystemTest, SelfReferencingRules) {
    auto kb = std::make_shared<RuleBasedKnowledgeBase>();
    
    // Add self-referencing rule: human(X) :- human(X)
    // This should be handled gracefully without infinite loops
    std::vector<std::unique_ptr<LogicFormula>> body;
    body.push_back(atomic_human_x->clone());
    auto self_rule = std::make_unique<LogicRule>(atomic_human_x->clone(), std::move(body));
    kb->add_rule(std::move(self_rule));
    
    SLDReasoner reasoner(kb);
    SLDReasoner::Config config;
    config.max_depth = 5; // Prevent infinite recursion
    
    auto result = reasoner.query(*atomic_human_x, config);
    EXPECT_LE(result.depth_reached, config.max_depth);
}
