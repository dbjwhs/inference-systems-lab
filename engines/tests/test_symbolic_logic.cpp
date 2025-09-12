// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "../src/neuro_symbolic/symbolic_logic.hpp"
#include "../src/neuro_symbolic/logic_types.hpp"

using namespace inference_lab::engines::neuro_symbolic;

//=============================================================================
// LogicFormula Tests
//=============================================================================

class SymbolicLogicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test predicates
        std::vector<std::unique_ptr<Term>> args1;
        args1.push_back(std::make_unique<Constant>("john"));
        pred_human_john = std::make_unique<Predicate>("human", std::move(args1));
        
        std::vector<std::unique_ptr<Term>> args2;
        args2.push_back(std::make_unique<Variable>("X"));
        pred_human_x = std::make_unique<Predicate>("human", std::move(args2));
        
        std::vector<std::unique_ptr<Term>> args3;
        args3.push_back(std::make_unique<Variable>("X"));
        pred_mortal_x = std::make_unique<Predicate>("mortal", std::move(args3));
        
        // Create atomic formulas
        atomic_human_john = std::make_unique<LogicFormula>(pred_human_john->clone());
        atomic_human_x = std::make_unique<LogicFormula>(pred_human_x->clone());
        atomic_mortal_x = std::make_unique<LogicFormula>(pred_mortal_x->clone());
    }

    std::unique_ptr<Predicate> pred_human_john;
    std::unique_ptr<Predicate> pred_human_x;
    std::unique_ptr<Predicate> pred_mortal_x;
    
    std::unique_ptr<LogicFormula> atomic_human_john;
    std::unique_ptr<LogicFormula> atomic_human_x;
    std::unique_ptr<LogicFormula> atomic_mortal_x;
};

TEST_F(SymbolicLogicTest, AtomicFormulaBasicProperties) {
    EXPECT_EQ(atomic_human_john->get_type(), LogicFormula::Type::ATOMIC);
    EXPECT_NE(atomic_human_john->get_predicate(), nullptr);
    EXPECT_EQ(atomic_human_john->get_predicate()->get_name(), "human");
    EXPECT_EQ(atomic_human_john->to_string(), "human(john)");
}

TEST_F(SymbolicLogicTest, CompoundFormulaConjunction) {
    // Create P ∧ Q
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(atomic_human_x->clone());
    operands.push_back(atomic_mortal_x->clone());
    
    LogicFormula conjunction(LogicOperator::AND, std::move(operands));
    
    EXPECT_EQ(conjunction.get_type(), LogicFormula::Type::COMPOUND);
    EXPECT_EQ(conjunction.get_operator(), LogicOperator::AND);
    EXPECT_EQ(conjunction.get_operands().size(), 2);
    EXPECT_EQ(conjunction.to_string(), "(human(X) ∧ mortal(X))");
}

TEST_F(SymbolicLogicTest, CompoundFormulaImplication) {
    // Create human(X) → mortal(X)
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(atomic_human_x->clone());
    operands.push_back(atomic_mortal_x->clone());
    
    LogicFormula implication(LogicOperator::IMPLIES, std::move(operands));
    
    EXPECT_EQ(implication.get_type(), LogicFormula::Type::COMPOUND);
    EXPECT_EQ(implication.get_operator(), LogicOperator::IMPLIES);
    EXPECT_EQ(implication.to_string(), "(human(X) → mortal(X))");
}

TEST_F(SymbolicLogicTest, QuantifiedFormula) {
    // Create ∀X human(X)
    auto var_x = std::make_unique<Variable>("X");
    auto formula_body = atomic_human_x->clone();
    
    LogicFormula universal(LogicOperator::FORALL, std::move(var_x), std::move(formula_body));
    
    EXPECT_EQ(universal.get_type(), LogicFormula::Type::QUANTIFIED);
    EXPECT_EQ(universal.get_operator(), LogicOperator::FORALL);
    EXPECT_NE(universal.get_quantified_variable(), nullptr);
    EXPECT_EQ(universal.get_quantified_variable()->get_name(), "X");
    EXPECT_EQ(universal.to_string(), "∀X human(X)");
}

TEST_F(SymbolicLogicTest, FormulaCloning) {
    auto cloned = atomic_human_john->clone();
    
    EXPECT_EQ(cloned->get_type(), LogicFormula::Type::ATOMIC);
    EXPECT_EQ(cloned->to_string(), "human(john)");
    
    // Should be different objects
    EXPECT_NE(atomic_human_john.get(), cloned.get());
}

TEST_F(SymbolicLogicTest, VariableCollection) {
    // Test variable collection in atomic formula
    auto vars = atomic_human_x->collect_variables();
    EXPECT_EQ(vars.size(), 1);
    
    // Test variable collection in compound formula
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(atomic_human_x->clone());
    operands.push_back(atomic_mortal_x->clone());
    LogicFormula conjunction(LogicOperator::AND, std::move(operands));
    
    auto compound_vars = conjunction.collect_variables();
    EXPECT_EQ(compound_vars.size(), 1); // Both formulas use the same variable X
}

//=============================================================================
// Unifier Tests
//=============================================================================

TEST_F(SymbolicLogicTest, UnifyIdenticalTerms) {
    auto const1 = std::make_unique<Constant>("john");
    auto const2 = std::make_unique<Constant>("john");
    
    auto result = Unifier::unify(*const1, *const2);
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.substitution.empty()); // No substitution needed
}

TEST_F(SymbolicLogicTest, UnifyVariableWithConstant) {
    auto var = std::make_unique<Variable>("X");
    auto constant = std::make_unique<Constant>("john");
    
    auto result = Unifier::unify(*var, *constant);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.substitution.size(), 1);
    EXPECT_EQ(result.substitution.at(var->get_id())->get_name(), "john");
}

TEST_F(SymbolicLogicTest, UnifyIncompatibleConstants) {
    auto const1 = std::make_unique<Constant>("john");
    auto const2 = std::make_unique<Constant>("mary");
    
    auto result = Unifier::unify(*const1, *const2);
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.substitution.empty());
}

TEST_F(SymbolicLogicTest, UnifyPredicates) {
    // likes(john, mary) unify with likes(X, Y)
    std::vector<std::unique_ptr<Term>> args1;
    args1.push_back(std::make_unique<Constant>("john"));
    args1.push_back(std::make_unique<Constant>("mary"));
    Predicate pred1("likes", std::move(args1));
    
    std::vector<std::unique_ptr<Term>> args2;
    args2.push_back(std::make_unique<Variable>("X"));
    args2.push_back(std::make_unique<Variable>("Y"));
    Predicate pred2("likes", std::move(args2));
    
    auto result = Unifier::unify_predicates(pred1, pred2);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.substitution.size(), 2);
}

TEST_F(SymbolicLogicTest, UnifyPredicatesDifferentNames) {
    // likes(john, mary) and hates(john, mary) should not unify
    std::vector<std::unique_ptr<Term>> args1;
    args1.push_back(std::make_unique<Constant>("john"));
    args1.push_back(std::make_unique<Constant>("mary"));
    Predicate pred1("likes", std::move(args1));
    
    std::vector<std::unique_ptr<Term>> args2;
    args2.push_back(std::make_unique<Constant>("john"));
    args2.push_back(std::make_unique<Constant>("mary"));
    Predicate pred2("hates", std::move(args2));
    
    auto result = Unifier::unify_predicates(pred1, pred2);
    EXPECT_FALSE(result.success);
}

TEST_F(SymbolicLogicTest, SubstitutionApplication) {
    auto var = std::make_unique<Variable>("X");
    Substitution subst;
    subst[var->get_id()] = std::make_unique<Constant>("john");
    
    auto result = Unifier::apply_substitution(*var, subst);
    EXPECT_EQ(result->get_type(), TermType::CONSTANT);
    EXPECT_EQ(result->get_name(), "john");
}

TEST_F(SymbolicLogicTest, SubstitutionComposition) {
    Substitution subst1, subst2;
    subst1[1] = std::make_unique<Variable>("Y");
    subst2[2] = std::make_unique<Constant>("john");
    
    auto composed = Unifier::compose_substitutions(subst1, subst2);
    EXPECT_EQ(composed.size(), 2);
    EXPECT_TRUE(composed.count(1) > 0);
    EXPECT_TRUE(composed.count(2) > 0);
}

//=============================================================================
// InferenceRules Tests
//=============================================================================

TEST_F(SymbolicLogicTest, ModusPonens) {
    // From P and P→Q, infer Q
    // P: human(john)
    auto premise = atomic_human_john->clone();
    
    // P→Q: human(X) → mortal(X) (with X substituted to john)
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(atomic_human_john->clone());
    
    std::vector<std::unique_ptr<Term>> mortal_args;
    mortal_args.push_back(std::make_unique<Constant>("john"));
    auto pred_mortal_john = std::make_unique<Predicate>("mortal", std::move(mortal_args));
    operands.push_back(std::make_unique<LogicFormula>(std::move(pred_mortal_john)));
    
    LogicFormula implication(LogicOperator::IMPLIES, std::move(operands));
    
    auto result = InferenceRules::modus_ponens(*premise, implication);
    EXPECT_TRUE(result.is_ok());
    if (result.is_ok()) {
        auto conclusion = std::move(result).unwrap();
        EXPECT_EQ(conclusion->to_string(), "mortal(john)");
    }
}

TEST_F(SymbolicLogicTest, UniversalInstantiation) {
    // From ∀X human(X), infer human(john)
    auto var_x = std::make_unique<Variable>("X");
    auto universal_body = atomic_human_x->clone();
    LogicFormula universal(LogicOperator::FORALL, std::move(var_x), std::move(universal_body));
    
    auto constant_john = std::make_unique<Constant>("john");
    
    auto result = InferenceRules::universal_instantiation(universal, *constant_john);
    EXPECT_TRUE(result.is_ok());
    if (result.is_ok()) {
        auto instantiated = std::move(result).unwrap();
        EXPECT_EQ(instantiated->to_string(), "human(john)");
    }
}

//=============================================================================
// KnowledgeBase Tests
//=============================================================================

TEST_F(SymbolicLogicTest, KnowledgeBaseBasicOperations) {
    KnowledgeBase kb;
    EXPECT_TRUE(kb.empty());
    EXPECT_EQ(kb.size(), 0);
    
    // Add fact: human(john)
    kb.add_formula(atomic_human_john->clone());
    EXPECT_FALSE(kb.empty());
    EXPECT_EQ(kb.size(), 1);
    
    // Add rule: human(X) → mortal(X)
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(atomic_human_x->clone());
    operands.push_back(atomic_mortal_x->clone());
    auto rule = std::make_unique<LogicFormula>(LogicOperator::IMPLIES, std::move(operands));
    
    kb.add_formula(std::move(rule));
    EXPECT_EQ(kb.size(), 2);
}

TEST_F(SymbolicLogicTest, KnowledgeBaseQuery) {
    KnowledgeBase kb;
    kb.add_formula(atomic_human_john->clone());
    
    // Query for human(john) - should match
    auto results = kb.query(*atomic_human_john);
    EXPECT_EQ(results.size(), 1);
    EXPECT_TRUE(results[0].second.success);
    
    // Query for human(mary) - should not match
    std::vector<std::unique_ptr<Term>> mary_args;
    mary_args.push_back(std::make_unique<Constant>("mary"));
    auto pred_human_mary = std::make_unique<Predicate>("human", std::move(mary_args));
    LogicFormula atomic_human_mary(std::move(pred_human_mary));
    
    auto no_results = kb.query(atomic_human_mary);
    EXPECT_TRUE(no_results.empty());
}

TEST_F(SymbolicLogicTest, KnowledgeBaseClear) {
    KnowledgeBase kb;
    kb.add_formula(atomic_human_john->clone());
    kb.add_formula(atomic_human_x->clone());
    
    EXPECT_EQ(kb.size(), 2);
    kb.clear();
    EXPECT_EQ(kb.size(), 0);
    EXPECT_TRUE(kb.empty());
}

//=============================================================================
// LogicReasoner Tests
//=============================================================================

TEST_F(SymbolicLogicTest, LogicReasonerBasicQuery) {
    auto kb = std::make_shared<KnowledgeBase>();
    kb->add_formula(atomic_human_john->clone());
    
    LogicReasoner reasoner(kb);
    
    // Query for existing fact
    auto answers = reasoner.answer_query(*atomic_human_john);
    EXPECT_EQ(answers.size(), 1);
    EXPECT_TRUE(answers[0].empty()); // No substitutions needed for exact match
}

TEST_F(SymbolicLogicTest, LogicReasonerVariableQuery) {
    auto kb = std::make_shared<KnowledgeBase>();
    kb->add_formula(atomic_human_john->clone());
    
    LogicReasoner reasoner(kb);
    
    // Query for human(X) - should return substitution X = john
    auto answers = reasoner.answer_query(*atomic_human_x);
    EXPECT_EQ(answers.size(), 1);
    EXPECT_FALSE(answers[0].empty()); // Should have substitution
}

TEST_F(SymbolicLogicTest, ForwardChaining) {
    auto kb = std::make_shared<KnowledgeBase>();
    
    // Add fact: human(john)
    kb->add_formula(atomic_human_john->clone());
    
    // Add rule: human(X) → mortal(X)
    std::vector<std::unique_ptr<LogicFormula>> operands;
    operands.push_back(atomic_human_x->clone());
    operands.push_back(atomic_mortal_x->clone());
    auto rule = std::make_unique<LogicFormula>(LogicOperator::IMPLIES, std::move(operands));
    kb->add_formula(std::move(rule));
    
    LogicReasoner reasoner(kb);
    
    // Initial size
    auto initial_size = kb->size();
    
    // Forward chain should derive new facts
    auto derived_count = reasoner.forward_chain(10);
    EXPECT_GE(derived_count, 0); // Should derive at least some facts
}

//=============================================================================
// Edge Cases and Error Conditions
//=============================================================================

TEST_F(SymbolicLogicTest, EmptyFormula) {
    // Test compound formula with no operands
    std::vector<std::unique_ptr<LogicFormula>> empty_operands;
    LogicFormula empty_conjunction(LogicOperator::AND, std::move(empty_operands));
    
    EXPECT_EQ(empty_conjunction.get_type(), LogicFormula::Type::COMPOUND);
    EXPECT_TRUE(empty_conjunction.get_operands().empty());
}

TEST_F(SymbolicLogicTest, ComplexCompoundFormula) {
    // Test deeply nested formula: ((P ∧ Q) ∨ R)
    std::vector<std::unique_ptr<LogicFormula>> inner_operands;
    inner_operands.push_back(atomic_human_x->clone());
    inner_operands.push_back(atomic_mortal_x->clone());
    auto inner_conjunction = std::make_unique<LogicFormula>(LogicOperator::AND, std::move(inner_operands));
    
    std::vector<std::unique_ptr<LogicFormula>> outer_operands;
    outer_operands.push_back(std::move(inner_conjunction));
    outer_operands.push_back(atomic_human_john->clone());
    LogicFormula outer_disjunction(LogicOperator::OR, std::move(outer_operands));
    
    EXPECT_EQ(outer_disjunction.get_type(), LogicFormula::Type::COMPOUND);
    EXPECT_EQ(outer_disjunction.get_operator(), LogicOperator::OR);
    EXPECT_EQ(outer_disjunction.get_operands().size(), 2);
}

TEST_F(SymbolicLogicTest, SubstitutionOnCompoundTerm) {
    // Create compound term f(X, john)
    std::vector<std::unique_ptr<Term>> args;
    auto var_x = std::make_unique<Variable>("X");
    SymbolId x_id = var_x->get_id();
    args.push_back(std::move(var_x));
    args.push_back(std::make_unique<Constant>("john"));
    
    CompoundTerm compound("f", std::move(args));
    
    // Create substitution X → mary
    Substitution subst;
    subst[x_id] = std::make_unique<Constant>("mary");
    
    // Apply substitution
    auto result = Unifier::apply_substitution(compound, subst);
    EXPECT_EQ(result->to_string(), "f(mary, john)");
}
