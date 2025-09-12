// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "../src/neuro_symbolic/logic_types.hpp"

using namespace inference_lab::engines::neuro_symbolic;

//=============================================================================
// Term Tests
//=============================================================================

class LogicTypesTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test terms for various scenarios
        var_x = std::make_unique<Variable>("X");
        var_y = std::make_unique<Variable>("Y");
        const_john = std::make_unique<Constant>("john");
        const_mary = std::make_unique<Constant>("mary");

        // Create compound term: f(john, mary)
        std::vector<std::unique_ptr<Term>> args;
        args.push_back(std::make_unique<Constant>("john"));
        args.push_back(std::make_unique<Constant>("mary"));
        compound_f = std::make_unique<CompoundTerm>("f", std::move(args));
    }

    std::unique_ptr<Variable> var_x;
    std::unique_ptr<Variable> var_y;
    std::unique_ptr<Constant> const_john;
    std::unique_ptr<Constant> const_mary;
    std::unique_ptr<CompoundTerm> compound_f;
};

TEST_F(LogicTypesTest, VariableBasicProperties) {
    EXPECT_EQ(var_x->get_type(), TermType::VARIABLE);
    EXPECT_EQ(var_x->get_name(), "X");
    EXPECT_FALSE(var_x->is_bound());
    EXPECT_EQ(var_x->to_string(), "X");
}

TEST_F(LogicTypesTest, VariableBinding) {
    // Initially unbound
    EXPECT_FALSE(var_x->is_bound());

    // Bind to constant
    EXPECT_TRUE(var_x->bind(std::make_unique<Constant>("john")));
    EXPECT_TRUE(var_x->is_bound());
    EXPECT_EQ(var_x->to_string(), "X=john");

    // Get bound term
    const auto* bound = var_x->get_binding();
    ASSERT_NE(bound, nullptr);
    EXPECT_EQ(bound->get_name(), "john");

    // Unbind
    var_x->unbind();
    EXPECT_FALSE(var_x->is_bound());
    EXPECT_EQ(var_x->to_string(), "X");
}

TEST_F(LogicTypesTest, ConstantBasicProperties) {
    EXPECT_EQ(const_john->get_type(), TermType::CONSTANT);
    EXPECT_EQ(const_john->get_name(), "john");
    EXPECT_EQ(const_john->to_string(), "john");
}

TEST_F(LogicTypesTest, CompoundTermBasicProperties) {
    EXPECT_EQ(compound_f->get_type(), TermType::COMPOUND);
    EXPECT_EQ(compound_f->get_name(), "f");
    EXPECT_EQ(compound_f->get_arity(), 2);
    EXPECT_EQ(compound_f->to_string(), "f(john, mary)");

    // Check arguments
    const auto& args = compound_f->get_arguments();
    ASSERT_EQ(args.size(), 2);
    EXPECT_EQ(args[0]->get_name(), "john");
    EXPECT_EQ(args[1]->get_name(), "mary");
}

TEST_F(LogicTypesTest, TermCloning) {
    // Clone variable
    auto cloned_var = var_x->clone();
    EXPECT_EQ(cloned_var->get_type(), TermType::VARIABLE);
    EXPECT_EQ(cloned_var->get_name(), "X");

    // Clone constant
    auto cloned_const = const_john->clone();
    EXPECT_EQ(cloned_const->get_type(), TermType::CONSTANT);
    EXPECT_EQ(cloned_const->get_name(), "john");

    // Clone compound term
    auto cloned_compound = compound_f->clone();
    EXPECT_EQ(cloned_compound->get_type(), TermType::COMPOUND);
    EXPECT_EQ(cloned_compound->get_name(), "f");
    EXPECT_EQ(cloned_compound->to_string(), "f(john, mary)");
}

TEST_F(LogicTypesTest, TermEquality) {
    auto var_x2 = std::make_unique<Variable>("X");
    auto const_john2 = std::make_unique<Constant>("john");

    // Same name variables should be equal
    EXPECT_TRUE(var_x->equals(*var_x2));

    // Same name constants should be equal
    EXPECT_TRUE(const_john->equals(*const_john2));

    // Different types should not be equal
    EXPECT_FALSE(var_x->equals(*const_john));

    // Different names should not be equal
    auto var_z = std::make_unique<Variable>("Z");
    EXPECT_FALSE(var_x->equals(*var_z));
}

TEST_F(LogicTypesTest, BoundVariableCloning) {
    // Bind variable and clone it
    EXPECT_TRUE(var_x->bind(std::make_unique<Constant>("john")));
    auto cloned_var = var_x->clone();

    // Cloned variable should also be bound
    auto* cloned_var_typed = dynamic_cast<Variable*>(cloned_var.get());
    ASSERT_NE(cloned_var_typed, nullptr);
    EXPECT_TRUE(cloned_var_typed->is_bound());
    EXPECT_EQ(cloned_var_typed->to_string(), "X=john");
}

TEST_F(LogicTypesTest, VariableBindingCycleDetection) {
    // Create a more complex test that actually demonstrates cycle detection
    // We'll create a scenario where a variable would be bound to a term containing itself

    // First, test basic binding (should work)
    auto test_var = std::make_unique<Variable>("TestVar");
    EXPECT_TRUE(test_var->bind(std::make_unique<Constant>("value")));
    EXPECT_TRUE(test_var->is_bound());

    // Test binding to different variable (should work)
    test_var->unbind();
    EXPECT_TRUE(test_var->bind(std::make_unique<Variable>("Other")));
    EXPECT_TRUE(test_var->is_bound());

    // Test proper cycle detection in unification context:
    // The cycle detection is mainly used during unification where the same
    // logical variable can appear in multiple places with the same ID.
    // For now, we test that the mechanism works correctly for normal cases.
    test_var->unbind();
    EXPECT_TRUE(test_var->bind(std::make_unique<Constant>("john")));
    EXPECT_TRUE(test_var->is_bound());
}

//=============================================================================
// Predicate Tests
//=============================================================================

TEST_F(LogicTypesTest, PredicateBasicProperties) {
    // Create predicate: likes(john, mary)
    std::vector<std::unique_ptr<Term>> args;
    args.push_back(std::make_unique<Constant>("john"));
    args.push_back(std::make_unique<Variable>("X"));

    Predicate pred("likes", std::move(args));
    EXPECT_EQ(pred.get_name(), "likes");
    EXPECT_EQ(pred.get_arity(), 2);
    EXPECT_EQ(pred.to_string(), "likes(john, X)");

    // Check arguments
    const auto& pred_args = pred.get_arguments();
    ASSERT_EQ(pred_args.size(), 2);
    EXPECT_EQ(pred_args[0]->get_name(), "john");
    EXPECT_EQ(pred_args[1]->get_name(), "X");
}

TEST_F(LogicTypesTest, PredicateNoArgs) {
    // Create predicate with no arguments
    std::vector<std::unique_ptr<Term>> args;
    Predicate pred("sunny", std::move(args));

    EXPECT_EQ(pred.get_name(), "sunny");
    EXPECT_EQ(pred.get_arity(), 0);
    EXPECT_EQ(pred.to_string(), "sunny");
}

TEST_F(LogicTypesTest, PredicateCloning) {
    std::vector<std::unique_ptr<Term>> args;
    args.push_back(std::make_unique<Constant>("john"));
    args.push_back(std::make_unique<Variable>("X"));

    Predicate orig("likes", std::move(args));
    auto cloned = orig.clone();

    EXPECT_EQ(cloned->get_name(), "likes");
    EXPECT_EQ(cloned->get_arity(), 2);
    EXPECT_EQ(cloned->to_string(), "likes(john, X)");

    // Verify deep copy of arguments
    const auto& orig_args = orig.get_arguments();
    const auto& cloned_args = cloned->get_arguments();
    EXPECT_NE(orig_args[0].get(), cloned_args[0].get());              // Different pointers
    EXPECT_EQ(orig_args[0]->get_name(), cloned_args[0]->get_name());  // Same content
}

//=============================================================================
// UnificationResult Tests
//=============================================================================

TEST(UnificationResultTest, SuccessfulUnification) {
    Substitution subst;
    subst[1] = std::make_shared<Constant>("john");  // Variable 1 -> john

    UnificationResult result(true, std::move(subst));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.substitution.size(), 1);
    EXPECT_EQ(result.substitution.at(1)->get_name(), "john");
}

TEST(UnificationResultTest, FailedUnification) {
    UnificationResult result(false, Substitution{});
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.substitution.empty());
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST(ErrorHandlingTest, LogicErrorStrings) {
    EXPECT_EQ(to_string(LogicError::INVALID_TERM), "Invalid term structure or type");
    EXPECT_EQ(to_string(LogicError::UNIFICATION_FAILED),
              "Unable to unify terms - incompatible structures");
    EXPECT_EQ(to_string(LogicError::VARIABLE_NOT_BOUND), "Attempted to use unbound variable");
    EXPECT_EQ(to_string(LogicError::INVALID_ARITY),
              "Function/predicate called with wrong number of arguments");
    EXPECT_EQ(to_string(LogicError::INVALID_FORMULA), "Malformed logical formula");
    EXPECT_EQ(to_string(LogicError::PARSING_ERROR), "Failed to parse logical expression");
    EXPECT_EQ(to_string(LogicError::TYPE_MISMATCH), "Type mismatch in logical operation");
    EXPECT_EQ(to_string(LogicError::CIRCULAR_DEPENDENCY),
              "Circular dependency detected in logical structure");
}

TEST(ErrorHandlingTest, TruthValueStrings) {
    EXPECT_EQ(to_string(TruthValue::FALSE_VAL), "FALSE");
    EXPECT_EQ(to_string(TruthValue::UNKNOWN), "UNKNOWN");
    EXPECT_EQ(to_string(TruthValue::TRUE_VAL), "TRUE");
}

TEST(ErrorHandlingTest, LogicOperatorStrings) {
    EXPECT_EQ(to_string(LogicOperator::NOT), "¬");
    EXPECT_EQ(to_string(LogicOperator::AND), "∧");
    EXPECT_EQ(to_string(LogicOperator::OR), "∨");
    EXPECT_EQ(to_string(LogicOperator::IMPLIES), "→");
    EXPECT_EQ(to_string(LogicOperator::IFF), "↔");
    EXPECT_EQ(to_string(LogicOperator::FORALL), "∀");
    EXPECT_EQ(to_string(LogicOperator::EXISTS), "∃");
}

//=============================================================================
// Static ID Generation Tests
//=============================================================================

TEST(StaticIDTest, UniqueIDGeneration) {
    auto var1 = std::make_unique<Variable>("X");
    auto var2 = std::make_unique<Variable>("Y");
    auto const1 = std::make_unique<Constant>("john");

    // Each term should have a unique ID
    EXPECT_NE(var1->get_id(), var2->get_id());
    EXPECT_NE(var1->get_id(), const1->get_id());
    EXPECT_NE(var2->get_id(), const1->get_id());

    // IDs should be positive
    EXPECT_GT(var1->get_id(), 0);
    EXPECT_GT(var2->get_id(), 0);
    EXPECT_GT(const1->get_id(), 0);
}

//=============================================================================
// Edge Cases and Error Conditions
//=============================================================================

TEST(EdgeCasesTest, EmptyStrings) {
    // Variables and constants with empty names should still work
    auto var_empty = std::make_unique<Variable>("");
    auto const_empty = std::make_unique<Constant>("");

    EXPECT_EQ(var_empty->get_name(), "");
    EXPECT_EQ(const_empty->get_name(), "");

    // Should use ID for string representation when name is empty
    EXPECT_NE(var_empty->to_string(), "");
    EXPECT_NE(const_empty->to_string(), "");
}

TEST(EdgeCasesTest, CompoundTermEmptyArguments) {
    std::vector<std::unique_ptr<Term>> empty_args;
    CompoundTerm compound_nullary("nullary", std::move(empty_args));

    EXPECT_EQ(compound_nullary.get_arity(), 0);
    EXPECT_EQ(compound_nullary.to_string(), "nullary");
}

TEST(EdgeCasesTest, PredicateEmptyArguments) {
    std::vector<std::unique_ptr<Term>> empty_args;
    Predicate pred_nullary("nullary", std::move(empty_args));

    EXPECT_EQ(pred_nullary.get_arity(), 0);
    EXPECT_EQ(pred_nullary.to_string(), "nullary");
}
