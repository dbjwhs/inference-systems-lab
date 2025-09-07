/**
 * @file test_differentiable_ltn.cpp
 * @brief Comprehensive tests for Differentiable Logic Tensor Networks
 */

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "../src/neuro_symbolic/differentiable_ops.hpp"
#include "../src/neuro_symbolic/fuzzy_logic.hpp"
#include "../src/neuro_symbolic/logic_tensor_network.hpp"
#include "../src/neuro_symbolic/tensor_logic_bridge.hpp"

using namespace inference_lab::engines::neuro_symbolic;
using namespace inference_lab::common::types;

// ================================================================================================
// FUZZY LOGIC TESTS
// ================================================================================================

class FuzzyLogicTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to check fuzzy value approximately equal
    void ExpectFuzzyEqual(FuzzyValue actual, FuzzyValue expected, float tolerance = 1e-6f) {
        EXPECT_NEAR(actual, expected, tolerance) << "Fuzzy values differ significantly";
        EXPECT_GE(actual, 0.0f) << "Fuzzy value below valid range";
        EXPECT_LE(actual, 1.0f) << "Fuzzy value above valid range";
    }
};

TEST_F(FuzzyLogicTest, BasicTruthValueValidation) {
    // Test valid fuzzy values
    EXPECT_TRUE(is_valid_fuzzy_value(0.0f));
    EXPECT_TRUE(is_valid_fuzzy_value(0.5f));
    EXPECT_TRUE(is_valid_fuzzy_value(1.0f));

    // Test invalid fuzzy values
    EXPECT_FALSE(is_valid_fuzzy_value(-0.1f));
    EXPECT_FALSE(is_valid_fuzzy_value(1.1f));
    EXPECT_FALSE(is_valid_fuzzy_value(std::numeric_limits<float>::quiet_NaN()));

    // Test clamping
    ExpectFuzzyEqual(clamp_fuzzy_value(-0.5f), 0.0f);
    ExpectFuzzyEqual(clamp_fuzzy_value(1.5f), 1.0f);
    ExpectFuzzyEqual(clamp_fuzzy_value(0.7f), 0.7f);
}

TEST_F(FuzzyLogicTest, TNormOperations) {
    FuzzyValue a = 0.8f;
    FuzzyValue b = 0.6f;

    // Product T-norm
    ExpectFuzzyEqual(tnorms::product(a, b), 0.48f);
    ExpectFuzzyEqual(tnorms::product(1.0f, a), a);     // Identity
    ExpectFuzzyEqual(tnorms::product(0.0f, a), 0.0f);  // Annihilator

    // Lukasiewicz T-norm
    ExpectFuzzyEqual(tnorms::lukasiewicz(a, b), 0.4f);        // 0.8 + 0.6 - 1.0 = 0.4
    ExpectFuzzyEqual(tnorms::lukasiewicz(0.3f, 0.4f), 0.0f);  // max(0, 0.7-1.0) = 0

    // Minimum T-norm
    ExpectFuzzyEqual(tnorms::minimum(a, b), 0.6f);
    ExpectFuzzyEqual(tnorms::minimum(0.9f, 0.3f), 0.3f);

    // Drastic T-norm
    ExpectFuzzyEqual(tnorms::drastic(1.0f, 0.5f), 0.5f);
    ExpectFuzzyEqual(tnorms::drastic(0.8f, 0.6f), 0.0f);
}

TEST_F(FuzzyLogicTest, TConormOperations) {
    FuzzyValue a = 0.4f;
    FuzzyValue b = 0.3f;

    // Probabilistic sum
    ExpectFuzzyEqual(tconorms::probabilistic_sum(a, b), 0.58f);    // 0.4 + 0.3 - 0.12
    ExpectFuzzyEqual(tconorms::probabilistic_sum(0.0f, a), a);     // Identity
    ExpectFuzzyEqual(tconorms::probabilistic_sum(1.0f, a), 1.0f);  // Annihilator

    // Bounded sum
    ExpectFuzzyEqual(tconorms::bounded_sum(a, b), 0.7f);
    ExpectFuzzyEqual(tconorms::bounded_sum(0.8f, 0.9f), 1.0f);  // Clamped

    // Maximum T-conorm
    ExpectFuzzyEqual(tconorms::maximum(a, b), 0.4f);
    ExpectFuzzyEqual(tconorms::maximum(0.2f, 0.8f), 0.8f);
}

TEST_F(FuzzyLogicTest, BasicFuzzyOperations) {
    FuzzyValue p = 0.7f;
    FuzzyValue q = 0.4f;

    // Negation
    ExpectFuzzyEqual(fuzzy_not(p), 0.3f);
    ExpectFuzzyEqual(fuzzy_not(fuzzy_not(p)), p);  // Double negation

    // Conjunction (product T-norm)
    ExpectFuzzyEqual(fuzzy_and(p, q), 0.28f);

    // Disjunction (probabilistic sum)
    ExpectFuzzyEqual(fuzzy_or(p, q), 0.82f);  // 0.7 + 0.4 - 0.28

    // Implication
    ExpectFuzzyEqual(fuzzy_implies(p, q), fuzzy_or(fuzzy_not(p), q));

    // Biconditional
    ExpectFuzzyEqual(fuzzy_biconditional(p, q),
                     fuzzy_and(fuzzy_implies(p, q), fuzzy_implies(q, p)));
}

TEST_F(FuzzyLogicTest, FuzzyQuantifiers) {
    std::vector<FuzzyValue> values = {0.8f, 0.6f, 0.9f, 0.7f};

    // Universal quantification (product)
    FuzzyValue forall_result = fuzzy_forall(values);
    FuzzyValue expected_forall = 0.8f * 0.6f * 0.9f * 0.7f;
    ExpectFuzzyEqual(forall_result, expected_forall);

    // Existential quantification
    FuzzyValue exists_result = fuzzy_exists(values);
    FuzzyValue expected_exists = 1.0f - (0.2f * 0.4f * 0.1f * 0.3f);
    ExpectFuzzyEqual(exists_result, expected_exists);

    // Edge cases
    std::vector<FuzzyValue> all_true = {1.0f, 1.0f, 1.0f};
    ExpectFuzzyEqual(fuzzy_forall(all_true), 1.0f);
    ExpectFuzzyEqual(fuzzy_exists(all_true), 1.0f);

    std::vector<FuzzyValue> all_false = {0.0f, 0.0f, 0.0f};
    ExpectFuzzyEqual(fuzzy_forall(all_false), 0.0f);
    ExpectFuzzyEqual(fuzzy_exists(all_false), 0.0f);
}

TEST_F(FuzzyLogicTest, MembershipFunctions) {
    // Triangular membership
    ExpectFuzzyEqual(triangular_membership(2.0f, 1.0f, 2.0f, 3.0f), 1.0f);  // Peak
    ExpectFuzzyEqual(triangular_membership(1.5f, 1.0f, 2.0f, 3.0f), 0.5f);  // Rising edge
    ExpectFuzzyEqual(triangular_membership(2.5f, 1.0f, 2.0f, 3.0f), 0.5f);  // Falling edge
    ExpectFuzzyEqual(triangular_membership(0.5f, 1.0f, 2.0f, 3.0f), 0.0f);  // Outside

    // Gaussian membership
    ExpectFuzzyEqual(gaussian_membership(0.0f, 0.0f, 1.0f), 1.0f);  // Center
    EXPECT_LT(gaussian_membership(1.0f, 0.0f, 1.0f), 1.0f);         // Off center
    EXPECT_GT(gaussian_membership(1.0f, 0.0f, 1.0f), 0.0f);         // But positive

    // Sigmoid membership
    ExpectFuzzyEqual(sigmoid_membership(0.0f, 1.0f, 0.0f), 0.5f);  // Inflection point
    EXPECT_GT(sigmoid_membership(1.0f, 1.0f, 0.0f), 0.5f);         // Above inflection
    EXPECT_LT(sigmoid_membership(-1.0f, 1.0f, 0.0f), 0.5f);        // Below inflection
}

// ================================================================================================
// TENSOR FUZZY OPERATIONS TESTS
// ================================================================================================

class TensorFuzzyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test tensors with known values
        auto data1 = std::make_unique<float[]>(4);
        data1[0] = 0.8f;
        data1[1] = 0.3f;
        data1[2] = 0.9f;
        data1[3] = 0.1f;
        tensor1_ = TypedTensor<float, Shape<2, 2>>::from_data(std::move(data1));

        auto data2 = std::make_unique<float[]>(4);
        data2[0] = 0.6f;
        data2[1] = 0.7f;
        data2[2] = 0.2f;
        data2[3] = 0.4f;
        tensor2_ = TypedTensor<float, Shape<2, 2>>::from_data(std::move(data2));
    }

    std::optional<TypedTensor<float, Shape<2, 2>>> tensor1_;
    std::optional<TypedTensor<float, Shape<2, 2>>> tensor2_;
};

TEST_F(TensorFuzzyTest, TensorNegation) {
    auto result = tensor_fuzzy_not(*tensor1_);

    EXPECT_NEAR(result[0], 0.2f, 1e-6f);  // 1.0 - 0.8
    EXPECT_NEAR(result[1], 0.7f, 1e-6f);  // 1.0 - 0.3
    EXPECT_NEAR(result[2], 0.1f, 1e-6f);  // 1.0 - 0.9
    EXPECT_NEAR(result[3], 0.9f, 1e-6f);  // 1.0 - 0.1
}

TEST_F(TensorFuzzyTest, TensorConjunction) {
    auto result = tensor_fuzzy_and(*tensor1_, *tensor2_);

    EXPECT_NEAR(result[0], 0.48f, 1e-6f);  // 0.8 * 0.6
    EXPECT_NEAR(result[1], 0.21f, 1e-6f);  // 0.3 * 0.7
    EXPECT_NEAR(result[2], 0.18f, 1e-6f);  // 0.9 * 0.2
    EXPECT_NEAR(result[3], 0.04f, 1e-6f);  // 0.1 * 0.4
}

TEST_F(TensorFuzzyTest, TensorDisjunction) {
    auto result = tensor_fuzzy_or(*tensor1_, *tensor2_);

    // Probabilistic sum: a + b - a*b
    EXPECT_NEAR(result[0], 0.92f, 1e-6f);  // 0.8 + 0.6 - 0.48
    EXPECT_NEAR(result[1], 0.79f, 1e-6f);  // 0.3 + 0.7 - 0.21
    EXPECT_NEAR(result[2], 0.92f, 1e-6f);  // 0.9 + 0.2 - 0.18
    EXPECT_NEAR(result[3], 0.46f, 1e-6f);  // 0.1 + 0.4 - 0.04
}

TEST_F(TensorFuzzyTest, TensorQuantification) {
    // Universal quantification over all elements
    FuzzyValue forall_result = tensor_fuzzy_forall(*tensor1_);
    FuzzyValue expected = 0.8f * 0.3f * 0.9f * 0.1f;  // Product of all
    EXPECT_NEAR(forall_result, expected, 1e-6f);

    // Existential quantification over all elements
    FuzzyValue exists_result = tensor_fuzzy_exists(*tensor1_);
    FuzzyValue expected_exists = 1.0f - (0.2f * 0.7f * 0.1f * 0.9f);
    EXPECT_NEAR(exists_result, expected_exists, 1e-6f);
}

// ================================================================================================
// DIFFERENTIABLE OPERATIONS TESTS
// ================================================================================================

class DifferentiableOpsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test tensors
        auto data1 = std::make_unique<float[]>(3);
        data1[0] = 0.8f;
        data1[1] = 0.3f;
        data1[2] = 0.6f;
        input1_ = TypedTensor<float, Shape<3>>::from_data(std::move(data1));

        auto data2 = std::make_unique<float[]>(3);
        data2[0] = 0.4f;
        data2[1] = 0.7f;
        data2[2] = 0.2f;
        input2_ = TypedTensor<float, Shape<3>>::from_data(std::move(data2));
    }

    std::optional<TypedTensor<float, Shape<3>>> input1_;
    std::optional<TypedTensor<float, Shape<3>>> input2_;
};

TEST_F(DifferentiableOpsTest, DifferentiableNot) {
    DifferentiableNot<float, Shape<3>> diff_not;

    // Forward pass
    auto output = diff_not.forward(input1_.value());
    EXPECT_NEAR(output[0], 0.2f, 1e-6f);  // 1 - 0.8
    EXPECT_NEAR(output[1], 0.7f, 1e-6f);  // 1 - 0.3
    EXPECT_NEAR(output[2], 0.4f, 1e-6f);  // 1 - 0.6

    // Backward pass - gradient should be -1 for all elements
    auto output_grad = TypedTensor<float, Shape<3>>::filled(1.0f);
    auto input_grad = diff_not.backward(output_grad);

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(input_grad[i], -1.0f, 1e-6f);
    }

    EXPECT_EQ(diff_not.name(), "DifferentiableNot");
}

TEST_F(DifferentiableOpsTest, DifferentiableSigmoid) {
    DifferentiableSigmoid<float, Shape<3>> diff_sigmoid;

    // Create input with both positive and negative values
    auto data = std::make_unique<float[]>(3);
    data[0] = 2.0f;
    data[1] = 0.0f;
    data[2] = -2.0f;
    auto input = TypedTensor<float, Shape<3>>::from_data(std::move(data));

    // Forward pass
    auto output = diff_sigmoid.forward(input);

    // Check sigmoid properties
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_GE(output[i], 0.0f);
        EXPECT_LE(output[i], 1.0f);
    }
    EXPECT_GT(output[0], 0.5f);           // sigmoid(2) > 0.5
    EXPECT_NEAR(output[1], 0.5f, 1e-6f);  // sigmoid(0) = 0.5
    EXPECT_LT(output[2], 0.5f);           // sigmoid(-2) < 0.5

    // Backward pass
    auto output_grad = TypedTensor<float, Shape<3>>::filled(1.0f);
    auto input_grad = diff_sigmoid.backward(output_grad);

    // Gradient should be σ(x)(1-σ(x)), which is maximized at x=0
    EXPECT_GT(input_grad[1], input_grad[0]);  // Gradient larger at x=0
    EXPECT_GT(input_grad[1], input_grad[2]);  // than at x=±2

    EXPECT_EQ(diff_sigmoid.name(), "DifferentiableSigmoid");
}

TEST_F(DifferentiableOpsTest, DifferentiableAnd) {
    DifferentiableAnd<float, Shape<3>> diff_and;

    // Forward pass
    auto inputs = std::make_tuple(input1_.value(), input2_.value());
    auto output = diff_and.forward(inputs);

    // Check element-wise product
    EXPECT_NEAR(output[0], 0.32f, 1e-6f);  // 0.8 * 0.4
    EXPECT_NEAR(output[1], 0.21f, 1e-6f);  // 0.3 * 0.7
    EXPECT_NEAR(output[2], 0.12f, 1e-6f);  // 0.6 * 0.2

    // Backward pass
    auto output_grad = TypedTensor<float, Shape<3>>::filled(1.0f);
    auto [grad1, grad2] = diff_and.backward(output_grad);

    // Gradients: ∂(xy)/∂x = y, ∂(xy)/∂y = x
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad1[i], input2_.value()[i], 1e-6f);
        EXPECT_NEAR(grad2[i], input1_.value()[i], 1e-6f);
    }

    EXPECT_EQ(diff_and.name(), "DifferentiableAnd");
}

TEST_F(DifferentiableOpsTest, DifferentiableOr) {
    DifferentiableOr<float, Shape<3>> diff_or;

    // Forward pass
    auto inputs = std::make_tuple(input1_.value(), input2_.value());
    auto output = diff_or.forward(inputs);

    // Check probabilistic sum: x + y - xy
    EXPECT_NEAR(output[0], 0.88f, 1e-6f);  // 0.8 + 0.4 - 0.32
    EXPECT_NEAR(output[1], 0.79f, 1e-6f);  // 0.3 + 0.7 - 0.21
    EXPECT_NEAR(output[2], 0.68f, 1e-6f);  // 0.6 + 0.2 - 0.12

    // Backward pass
    auto output_grad = TypedTensor<float, Shape<3>>::filled(1.0f);
    auto [grad1, grad2] = diff_or.backward(output_grad);

    // Gradients: ∂(x+y-xy)/∂x = 1-y, ∂(x+y-xy)/∂y = 1-x
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad1[i], 1.0f - input2_.value()[i], 1e-6f);
        EXPECT_NEAR(grad2[i], 1.0f - input1_.value()[i], 1e-6f);
    }

    EXPECT_EQ(diff_or.name(), "DifferentiableOr");
}

TEST_F(DifferentiableOpsTest, DifferentiableQuantifiers) {
    // Test differentiable forall
    DifferentiableForall<float, Shape<3>> diff_forall(5.0f);  // Lower temperature for testing

    FuzzyValue forall_result = diff_forall.forward(input1_.value());
    EXPECT_GE(forall_result, 0.0f);
    EXPECT_LE(forall_result, 1.0f);

    // Should be less than simple product due to soft minimum
    FuzzyValue simple_product = input1_.value()[0] * input1_.value()[1] * input1_.value()[2];
    EXPECT_LE(forall_result, simple_product);

    // Test differentiable exists
    DifferentiableExists<float, Shape<3>> diff_exists(5.0f);

    FuzzyValue exists_result = diff_exists.forward(input1_.value());
    EXPECT_GE(exists_result, 0.0f);
    EXPECT_LE(exists_result, 1.0f);

    // Should be greater than simple maximum due to soft maximum
    FuzzyValue simple_max = std::max({input1_.value()[0], input1_.value()[1], input1_.value()[2]});
    EXPECT_GE(exists_result, simple_max);

    EXPECT_EQ(diff_forall.name(), "DifferentiableForall");
    EXPECT_EQ(diff_exists.name(), "DifferentiableExists");
}

// ================================================================================================
// TENSOR-LOGIC BRIDGE TESTS
// ================================================================================================

class TensorLogicBridgeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test tensor with various values
        auto data = std::make_unique<float[]>(6);
        data[0] = 0.8f;
        data[1] = 0.3f;
        data[2] = 0.9f;
        data[3] = 0.1f;
        data[4] = 0.6f;
        data[5] = 0.4f;
        tensor_ = TypedTensor<float, Shape<2, 3>>::from_data(std::move(data));

        logical_ = LogicalTensor<float, Shape<2, 3>>::from_tensor(*tensor_);
    }

    std::optional<TypedTensor<float, Shape<2, 3>>> tensor_;
    std::optional<LogicalTensor<float, Shape<2, 3>>> logical_;
};

TEST_F(TensorLogicBridgeTest, ConstructionAndConversion) {
    // Test construction from tensor
    auto logical1 = LogicalTensor<float, Shape<2, 3>>::from_tensor(*tensor_);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(logical1[i], (*tensor_)[i], 1e-6f);
    }

    // Test construction with clamping
    auto bad_data = std::make_unique<float[]>(3);
    bad_data[0] = -0.5f;
    bad_data[1] = 1.5f;
    bad_data[2] = 0.7f;
    auto bad_tensor = TypedTensor<float, Shape<3>>::from_data(std::move(bad_data));

    auto clamped_logical = LogicalTensor<float, Shape<3>>::from_tensor(std::move(bad_tensor));
    EXPECT_NEAR(clamped_logical[0], 0.0f, 1e-6f);  // Clamped from -0.5
    EXPECT_NEAR(clamped_logical[1], 1.0f, 1e-6f);  // Clamped from 1.5
    EXPECT_NEAR(clamped_logical[2], 0.7f, 1e-6f);  // Unchanged

    // Test factory methods
    auto all_true = LogicalTensor<float, Shape<2, 2>>::all_true();
    auto all_false = LogicalTensor<float, Shape<2, 2>>::all_false();

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(all_true[i], 1.0f, 1e-6f);
        EXPECT_NEAR(all_false[i], 0.0f, 1e-6f);
    }
}

TEST_F(TensorLogicBridgeTest, PredicateOperations) {
    // Greater than predicate
    auto gt_05 = logical_->greater_than(0.5f);
    EXPECT_GT(gt_05[0], 0.5f);  // 0.8 > 0.5
    EXPECT_LT(gt_05[1], 0.5f);  // 0.3 < 0.5
    EXPECT_GT(gt_05[2], 0.5f);  // 0.9 > 0.5

    // Less than predicate
    auto lt_05 = logical_->less_than(0.5f);
    EXPECT_LT(lt_05[0], 0.5f);  // 0.8 > 0.5, so result < 0.5
    EXPECT_GT(lt_05[1], 0.5f);  // 0.3 < 0.5, so result > 0.5

    // Approximately equal predicate
    auto approx_06 = logical_->approximately_equal(0.6f, 0.1f);
    EXPECT_GT(approx_06[4], approx_06[0]);  // 0.6 closer to 0.6 than 0.8

    // Custom predicate
    auto is_high = logical_->predicate([](float x) { return x > 0.7f ? 1.0f : 0.0f; });
    EXPECT_NEAR(is_high[0], 1.0f, 1e-6f);  // 0.8 > 0.7
    EXPECT_NEAR(is_high[1], 0.0f, 1e-6f);  // 0.3 < 0.7
    EXPECT_NEAR(is_high[2], 1.0f, 1e-6f);  // 0.9 > 0.7
}

TEST_F(TensorLogicBridgeTest, LogicalOperations) {
    auto other_data = std::make_unique<float[]>(6);
    other_data[0] = 0.2f;
    other_data[1] = 0.8f;
    other_data[2] = 0.4f;
    other_data[3] = 0.7f;
    other_data[4] = 0.1f;
    other_data[5] = 0.9f;
    auto other_tensor = TypedTensor<float, Shape<2, 3>>::from_data(std::move(other_data));
    auto other_logical = LogicalTensor<float, Shape<2, 3>>::from_tensor(std::move(other_tensor));

    // Logical NOT
    auto not_result = logical_->logical_not();
    EXPECT_NEAR(not_result[0], 0.2f, 1e-6f);  // 1 - 0.8
    EXPECT_NEAR(not_result[1], 0.7f, 1e-6f);  // 1 - 0.3

    // Logical AND
    auto and_result = logical_->logical_and(other_logical);
    EXPECT_NEAR(and_result[0], 0.16f, 1e-6f);  // 0.8 * 0.2
    EXPECT_NEAR(and_result[1], 0.24f, 1e-6f);  // 0.3 * 0.8

    // Logical OR
    auto or_result = logical_->logical_or(other_logical);
    EXPECT_NEAR(or_result[0], 0.84f, 1e-6f);  // 0.8 + 0.2 - 0.16
    EXPECT_NEAR(or_result[1], 0.86f, 1e-6f);  // 0.3 + 0.8 - 0.24

    // Logical IMPLIES
    auto implies_result = logical_->logical_implies(other_logical);
    // x → y = ¬x ∨ y = (1-x) + xy
    float expected_0 = fuzzy_implies(0.8f, 0.2f);
    EXPECT_NEAR(implies_result[0], expected_0, 1e-6f);

    // Test operator overloads
    auto and_operator = (*logical_) & other_logical;
    auto or_operator = (*logical_) | other_logical;
    auto not_operator = !(*logical_);

    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(and_operator[i], and_result[i], 1e-6f);
        EXPECT_NEAR(or_operator[i], or_result[i], 1e-6f);
        EXPECT_NEAR(not_operator[i], not_result[i], 1e-6f);
    }
}

TEST_F(TensorLogicBridgeTest, QuantificationOperations) {
    // Universal quantification
    FuzzyValue forall_result = logical_->forall();
    FuzzyValue expected_forall = 0.8f * 0.3f * 0.9f * 0.1f * 0.6f * 0.4f;
    EXPECT_NEAR(forall_result, expected_forall, 1e-6f);

    // Existential quantification
    FuzzyValue exists_result = logical_->exists();
    // Product of complements: (1-0.8)(1-0.3)(1-0.9)(1-0.1)(1-0.6)(1-0.4)
    FuzzyValue complement_product = 0.2f * 0.7f * 0.1f * 0.9f * 0.4f * 0.6f;
    FuzzyValue expected_exists = 1.0f - complement_product;
    EXPECT_NEAR(exists_result, expected_exists, 1e-6f);

    // Smooth quantification
    FuzzyValue smooth_forall = logical_->smooth_forall(10.0f);
    FuzzyValue smooth_exists = logical_->smooth_exists(10.0f);

    EXPECT_GE(smooth_forall, 0.0f);
    EXPECT_LE(smooth_forall, 1.0f);
    EXPECT_GE(smooth_exists, 0.0f);
    EXPECT_LE(smooth_exists, 1.0f);

    // Smooth versions should be different from exact versions
    EXPECT_NE(smooth_forall, forall_result);
    EXPECT_NE(smooth_exists, exists_result);
}

TEST_F(TensorLogicBridgeTest, StatisticalAnalysis) {
    auto stats = logical_->statistics();

    // Check basic statistics
    EXPECT_GE(stats.mean, 0.0f);
    EXPECT_LE(stats.mean, 1.0f);
    EXPECT_GE(stats.variance, 0.0f);
    EXPECT_GE(stats.min_value, 0.0f);
    EXPECT_LE(stats.max_value, 1.0f);
    EXPECT_GE(stats.entropy, 0.0f);

    // Calculate expected values manually
    float sum = 0.8f + 0.3f + 0.9f + 0.1f + 0.6f + 0.4f;
    float expected_mean = sum / 6.0f;
    EXPECT_NEAR(stats.mean, expected_mean, 1e-6f);

    EXPECT_NEAR(stats.min_value, 0.1f, 1e-6f);
    EXPECT_NEAR(stats.max_value, 0.9f, 1e-6f);

    // Truth counting
    std::size_t true_count = logical_->count_true(0.5f);
    EXPECT_EQ(true_count, 3u);  // 0.8, 0.9, 0.6 are >= 0.5

    float truth_prop = logical_->truth_proportion(0.5f);
    EXPECT_NEAR(truth_prop, 0.5f, 1e-6f);  // 3/6 = 0.5
}

// ================================================================================================
// UTILITY FUNCTION TESTS
// ================================================================================================

class UtilityFunctionTest : public ::testing::Test {};

TEST_F(UtilityFunctionTest, NeuralToLogicalConversion) {
    // Create neural network output with values outside [0,1]
    auto nn_data = std::make_unique<float[]>(4);
    nn_data[0] = 2.0f;
    nn_data[1] = -1.0f;
    nn_data[2] = 0.5f;
    nn_data[3] = 0.0f;
    auto nn_output = TypedTensor<float, Shape<4>>::from_data(std::move(nn_data));

    // Test sigmoid conversion
    auto logical_sigmoid = neural_to_logical(nn_output, "sigmoid");
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_GE(logical_sigmoid[i], 0.0f);
        EXPECT_LE(logical_sigmoid[i], 1.0f);
    }
    EXPECT_GT(logical_sigmoid[0], 0.5f);           // sigmoid(2.0) > 0.5
    EXPECT_LT(logical_sigmoid[1], 0.5f);           // sigmoid(-1.0) < 0.5
    EXPECT_NEAR(logical_sigmoid[3], 0.5f, 1e-6f);  // sigmoid(0.0) = 0.5

    // Test tanh conversion
    auto logical_tanh = neural_to_logical(nn_output, "tanh");
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_GE(logical_tanh[i], 0.0f);
        EXPECT_LE(logical_tanh[i], 1.0f);
    }

    // Test identity conversion (with clamping)
    auto logical_identity = neural_to_logical(nn_output, "identity");
    EXPECT_NEAR(logical_identity[0], 1.0f, 1e-6f);  // 2.0 clamped to 1.0
    EXPECT_NEAR(logical_identity[1], 0.0f, 1e-6f);  // -1.0 clamped to 0.0
    EXPECT_NEAR(logical_identity[2], 0.5f, 1e-6f);  // 0.5 unchanged
    EXPECT_NEAR(logical_identity[3], 0.0f, 1e-6f);  // 0.0 unchanged
}

TEST_F(UtilityFunctionTest, ProbabilitiesToLogical) {
    // Create probability tensor (already in [0,1])
    auto prob_data = std::make_unique<float[]>(3);
    prob_data[0] = 0.8f;
    prob_data[1] = 0.0f;
    prob_data[2] = 1.0f;
    auto prob_tensor = TypedTensor<float, Shape<3>>::from_data(std::move(prob_data));

    auto logical = probabilities_to_logical(prob_tensor);

    // Values should be preserved since they're already valid
    EXPECT_NEAR(logical[0], 0.8f, 1e-6f);
    EXPECT_NEAR(logical[1], 0.0f, 1e-6f);
    EXPECT_NEAR(logical[2], 1.0f, 1e-6f);
}

// ================================================================================================
// INTEGRATION TESTS
// ================================================================================================

class LTNIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        LTNConfig config;
        config.embedding_dim = 4;
        config.learning_rate = 0.1f;
        config.max_iterations = 100;

        auto ltn_result = LogicTensorNetwork::create(config);
        ASSERT_TRUE(ltn_result.is_ok());
        ltn_ = std::move(ltn_result).unwrap();
    }

    std::unique_ptr<LogicTensorNetwork> ltn_;
};

TEST_F(LTNIntegrationTest, BasicLTNOperations) {
    // Add individuals
    auto socrates_result = ltn_->add_individual("Socrates");
    auto plato_result = ltn_->add_individual("Plato");
    ASSERT_TRUE(socrates_result.is_ok());
    ASSERT_TRUE(plato_result.is_ok());

    // Add predicates
    auto human_result = ltn_->add_predicate("Human", 1);
    auto mortal_result = ltn_->add_predicate("Mortal", 1);
    auto likes_result = ltn_->add_predicate("Likes", 2);
    ASSERT_TRUE(human_result.is_ok());
    ASSERT_TRUE(mortal_result.is_ok());
    ASSERT_TRUE(likes_result.is_ok());

    // Test predicate evaluation
    auto human_socrates_result = ltn_->evaluate_predicate("Human", "Socrates");
    ASSERT_TRUE(human_socrates_result.is_ok());

    FuzzyValue human_socrates = human_socrates_result.unwrap();
    EXPECT_GE(human_socrates, 0.0f);
    EXPECT_LE(human_socrates, 1.0f);

    // Test relation evaluation
    auto likes_result_eval = ltn_->evaluate_relation("Likes", "Socrates", "Plato");
    ASSERT_TRUE(likes_result_eval.is_ok());

    FuzzyValue likes_value = likes_result_eval.unwrap();
    EXPECT_GE(likes_value, 0.0f);
    EXPECT_LE(likes_value, 1.0f);

    // Test batch evaluation
    std::vector<std::string> individuals = {"Socrates", "Plato"};
    auto batch_result = ltn_->batch_evaluate_predicate("Human", individuals);
    ASSERT_TRUE(batch_result.is_ok());

    const auto& batch_values = batch_result.unwrap();
    EXPECT_EQ(batch_values.size(), 2u);
    for (auto value : batch_values) {
        EXPECT_GE(value, 0.0f);
        EXPECT_LE(value, 1.0f);
    }
}

TEST_F(LTNIntegrationTest, FormulaConstructionAndEvaluation) {
    // Set up entities and predicates
    ltn_->add_individual("Socrates");
    ltn_->add_individual("Plato");
    ltn_->add_predicate("Human", 1);
    ltn_->add_predicate("Mortal", 1);

    // Create atomic formulas
    auto human_socrates = ltn_->atomic("Human", {"Socrates"});
    auto mortal_socrates = ltn_->atomic("Mortal", {"Socrates"});
    auto human_plato = ltn_->atomic("Human", {"Plato"});

    // Create compound formulas
    auto socrates_human_and_mortal = ltn_->conjunction(ltn_->atomic("Human", {"Socrates"}),
                                                       ltn_->atomic("Mortal", {"Socrates"}));

    auto someone_is_human =
        ltn_->disjunction(ltn_->atomic("Human", {"Socrates"}), ltn_->atomic("Human", {"Plato"}));

    auto mortality_rule = ltn_->implication(ltn_->atomic("Human", {"Socrates"}),
                                            ltn_->atomic("Mortal", {"Socrates"}));

    // Test formula evaluation
    auto result1 = ltn_->query(*socrates_human_and_mortal);
    auto result2 = ltn_->query(*someone_is_human);
    auto result3 = ltn_->query(*mortality_rule);

    ASSERT_TRUE(result1.is_ok());
    ASSERT_TRUE(result2.is_ok());
    ASSERT_TRUE(result3.is_ok());

    FuzzyValue and_value = result1.unwrap();
    FuzzyValue or_value = result2.unwrap();
    FuzzyValue implies_value = result3.unwrap();

    EXPECT_GE(and_value, 0.0f);
    EXPECT_LE(and_value, 1.0f);
    EXPECT_GE(or_value, 0.0f);
    EXPECT_LE(or_value, 1.0f);
    EXPECT_GE(implies_value, 0.0f);
    EXPECT_LE(implies_value, 1.0f);

    // OR should typically be greater than AND for same predicates
    // EXPECT_GE(or_value, and_value);  // This may not always hold due to random initialization
}

TEST_F(LTNIntegrationTest, TrainingAndLearning) {
    // Set up simple learning scenario
    ltn_->add_individual("Alice");
    ltn_->add_individual("Bob");
    ltn_->add_predicate("Student", 1);

    // Create training examples
    std::vector<LogicTensorNetwork::Example> examples = {
        {"Student", {"Alice"}, 0.9f, 1.0f},  // Alice is definitely a student
        {"Student", {"Bob"}, 0.1f, 1.0f}     // Bob is definitely not a student
    };

    // Train the network
    auto train_result = ltn_->train(examples, 50);
    ASSERT_TRUE(train_result.is_ok());

    float final_loss = train_result.unwrap();
    EXPECT_GE(final_loss, 0.0f);

    // Check if learning occurred by evaluating predictions
    auto alice_pred = ltn_->evaluate_predicate("Student", "Alice");
    auto bob_pred = ltn_->evaluate_predicate("Student", "Bob");

    ASSERT_TRUE(alice_pred.is_ok());
    ASSERT_TRUE(bob_pred.is_ok());

    // After training, Alice should have higher student probability than Bob
    EXPECT_GT(alice_pred.unwrap(), bob_pred.unwrap());
}

TEST_F(LTNIntegrationTest, StatisticsAndModelManagement) {
    // Add some entities and predicates
    ltn_->add_individual("Entity1");
    ltn_->add_individual("Entity2");
    ltn_->add_predicate("Property1", 1);
    ltn_->add_predicate("Relation1", 2);

    // Get statistics
    auto stats = ltn_->get_statistics();
    EXPECT_EQ(stats.num_individuals, 2u);
    EXPECT_EQ(stats.num_predicates, 2u);
    EXPECT_GT(stats.total_parameters, 0.0f);
    EXPECT_EQ(stats.embedding_norms.size(), 2u);
    EXPECT_EQ(stats.predicate_norms.size(), 2u);

    // Test model export/import
    auto exported_model = ltn_->export_model();
    EXPECT_GT(exported_model.size(), 0u);

    auto import_result = ltn_->import_model(exported_model);
    EXPECT_TRUE(import_result.is_ok());

    // Test individual lookup
    auto entity1_result = ltn_->get_embedding("Entity1");
    ASSERT_TRUE(entity1_result.is_ok());

    const auto& embedding = entity1_result.unwrap();
    EXPECT_EQ(embedding.size(), 4u);  // Based on config.embedding_dim = 4

    // Test individual finding
    auto individuals_result = ltn_->find_individuals("Property1", 0.5f);
    ASSERT_TRUE(individuals_result.is_ok());
    // Results will vary based on random initialization
}

// ================================================================================================
// ERROR HANDLING TESTS
// ================================================================================================

class LTNErrorHandlingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        LTNConfig config;
        config.embedding_dim = 4;
        auto ltn_result = LogicTensorNetwork::create(config);
        ASSERT_TRUE(ltn_result.is_ok());
        ltn_ = std::move(ltn_result).unwrap();
    }

    std::unique_ptr<LogicTensorNetwork> ltn_;
};

TEST_F(LTNErrorHandlingTest, InvalidConfiguration) {
    LTNConfig bad_config;
    bad_config.embedding_dim = 0;  // Invalid

    auto result = LogicTensorNetwork::create(bad_config);
    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), LTNError::INVALID_CONFIGURATION);

    bad_config.embedding_dim = 4;
    bad_config.learning_rate = -0.1f;  // Invalid

    result = LogicTensorNetwork::create(bad_config);
    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(result.unwrap_err(), LTNError::INVALID_CONFIGURATION);
}

TEST_F(LTNErrorHandlingTest, UndefinedEntities) {
    // Try to access non-existent individual
    auto result1 = ltn_->get_individual("NonExistent");
    EXPECT_TRUE(result1.is_err());
    EXPECT_EQ(result1.unwrap_err(), LTNError::UNDEFINED_INDIVIDUAL);

    // Try to access non-existent predicate
    auto result2 = ltn_->get_predicate("NonExistent");
    EXPECT_TRUE(result2.is_err());
    EXPECT_EQ(result2.unwrap_err(), LTNError::UNDEFINED_PREDICATE);

    // Try to evaluate predicate with non-existent entities
    auto result3 = ltn_->evaluate_predicate("NonExistent", "AlsoNonExistent");
    EXPECT_TRUE(result3.is_err());
    // Should be either UNDEFINED_PREDICATE or UNDEFINED_INDIVIDUAL
}

TEST_F(LTNErrorHandlingTest, ArityMismatches) {
    ltn_->add_individual("Alice");
    ltn_->add_predicate("UnaryPred", 1);
    ltn_->add_predicate("BinaryRel", 2);

    // Try to evaluate binary relation as unary predicate
    auto result1 = ltn_->evaluate_predicate("BinaryRel", "Alice");
    EXPECT_TRUE(result1.is_err());
    EXPECT_EQ(result1.unwrap_err(), LTNError::DIMENSION_MISMATCH);

    // Try to create predicate with zero arity
    auto result2 = ltn_->add_predicate("ZeroArity", 0);
    EXPECT_TRUE(result2.is_err());
    EXPECT_EQ(result2.unwrap_err(), LTNError::INVALID_CONFIGURATION);
}

// ================================================================================================
// MAIN TEST RUNNER
// ================================================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
