/**
 * @file test_type_system.cpp
 * @brief Comprehensive tests for the ML type system with compile-time shape checking
 *
 * Tests the advanced type system implementation from type_system.hpp:
 * - Compile-time shape verification
 * - Strong type aliases for ML concepts
 * - TypedTensor operations with shape checking
 * - Layer composition and sequential models
 * - Automatic differentiation framework
 */

#include <type_traits>

#include <common/src/type_system.hpp>
#include <gtest/gtest.h>

using namespace inference_lab::common::types;

// ================================================================================================
// COMPILE-TIME SHAPE TESTS
// ================================================================================================

class ShapeTest : public ::testing::Test {
  protected:
    using ImageShape = Shape<224, 224, 3>;
    using VectorShape = Shape<1000>;
    using MatrixShape = Shape<10, 20>;
    using DynamicTestShape = DynamicShape;
};

TEST_F(ShapeTest, CompileTimeProperties) {
    // Test compile-time shape properties
    EXPECT_EQ(ImageShape::rank, 3);
    EXPECT_EQ(ImageShape::size, 224 * 224 * 3);
    EXPECT_TRUE(ImageShape::is_static);

    EXPECT_EQ(ImageShape::template dim<0>(), 224);
    EXPECT_EQ(ImageShape::template dim<1>(), 224);
    EXPECT_EQ(ImageShape::template dim<2>(), 3);

    // Test vector conversion
    auto dims = ImageShape::to_vector();
    EXPECT_EQ(dims.size(), 3);
    EXPECT_EQ(dims[0], 224);
    EXPECT_EQ(dims[1], 224);
    EXPECT_EQ(dims[2], 3);
}

TEST_F(ShapeTest, DynamicShapeOperations) {
    DynamicShape shape({10, 20, 30});

    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.size(), 10 * 20 * 30);
    EXPECT_EQ(shape.dim(0), 10);
    EXPECT_EQ(shape.dim(1), 20);
    EXPECT_EQ(shape.dim(2), 30);

    DynamicShape other({10, 20, 30});
    EXPECT_TRUE(shape == other);

    DynamicShape different({10, 20, 31});
    EXPECT_FALSE(shape == different);
}

TEST_F(ShapeTest, ShapeMetaprogramming) {
    // Test shape concatenation
    using Shape1 = Shape<2, 3>;
    using Shape2 = Shape<4, 5>;
    using ConcatenatedShape = ConcatShapes<Shape1, Shape2>::type;

    EXPECT_EQ(ConcatenatedShape::rank, 4);
    EXPECT_EQ(ConcatenatedShape::template dim<0>(), 2);
    EXPECT_EQ(ConcatenatedShape::template dim<1>(), 3);
    EXPECT_EQ(ConcatenatedShape::template dim<2>(), 4);
    EXPECT_EQ(ConcatenatedShape::template dim<3>(), 5);

    // Test reshape compatibility
    using OriginalShape = Shape<2, 6>;
    using NewShape = Shape<3, 4>;
    EXPECT_TRUE((CanReshape<OriginalShape, NewShape>::value));

    using IncompatibleShape = Shape<3, 5>;
    EXPECT_FALSE((CanReshape<OriginalShape, IncompatibleShape>::value));
}

TEST_F(ShapeTest, MatrixMultiplicationShapes) {
    using LeftMatrix = Shape<5, 3>;
    using RightMatrix = Shape<3, 7>;
    using ResultMatrix = MatMulShape<LeftMatrix, RightMatrix>::type;

    EXPECT_EQ(ResultMatrix::rank, 2);
    EXPECT_EQ(ResultMatrix::template dim<0>(), 5);
    EXPECT_EQ(ResultMatrix::template dim<1>(), 7);

    // This should fail to compile with incompatible dimensions
    // using IncompatibleRight = Shape<4, 7>;
    // using BadResult = MatMulShape<LeftMatrix, IncompatibleRight>::type; // Compile error
}

TEST_F(ShapeTest, BroadcastShapes) {
    using Shape1 = Shape<3, 4>;
    using Shape2 = Shape<3, 4>;
    using BroadcastResult = BroadcastShape<Shape1, Shape2>::type;

    EXPECT_EQ(BroadcastResult::rank, 2);
    EXPECT_EQ(BroadcastResult::template dim<0>(), 3);
    EXPECT_EQ(BroadcastResult::template dim<1>(), 4);
}

// ================================================================================================
// STRONG TYPE TESTS
// ================================================================================================

class StrongTypeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        weights_ = Weights<float>(3.14f);
        bias_ = Bias<float>(2.71f);
        learning_rate_ = LearningRate(0.001f);
        momentum_ = Momentum(0.9f);
    }

    Weights<float> weights_{0.0f};
    Bias<float> bias_{0.0f};
    LearningRate learning_rate_{0.0f};
    Momentum momentum_{0.0f};
};

TEST_F(StrongTypeTest, BasicOperations) {
    EXPECT_FLOAT_EQ(*weights_, 3.14f);
    EXPECT_FLOAT_EQ(weights_.get(), 3.14f);

    EXPECT_FLOAT_EQ(*bias_, 2.71f);
    EXPECT_FLOAT_EQ(*learning_rate_, 0.001f);
    EXPECT_FLOAT_EQ(*momentum_, 0.9f);
}

TEST_F(StrongTypeTest, TypeSafety) {
    // These should be different types, preventing accidental mixing
    EXPECT_FALSE((std::is_same_v<decltype(weights_), decltype(bias_)>));
    EXPECT_FALSE((std::is_same_v<decltype(learning_rate_), decltype(momentum_)>));

    // Cannot assign different strong types to each other
    // weights_ = bias_;  // Should not compile
    // learning_rate_ = momentum_;  // Should not compile
}

TEST_F(StrongTypeTest, PointerSemantics) {
    // Test pointer-like access
    *weights_ = 1.5f;
    EXPECT_FLOAT_EQ(weights_.get(), 1.5f);

    weights_.get() = 2.5f;
    EXPECT_FLOAT_EQ(*weights_, 2.5f);
}

// ================================================================================================
// TYPED TENSOR TESTS
// ================================================================================================

class TypedTensorTest : public ::testing::Test {
  protected:
    using SmallShape = Shape<2, 3>;
    using SmallTensor = TypedTensor<float, SmallShape>;
    using VectorTensor = TypedTensor<float, Shape<6>>;
    using MatrixTensor = TypedTensor<float, Shape<3, 2>>;
};

TEST_F(TypedTensorTest, BasicConstruction) {
    auto tensor = SmallTensor::zeros();

    EXPECT_EQ(SmallTensor::rank, 2);
    EXPECT_EQ(SmallTensor::size, 6);
    EXPECT_TRUE(SmallTensor::is_static);

    // Check zero initialization
    for (std::size_t i = 0; i < SmallTensor::size; ++i) {
        EXPECT_FLOAT_EQ(tensor[i], 0.0f);
    }
}

TEST_F(TypedTensorTest, FilledConstruction) {
    auto tensor = SmallTensor::filled(3.14f);

    for (std::size_t i = 0; i < SmallTensor::size; ++i) {
        EXPECT_FLOAT_EQ(tensor[i], 3.14f);
    }
}

TEST_F(TypedTensorTest, DataAccess) {
    auto tensor = SmallTensor::zeros();

    // Test element access
    tensor[0] = 1.0f;
    tensor[5] = 2.0f;

    EXPECT_FLOAT_EQ(tensor[0], 1.0f);
    EXPECT_FLOAT_EQ(tensor[5], 2.0f);

    // Test data pointer access
    const float* data = tensor.data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[5], 2.0f);
}

TEST_F(TypedTensorTest, ReshapeOperation) {
    auto original = SmallTensor::filled(1.5f);
    auto reshaped = original.template reshape<Shape<6>>();

    EXPECT_EQ(decltype(reshaped)::rank, 1);
    EXPECT_EQ(decltype(reshaped)::size, 6);

    // Data should be preserved
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(reshaped[i], 1.5f);
    }
}

TEST_F(TypedTensorTest, ElementWiseAddition) {
    auto tensor1 = SmallTensor::filled(1.0f);
    auto tensor2 = SmallTensor::filled(2.0f);

    auto result = tensor1 + tensor2;

    // Result should have same shape
    EXPECT_EQ(decltype(result)::rank, 2);
    EXPECT_EQ(decltype(result)::size, 6);

    // Check addition results
    for (std::size_t i = 0; i < SmallTensor::size; ++i) {
        EXPECT_FLOAT_EQ(result[i], 3.0f);
    }
}

TEST_F(TypedTensorTest, MatrixMultiplication) {
    // Create 2x3 matrix
    auto left = TypedTensor<float, Shape<2, 3>>::zeros();
    left[0] = 1;
    left[1] = 2;
    left[2] = 3;  // First row
    left[3] = 4;
    left[4] = 5;
    left[5] = 6;  // Second row

    // Create 3x2 matrix
    auto right = TypedTensor<float, Shape<3, 2>>::zeros();
    right[0] = 7;
    right[1] = 8;  // First row
    right[2] = 9;
    right[3] = 10;  // Second row
    right[4] = 11;
    right[5] = 12;  // Third row

    auto result = left.matmul(right);

    // Result should be 2x2
    EXPECT_EQ(decltype(result)::rank, 2);
    EXPECT_EQ(decltype(result)::template dim<0>(), 2);
    EXPECT_EQ(decltype(result)::template dim<1>(), 2);

    // Verify matrix multiplication results
    // [1 2 3] [7  8 ]   [58  64]
    // [4 5 6] [9  10] = [139 154]
    //         [11 12]
    EXPECT_FLOAT_EQ(result[0], 58.0f);   // (1*7 + 2*9 + 3*11)
    EXPECT_FLOAT_EQ(result[1], 64.0f);   // (1*8 + 2*10 + 3*12)
    EXPECT_FLOAT_EQ(result[2], 139.0f);  // (4*7 + 5*9 + 6*11)
    EXPECT_FLOAT_EQ(result[3], 154.0f);  // (4*8 + 5*10 + 6*12)
}

// ================================================================================================
// LAYER TESTS
// ================================================================================================

class LayerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        dense_layer_ = std::make_unique<DenseLayer<4, 2>>();
        relu_layer_ = std::make_unique<ReLULayer<Shape<2>>>();
    }

    std::unique_ptr<DenseLayer<4, 2>> dense_layer_;
    std::unique_ptr<ReLULayer<Shape<2>>> relu_layer_;
};

TEST_F(LayerTest, DenseLayerForward) {
    // Create input tensor
    auto input = TypedTensor<float, Shape<4>>::zeros();
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    input[3] = 4.0f;

    // Initialize weights to identity-like pattern for predictable output
    auto& weights = dense_layer_->weights();
    auto& bias = dense_layer_->bias();

    // Set simple weights: output[i] = sum of inputs + bias[i]
    for (std::size_t out = 0; out < 2; ++out) {
        bias[out] = 0.0f;
        for (std::size_t in = 0; in < 4; ++in) {
            weights[in * 2 + out] = 1.0f;
        }
    }

    auto output = dense_layer_->forward(input);

    EXPECT_EQ(decltype(output)::rank, 1);
    EXPECT_EQ(decltype(output)::size, 2);

    // Each output should be sum of inputs (1+2+3+4 = 10)
    EXPECT_FLOAT_EQ(output[0], 10.0f);
    EXPECT_FLOAT_EQ(output[1], 10.0f);
}

TEST_F(LayerTest, ReLULayerForward) {
    // Create input with both positive and negative values
    auto input = TypedTensor<float, Shape<2>>::zeros();
    input[0] = -1.5f;
    input[1] = 2.5f;

    auto output = relu_layer_->forward(input);

    EXPECT_FLOAT_EQ(output[0], 0.0f);  // ReLU(-1.5) = 0
    EXPECT_FLOAT_EQ(output[1], 2.5f);  // ReLU(2.5) = 2.5
}

// ================================================================================================
// SEQUENTIAL MODEL TESTS
// ================================================================================================

TEST(SequentialModelTest, LayerComposition) {
    // Create a simple 2-layer network: Dense(4->3) -> ReLU -> Dense(3->1)
    auto model = make_sequential(DenseLayer<4, 3>(), ReLULayer<Shape<3>>(), DenseLayer<3, 1>());

    // Create test input
    auto input = TypedTensor<float, Shape<4>>::filled(1.0f);

    // Forward pass through entire model
    auto output = model.forward(input);

    // Output should be 1D with size 1
    EXPECT_EQ(decltype(output)::rank, 1);
    EXPECT_EQ(decltype(output)::size, 1);

    // Should produce some output (exact value depends on random initialization)
    EXPECT_TRUE(std::isfinite(output[0]));
}

TEST(SequentialModelTest, CompileTimeShapeVerification) {
    // This should compile successfully
    auto valid_model =
        make_sequential(DenseLayer<10, 5>(), ReLULayer<Shape<5>>(), DenseLayer<5, 1>());

    // This would fail to compile due to shape mismatch:
    // auto invalid_model = make_sequential(
    //     DenseLayer<10, 5>(),
    //     ReLULayer<Shape<3>>(),  // Wrong shape - expects 5, not 3
    //     DenseLayer<5, 1>()
    // );

    SUCCEED();  // Test passes if compilation succeeds
}

// ================================================================================================
// TYPE TRAITS TESTS
// ================================================================================================

TEST(TypeTraitsTest, TensorTypeTraits) {
    using TestTensor = TypedTensor<float, Shape<3, 4>>;

    EXPECT_TRUE(is_typed_tensor_v<TestTensor>);
    EXPECT_FALSE(is_typed_tensor_v<int>);
    EXPECT_FALSE(is_typed_tensor_v<std::vector<float>>);

    EXPECT_TRUE((std::is_same_v<tensor_element_type_t<TestTensor>, float>));
    EXPECT_TRUE((std::is_same_v<tensor_shape_type_t<TestTensor>, Shape<3, 4>>));
}

TEST(TypeTraitsTest, ShapeTypeTraits) {
    using TestShape = Shape<3, 4, 5>;
    EXPECT_TRUE(is_static_shape_v<TestShape>);
    EXPECT_FALSE(is_static_shape_v<DynamicShape>);
    EXPECT_FALSE(is_static_shape_v<int>);
}

// ================================================================================================
// AUTOMATIC DIFFERENTIATION TESTS
// ================================================================================================

class AutoDiffTest : public ::testing::Test {
  protected:
    void SetUp() override {
        x_ = DualFloat(2.0f, 1.0f);  // Value = 2, gradient = 1 (dx/dx = 1)
        y_ = DualFloat(3.0f, 0.0f);  // Value = 3, gradient = 0 (dy/dx = 0)
    }

    DualFloat x_{0.0f, 0.0f};
    DualFloat y_{0.0f, 0.0f};
};

TEST_F(AutoDiffTest, BasicArithmetic) {
    // Test addition: (x + y)' = x' + y'
    auto sum = x_ + y_;
    EXPECT_FLOAT_EQ(sum.value, 5.0f);     // 2 + 3 = 5
    EXPECT_FLOAT_EQ(sum.gradient, 1.0f);  // 1 + 0 = 1

    // Test multiplication: (x * y)' = x' * y + x * y'
    auto product = x_ * y_;
    EXPECT_FLOAT_EQ(product.value, 6.0f);     // 2 * 3 = 6
    EXPECT_FLOAT_EQ(product.gradient, 3.0f);  // 1 * 3 + 2 * 0 = 3
}

TEST_F(AutoDiffTest, ActivationFunctions) {
    // Test ReLU and its derivative
    DualFloat positive(2.0f, 1.0f);
    DualFloat negative(-1.0f, 1.0f);

    auto relu_pos = relu(positive);
    EXPECT_FLOAT_EQ(relu_pos.value, 2.0f);
    EXPECT_FLOAT_EQ(relu_pos.gradient, 1.0f);  // Derivative of ReLU(x>0) = 1

    auto relu_neg = relu(negative);
    EXPECT_FLOAT_EQ(relu_neg.value, 0.0f);
    EXPECT_FLOAT_EQ(relu_neg.gradient, 0.0f);  // Derivative of ReLU(x<0) = 0

    // Test sigmoid
    DualFloat input(0.0f, 1.0f);
    auto sig = sigmoid(input);
    EXPECT_FLOAT_EQ(sig.value, 0.5f);      // sigmoid(0) = 0.5
    EXPECT_FLOAT_EQ(sig.gradient, 0.25f);  // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
}

TEST_F(AutoDiffTest, ChainRule) {
    // Test chain rule: f(g(x)) where f(u) = u^2, g(x) = x + 1
    // f'(g(x)) * g'(x) = 2*(x+1) * 1 = 2*(x+1)

    DualFloat x(2.0f, 1.0f);
    auto g = x + DualFloat(1.0f, 0.0f);  // g(x) = x + 1
    auto f = g * g;                      // f(g) = g^2

    EXPECT_FLOAT_EQ(f.value, 9.0f);     // (2+1)^2 = 9
    EXPECT_FLOAT_EQ(f.gradient, 6.0f);  // 2*(2+1)*1 = 6
}

// ================================================================================================
// PERFORMANCE AND MEMORY TESTS
// ================================================================================================

TEST(PerformanceTest, CompileTimeOptimizations) {
    // These operations should be optimizable at compile time
    constexpr auto shape_size = Shape<100, 200>::size;
    EXPECT_EQ(shape_size, 20000);

    constexpr auto shape_rank = Shape<1, 2, 3, 4>::rank;
    EXPECT_EQ(shape_rank, 4);

    constexpr auto dim_check = Shape<10, 20, 30>::template dim<1>();
    EXPECT_EQ(dim_check, 20);
}

TEST(PerformanceTest, ZeroCostAbstractions) {
    // Verify that our type system doesn't add runtime overhead
    const int iterations = 1000000;

    // Test TypedTensor performance vs raw array
    auto tensor = TypedTensor<float, Shape<1000>>::zeros();
    std::array<float, 1000> raw_array{};

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (std::size_t j = 0; j < 1000; ++j) {
            tensor[j] = static_cast<float>(j);
        }
    }
    auto tensor_time = std::chrono::high_resolution_clock::now() - start;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (std::size_t j = 0; j < 1000; ++j) {
            raw_array[j] = static_cast<float>(j);
        }
    }
    auto array_time = std::chrono::high_resolution_clock::now() - start;

    auto tensor_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tensor_time).count();
    auto array_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(array_time).count();

    // TypedTensor should have similar performance to raw array
    double overhead_ratio = static_cast<double>(tensor_ns) / array_ns;
    EXPECT_LT(overhead_ratio, 1.1);  // Less than 10% overhead

    std::cout << "TypedTensor time: " << tensor_ns << " ns\n";
    std::cout << "Raw array time: " << array_ns << " ns\n";
    std::cout << "Overhead ratio: " << overhead_ratio << "\n";
}

// ================================================================================================
// INTEGRATION TESTS
// ================================================================================================

TEST(TypeSystemIntegration, FullMLWorkflow) {
    // Demonstrate a complete ML workflow using the type system

    // 1. Define model architecture at compile time
    using InputShape = Shape<4>;
    using HiddenShape = Shape<8>;
    using OutputShape = Shape<2>;

    // 2. Create model with compile-time shape verification
    auto model = make_sequential(DenseLayer<4, 8>(), ReLULayer<HiddenShape>(), DenseLayer<8, 2>());

    // 3. Create input data
    auto input = TypedTensor<float, InputShape>::zeros();
    input[0] = 1.0f;
    input[1] = 0.5f;
    input[2] = -0.3f;
    input[3] = 0.8f;

    // 4. Forward pass with automatic shape verification
    auto output = model.forward(input);

    // 5. Verify output properties
    EXPECT_EQ(decltype(output)::rank, 1);
    EXPECT_EQ(decltype(output)::size, 2);

    // 6. Use strong types for training parameters
    LearningRate lr(0.001f);
    Momentum momentum(0.9f);

    EXPECT_FLOAT_EQ(*lr, 0.001f);
    EXPECT_FLOAT_EQ(*momentum, 0.9f);

    // 7. Automatic differentiation for gradient computation
    DualFloat loss_input(output[0], 1.0f);  // Assume gradient w.r.t. first output
    auto sigmoid_output = sigmoid(loss_input);

    EXPECT_TRUE(std::isfinite(sigmoid_output.value));
    EXPECT_TRUE(std::isfinite(sigmoid_output.gradient));

    std::cout << "ML workflow completed with compile-time shape verification\n";
    std::cout << "Output: [" << output[0] << ", " << output[1] << "]\n";
    std::cout << "Sigmoid output: " << sigmoid_output.value << " (grad: " << sigmoid_output.gradient
              << ")\n";
}
