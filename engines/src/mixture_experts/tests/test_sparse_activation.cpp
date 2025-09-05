#include <chrono>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "../moe_engine.hpp"  // For MoEError definition
#include "../sparse_activation.hpp"

namespace engines::mixture_experts {

class SparseActivationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Default test configuration
        config_.sparsity_threshold = 0.01f;
        config_.block_size = 64;
        config_.vector_alignment = 32;
        config_.enable_simd_optimization = true;
        config_.enable_dynamic_sparsity = false;
        config_.target_sparsity_ratio = 0.7f;
    }

    SparseConfig config_;

    // Helper function to create test vector with known sparsity pattern
    std::vector<float> create_test_vector(std::size_t size, float sparsity_ratio = 0.5f) {
        std::vector<float> vec(size, 0.0f);
        std::size_t non_zero_count = static_cast<std::size_t>(size * (1.0f - sparsity_ratio));

        // Fill first part with non-zero values
        std::iota(vec.begin(), vec.begin() + non_zero_count, 1.0f);

        return vec;
    }
};

TEST_F(SparseActivationTest, CreateSparseActivationWithValidConfig) {
    auto sparse_result = SparseActivation::create(config_);
    ASSERT_TRUE(sparse_result.is_ok()) << "Failed to create SparseActivation with valid config";

    auto sparse_activation = std::move(sparse_result).unwrap();
    ASSERT_NE(sparse_activation, nullptr);
}

TEST_F(SparseActivationTest, CreateSparseActivationWithInvalidThreshold) {
    // Test with invalid sparsity threshold (too high)
    config_.sparsity_threshold = 1.5f;

    auto sparse_result = SparseActivation::create(config_);
    ASSERT_TRUE(sparse_result.is_err()) << "Should fail with invalid sparsity threshold";
    EXPECT_EQ(sparse_result.unwrap_err(), MoEError::SPARSE_ACTIVATION_ERROR);
}

TEST_F(SparseActivationTest, CreateSparseActivationWithZeroBlockSize) {
    // Test with zero block size (invalid)
    config_.block_size = 0;

    auto sparse_result = SparseActivation::create(config_);
    ASSERT_TRUE(sparse_result.is_err()) << "Should fail with zero block size";
    EXPECT_EQ(sparse_result.unwrap_err(), MoEError::SPARSE_ACTIVATION_ERROR);
}

TEST_F(SparseActivationTest, SparsePatternFromDense) {
    std::vector<float> dense_input = {1.0f, 0.005f, 2.0f, 0.001f, 3.0f};  // threshold = 0.01f

    auto pattern = SparsePattern::from_dense(dense_input, 0.01f);

    // Should keep values > 0.01f: 1.0f, 2.0f, 3.0f (indices 0, 2, 4)
    EXPECT_EQ(pattern.get_nnz(), 3);
    EXPECT_NEAR(pattern.get_sparsity_ratio(), 0.4f, 1e-6);  // 2 out of 5 are zero

    const auto& values = pattern.get_values();
    const auto& indices = pattern.get_indices();

    EXPECT_EQ(values.size(), 3);
    EXPECT_EQ(indices.size(), 3);

    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);

    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 4);
}

TEST_F(SparseActivationTest, SparsePatternToDense) {
    // Create a sparse pattern from dense data
    std::vector<float> dense_input = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};
    auto pattern = SparsePattern::from_dense(dense_input, 0.01f);

    auto dense_output = pattern.to_dense(6);

    EXPECT_EQ(dense_output.size(), 6);
    EXPECT_FLOAT_EQ(dense_output[0], 1.0f);
    EXPECT_FLOAT_EQ(dense_output[1], 0.0f);
    EXPECT_FLOAT_EQ(dense_output[2], 2.0f);
    EXPECT_FLOAT_EQ(dense_output[3], 0.0f);
    EXPECT_FLOAT_EQ(dense_output[4], 3.0f);
    EXPECT_FLOAT_EQ(dense_output[5], 0.0f);
}

TEST_F(SparseActivationTest, ApplySparseActivationBasic) {
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    std::vector<float> input = create_test_vector(16, 0.5f);  // 50% sparsity
    std::vector<float> expert_weights = {1.0f, 0.8f};

    auto pattern_result = sparse_activation->apply_sparse_activation(input, expert_weights);
    ASSERT_TRUE(pattern_result.is_ok()) << "Failed to apply sparse activation";

    auto pattern = std::move(pattern_result).unwrap();
    EXPECT_GT(pattern.get_nnz(), 0) << "Should have some non-zero elements";
    EXPECT_LT(pattern.get_nnz(), input.size()) << "Should be sparse";
}

TEST_F(SparseActivationTest, ApplySparseActivationEmptyInput) {
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    std::vector<float> empty_input;
    std::vector<float> expert_weights = {1.0f};

    auto pattern_result = sparse_activation->apply_sparse_activation(empty_input, expert_weights);
    ASSERT_TRUE(pattern_result.is_err()) << "Should fail with empty input";
    EXPECT_EQ(pattern_result.unwrap_err(), MoEError::SPARSE_ACTIVATION_ERROR);
}

TEST_F(SparseActivationTest, SparseMatrixVectorMultiply) {
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    // Create sparse pattern: [1.0, 0.0, 2.0, 0.0] from dense data
    std::vector<float> dense_input = {1.0f, 0.0f, 2.0f, 0.0f};
    auto sparse_pattern = SparsePattern::from_dense(dense_input, 0.01f);

    // Weight matrix: 2x4 matrix [[1,2,3,4], [5,6,7,8]]
    std::vector<float> weight_matrix = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::size_t rows = 2, cols = 4;

    auto result =
        sparse_activation->sparse_matrix_vector_multiply(sparse_pattern, weight_matrix, rows, cols);
    ASSERT_TRUE(result.is_ok()) << "Sparse matrix-vector multiply should succeed";

    auto output = std::move(result).unwrap();
    EXPECT_EQ(output.size(), rows);

    // Expected: [1*1 + 2*3, 1*5 + 2*7] = [7, 19]
    EXPECT_NEAR(output[0], 7.0f, 1e-6);
    EXPECT_NEAR(output[1], 19.0f, 1e-6);
}

TEST_F(SparseActivationTest, SparseMatrixVectorMultiplyInvalidDimensions) {
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    std::vector<float> dense_input = {1.0f};
    auto sparse_pattern = SparsePattern::from_dense(dense_input, 0.01f);

    // Incorrect matrix size: should be rows * cols = 2 * 4 = 8
    std::vector<float> weight_matrix = {1.0f, 2.0f, 3.0f};  // Size 3, not 8
    std::size_t rows = 2, cols = 4;

    auto result =
        sparse_activation->sparse_matrix_vector_multiply(sparse_pattern, weight_matrix, rows, cols);
    ASSERT_TRUE(result.is_err()) << "Should fail with invalid matrix dimensions";
    EXPECT_EQ(result.unwrap_err(), MoEError::SPARSE_ACTIVATION_ERROR);
}

TEST_F(SparseActivationTest, SIMDOptimizationEnabled) {
    // Test with SIMD enabled
    config_.enable_simd_optimization = true;
    auto simd_sparse = SparseActivation::create(config_).unwrap();

    // Create sparse pattern that will trigger SIMD code path (size >= 8)
    std::vector<float> large_input = create_test_vector(32, 0.5f);  // 50% sparsity, 32 elements
    auto sparse_pattern = simd_sparse->apply_sparse_activation(large_input, {});

    ASSERT_TRUE(sparse_pattern.is_ok()) << "SIMD sparse activation should succeed";
    auto pattern = std::move(sparse_pattern).unwrap();
    EXPECT_GT(pattern.get_nnz(), 0) << "Should have non-zero elements";
}

TEST_F(SparseActivationTest, SIMDOptimizationDisabled) {
    // Test with SIMD disabled (scalar fallback)
    config_.enable_simd_optimization = false;
    auto scalar_sparse = SparseActivation::create(config_).unwrap();

    // Create sparse pattern using scalar code path
    std::vector<float> large_input = create_test_vector(32, 0.5f);
    auto sparse_pattern = scalar_sparse->apply_sparse_activation(large_input, {});

    ASSERT_TRUE(sparse_pattern.is_ok()) << "Scalar sparse activation should succeed";
    auto pattern = std::move(sparse_pattern).unwrap();
    EXPECT_GT(pattern.get_nnz(), 0) << "Should have non-zero elements";
}

TEST_F(SparseActivationTest, SIMDConsistencyLargeVector) {
    // Test that SIMD and scalar implementations give consistent results
    std::vector<float> test_input = create_test_vector(64, 0.3f);  // 30% sparsity
    std::vector<float> expert_weights = {1.0f, 0.5f};

    // SIMD version
    config_.enable_simd_optimization = true;
    auto simd_sparse = SparseActivation::create(config_).unwrap();
    auto simd_result = simd_sparse->apply_sparse_activation(test_input, expert_weights);
    ASSERT_TRUE(simd_result.is_ok());
    auto simd_pattern = std::move(simd_result).unwrap();

    // Scalar version
    config_.enable_simd_optimization = false;
    auto scalar_sparse = SparseActivation::create(config_).unwrap();
    auto scalar_result = scalar_sparse->apply_sparse_activation(test_input, expert_weights);
    ASSERT_TRUE(scalar_result.is_ok());
    auto scalar_pattern = std::move(scalar_result).unwrap();

    // Results should be essentially identical
    EXPECT_EQ(simd_pattern.get_nnz(), scalar_pattern.get_nnz()) << "Same number of non-zeros";
    EXPECT_NEAR(simd_pattern.get_sparsity_ratio(), scalar_pattern.get_sparsity_ratio(), 1e-6)
        << "Same sparsity ratio";

    // Compare actual values
    const auto& simd_values = simd_pattern.get_values();
    const auto& scalar_values = scalar_pattern.get_values();
    const auto& simd_indices = simd_pattern.get_indices();
    const auto& scalar_indices = scalar_pattern.get_indices();

    ASSERT_EQ(simd_values.size(), scalar_values.size());
    ASSERT_EQ(simd_indices.size(), scalar_indices.size());

    for (std::size_t i = 0; i < simd_values.size(); ++i) {
        EXPECT_NEAR(simd_values[i], scalar_values[i], 1e-6) << "Value mismatch at index " << i;
        EXPECT_EQ(simd_indices[i], scalar_indices[i]) << "Index mismatch at position " << i;
    }
}

TEST_F(SparseActivationTest, SIMDCapabilityDetection) {
    // Test with SIMD optimization disabled - should always succeed
    config_.enable_simd_optimization = false;
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    auto validation_result = sparse_activation->validate_simd_capabilities();
    EXPECT_TRUE(validation_result.is_ok()) << "SIMD validation should succeed when SIMD disabled";

    // Test with SIMD optimization enabled - may fail if no SIMD hardware available
    config_.enable_simd_optimization = true;
    auto simd_sparse_activation = SparseActivation::create(config_).unwrap();

    // Test validation without asserting - function should not crash
    // On platforms without SIMD support, this may return an error
    (void)simd_sparse_activation->validate_simd_capabilities();
}

TEST_F(SparseActivationTest, SparseElementwiseMultiply) {
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    // Create two sparse patterns with some overlap
    // lhs: [1.0, 0.0, 2.0, 0.0, 3.0] -> indices {0, 2, 4}
    // rhs: [0.0, 4.0, 5.0, 0.0, 6.0] -> indices {1, 2, 4}
    std::vector<float> lhs_dense = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
    std::vector<float> rhs_dense = {0.0f, 4.0f, 5.0f, 0.0f, 6.0f};

    auto lhs = SparsePattern::from_dense(lhs_dense, 0.01f);
    auto rhs = SparsePattern::from_dense(rhs_dense, 0.01f);

    auto result = sparse_activation->sparse_elementwise_multiply(lhs, rhs);
    ASSERT_TRUE(result.is_ok()) << "Sparse elementwise multiply should succeed";

    auto product = std::move(result).unwrap();

    // Should have intersection at indices 2 and 4: 2*5=10, 3*6=18
    EXPECT_EQ(product.get_nnz(), 2);

    const auto& values = product.get_values();
    const auto& indices = product.get_indices();

    EXPECT_FLOAT_EQ(values[0], 10.0f);  // 2 * 5
    EXPECT_FLOAT_EQ(values[1], 18.0f);  // 3 * 6
    EXPECT_EQ(indices[0], 2);
    EXPECT_EQ(indices[1], 4);
}

TEST_F(SparseActivationTest, PerformanceBenchmark) {
    auto sparse_activation = SparseActivation::create(config_).unwrap();

    // Run a small benchmark to ensure no crashes
    std::size_t vector_size = 128;
    float sparsity_ratio = 0.5f;

    auto benchmark_result =
        sparse_activation->benchmark_sparse_performance(vector_size, sparsity_ratio);
    ASSERT_TRUE(benchmark_result.is_ok()) << "Performance benchmark should succeed";

    float gflops = std::move(benchmark_result).unwrap();
    EXPECT_GT(gflops, 0.0f) << "Should report positive GFLOPS";
}

}  // namespace engines::mixture_experts
