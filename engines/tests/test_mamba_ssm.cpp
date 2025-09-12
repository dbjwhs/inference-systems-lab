// MIT License
// Copyright (c) 2025 dbjwhs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include "../src/mamba_ssm/mamba_ssm.hpp"

using namespace inference_lab::engines::mamba_ssm;

class MambaSSMTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.d_model = 64;
        config_.d_state = 16;
        config_.d_inner = 128;
        config_.d_conv = 4;
        config_.max_seq_len = 256;
        config_.batch_size = 2;
        config_.use_simd_kernels = true;
        config_.activation = MambaSSMConfig::ActivationType::SILU;
    }

    MambaSSMConfig config_;
};

TEST_F(MambaSSMTest, EngineCreation) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->is_ready());
}

TEST_F(MambaSSMTest, InvalidConfiguration) {
    MambaSSMConfig invalid_config;
    invalid_config.d_model = 0;  // Invalid: must be positive
    invalid_config.d_state = 16;
    invalid_config.d_inner = 128;

    auto engine_result = create_mamba_ssm_engine(invalid_config);
    EXPECT_FALSE(engine_result.is_ok());
    EXPECT_EQ(engine_result.unwrap_err(), MambaSSMError::INVALID_CONFIGURATION);
}

TEST_F(MambaSSMTest, BackendInfo) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();
    std::string info = engine->get_backend_info();

    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("Mamba State Space Model"), std::string::npos);
    EXPECT_NE(info.find("64"), std::string::npos);   // d_model
    EXPECT_NE(info.find("16"), std::string::npos);   // d_state
    EXPECT_NE(info.find("128"), std::string::npos);  // d_inner
    EXPECT_NE(info.find("SIMD"), std::string::npos);
}

TEST_F(MambaSSMTest, BasicSequenceProcessing) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Create test input sequence [batch=2, seq_len=32, d_model=64]
    const size_t batch = 2;
    const size_t seq_len = 32;
    FloatTensor input(Shape{batch, seq_len, config_.d_model});

    // Fill with test pattern
    auto input_data = input.data();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t d = 0; d < config_.d_model; ++d) {
                size_t idx = (b * seq_len + t) * config_.d_model + d;
                input_data[idx] = 0.1f * static_cast<float>(t + d) + 0.01f * static_cast<float>(b);
            }
        }
    }

    // Run inference
    auto result = engine->run_mamba_ssm(input);
    ASSERT_TRUE(result.is_ok());

    auto output = std::move(result).unwrap();
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], config_.d_model);

    // Verify output contains reasonable values
    auto output_data = output.data();
    bool has_non_zero = false;
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i])) << "NaN found at index " << i;
        EXPECT_FALSE(std::isinf(output_data[i])) << "Inf found at index " << i;
        if (std::abs(output_data[i]) > 1e-6) {
            has_non_zero = true;
        }
    }
    EXPECT_TRUE(has_non_zero) << "Output should contain non-zero values";

    // Check metrics
    auto metrics = engine->get_metrics();
    EXPECT_EQ(metrics.sequence_length, seq_len);
    EXPECT_GT(metrics.inference_time_ms.count(), 0);
    EXPECT_GT(metrics.selective_scan_time_ms.count(), 0);
    EXPECT_GT(metrics.throughput_tokens_per_sec, 0.0);
    EXPECT_GT(metrics.total_flops, 0);
}

TEST_F(MambaSSMTest, SequenceLengthLimits) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Test sequence that exceeds maximum length
    const size_t excessive_seq_len = config_.max_seq_len + 100;
    FloatTensor long_input(Shape{1, excessive_seq_len, config_.d_model});

    auto result = engine->run_mamba_ssm(long_input);
    EXPECT_FALSE(result.is_ok());
    EXPECT_EQ(result.unwrap_err(), MambaSSMError::SEQUENCE_TOO_LONG);
}

TEST_F(MambaSSMTest, DimensionMismatch) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Test input with wrong model dimension
    FloatTensor wrong_input(Shape{1, 32, config_.d_model + 10});

    auto result = engine->run_mamba_ssm(wrong_input);
    EXPECT_FALSE(result.is_ok());
    EXPECT_EQ(result.unwrap_err(), MambaSSMError::DIMENSION_MISMATCH);

    // Test 2D input (should be 3D)
    FloatTensor wrong_dims(Shape{32, config_.d_model});
    auto result2 = engine->run_mamba_ssm(wrong_dims);
    EXPECT_FALSE(result2.is_ok());
    EXPECT_EQ(result2.unwrap_err(), MambaSSMError::DIMENSION_MISMATCH);
}

TEST_F(MambaSSMTest, StateManagement) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Get initial state
    const auto& initial_state = engine->get_state();
    EXPECT_EQ(initial_state.sequence_length, 0);

    // Process a sequence
    FloatTensor input = inference_lab::engines::mamba_ssm::testing::generate_random_sequence(
        1, 16, config_.d_model);
    auto result = engine->run_mamba_ssm(input);
    ASSERT_TRUE(result.is_ok());

    // Check state was updated
    const auto& updated_state = engine->get_state();
    EXPECT_EQ(updated_state.sequence_length, 0);  // Batch processing doesn't update sequence_length

    // Clone state
    auto cloned_state = updated_state.clone();
    ASSERT_NE(cloned_state, nullptr);
    EXPECT_EQ(cloned_state->sequence_length, updated_state.sequence_length);

    // Reset state
    engine->reset_state();
    const auto& reset_state = engine->get_state();
    EXPECT_EQ(reset_state.sequence_length, 0);
}

TEST_F(MambaSSMTest, ConfigurationUpdate) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Update configuration
    MambaSSMConfig new_config = config_;
    new_config.d_model = 32;
    new_config.d_state = 8;
    new_config.d_inner = 64;

    engine->update_config(new_config);
    const auto& updated_config = engine->get_config();

    EXPECT_EQ(updated_config.d_model, 32);
    EXPECT_EQ(updated_config.d_state, 8);
    EXPECT_EQ(updated_config.d_inner, 64);

    // Verify backend info reflects new config
    std::string info = engine->get_backend_info();
    EXPECT_NE(info.find("32"), std::string::npos);  // new d_model
    EXPECT_NE(info.find("8"), std::string::npos);   // new d_state
    EXPECT_NE(info.find("64"), std::string::npos);  // new d_inner
}

TEST_F(MambaSSMTest, SelectiveParameters) {
    const size_t batch = 2;
    const size_t seq_len = 16;
    const size_t d_inner = 64;
    const size_t d_state = 16;

    SelectiveParameters params(batch, seq_len, d_inner, d_state);

    // Check tensor shapes
    EXPECT_EQ(params.delta.shape().size(), 3);
    EXPECT_EQ(params.delta.shape()[0], batch);
    EXPECT_EQ(params.delta.shape()[1], seq_len);
    EXPECT_EQ(params.delta.shape()[2], d_inner);

    EXPECT_EQ(params.B_matrix.shape().size(), 3);
    EXPECT_EQ(params.B_matrix.shape()[0], batch);
    EXPECT_EQ(params.B_matrix.shape()[1], seq_len);
    EXPECT_EQ(params.B_matrix.shape()[2], d_state);

    EXPECT_EQ(params.C_matrix.shape().size(), 3);
    EXPECT_EQ(params.C_matrix.shape()[0], batch);
    EXPECT_EQ(params.C_matrix.shape()[1], seq_len);
    EXPECT_EQ(params.C_matrix.shape()[2], d_state);

    // Check that delta values are positive (initialized with abs + 0.1)
    auto delta_data = params.delta.data();
    for (size_t i = 0; i < params.delta.size(); ++i) {
        EXPECT_GT(delta_data[i], 0.0f) << "Delta values should be positive";
    }
}

TEST_F(MambaSSMTest, SSMStateOperations) {
    const size_t batch = 2;
    const size_t d_inner = 64;
    const size_t d_state = 16;
    const size_t conv_width = 4;

    SSMState state(batch, d_inner, d_state, conv_width);

    // Check initial state
    EXPECT_EQ(state.sequence_length, 0);
    EXPECT_EQ(state.last_update_time.count(), 0);

    // Check tensor shapes
    EXPECT_EQ(state.hidden_state.shape().size(), 3);
    EXPECT_EQ(state.hidden_state.shape()[0], batch);
    EXPECT_EQ(state.hidden_state.shape()[1], d_inner);
    EXPECT_EQ(state.hidden_state.shape()[2], d_state);

    EXPECT_EQ(state.conv_state.shape().size(), 3);
    EXPECT_EQ(state.conv_state.shape()[0], batch);
    EXPECT_EQ(state.conv_state.shape()[1], d_inner);
    EXPECT_EQ(state.conv_state.shape()[2], conv_width);

    // Check that state is zero-initialized
    auto hidden_data = state.hidden_state.data();
    auto conv_data = state.conv_state.data();

    for (size_t i = 0; i < state.hidden_state.size(); ++i) {
        EXPECT_EQ(hidden_data[i], 0.0f);
    }
    for (size_t i = 0; i < state.conv_state.size(); ++i) {
        EXPECT_EQ(conv_data[i], 0.0f);
    }

    // Test state cloning
    auto cloned = state.clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->sequence_length, state.sequence_length);
    EXPECT_EQ(cloned->last_update_time, state.last_update_time);

    // Modify original and verify clone is independent
    state.sequence_length = 10;
    EXPECT_NE(cloned->sequence_length, state.sequence_length);

    // Test reset
    state.reset();
    EXPECT_EQ(state.sequence_length, 0);
    EXPECT_EQ(state.last_update_time.count(), 0);
}

TEST_F(MambaSSMTest, UnifiedInferenceInterface) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Test unified InferenceEngine interface
    inference_lab::engines::InferenceRequest request;
    // Request details are ignored in favor of demo model

    auto response_result = engine->run_inference(request);
    ASSERT_TRUE(response_result.is_ok());

    auto response = std::move(response_result).unwrap();

    // Should have output tensors
    EXPECT_EQ(response.output_tensors.size(), 1);
    EXPECT_EQ(response.output_names.size(), 1);
    EXPECT_EQ(response.output_names[0], "mamba_output");

    // Performance stats should be available
    std::string stats = engine->get_performance_stats();
    EXPECT_FALSE(stats.empty());
    EXPECT_NE(stats.find("Inference Time"), std::string::npos);
    EXPECT_NE(stats.find("Throughput"), std::string::npos);
    EXPECT_NE(stats.find("GFLOPS"), std::string::npos);
}

TEST_F(MambaSSMTest, PerformanceMetrics) {
    auto engine_result = create_mamba_ssm_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Initial metrics should be zero/default
    auto initial_metrics = engine->get_metrics();
    EXPECT_EQ(initial_metrics.sequence_length, 0);
    EXPECT_EQ(initial_metrics.inference_time_ms.count(), 0);
    EXPECT_EQ(initial_metrics.total_flops, 0);

    // Run inference to generate metrics
    FloatTensor input = inference_lab::engines::mamba_ssm::testing::generate_random_sequence(
        1, 64, config_.d_model);
    auto result = engine->run_mamba_ssm(input);
    ASSERT_TRUE(result.is_ok());

    // Check metrics were updated
    auto metrics = engine->get_metrics();
    EXPECT_EQ(metrics.sequence_length, 64);
    EXPECT_GT(metrics.inference_time_ms.count(), 0);
    EXPECT_GT(metrics.selective_scan_time_ms.count(), 0);
    EXPECT_GT(metrics.total_flops, 0);
    EXPECT_GT(metrics.flops_per_second, 0.0);
    EXPECT_GT(metrics.throughput_tokens_per_sec, 0.0);
    EXPECT_GT(metrics.average_step_size, 0.0);
    EXPECT_GT(metrics.selective_updates, 0);
    EXPECT_TRUE(metrics.converged);

    // Reset metrics
    engine->reset_metrics();
    auto reset_metrics = engine->get_metrics();
    EXPECT_EQ(reset_metrics.sequence_length, 0);
    EXPECT_EQ(reset_metrics.inference_time_ms.count(), 0);
    EXPECT_EQ(reset_metrics.total_flops, 0);
}

TEST_F(MambaSSMTest, LinearComplexityScaling) {
    // Test that inference time scales linearly with sequence length
    std::vector<size_t> sequence_lengths = {16, 32, 64, 128};
    std::vector<double> inference_times;

    for (size_t seq_len : sequence_lengths) {
        MambaSSMConfig test_config = config_;
        test_config.max_seq_len = std::max(seq_len, test_config.max_seq_len);

        auto engine_result = create_mamba_ssm_engine(test_config);
        ASSERT_TRUE(engine_result.is_ok());

        auto engine = std::move(engine_result).unwrap();
        FloatTensor input = inference_lab::engines::mamba_ssm::testing::generate_random_sequence(
            1, seq_len, config_.d_model);

        auto result = engine->run_mamba_ssm(input);
        ASSERT_TRUE(result.is_ok());

        auto metrics = engine->get_metrics();
        inference_times.push_back(static_cast<double>(metrics.inference_time_ms.count()));
    }

    // Verify roughly linear scaling (allowing for measurement noise)
    ASSERT_GE(inference_times.size(), 2);

    // Check that time roughly doubles when sequence length doubles
    for (size_t i = 1; i < inference_times.size(); ++i) {
        double length_ratio = static_cast<double>(sequence_lengths[i]) / sequence_lengths[i - 1];
        double time_ratio = inference_times[i] / std::max(inference_times[i - 1], 1.0);

        // Allow for significant variation due to measurement noise
        EXPECT_LT(time_ratio, length_ratio * 3.0) << "Time scaling appears worse than linear";
    }
}

TEST_F(MambaSSMTest, DifferentActivationTypes) {
    // Test different activation types
    std::vector<MambaSSMConfig::ActivationType> activations = {
        MambaSSMConfig::ActivationType::SILU,
        MambaSSMConfig::ActivationType::GELU,
        MambaSSMConfig::ActivationType::RELU};

    for (auto activation : activations) {
        MambaSSMConfig test_config = config_;
        test_config.activation = activation;

        auto engine_result = create_mamba_ssm_engine(test_config);
        ASSERT_TRUE(engine_result.is_ok());

        auto engine = std::move(engine_result).unwrap();
        FloatTensor input = inference_lab::engines::mamba_ssm::testing::generate_random_sequence(
            1, 32, config_.d_model);

        auto result = engine->run_mamba_ssm(input);
        EXPECT_TRUE(result.is_ok())
            << "Failed with activation type " << static_cast<int>(activation);

        if (result.is_ok()) {
            auto output = std::move(result).unwrap();
            EXPECT_EQ(output.shape()[1], 32);  // sequence length preserved
        }
    }
}

TEST_F(MambaSSMTest, ErrorStringConversions) {
    EXPECT_EQ(to_string(MambaSSMError::INVALID_CONFIGURATION),
              "Invalid Mamba SSM configuration parameters");
    EXPECT_EQ(to_string(MambaSSMError::SEQUENCE_TOO_LONG),
              "Input sequence exceeds maximum allowed length");
    EXPECT_EQ(to_string(MambaSSMError::DIMENSION_MISMATCH),
              "Tensor dimension mismatch in SSM operations");
    EXPECT_EQ(to_string(MambaSSMError::MEMORY_ALLOCATION_FAILED),
              "Failed to allocate memory for SSM computations");
    EXPECT_EQ(to_string(MambaSSMError::NUMERICAL_INSTABILITY),
              "Numerical instability detected in SSM computations");
    EXPECT_EQ(to_string(MambaSSMError::UNKNOWN_ERROR), "Unknown Mamba SSM error");
}

TEST_F(MambaSSMTest, TestingUtilities) {
    // Test random sequence generation
    auto sequence = inference_lab::engines::mamba_ssm::testing::generate_random_sequence(2, 32, 64);
    EXPECT_EQ(sequence.shape().size(), 3);
    EXPECT_EQ(sequence.shape()[0], 2);   // batch
    EXPECT_EQ(sequence.shape()[1], 32);  // seq_len
    EXPECT_EQ(sequence.shape()[2], 64);  // d_model

    // Verify contains random data (not all zeros)
    auto data = sequence.data();
    bool has_variation = false;
    float first_val = data[0];
    for (size_t i = 1; i < std::min(sequence.size(), 100UL); ++i) {
        if (std::abs(data[i] - first_val) > 1e-6) {
            has_variation = true;
            break;
        }
    }
    EXPECT_TRUE(has_variation) << "Generated sequence should have variation";

    // Test model configuration creation
    auto test_config = inference_lab::engines::mamba_ssm::testing::create_test_model(128, 32);
    EXPECT_EQ(test_config.d_model, 128);
    EXPECT_EQ(test_config.d_state, 32);
    EXPECT_EQ(test_config.d_inner, 256);  // 2 * d_model
    EXPECT_EQ(test_config.batch_size, 1);
    EXPECT_TRUE(test_config.use_simd_kernels);
}

TEST_F(MambaSSMTest, BenchmarkSequenceLengths) {
    // Test the benchmarking utility
    std::vector<size_t> lengths = {16, 32, 64};
    auto benchmark_results =
        inference_lab::engines::mamba_ssm::testing::benchmark_sequence_lengths(lengths);

    // Should have results for each length (if no errors occurred)
    EXPECT_LE(benchmark_results.size(), lengths.size());

    for (const auto& [length, metrics] : benchmark_results) {
        EXPECT_GT(length, 0);
        EXPECT_EQ(metrics.sequence_length, length);
        EXPECT_GT(metrics.inference_time_ms.count(), 0);
        EXPECT_TRUE(metrics.converged);
    }

    // If we have multiple results, verify they're ordered by sequence length
    for (size_t i = 1; i < benchmark_results.size(); ++i) {
        EXPECT_GT(benchmark_results[i].first, benchmark_results[i - 1].first);
    }
}
