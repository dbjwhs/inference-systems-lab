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

/**
 * @file test_ml_types.cpp
 * @brief Comprehensive tests for ML-specific type definitions
 *
 * This file contains tests for the ML types and utilities defined in ml_types.hpp.
 * Tests cover tensor operations, model configuration, inference requests/responses,
 * and utility functions to ensure robust ML inference capabilities.
 */

#include <chrono>
#include <random>

#include <gtest/gtest.h>

#include "../src/ml_types.hpp"

namespace inference_lab::common::ml::tests {

//=============================================================================
// Data Type Tests
//=============================================================================

TEST(DataTypeTest, SizeCalculation) {
    EXPECT_EQ(get_dtype_size(DataType::FLOAT32), 4);
    EXPECT_EQ(get_dtype_size(DataType::FLOAT16), 2);
    EXPECT_EQ(get_dtype_size(DataType::INT32), 4);
    EXPECT_EQ(get_dtype_size(DataType::INT16), 2);
    EXPECT_EQ(get_dtype_size(DataType::INT8), 1);
    EXPECT_EQ(get_dtype_size(DataType::UINT8), 1);
    EXPECT_EQ(get_dtype_size(DataType::BOOL), 1);
    EXPECT_EQ(get_dtype_size(DataType::FLOAT64), 8);
    EXPECT_EQ(get_dtype_size(DataType::COMPLEX64), 4);
    EXPECT_EQ(get_dtype_size(DataType::COMPLEX128), 8);
}

TEST(DataTypeTest, StringConversion) {
    EXPECT_STREQ(dtype_to_string(DataType::FLOAT32), "float32");
    EXPECT_STREQ(dtype_to_string(DataType::FLOAT16), "float16");
    EXPECT_STREQ(dtype_to_string(DataType::INT32), "int32");
    EXPECT_STREQ(dtype_to_string(DataType::INT16), "int16");
    EXPECT_STREQ(dtype_to_string(DataType::INT8), "int8");
    EXPECT_STREQ(dtype_to_string(DataType::UINT8), "uint8");
    EXPECT_STREQ(dtype_to_string(DataType::BOOL), "bool");
    EXPECT_STREQ(dtype_to_string(DataType::FLOAT64), "float64");
    EXPECT_STREQ(dtype_to_string(DataType::COMPLEX64), "complex64");
    EXPECT_STREQ(dtype_to_string(DataType::COMPLEX128), "complex128");
}

//=============================================================================
// MLTensor Tests
//=============================================================================

class MLTensorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create various tensor shapes for testing
        tensor_1d_ = std::make_unique<FloatTensor>(Shape{10});
        tensor_2d_ = std::make_unique<FloatTensor>(Shape{3, 4});
        tensor_3d_ = std::make_unique<FloatTensor>(Shape{2, 3, 4});
    }

    std::unique_ptr<FloatTensor> tensor_1d_;
    std::unique_ptr<FloatTensor> tensor_2d_;
    std::unique_ptr<FloatTensor> tensor_3d_;
};

TEST_F(MLTensorTest, BasicConstruction) {
    // Test type aliases
    FloatTensor float_tensor({2, 2});
    IntTensor int_tensor({3, 3});
    ByteTensor byte_tensor({4, 4});
    BoolTensor bool_tensor({5, 5});

    EXPECT_EQ(float_tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(int_tensor.dtype(), DataType::INT32);
    EXPECT_EQ(byte_tensor.dtype(), DataType::UINT8);
    EXPECT_EQ(bool_tensor.dtype(), DataType::BOOL);
}

TEST_F(MLTensorTest, CreateWithDataType) {
    auto result = FloatTensor::create({3, 3}, DataType::FLOAT32);
    ASSERT_TRUE(result.is_ok());

    auto tensor = std::move(result).unwrap();
    EXPECT_EQ(tensor.size(), 9);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
}

TEST_F(MLTensorTest, SafeReshape) {
    // Valid reshape
    auto result = tensor_2d_->reshape_safe({2, 6});
    EXPECT_TRUE(result.is_ok());
    EXPECT_TRUE(result.unwrap());
    EXPECT_EQ(tensor_2d_->shape()[0], 2);
    EXPECT_EQ(tensor_2d_->shape()[1], 6);

    // Invalid reshape (different total size)
    result = tensor_2d_->reshape_safe({3, 5});
    EXPECT_TRUE(result.is_err());

    // Invalid reshape (zero dimension)
    result = tensor_2d_->reshape_safe({0, 12});
    EXPECT_TRUE(result.is_err());
}

TEST_F(MLTensorTest, BatchExtraction) {
    // Create batch tensor (batch_size=4, feature_size=3)
    FloatTensor batch_tensor({4, 3});

    // Fill with test data
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            batch_tensor(i, j) = static_cast<float>((i * 3) + j);
        }
    }

    // Extract single batch
    auto result = batch_tensor.extract_batch(1, 1);
    ASSERT_TRUE(result.is_ok());

    auto single_batch = std::move(result).unwrap();
    EXPECT_EQ(single_batch.shape()[0], 1);
    EXPECT_EQ(single_batch.shape()[1], 3);

    // Verify data
    EXPECT_FLOAT_EQ(single_batch(0, 0), 3.0F);
    EXPECT_FLOAT_EQ(single_batch(0, 1), 4.0F);
    EXPECT_FLOAT_EQ(single_batch(0, 2), 5.0F);

    // Extract multiple batches
    result = batch_tensor.extract_batch(0, 2);
    ASSERT_TRUE(result.is_ok());

    auto multi_batch = std::move(result).unwrap();
    EXPECT_EQ(multi_batch.shape()[0], 2);
    EXPECT_EQ(multi_batch.shape()[1], 3);

    // Test error cases
    result = batch_tensor.extract_batch(4, 1);  // Out of range
    EXPECT_TRUE(result.is_err());

    result = batch_tensor.extract_batch(3, 2);  // Exceeds bounds
    EXPECT_TRUE(result.is_err());
}

TEST_F(MLTensorTest, MemoryUsage) {
    auto memory_usage = tensor_2d_->total_memory_usage();
    auto expected_data_size = 3 * 4 * sizeof(float);

    EXPECT_GE(memory_usage, expected_data_size);
    EXPECT_LT(memory_usage, expected_data_size + 1024);  // Reasonable overhead
}

//=============================================================================
// Tensor Factory Tests
//=============================================================================

TEST(TensorFactoryTest, ZerosAndOnes) {
    auto zeros = tensor_factory::zeros<float>({3, 3});
    auto ones = tensor_factory::ones<float>({2, 4});

    // Verify zeros
    for (std::size_t i = 0; i < zeros.size(); ++i) {
        EXPECT_FLOAT_EQ(zeros.data()[i], 0.0F);
    }

    // Verify ones
    for (std::size_t i = 0; i < ones.size(); ++i) {
        EXPECT_FLOAT_EQ(ones.data()[i], 1.0F);
    }
}

TEST(TensorFactoryTest, RandomUniform) {
    auto random_tensor = tensor_factory::random_uniform<float>({100}, -1.0F, 1.0F);

    // Check all values are in range
    for (std::size_t i = 0; i < random_tensor.size(); ++i) {
        EXPECT_GE(random_tensor.data()[i], -1.0F);
        EXPECT_LE(random_tensor.data()[i], 1.0F);
    }

    // Basic randomness check - should have some variation
    auto stats = calculate_stats(random_tensor);
    EXPECT_GT(stats.max_val - stats.min_val, 0.5F);  // Should have some spread
}

TEST(TensorFactoryTest, FromData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto result = tensor_factory::from_data<float>({2, 3}, data);
    ASSERT_TRUE(result.is_ok());

    auto tensor = std::move(result).unwrap();
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);

    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], data[i]);
    }

    // Test size mismatch
    result = tensor_factory::from_data<float>({3, 3}, data);  // Wrong size
    EXPECT_TRUE(result.is_err());
}

//=============================================================================
// TensorSpec Tests
//=============================================================================

TEST(TensorSpecTest, Validation) {
    // Valid spec
    TensorSpec valid_spec{.name = "input",
                          .shape = {1, 3, 224, 224},
                          .dtype = DataType::FLOAT32,
                          .is_dynamic = false};
    EXPECT_TRUE(valid_spec.is_valid());

    // Invalid specs
    TensorSpec empty_name = valid_spec;
    empty_name.name = "";
    EXPECT_FALSE(empty_name.is_valid());

    TensorSpec empty_shape = valid_spec;
    empty_shape.shape = {};
    EXPECT_FALSE(empty_shape.is_valid());

    TensorSpec zero_dim = valid_spec;
    zero_dim.shape = {1, 0, 224, 224};
    EXPECT_FALSE(zero_dim.is_valid());
}

TEST(TensorSpecTest, MemoryCalculation) {
    TensorSpec spec{.name = "input",
                    .shape = {1, 3, 224, 224},
                    .dtype = DataType::FLOAT32,
                    .is_dynamic = false};

    auto num_elements = spec.num_elements();
    auto memory_size = spec.memory_size();

    EXPECT_EQ(num_elements, 1 * 3 * 224 * 224);
    EXPECT_EQ(memory_size, num_elements * sizeof(float));
}

//=============================================================================
// ModelConfig Tests
//=============================================================================

TEST(ModelConfigTest, Validation) {
    ModelConfig valid_config{.name = "test_model",
                             .input_specs = {{.name = "input",
                                              .shape = {1, 3, 224, 224},
                                              .dtype = DataType::FLOAT32,
                                              .is_dynamic = false}},
                             .output_specs = {{.name = "output",
                                               .shape = {1, 1000},
                                               .dtype = DataType::FLOAT32,
                                               .is_dynamic = false}},
                             .model_path = "/path/to/model.onnx",
                             .max_batch_size = 8};

    auto result = valid_config.validate();
    EXPECT_TRUE(result.is_ok());
    EXPECT_TRUE(result.unwrap());

    // Test invalid configurations
    ModelConfig empty_name = valid_config;
    empty_name.name = "";
    EXPECT_TRUE(empty_name.validate().is_err());

    ModelConfig empty_path = valid_config;
    empty_path.model_path = "";
    EXPECT_TRUE(empty_path.validate().is_err());

    ModelConfig no_inputs = valid_config;
    no_inputs.input_specs.clear();
    EXPECT_TRUE(no_inputs.validate().is_err());

    ModelConfig zero_batch = valid_config;
    zero_batch.max_batch_size = 0;
    EXPECT_TRUE(zero_batch.validate().is_err());
}

TEST(ModelConfigTest, MemoryCalculation) {
    ModelConfig config{
        .name = "test_model",
        .input_specs =
            {{.name = "input", .shape = {1, 100}, .dtype = DataType::FLOAT32, .is_dynamic = false}},
        .output_specs =
            {{.name = "output", .shape = {1, 10}, .dtype = DataType::FLOAT32, .is_dynamic = false}},
        .model_path = "/path/to/model.onnx",
        .max_batch_size = 4};

    auto input_memory = config.total_input_memory();
    auto output_memory = config.total_output_memory();

    EXPECT_EQ(input_memory, 4 * 100 * sizeof(float));  // batch_size * elements * dtype_size
    EXPECT_EQ(output_memory, 4 * 10 * sizeof(float));
}

//=============================================================================
// InferenceRequest Tests
//=============================================================================

TEST(InferenceRequestTest, Validation) {
    // Create model config
    ModelConfig config{
        .name = "test_model",
        .input_specs =
            {{.name = "input", .shape = {1, 3}, .dtype = DataType::FLOAT32, .is_dynamic = false}},
        .output_specs =
            {{.name = "output", .shape = {1, 1}, .dtype = DataType::FLOAT32, .is_dynamic = false}},
        .model_path = "/path/to/model.onnx",
        .max_batch_size = 4};

    // Create valid request
    FloatTensor input_tensor({1, 3});
    input_tensor.fill(1.0F);

    TensorInput tensor_input;
    tensor_input.name = "input";
    tensor_input.tensor = std::move(input_tensor);

    InferenceRequest valid_request;
    valid_request.batch_size = 1;
    valid_request.inputs.push_back(std::move(tensor_input));

    auto result = valid_request.validate(config);
    EXPECT_TRUE(result.is_ok());
    EXPECT_TRUE(result.unwrap());

    // Test invalid requests
    InferenceRequest no_inputs;
    no_inputs.batch_size = 1;
    EXPECT_TRUE(no_inputs.validate(config).is_err());

    InferenceRequest zero_batch;
    zero_batch.batch_size = 0;
    EXPECT_TRUE(zero_batch.validate(config).is_err());

    InferenceRequest large_batch;
    large_batch.batch_size = 10;  // Exceeds max_batch_size
    EXPECT_TRUE(large_batch.validate(config).is_err());
}

//=============================================================================
// InferenceResponse Tests
//=============================================================================

TEST(InferenceResponseTest, OutputRetrieval) {
    FloatTensor output_tensor({1, 3});
    output_tensor.fill(0.5F);

    TensorOutput tensor_output;
    tensor_output.name = "logits";
    tensor_output.tensor = std::move(output_tensor);
    tensor_output.confidence = 0.95F;

    InferenceResponse response;
    response.inference_time = std::chrono::milliseconds(50);
    response.overall_confidence = 0.95f;
    response.outputs.push_back(std::move(tensor_output));

    // Test successful retrieval
    auto output_opt = response.get_output("logits");
    ASSERT_TRUE(output_opt.has_value());

    const auto& output = output_opt.value().get();
    EXPECT_EQ(output.name, "logits");
    EXPECT_FLOAT_EQ(output.confidence, 0.95F);

    // Test missing output
    auto missing_opt = response.get_output("missing");
    EXPECT_FALSE(missing_opt.has_value());
}

TEST(InferenceResponseTest, MemoryCalculation) {
    FloatTensor output1({10});
    FloatTensor output2({20});

    TensorOutput tensor_output1;
    tensor_output1.name = "output1";
    tensor_output1.tensor = std::move(output1);

    TensorOutput tensor_output2;
    tensor_output2.name = "output2";
    tensor_output2.tensor = std::move(output2);

    InferenceResponse response;
    response.inference_time = std::chrono::milliseconds(100);
    response.outputs.push_back(std::move(tensor_output1));
    response.outputs.push_back(std::move(tensor_output2));

    auto total_memory = response.total_output_memory();
    EXPECT_EQ(total_memory, (10 + 20) * sizeof(float));
}

//=============================================================================
// Classification Tests
//=============================================================================

TEST(ClassificationResultTest, TopKPredictions) {
    ClassificationResult result{.probabilities = {0.1F, 0.7F, 0.05F, 0.15F},
                                .labels = {"cat", "dog", "bird", "fish"},
                                .predicted_class = 1,
                                .max_probability = 0.7F};

    auto top_2 = result.top_k(2);
    ASSERT_EQ(top_2.size(), 2);

    // Should be sorted by probability (descending)
    EXPECT_EQ(top_2[0].first, 1);  // dog
    EXPECT_FLOAT_EQ(top_2[0].second, 0.7F);
    EXPECT_EQ(top_2[1].first, 3);  // fish
    EXPECT_FLOAT_EQ(top_2[1].second, 0.15F);

    // Test k larger than available classes
    auto top_10 = result.top_k(10);
    EXPECT_EQ(top_10.size(), 4);  // Should return all 4 classes
}

//=============================================================================
// Uncertainty Tests
//=============================================================================

TEST(UncertaintyEstimateTest, ReliabilityCheck) {
    UncertaintyEstimate low_uncertainty{.epistemic_uncertainty = 0.02F,
                                        .aleatoric_uncertainty = 0.03F,
                                        .total_uncertainty = 0.05F,
                                        .confidence_interval_95 = 0.1F};

    UncertaintyEstimate high_uncertainty{.epistemic_uncertainty = 0.15F,
                                         .aleatoric_uncertainty = 0.2F,
                                         .total_uncertainty = 0.35F,
                                         .confidence_interval_95 = 0.7F};

    EXPECT_TRUE(low_uncertainty.is_reliable(0.1F));    // Below threshold
    EXPECT_FALSE(high_uncertainty.is_reliable(0.1F));  // Above threshold
}

//=============================================================================
// Batch Processing Tests
//=============================================================================

TEST(BatchResultTest, ThroughputCalculation) {
    std::vector<TensorOutput> outputs;
    outputs.reserve(5);  // 5 samples

    // Create 5 empty outputs
    for (int i = 0; i < 5; ++i) {
        TensorOutput output;
        output.name = "output_" + std::to_string(i);
        output.tensor = FloatTensor({1});
        outputs.push_back(std::move(output));
    }

    BatchResult result{.batch_outputs = std::move(outputs),
                       .total_time = std::chrono::milliseconds(100),  // 100ms total
                       .avg_per_sample_time = std::chrono::milliseconds(20),
                       .batch_efficiency = 0.8f};

    auto throughput = result.get_throughput();
    EXPECT_FLOAT_EQ(throughput, 50.0F);  // 5 samples / 0.1 seconds = 50 samples/sec

    // Test zero time case
    BatchResult zero_time_result{.total_time = std::chrono::milliseconds(0),
                                 .avg_per_sample_time = std::chrono::milliseconds(20),
                                 .batch_efficiency = 0.8f};
    EXPECT_FLOAT_EQ(zero_time_result.get_throughput(), 0.0F);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST(UtilityTest, TensorConversion) {
    FloatTensor float_tensor({2, 2});
    float_tensor.fill(3.14F);

    auto int_tensor = convert_tensor<float, int>(float_tensor);

    EXPECT_EQ(int_tensor.shape(), float_tensor.shape());
    EXPECT_EQ(int_tensor.dtype(), DataType::INT32);

    for (std::size_t i = 0; i < int_tensor.size(); ++i) {
        EXPECT_EQ(int_tensor.data()[i], 3);  // Truncated to int
    }
}

TEST(UtilityTest, TensorStatistics) {
    FloatTensor tensor({5});
    tensor.data()[0] = 1.0F;
    tensor.data()[1] = 2.0F;
    tensor.data()[2] = 3.0F;
    tensor.data()[3] = 4.0F;
    tensor.data()[4] = 5.0F;

    auto stats = calculate_stats(tensor);

    EXPECT_FLOAT_EQ(stats.mean, 3.0F);
    EXPECT_FLOAT_EQ(stats.min_val, 1.0F);
    EXPECT_FLOAT_EQ(stats.max_val, 5.0F);
    EXPECT_EQ(stats.non_zero_count, 5);
    EXPECT_GT(stats.std_dev, 0.0F);  // Should have some variance

    // Test zero-size tensor (single element with value 0)
    FloatTensor zero_tensor({1});
    zero_tensor.zero();
    auto zero_stats = calculate_stats(zero_tensor);
    EXPECT_FLOAT_EQ(zero_stats.mean, 0.0F);
    EXPECT_FLOAT_EQ(zero_stats.std_dev, 0.0F);
    EXPECT_EQ(zero_stats.non_zero_count, 0);
}

}  // namespace inference_lab::common::ml::tests
