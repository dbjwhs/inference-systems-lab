/**
 * @file test_advanced_containers.cpp
 * @brief Comprehensive tests for advanced ML-specific containers
 *
 * Tests the advanced container implementations added to containers.hpp:
 * - BatchContainer for ML inference batching
 * - RealtimeCircularBuffer for streaming data
 * - FeatureCache for ML feature caching
 * - SIMD operations for vectorized computations
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <common/src/containers.hpp>
#include <gtest/gtest.h>

#ifdef __AVX2__
    #include <immintrin.h>
#elif defined(__SSE2__)
    #include <emmintrin.h>
#endif

using namespace inference_lab::common;

// ================================================================================================
// BATCH CONTAINER TESTS
// ================================================================================================

class BatchContainerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        sample_shape_ = {224, 224, 3};  // Image shape: height x width x channels
        elements_per_sample_ = 224 * 224 * 3;
    }

    std::vector<std::size_t> sample_shape_;
    std::size_t elements_per_sample_;
};

TEST_F(BatchContainerTest, BasicConstruction) {
    BatchContainer<float, 32> batch(sample_shape_);

    EXPECT_EQ(batch.size(), 0);
    EXPECT_EQ(batch.capacity(), 32);
    EXPECT_TRUE(batch.empty());
    EXPECT_FALSE(batch.full());
    EXPECT_EQ(batch.sample_shape(), sample_shape_);
    EXPECT_EQ(batch.elements_per_sample(), elements_per_sample_);
}

TEST_F(BatchContainerTest, AddSamples) {
    BatchContainer<float, 4> batch(sample_shape_);

    // Create test data
    std::vector<float> sample_data(elements_per_sample_, 1.0f);

    // Add samples
    EXPECT_TRUE(batch.add_sample(sample_data.data()));
    EXPECT_EQ(batch.size(), 1);
    EXPECT_FALSE(batch.empty());

    EXPECT_TRUE(batch.add_sample(sample_data.data()));
    EXPECT_TRUE(batch.add_sample(sample_data.data()));
    EXPECT_TRUE(batch.add_sample(sample_data.data()));
    EXPECT_EQ(batch.size(), 4);
    EXPECT_TRUE(batch.full());

    // Should fail to add more samples
    EXPECT_FALSE(batch.add_sample(sample_data.data()));
    EXPECT_EQ(batch.size(), 4);
}

TEST_F(BatchContainerTest, SampleDataAccess) {
    BatchContainer<float, 2> batch(sample_shape_);

    // Create different test data for each sample
    std::vector<float> sample1(elements_per_sample_, 1.0f);
    std::vector<float> sample2(elements_per_sample_, 2.0f);

    batch.add_sample(sample1.data());
    batch.add_sample(sample2.data());

    // Verify data integrity
    const float* first_sample = batch.sample_data(0);
    const float* second_sample = batch.sample_data(1);

    EXPECT_FLOAT_EQ(first_sample[0], 1.0f);
    EXPECT_FLOAT_EQ(first_sample[elements_per_sample_ - 1], 1.0f);
    EXPECT_FLOAT_EQ(second_sample[0], 2.0f);
    EXPECT_FLOAT_EQ(second_sample[elements_per_sample_ - 1], 2.0f);
}

TEST_F(BatchContainerTest, UtilizationTracking) {
    BatchContainer<float, 10> batch(sample_shape_);
    std::vector<float> sample_data(elements_per_sample_, 1.0f);

    EXPECT_DOUBLE_EQ(batch.utilization(), 0.0);

    batch.add_sample(sample_data.data());
    EXPECT_DOUBLE_EQ(batch.utilization(), 0.1);

    for (int i = 1; i < 5; ++i) {
        batch.add_sample(sample_data.data());
    }
    EXPECT_DOUBLE_EQ(batch.utilization(), 0.5);

    batch.clear();
    EXPECT_DOUBLE_EQ(batch.utilization(), 0.0);
    EXPECT_TRUE(batch.empty());
}

TEST_F(BatchContainerTest, TensorContainerIntegration) {
    BatchContainer<float, 2> batch(sample_shape_);
    MemoryPool<float> pool(elements_per_sample_ * 4);
    TensorContainer<float> tensor(sample_shape_, std::move(pool));

    // Fill tensor with test data
    std::fill(tensor.begin(), tensor.end(), 3.14f);

    EXPECT_TRUE(batch.add_sample(tensor));
    EXPECT_EQ(batch.size(), 1);

    const float* sample = batch.sample_data(0);
    EXPECT_FLOAT_EQ(sample[0], 3.14f);
    EXPECT_FLOAT_EQ(sample[elements_per_sample_ - 1], 3.14f);
}

TEST_F(BatchContainerTest, MemoryAlignment) {
    BatchContainer<float, 8> batch(sample_shape_);

    // Check that data pointer is properly aligned
    auto* data_ptr = batch.data();
    auto address = reinterpret_cast<std::uintptr_t>(data_ptr);
    EXPECT_EQ(address % 64, 0);  // Should be 64-byte aligned
}

// ================================================================================================
// REALTIME CIRCULAR BUFFER TESTS
// ================================================================================================

class RealtimeCircularBufferTest : public ::testing::Test {
  protected:
    using Buffer = RealtimeCircularBuffer<int, 8>;  // Power of 2 capacity
};

TEST_F(RealtimeCircularBufferTest, BasicOperations) {
    Buffer buffer;

    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 8);

    // Push elements
    EXPECT_TRUE(buffer.push(1));
    EXPECT_TRUE(buffer.push(2));
    EXPECT_EQ(buffer.size(), 2);
    EXPECT_FALSE(buffer.empty());

    // Pop elements
    int value;
    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 1);
    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 2);
    EXPECT_TRUE(buffer.empty());

    // Pop from empty buffer
    EXPECT_FALSE(buffer.pop(value));
}

TEST_F(RealtimeCircularBufferTest, FullBufferBehavior) {
    Buffer buffer;

    // Fill buffer to capacity - 1 (one slot reserved)
    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(buffer.push(i));
    }
    EXPECT_TRUE(buffer.full());

    // Should fail to push when full
    EXPECT_FALSE(buffer.push(999));

    // Pop one element, should be able to push again
    int value;
    EXPECT_TRUE(buffer.pop(value));
    EXPECT_EQ(value, 0);
    EXPECT_FALSE(buffer.full());
    EXPECT_TRUE(buffer.push(7));
}

TEST_F(RealtimeCircularBufferTest, MoveSemantics) {
    Buffer buffer;

    std::string movable_value = "test_string";
    RealtimeCircularBuffer<std::string, 4> string_buffer;

    EXPECT_TRUE(string_buffer.push(std::move(movable_value)));
    EXPECT_TRUE(movable_value.empty());  // Should be moved

    std::string result;
    EXPECT_TRUE(string_buffer.pop(result));
    EXPECT_EQ(result, "test_string");
}

TEST_F(RealtimeCircularBufferTest, ConcurrentAccess) {
    Buffer buffer;
    std::atomic<int> producer_count{0};
    std::atomic<int> consumer_count{0};
    const int num_items = 1000;

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_items; ++i) {
            while (!buffer.push(i)) {
                std::this_thread::yield();
            }
            producer_count.fetch_add(1);
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        int value;
        for (int i = 0; i < num_items; ++i) {
            while (!buffer.pop(value)) {
                std::this_thread::yield();
            }
            EXPECT_EQ(value, i);
            consumer_count.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(producer_count.load(), num_items);
    EXPECT_EQ(consumer_count.load(), num_items);
    EXPECT_TRUE(buffer.empty());
}

// ================================================================================================
// FEATURE CACHE TESTS
// ================================================================================================

class FeatureCacheTest : public ::testing::Test {
  protected:
    void SetUp() override { cache_ = std::make_unique<FeatureCache<std::string, float>>(64); }

    std::unique_ptr<FeatureCache<std::string, float>> cache_;
};

TEST_F(FeatureCacheTest, BasicInsertAndFind) {
    auto result = cache_->insert("feature1", 1.5f);
    EXPECT_TRUE(result.first);  // Insert successful
    EXPECT_EQ(cache_->size(), 1);

    const float* value = cache_->find("feature1");
    ASSERT_NE(value, nullptr);
    EXPECT_FLOAT_EQ(*value, 1.5f);

    // Find non-existent key
    const float* missing = cache_->find("nonexistent");
    EXPECT_EQ(missing, nullptr);
}

TEST_F(FeatureCacheTest, UpdateExistingKey) {
    cache_->insert("feature1", 1.0f);
    auto result = cache_->insert("feature1", 2.0f);

    EXPECT_FALSE(result.first);  // Update, not insert
    EXPECT_EQ(cache_->size(), 1);

    const float* value = cache_->find("feature1");
    ASSERT_NE(value, nullptr);
    EXPECT_FLOAT_EQ(*value, 2.0f);
}

TEST_F(FeatureCacheTest, SubscriptOperator) {
    (*cache_)["new_feature"] = 3.14f;
    EXPECT_EQ(cache_->size(), 1);

    // Debug: check with find first
    const float* found_value = cache_->find("new_feature");
    ASSERT_NE(found_value, nullptr);
    EXPECT_FLOAT_EQ(*found_value, 3.14f);

    // Now test subscript operator
    float& value = (*cache_)["new_feature"];
    EXPECT_FLOAT_EQ(value, 3.14f);

    value = 2.71f;
    const float* found = cache_->find("new_feature");
    ASSERT_NE(found, nullptr);
    EXPECT_FLOAT_EQ(*found, 2.71f);
}

TEST_F(FeatureCacheTest, LoadFactorAndResize) {
    const std::size_t initial_capacity = cache_->capacity();

    // Fill cache beyond load factor threshold
    for (int i = 0; i < 100; ++i) {
        cache_->insert("feature" + std::to_string(i), static_cast<float>(i));
    }

    EXPECT_EQ(cache_->size(), 100);
    EXPECT_GT(cache_->capacity(), initial_capacity);  // Should have resized
    EXPECT_LT(cache_->load_factor(), 0.8);            // Should maintain load factor

    // Verify all values are still accessible
    for (int i = 0; i < 100; ++i) {
        const float* value = cache_->find("feature" + std::to_string(i));
        ASSERT_NE(value, nullptr);
        EXPECT_FLOAT_EQ(*value, static_cast<float>(i));
    }
}

TEST_F(FeatureCacheTest, ClearOperation) {
    cache_->insert("feature1", 1.0f);
    cache_->insert("feature2", 2.0f);
    EXPECT_EQ(cache_->size(), 2);

    cache_->clear();
    EXPECT_EQ(cache_->size(), 0);
    EXPECT_TRUE(cache_->empty());

    EXPECT_EQ(cache_->find("feature1"), nullptr);
    EXPECT_EQ(cache_->find("feature2"), nullptr);
}

TEST_F(FeatureCacheTest, PerformanceBenchmark) {
    const int num_operations = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, num_operations - 1);

    // Insert benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        cache_->insert("key" + std::to_string(i), static_cast<float>(i));
    }
    auto insert_time = std::chrono::high_resolution_clock::now() - start;

    // Lookup benchmark
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_operations; ++i) {
        int key_index = dis(gen);
        cache_->find("key" + std::to_string(key_index));
    }
    auto lookup_time = std::chrono::high_resolution_clock::now() - start;

    auto insert_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_time).count();
    auto lookup_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(lookup_time).count();

    // Performance assertions (adjust based on expected performance)
    EXPECT_LT(insert_ns / num_operations, 1000);  // < 1µs per insert
    EXPECT_LT(lookup_ns / num_operations, 500);   // < 0.5µs per lookup

    std::cout << "Average insert time: " << (insert_ns / num_operations) << " ns\n";
    std::cout << "Average lookup time: " << (lookup_ns / num_operations) << " ns\n";
}

// ================================================================================================
// SIMD OPERATIONS TESTS
// ================================================================================================

class SIMDOperationsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        size_ = 1024;
        a_.resize(size_, 1.0f);
        b_.resize(size_, 2.0f);
        result_.resize(size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (auto& val : a_)
            val = dis(gen);
        for (auto& val : b_)
            val = dis(gen);
    }

    std::size_t size_;
    std::vector<float> a_, b_, result_;
};

TEST_F(SIMDOperationsTest, VectorizedAddition) {
    simd_ops::vectorized_add(a_.data(), b_.data(), result_.data(), size_);

    // Verify results
    for (std::size_t i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(result_[i], a_[i] + b_[i]);
    }
}

TEST_F(SIMDOperationsTest, VectorizedReLU) {
    // Create input with both positive and negative values
    std::vector<float> input(size_);
    for (std::size_t i = 0; i < size_; ++i) {
        input[i] = static_cast<float>(i) - static_cast<float>(size_ / 2);
    }

    simd_ops::vectorized_relu(input.data(), result_.data(), size_);

    // Verify ReLU operation
    for (std::size_t i = 0; i < size_; ++i) {
        float expected = std::max(0.0f, input[i]);
        EXPECT_FLOAT_EQ(result_[i], expected);
    }
}

TEST_F(SIMDOperationsTest, VectorizedSum) {
    // Create simple test data
    std::vector<float> test_data(100, 1.0f);

    float sum = simd_ops::vectorized_sum(test_data.data(), test_data.size());
    EXPECT_FLOAT_EQ(sum, 100.0f);

    // Test with actual random data
    float expected_sum = std::accumulate(a_.begin(), a_.end(), 0.0f);
    float simd_sum = simd_ops::vectorized_sum(a_.data(), a_.size());

    // Allow for small floating point differences
    EXPECT_NEAR(simd_sum, expected_sum, 1e-5);
}

TEST_F(SIMDOperationsTest, PerformanceComparison) {
    const int iterations = 1000;

    // Scalar addition benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (std::size_t i = 0; i < size_; ++i) {
            result_[i] = a_[i] + b_[i];
        }
    }
    auto scalar_time = std::chrono::high_resolution_clock::now() - start;

    // Vectorized addition benchmark
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        simd_ops::vectorized_add(a_.data(), b_.data(), result_.data(), size_);
    }
    auto simd_time = std::chrono::high_resolution_clock::now() - start;

    auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(scalar_time).count();
    auto simd_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(simd_time).count();

    std::cout << "Scalar time: " << scalar_ns << " ns\n";
    std::cout << "SIMD time: " << simd_ns << " ns\n";
    std::cout << "Speedup: " << static_cast<double>(scalar_ns) / simd_ns << "x\n";

    // SIMD should be at least as fast as scalar (may be better with actual SIMD)
    EXPECT_LE(simd_ns, scalar_ns * 1.1);  // Allow 10% margin for test variability
}

TEST_F(SIMDOperationsTest, AlignmentRequirements) {
    // Test with unaligned data to ensure robustness
    std::vector<float> unaligned_a(size_ + 1);
    std::vector<float> unaligned_b(size_ + 1);
    std::vector<float> unaligned_result(size_ + 1);

    // Use offset pointers to create misalignment
    float* a_ptr = unaligned_a.data() + 1;
    float* b_ptr = unaligned_b.data() + 1;
    float* result_ptr = unaligned_result.data() + 1;

    std::fill_n(a_ptr, size_, 1.0f);
    std::fill_n(b_ptr, size_, 2.0f);

    // Should handle unaligned data gracefully
    EXPECT_NO_THROW({ simd_ops::vectorized_add(a_ptr, b_ptr, result_ptr, size_); });

    // Verify correctness with unaligned data
    for (std::size_t i = 0; i < size_; ++i) {
        EXPECT_FLOAT_EQ(result_ptr[i], 3.0f);
    }
}

// ================================================================================================
// INTEGRATION TESTS
// ================================================================================================

TEST(AdvancedContainersIntegration, MLPipelineSimulation) {
    // Simulate a complete ML inference pipeline using all advanced containers

    constexpr std::size_t batch_size = 4;
    constexpr std::size_t feature_dim = 512;

    // Setup feature cache for feature lookup
    FeatureCache<std::string, std::vector<float>> feature_cache(1024);

    // Setup batch container for inference
    BatchContainer<float, batch_size> batch({feature_dim});

    // Setup circular buffer for streaming requests
    RealtimeCircularBuffer<std::string, 16> request_buffer;

    // Simulate incoming requests
    std::vector<std::string> request_ids = {"req1", "req2", "req3", "req4"};
    for (const auto& id : request_ids) {
        EXPECT_TRUE(request_buffer.push(id));
    }

    // Process requests: lookup features and build batch
    std::string request_id;
    while (request_buffer.pop(request_id) && !batch.full()) {
        // Simulate feature lookup (or creation for new features)
        auto& features = feature_cache[request_id];
        if (features.empty()) {
            features.resize(feature_dim, 1.0f);  // Generate dummy features
        }

        EXPECT_TRUE(batch.add_sample(features.data()));
    }

    EXPECT_EQ(batch.size(), batch_size);
    EXPECT_TRUE(batch.full());

    // Simulate SIMD-accelerated preprocessing
    auto batch_data = batch.data();
    std::vector<float> preprocessed(batch.size() * batch.elements_per_sample());

    // Apply ReLU activation to entire batch
    simd_ops::vectorized_relu(batch_data, preprocessed.data(), preprocessed.size());

    // Verify preprocessing results
    for (std::size_t i = 0; i < preprocessed.size(); ++i) {
        EXPECT_GE(preprocessed[i], 0.0f);  // ReLU ensures non-negative values
    }

    std::cout << "ML Pipeline simulation completed successfully\n";
    std::cout << "Processed batch size: " << batch.size() << "\n";
    std::cout << "Feature cache size: " << feature_cache.size() << "\n";
    std::cout << "Remaining requests: " << request_buffer.size() << "\n";
}
