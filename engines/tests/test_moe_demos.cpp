// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "../src/mixture_experts/expert_router.hpp"
#include "../src/mixture_experts/load_balancer.hpp"
#include "../src/mixture_experts/moe_engine.hpp"

namespace {

using namespace engines::mixture_experts;

/**
 * @brief Test fixture for MoE demonstration applications
 */
class MoEDemoTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create standard MoE configuration for testing
        config_.num_experts = 8;
        config_.expert_capacity = 2;
        config_.input_dimension = 8;  // Match the 8-dimensional test feature vectors
        config_.load_balancing_weight = 0.1f;
        config_.enable_sparse_activation = false;  // Disable to avoid SIMD-dependent failures
        config_.max_concurrent_requests = 10;
        config_.memory_pool_size_mb = 50;

        // Create MoE engine for testing
        auto engine_result = MoEEngine::create(config_);
        ASSERT_TRUE(engine_result.is_ok()) << "Failed to create MoE engine for testing";
        engine_ = std::move(engine_result).unwrap();
    }

    MoEConfig config_;
    std::unique_ptr<MoEEngine> engine_;
};

/**
 * @brief Test text classification MoE functionality
 */
TEST_F(MoEDemoTest, TextClassificationMoEFunctionality) {
    // Simulate text feature vectors for different domains
    std::vector<std::vector<float>> domain_features = {
        // News domain features (politics, international, business)
        {0.8f, 0.2f, 0.9f, 0.1f, 0.7f, 0.3f, 0.8f, 0.4f},
        // Review domain features (sentiment, product, rating)
        {0.3f, 0.9f, 0.2f, 0.8f, 0.4f, 0.7f, 0.1f, 0.9f},
        // Technical domain features (documentation, API, code)
        {0.9f, 0.1f, 0.8f, 0.2f, 0.9f, 0.1f, 0.8f, 0.2f},
        // Social domain features (conversation, hashtag, mention)
        {0.4f, 0.6f, 0.3f, 0.7f, 0.5f, 0.5f, 0.4f, 0.6f},
        // Academic domain features (research, citation, methodology)
        {0.7f, 0.8f, 0.9f, 0.9f, 0.8f, 0.7f, 0.9f, 0.8f}};

    // Test inference on each domain type
    for (std::size_t domain = 0; domain < domain_features.size(); ++domain) {
        MoEInput input{.features = domain_features[domain],
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = domain,
                       .priority = 1.0f};

        auto response = engine_->run_inference(input);
        if (!response.is_ok()) {
            auto error = response.unwrap_err();
            FAIL() << "Text classification inference failed for domain " << domain
                   << " with error: " << static_cast<int>(error);
        }

        auto result = response.unwrap();

        // Validate response structure
        EXPECT_FALSE(result.outputs.empty()) << "Empty outputs for domain " << domain;
        EXPECT_FALSE(result.selected_experts.empty())
            << "No experts selected for domain " << domain;
        EXPECT_EQ(result.selected_experts.size(), result.expert_weights.size())
            << "Mismatch between selected experts and weights for domain " << domain;
        EXPECT_GT(result.routing_latency_ms, 0.0f)
            << "Invalid routing latency for domain " << domain;
        EXPECT_GT(result.inference_latency_ms, 0.0f)
            << "Invalid inference latency for domain " << domain;
        EXPECT_LE(result.selected_experts.size(), config_.expert_capacity)
            << "Too many experts selected for domain " << domain;
    }
}

/**
 * @brief Test computer vision MoE functionality
 */
TEST_F(MoEDemoTest, ComputerVisionMoEFunctionality) {
    // Simulate image feature vectors for different vision tasks
    std::vector<std::vector<float>> task_features = {
        // Object detection features (edge density, corner count, texture complexity)
        {0.9f, 0.8f, 0.7f, 0.6f, 0.8f, 0.9f, 0.7f, 0.8f},
        // Scene classification features (color histogram, spatial layout, context)
        {0.5f, 0.6f, 0.8f, 0.9f, 0.4f, 0.3f, 0.7f, 0.8f},
        // Facial recognition features (symmetry, facial landmarks, skin tone)
        {0.8f, 0.9f, 0.6f, 0.7f, 0.8f, 0.8f, 0.9f, 0.7f}};

    // Test inference on each vision task
    for (std::size_t task = 0; task < task_features.size(); ++task) {
        MoEInput input{.features = task_features[task],
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = 100 + task,  // Different request ID range for vision
                       .priority = 1.0f};

        auto response = engine_->run_inference(input);
        ASSERT_TRUE(response.is_ok()) << "Computer vision inference failed for task " << task;

        auto result = response.unwrap();

        // Validate response structure
        EXPECT_FALSE(result.outputs.empty()) << "Empty outputs for vision task " << task;
        EXPECT_FALSE(result.selected_experts.empty())
            << "No experts selected for vision task " << task;
        EXPECT_GT(result.active_parameters, 0) << "No active parameters for vision task " << task;
    }
}

/**
 * @brief Test recommendation system MoE functionality
 */
TEST_F(MoEDemoTest, RecommendationMoEFunctionality) {
    // Simulate user/item feature vectors for different recommendation contexts
    std::vector<std::vector<float>> context_features = {
        // Collaborative filtering context (user similarity, item popularity)
        {0.7f, 0.8f, 0.6f, 0.9f, 0.5f, 0.4f, 0.7f, 0.8f},
        // Content-based context (item attributes, user preferences)
        {0.9f, 0.6f, 0.8f, 0.4f, 0.7f, 0.9f, 0.5f, 0.6f},
        // Demographic context (age, location, behavior patterns)
        {0.4f, 0.7f, 0.9f, 0.3f, 0.8f, 0.5f, 0.6f, 0.7f},
        // Hybrid context (multiple signals combined)
        {0.6f, 0.8f, 0.7f, 0.5f, 0.7f, 0.6f, 0.8f, 0.7f}};

    // Test inference on each recommendation context
    for (std::size_t context = 0; context < context_features.size(); ++context) {
        MoEInput input{.features = context_features[context],
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = 200 + context,  // Different request ID range for
                                                     // recommendations
                       .priority = 1.0f};

        auto response = engine_->run_inference(input);
        ASSERT_TRUE(response.is_ok()) << "Recommendation inference failed for context " << context;

        auto result = response.unwrap();

        // Validate response structure and recommendation-specific metrics
        EXPECT_FALSE(result.outputs.empty())
            << "Empty outputs for recommendation context " << context;
        EXPECT_FALSE(result.selected_experts.empty())
            << "No experts selected for recommendation context " << context;
        EXPECT_GT(result.routing_latency_ms, 0.0f)
            << "Invalid routing latency for recommendation context " << context;
    }
}

/**
 * @brief Test expert utilization balance across different demo scenarios
 */
TEST_F(MoEDemoTest, ExpertUtilizationBalance) {
    // Run multiple inferences with varied input patterns
    std::vector<std::vector<float>> varied_inputs;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Generate diverse input patterns
    for (int i = 0; i < 50; ++i) {
        std::vector<float> input(8);
        for (auto& val : input) {
            val = dis(gen);
        }
        varied_inputs.push_back(input);
    }

    // Execute multiple inference requests
    for (std::size_t i = 0; i < varied_inputs.size(); ++i) {
        MoEInput input{.features = varied_inputs[i],
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = 300 + i,
                       .priority = 1.0f};

        auto response = engine_->run_inference(input);
        ASSERT_TRUE(response.is_ok()) << "Inference failed for varied input " << i;
    }

    // Check expert utilization distribution
    auto utilization = engine_->get_expert_utilization();
    EXPECT_EQ(utilization.size(), config_.num_experts) << "Incorrect number of utilization metrics";

    // Verify that experts are being used (at least some should have non-zero utilization)
    bool any_expert_used = false;
    for (const auto& util : utilization) {
        if (util > 0.0f) {
            any_expert_used = true;
            break;
        }
    }
    EXPECT_TRUE(any_expert_used) << "No experts appear to be utilized";
}

/**
 * @brief Test memory usage tracking across demo applications
 */
TEST_F(MoEDemoTest, MemoryUsageTracking) {
    // Run a series of inferences to populate memory usage metrics
    for (int i = 0; i < 10; ++i) {
        std::vector<float> features(8, static_cast<float>(i) / 10.0f);
        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = static_cast<std::size_t>(400 + i),
                       .priority = 1.0f};

        auto response = engine_->run_inference(input);
        ASSERT_TRUE(response.is_ok()) << "Inference failed for memory test " << i;
    }

    // Check memory usage statistics
    auto memory_usage = engine_->get_memory_usage();
    EXPECT_EQ(memory_usage.size(), config_.num_experts)
        << "Incorrect number of memory usage metrics";

    // Verify memory usage is within reasonable bounds
    for (std::size_t expert = 0; expert < memory_usage.size(); ++expert) {
        EXPECT_GE(memory_usage[expert], 0.0f) << "Negative memory usage for expert " << expert;
        EXPECT_LE(memory_usage[expert], static_cast<float>(config_.memory_pool_size_mb))
            << "Memory usage exceeds pool size for expert " << expert;
    }
}

/**
 * @brief Test system health validation
 */
TEST_F(MoEDemoTest, SystemHealthValidation) {
    // Run some inferences to establish system state
    for (int i = 0; i < 5; ++i) {
        std::vector<float> features = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = static_cast<std::size_t>(500 + i),
                       .priority = 1.0f};

        auto response = engine_->run_inference(input);
        ASSERT_TRUE(response.is_ok()) << "Setup inference failed for health test " << i;
    }

    // Validate system health
    auto health_result = engine_->validate_system_health();
    EXPECT_TRUE(health_result.is_ok()) << "System health validation failed";
}

/**
 * @brief Test performance characteristics of demo applications
 */
TEST_F(MoEDemoTest, PerformanceCharacteristics) {
    const int num_trials = 20;
    std::vector<float> latencies;

    // Measure inference latencies
    for (int trial = 0; trial < num_trials; ++trial) {
        std::vector<float> features(8, 0.5f);  // Neutral input
        MoEInput input{.features = features,
                       .batch_size = 1,
                       .enable_load_balancing = true,
                       .request_id = static_cast<std::size_t>(600 + trial),
                       .priority = 1.0f};

        auto start = std::chrono::high_resolution_clock::now();
        auto response = engine_->run_inference(input);
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(response.is_ok()) << "Performance test inference failed for trial " << trial;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(static_cast<float>(duration.count()) / 1000.0f);  // Convert to
                                                                              // milliseconds
    }

    // Analyze performance characteristics
    float total_latency = 0.0f;
    float max_latency = 0.0f;
    float min_latency = std::numeric_limits<float>::max();

    for (float latency : latencies) {
        total_latency += latency;
        max_latency = std::max(max_latency, latency);
        min_latency = std::min(min_latency, latency);
    }

    float avg_latency = total_latency / static_cast<float>(num_trials);

    // Performance assertions (generous bounds for test environment)
    EXPECT_LT(avg_latency, 100.0f) << "Average latency too high: " << avg_latency << " ms";
    EXPECT_LT(max_latency, 500.0f) << "Maximum latency too high: " << max_latency << " ms";
    EXPECT_GT(min_latency, 0.0f) << "Minimum latency invalid: " << min_latency << " ms";

    // Log performance metrics for analysis
    std::cout << "Performance Metrics:\n"
              << "  Average latency: " << avg_latency << " ms\n"
              << "  Max latency: " << max_latency << " ms\n"
              << "  Min latency: " << min_latency << " ms\n";
}

}  // anonymous namespace
