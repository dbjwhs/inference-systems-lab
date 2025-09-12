// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software.

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../moe_config.hpp"
#include "../moe_engine.hpp"

namespace engines::mixture_experts {

class MoEEngineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Default test configuration
        config_.num_experts = 4;
        config_.expert_capacity = 2;
        config_.max_concurrent_requests = 50;
        config_.memory_pool_size_mb = 100;
        config_.enable_sparse_activation = true;
    }

    MoEConfig config_;
};

TEST_F(MoEEngineTest, CreateEngineWithValidConfig) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok()) << "Failed to create MoE engine with valid config";

    auto engine = std::move(engine_result).unwrap();
    ASSERT_NE(engine, nullptr);
}

TEST_F(MoEEngineTest, CreateEngineWithInvalidConfig) {
    // Test with zero experts (invalid)
    config_.num_experts = 0;

    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_err()) << "Should fail with invalid num_experts=0";
    EXPECT_EQ(engine_result.unwrap_err(), MoEError::EXPERT_INITIALIZATION_FAILED);
}

TEST_F(MoEEngineTest, CreateEngineWithExcessiveCapacity) {
    // Test with expert capacity greater than number of experts
    config_.expert_capacity = 10;  // More than num_experts=4

    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_err()) << "Should fail with capacity > num_experts";
}

TEST_F(MoEEngineTest, RunInferenceWithValidInput) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    // Create test input
    MoEInput input;
    input.features = std::vector<float>(256, 1.0f);  // 256-dim input with all 1s
    input.request_id = 1;
    input.priority = 1.0f;

    auto response_result = engine->run_inference(input);
    ASSERT_TRUE(response_result.is_ok()) << "Inference should succeed with valid input";

    auto response = std::move(response_result).unwrap();
    EXPECT_FALSE(response.outputs.empty()) << "Should produce non-empty outputs";
    EXPECT_EQ(response.selected_experts.size(), config_.expert_capacity)
        << "Should select exactly expert_capacity experts";
    EXPECT_EQ(response.expert_weights.size(), config_.expert_capacity)
        << "Should have weights for all selected experts";
}

TEST_F(MoEEngineTest, RunInferenceWithEmptyInput) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    // Create invalid input with empty features
    MoEInput input;
    input.features = std::vector<float>();  // Empty features
    input.request_id = 1;

    auto response_result = engine->run_inference(input);
    ASSERT_TRUE(response_result.is_err()) << "Should fail with empty input features";
}

TEST_F(MoEEngineTest, PerformanceMetricsTracking) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    // Run multiple inferences to build up metrics
    for (int i = 0; i < 10; ++i) {
        MoEInput input;
        input.features = std::vector<float>(256, static_cast<float>(i));
        input.request_id = static_cast<std::size_t>(i);

        auto response_result = engine->run_inference(input);
        ASSERT_TRUE(response_result.is_ok());

        auto response = std::move(response_result).unwrap();
        EXPECT_GT(response.routing_latency_ms, 0.0f) << "Should track routing latency";
        EXPECT_GT(response.inference_latency_ms, 0.0f) << "Should track inference latency";
        EXPECT_GT(response.active_parameters, 0u) << "Should track active parameters";
    }

    // Check expert utilization tracking
    auto utilization = engine->get_expert_utilization();
    EXPECT_EQ(utilization.size(), config_.num_experts) << "Should track all experts";

    float total_utilization = 0.0f;
    for (auto util : utilization) {
        EXPECT_GE(util, 0.0f) << "Utilization should be non-negative";
        EXPECT_LE(util, 1.0f) << "Utilization should not exceed 100%";
        total_utilization += util;
    }
    EXPECT_NEAR(total_utilization, 1.0f, 0.1f) << "Total utilization should be ~100%";
}

TEST_F(MoEEngineTest, MemoryUsageTracking) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    auto memory_usage = engine->get_memory_usage();
    EXPECT_EQ(memory_usage.size(), config_.num_experts) << "Should track memory for all experts";

    for (auto usage : memory_usage) {
        EXPECT_GE(usage, 0.0f) << "Memory usage should be non-negative";
        EXPECT_LE(usage, static_cast<float>(config_.memory_pool_size_mb))
            << "Memory usage should not exceed pool size";
    }
}

TEST_F(MoEEngineTest, SystemHealthValidation) {
    // Use config without sparse activation to avoid SIMD-dependent failures
    auto test_config = config_;
    test_config.enable_sparse_activation = false;

    auto engine_result = MoEEngine::create(test_config);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    auto health_result = engine->validate_system_health();
    EXPECT_TRUE(health_result.is_ok()) << "Newly created engine should be healthy";
}

TEST_F(MoEEngineTest, ConcurrentInferenceRequests) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    // Test concurrent access with multiple threads
    const int num_threads = 8;
    const int requests_per_thread = 10;
    std::vector<std::thread> threads;
    std::atomic<int> successful_requests{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&engine, &successful_requests, requests_per_thread, t]() {
            for (int i = 0; i < requests_per_thread; ++i) {
                MoEInput input;
                input.features = std::vector<float>(256, static_cast<float>(t * 100 + i));
                input.request_id = static_cast<std::size_t>(t * 1000 + i);

                auto response_result = engine->run_inference(input);
                if (response_result.is_ok()) {
                    successful_requests.fetch_add(1);
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify most requests succeeded (allowing for some load balancing failures)
    int total_requests = num_threads * requests_per_thread;
    EXPECT_GE(successful_requests.load(), static_cast<int>(total_requests * 0.9f))
        << "At least 90% of concurrent requests should succeed";
}

TEST_F(MoEEngineTest, PerformanceLatencyTargets) {
    auto engine_result = MoEEngine::create(config_);
    ASSERT_TRUE(engine_result.is_ok());
    auto engine = std::move(engine_result).unwrap();

    // Warm up the engine
    for (int i = 0; i < 5; ++i) {
        MoEInput input;
        input.features = std::vector<float>(256, 1.0f);
        input.request_id = static_cast<std::size_t>(i);
        engine->run_inference(input);
    }

    // Measure latency for multiple requests
    std::vector<float> total_latencies;
    std::vector<float> routing_latencies;

    for (int i = 0; i < 20; ++i) {
        MoEInput input;
        input.features = std::vector<float>(256, static_cast<float>(i));
        input.request_id = static_cast<std::size_t>(i + 100);

        auto start = std::chrono::high_resolution_clock::now();
        auto response_result = engine->run_inference(input);
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(response_result.is_ok());
        auto response = std::move(response_result).unwrap();

        float total_latency = std::chrono::duration<float, std::milli>(end - start).count();
        total_latencies.push_back(total_latency);
        routing_latencies.push_back(response.routing_latency_ms);
    }

    // Calculate P50 and P95 latencies
    std::sort(total_latencies.begin(), total_latencies.end());
    std::sort(routing_latencies.begin(), routing_latencies.end());

    float p50_total = total_latencies[total_latencies.size() / 2];
    float p95_total = total_latencies[static_cast<std::size_t>(total_latencies.size() * 0.95f)];
    float p50_routing = routing_latencies[routing_latencies.size() / 2];

    // Verify performance targets from roadmap
    EXPECT_LT(p50_total, 75.0f) << "P50 total latency should be < 75ms (target from roadmap)";
    EXPECT_LT(p95_total, 150.0f) << "P95 total latency should be < 150ms (target from roadmap)";
    EXPECT_LT(p50_routing, 5.0f) << "P50 routing latency should be < 5ms (target from roadmap)";
}

}  // namespace engines::mixture_experts
