#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../moe_config.hpp"
#include "../moe_engine.hpp"

namespace engines::mixture_experts {

class MoEIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create integration test configuration
        config_.num_experts = 6;
        config_.expert_capacity = 3;
        config_.max_concurrent_requests = 50;
        config_.memory_pool_size_mb = 200;
        config_.enable_sparse_activation = true;
    }

    MoEConfig config_;
    std::unique_ptr<MoEEngine> createTestEngine() {
        auto engine_result = MoEEngine::create(config_);
        EXPECT_TRUE(engine_result.is_ok());
        return std::move(engine_result).unwrap();
    }
};

TEST_F(MoEIntegrationTest, EndToEndInferencePipeline) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    // Test complete inference pipeline
    MoEInput input;
    input.features = std::vector<float>(512, 1.0f);  // Larger feature vector
    input.request_id = 1;
    input.priority = 1.0f;

    auto response_result = engine->run_inference(input);
    ASSERT_TRUE(response_result.is_ok()) << "End-to-end inference should succeed";

    auto response = response_result.unwrap();

    // Validate complete response structure
    EXPECT_FALSE(response.outputs.empty()) << "Should produce outputs";
    EXPECT_EQ(response.selected_experts.size(), config_.expert_capacity)
        << "Should select correct number of experts";
    EXPECT_EQ(response.expert_weights.size(), config_.expert_capacity)
        << "Should have weights for all selected experts";
    EXPECT_GT(response.routing_latency_ms, 0.0f) << "Should measure routing latency";
    EXPECT_GT(response.inference_latency_ms, 0.0f) << "Should measure inference latency";
    EXPECT_GT(response.active_parameters, 0u) << "Should track active parameters";

    // Validate routing consistency
    for (auto expert_id : response.selected_experts) {
        EXPECT_LT(expert_id, config_.num_experts) << "Expert IDs should be valid";
    }

    // Validate weight normalization
    float total_weight = 0.0f;
    for (auto weight : response.expert_weights) {
        EXPECT_GE(weight, 0.0f) << "Weights should be non-negative";
        EXPECT_LE(weight, 1.0f) << "Weights should not exceed 1.0";
        total_weight += weight;
    }
    EXPECT_NEAR(total_weight, 1.0f, 1e-4f) << "Weights should sum to 1.0";
}

TEST_F(MoEIntegrationTest, MultipleRequestsConsistency) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    const int num_requests = 20;
    std::vector<MoEResponse> responses;

    // Execute multiple requests with varied inputs
    for (int i = 0; i < num_requests; ++i) {
        MoEInput input;
        input.features = std::vector<float>(512);

        // Generate varied but deterministic input patterns
        for (std::size_t j = 0; j < input.features.size(); ++j) {
            input.features[j] = std::sin(static_cast<float>(i * j)) * 0.5f + 0.5f;
        }

        input.request_id = static_cast<std::size_t>(i);
        input.priority = 1.0f;

        auto response_result = engine->run_inference(input);
        ASSERT_TRUE(response_result.is_ok()) << "Request " << i << " should succeed";
        responses.push_back(response_result.unwrap());
    }

    // Validate consistency across all responses
    for (std::size_t i = 0; i < responses.size(); ++i) {
        const auto& response = responses[i];

        EXPECT_EQ(response.selected_experts.size(), config_.expert_capacity)
            << "Request " << i << " should select correct number of experts";
        EXPECT_FALSE(response.outputs.empty()) << "Request " << i << " should produce outputs";

        // Check performance metrics are reasonable
        EXPECT_LT(response.routing_latency_ms, 50.0f)
            << "Request " << i << " routing should be fast";
        EXPECT_LT(response.inference_latency_ms, 100.0f)
            << "Request " << i << " inference should be reasonable";
    }

    // Validate expert diversity across requests
    std::set<std::size_t> all_selected_experts;
    for (const auto& response : responses) {
        for (auto expert_id : response.selected_experts) {
            all_selected_experts.insert(expert_id);
        }
    }

    // Should use multiple different experts across all requests
    EXPECT_GT(all_selected_experts.size(), 1u)
        << "Should utilize multiple different experts across requests";
}

TEST_F(MoEIntegrationTest, ConcurrentRequestProcessing) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    const int num_threads = 10;
    const int requests_per_thread = 5;
    std::atomic<int> successful_requests{0};
    std::atomic<int> failed_requests{0};

    std::vector<std::thread> threads;
    std::vector<std::vector<MoEResponse>> thread_responses(num_threads);

    // Launch concurrent threads making requests
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < requests_per_thread; ++i) {
                MoEInput input;
                input.features = std::vector<float>(512);

                // Generate thread-specific patterns
                for (std::size_t j = 0; j < input.features.size(); ++j) {
                    input.features[j] = std::cos(static_cast<float>(t * 100 + i * j)) * 0.3f + 0.7f;
                }

                input.request_id = static_cast<std::size_t>(t * 1000 + i);
                input.priority = 1.0f + static_cast<float>(t) * 0.1f;

                auto response_result = engine->run_inference(input);
                if (response_result.is_ok()) {
                    thread_responses[t].push_back(response_result.unwrap());
                    successful_requests.fetch_add(1);
                } else {
                    failed_requests.fetch_add(1);
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Validate concurrent execution results
    int total_requests = num_threads * requests_per_thread;
    EXPECT_GE(successful_requests.load(), static_cast<int>(total_requests * 0.95f))
        << "At least 95% of concurrent requests should succeed";
    EXPECT_LE(failed_requests.load(), static_cast<int>(total_requests * 0.05f))
        << "At most 5% of concurrent requests should fail";

    // Validate response quality from all threads
    for (int t = 0; t < num_threads; ++t) {
        for (const auto& response : thread_responses[t]) {
            EXPECT_EQ(response.selected_experts.size(), config_.expert_capacity);
            EXPECT_FALSE(response.outputs.empty());
            EXPECT_GT(response.routing_latency_ms, 0.0f);
            EXPECT_GT(response.inference_latency_ms, 0.0f);
        }
    }
}

TEST_F(MoEIntegrationTest, ExpertUtilizationBalancing) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    // Execute many requests to build up utilization statistics
    const int total_requests = 100;

    for (int i = 0; i < total_requests; ++i) {
        MoEInput input;
        input.features = std::vector<float>(512);

        // Generate diverse input patterns to encourage expert diversity
        float pattern_phase = static_cast<float>(i) / 10.0f;
        for (std::size_t j = 0; j < input.features.size(); ++j) {
            input.features[j] =
                std::sin(pattern_phase + static_cast<float>(j) * 0.01f) * 0.4f + 0.5f;
        }

        input.request_id = static_cast<std::size_t>(i);
        input.priority = 1.0f;

        auto response_result = engine->run_inference(input);
        ASSERT_TRUE(response_result.is_ok()) << "Request " << i << " should succeed";
    }

    // Check expert utilization distribution
    auto utilization = engine->get_expert_utilization();
    ASSERT_EQ(utilization.size(), config_.num_experts);

    // Calculate utilization statistics
    float total_utilization = 0.0f;
    float min_utilization = 1.0f;
    float max_utilization = 0.0f;

    for (auto util : utilization) {
        total_utilization += util;
        min_utilization = std::min(min_utilization, util);
        max_utilization = std::max(max_utilization, util);
    }

    // Validate balanced utilization
    EXPECT_NEAR(total_utilization, 1.0f, 0.1f) << "Total utilization should be close to 100%";

    float utilization_ratio = max_utilization / std::max(min_utilization, 0.01f);
    EXPECT_LT(utilization_ratio, 5.0f) << "Expert utilization should be reasonably balanced";

    // All experts should be used at least occasionally
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        EXPECT_GT(utilization[i], 0.001f) << "Expert " << i << " should be used occasionally";
    }
}

TEST_F(MoEIntegrationTest, MemoryUsageTracking) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    // Initial memory usage
    auto initial_memory = engine->get_memory_usage();
    ASSERT_EQ(initial_memory.size(), config_.num_experts);

    for (auto usage : initial_memory) {
        EXPECT_GE(usage, 0.0f) << "Memory usage should be non-negative";
        EXPECT_LT(usage, static_cast<float>(config_.memory_pool_size_mb))
            << "Memory usage should be within pool limits";
    }

    // Execute requests and monitor memory usage changes
    for (int i = 0; i < 20; ++i) {
        MoEInput input;
        input.features = std::vector<float>(512, static_cast<float>(i) * 0.1f);
        input.request_id = static_cast<std::size_t>(i + 100);

        auto response_result = engine->run_inference(input);
        ASSERT_TRUE(response_result.is_ok());

        // Check memory after each request
        auto current_memory = engine->get_memory_usage();
        ASSERT_EQ(current_memory.size(), config_.num_experts);

        for (auto usage : current_memory) {
            EXPECT_LT(usage, static_cast<float>(config_.memory_pool_size_mb))
                << "Memory usage should stay within limits during operation";
        }
    }

    // Final memory usage should be stable
    auto final_memory = engine->get_memory_usage();
    ASSERT_EQ(final_memory.size(), config_.num_experts);

    float total_memory = 0.0f;
    for (auto usage : final_memory) {
        total_memory += usage;
    }

    EXPECT_LT(total_memory, static_cast<float>(config_.num_experts * config_.memory_pool_size_mb))
        << "Total memory usage should be reasonable";
}

TEST_F(MoEIntegrationTest, SystemHealthMonitoring) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    // Initial health check
    auto health_result = engine->validate_system_health();
    EXPECT_TRUE(health_result.is_ok()) << "System should be healthy initially";

    // Execute some requests
    for (int i = 0; i < 30; ++i) {
        MoEInput input;
        input.features = std::vector<float>(512, static_cast<float>(i % 10) * 0.1f + 0.1f);
        input.request_id = static_cast<std::size_t>(i + 200);

        auto response_result = engine->run_inference(input);
        ASSERT_TRUE(response_result.is_ok()) << "Request " << i << " should succeed";

        // Periodic health checks during operation
        if (i % 10 == 9) {
            health_result = engine->validate_system_health();
            EXPECT_TRUE(health_result.is_ok()) << "System should remain healthy during operation";
        }
    }

    // Final health check
    health_result = engine->validate_system_health();
    EXPECT_TRUE(health_result.is_ok()) << "System should be healthy after normal operation";
}

TEST_F(MoEIntegrationTest, PerformanceTargetsValidation) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    std::vector<float> total_latencies;
    std::vector<float> routing_latencies;
    std::vector<float> inference_latencies;

    // Warm up the system
    for (int i = 0; i < 10; ++i) {
        MoEInput input;
        input.features = std::vector<float>(512, 1.0f);
        input.request_id = static_cast<std::size_t>(i);
        engine->run_inference(input);
    }

    // Measure performance across multiple requests
    const int measurement_requests = 50;
    for (int i = 0; i < measurement_requests; ++i) {
        MoEInput input;
        input.features = std::vector<float>(512);

        // Generate varied input for realistic measurements
        for (std::size_t j = 0; j < input.features.size(); ++j) {
            input.features[j] = std::sin(static_cast<float>(i + j) * 0.1f) * 0.5f + 0.5f;
        }

        input.request_id = static_cast<std::size_t>(i + 300);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto response_result = engine->run_inference(input);
        auto end_time = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(response_result.is_ok()) << "Performance measurement request should succeed";

        auto response = response_result.unwrap();
        float total_latency =
            std::chrono::duration<float, std::milli>(end_time - start_time).count();

        total_latencies.push_back(total_latency);
        routing_latencies.push_back(response.routing_latency_ms);
        inference_latencies.push_back(response.inference_latency_ms);
    }

    // Calculate performance percentiles
    auto calculate_percentile = [](std::vector<float>& values, float percentile) {
        std::sort(values.begin(), values.end());
        std::size_t index = static_cast<std::size_t>(values.size() * percentile);
        return values[std::min(index, values.size() - 1)];
    };

    float p50_total = calculate_percentile(total_latencies, 0.5f);
    float p95_total = calculate_percentile(total_latencies, 0.95f);
    float p50_routing = calculate_percentile(routing_latencies, 0.5f);

    // Validate against roadmap performance targets
    EXPECT_LT(p50_total, MoEConstants::TARGET_P50_LATENCY_MS)
        << "P50 latency should meet target: " << p50_total << "ms vs "
        << MoEConstants::TARGET_P50_LATENCY_MS << "ms target";

    EXPECT_LT(p95_total, MoEConstants::TARGET_P95_LATENCY_MS)
        << "P95 latency should meet target: " << p95_total << "ms vs "
        << MoEConstants::TARGET_P95_LATENCY_MS << "ms target";

    EXPECT_LT(p50_routing, MoEConstants::TARGET_EXPERT_SELECTION_MS)
        << "Routing latency should meet target: " << p50_routing << "ms vs "
        << MoEConstants::TARGET_EXPERT_SELECTION_MS << "ms target";
}

TEST_F(MoEIntegrationTest, ErrorRecoveryAndRobustness) {
    auto engine = createTestEngine();
    ASSERT_NE(engine, nullptr);

    // Test with various edge case inputs
    std::vector<std::vector<float>> edge_case_inputs = {
        std::vector<float>(512, 0.0f),   // All zeros
        std::vector<float>(512, 1.0f),   // All ones
        std::vector<float>(512, -1.0f),  // All negative
        std::vector<float>(512, 1e6f),   // Large values
        std::vector<float>(512, 1e-6f),  // Small values
    };

    // Add some NaN/inf patterns (should be handled gracefully)
    std::vector<float> nan_input(512, 1.0f);
    nan_input[100] = std::numeric_limits<float>::quiet_NaN();
    edge_case_inputs.push_back(nan_input);

    std::vector<float> inf_input(512, 1.0f);
    inf_input[200] = std::numeric_limits<float>::infinity();
    edge_case_inputs.push_back(inf_input);

    int successful_edge_cases = 0;
    int failed_edge_cases = 0;

    for (std::size_t i = 0; i < edge_case_inputs.size(); ++i) {
        MoEInput input;
        input.features = edge_case_inputs[i];
        input.request_id = static_cast<std::size_t>(i + 400);
        input.priority = 1.0f;

        auto response_result = engine->run_inference(input);
        if (response_result.is_ok()) {
            successful_edge_cases++;

            // Validate response quality even for edge cases
            auto response = response_result.unwrap();
            EXPECT_FALSE(response.outputs.empty())
                << "Edge case " << i << " should produce outputs";
            EXPECT_EQ(response.selected_experts.size(), config_.expert_capacity)
                << "Edge case " << i << " should select correct number of experts";
        } else {
            failed_edge_cases++;
            // Graceful failure is acceptable for truly invalid inputs (NaN, inf)
        }
    }

    // Most edge cases should be handled successfully
    EXPECT_GE(successful_edge_cases, static_cast<int>(edge_case_inputs.size() * 0.7f))
        << "At least 70% of edge cases should be handled successfully";

    // System should remain operational after edge cases
    auto health_result = engine->validate_system_health();
    EXPECT_TRUE(health_result.is_ok()) << "System should remain healthy after edge case testing";
}

}  // namespace engines::mixture_experts
