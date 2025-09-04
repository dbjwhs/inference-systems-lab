#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../load_balancer.hpp"

namespace engines::mixture_experts {

class LoadBalancerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.num_experts = 6;
        config_.load_balance_weight = 0.1f;
        config_.monitoring_window_ms = 1000;
        config_.max_queue_size_per_expert = 20;
        config_.overload_threshold = 0.8f;
        config_.enable_adaptive_routing = true;
        config_.expert_capacity_factor = 1.2f;
    }

    LoadBalancerConfig config_;
};

TEST_F(LoadBalancerTest, CreateLoadBalancerWithValidConfig) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok()) << "Should create load balancer with valid config";

    auto load_balancer = std::move(lb_result).unwrap();
    ASSERT_NE(load_balancer, nullptr);
}

TEST_F(LoadBalancerTest, CreateLoadBalancerWithInvalidConfig) {
    // Test with zero experts
    config_.num_experts = 0;

    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_err()) << "Should fail with zero experts";
}

TEST_F(LoadBalancerTest, SelectOptimalExpertBasicSelection) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    std::vector<std::size_t> candidates = {0, 1, 2};
    std::vector<float> weights = {0.4f, 0.3f, 0.3f};

    auto expert_result = load_balancer->select_optimal_expert(candidates, weights);
    ASSERT_TRUE(expert_result.is_ok()) << "Expert selection should succeed";

    auto selected_expert = std::move(expert_result).unwrap();
    EXPECT_TRUE(std::find(candidates.begin(), candidates.end(), selected_expert) !=
                candidates.end())
        << "Selected expert should be from candidates list";
}

TEST_F(LoadBalancerTest, SelectOptimalExpertWithMismatchedInputs) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    std::vector<std::size_t> candidates = {0, 1, 2};
    std::vector<float> weights = {0.5f, 0.5f};  // Only 2 weights for 3 candidates

    auto expert_result = load_balancer->select_optimal_expert(candidates, weights);
    ASSERT_TRUE(expert_result.is_err()) << "Should fail with mismatched candidate/weight sizes";
}

TEST_F(LoadBalancerTest, SelectOptimalExpertWithInvalidExpertIds) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    std::vector<std::size_t> candidates = {0, 10, 20};  // Experts 10, 20 don't exist
    std::vector<float> weights = {0.33f, 0.33f, 0.34f};

    auto expert_result = load_balancer->select_optimal_expert(candidates, weights);
    ASSERT_TRUE(expert_result.is_err()) << "Should fail with invalid expert IDs";
}

TEST_F(LoadBalancerTest, RequestRegistrationAndCompletion) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    std::size_t expert_id = 1;
    std::size_t request_id = 101;

    // Register request start
    auto start_result = load_balancer->register_request_start(expert_id, request_id);
    EXPECT_TRUE(start_result.is_ok()) << "Request start registration should succeed";

    // Simulate processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Register request completion
    auto completion_result =
        load_balancer->register_request_completion(expert_id, request_id, 10.0f);
    EXPECT_TRUE(completion_result.is_ok()) << "Request completion registration should succeed";
}

TEST_F(LoadBalancerTest, RegisterRequestWithInvalidExpertId) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    std::size_t invalid_expert_id = 100;  // Doesn't exist
    std::size_t request_id = 101;

    auto start_result = load_balancer->register_request_start(invalid_expert_id, request_id);
    EXPECT_TRUE(start_result.is_err()) << "Should fail with invalid expert ID";
}

TEST_F(LoadBalancerTest, LoadBalancingStatisticsTracking) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    // Simulate some requests on different experts
    for (int i = 0; i < 10; ++i) {
        std::size_t expert_id = static_cast<std::size_t>(i % config_.num_experts);
        std::size_t request_id = static_cast<std::size_t>(i + 100);

        load_balancer->register_request_start(expert_id, request_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        load_balancer->register_request_completion(expert_id, request_id, 5.0f);
    }

    auto stats = load_balancer->get_load_balancing_stats();
    EXPECT_EQ(stats.expert_utilization_rates.size(), config_.num_experts);
    EXPECT_EQ(stats.expert_request_counts.size(), config_.num_experts);
    EXPECT_EQ(stats.expert_average_latencies.size(), config_.num_experts);
    EXPECT_GT(stats.total_requests_processed, 0u);
    EXPECT_GE(stats.load_balance_coefficient, 0.0f);
}

TEST_F(LoadBalancerTest, ExpertLoadTracking) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    // Get initial expert loads
    auto expert_loads = load_balancer->get_expert_loads();
    EXPECT_EQ(expert_loads.size(), config_.num_experts);

    for (const auto& load : expert_loads) {
        EXPECT_LT(load.expert_id, config_.num_experts);
        EXPECT_EQ(load.active_requests, 0u);
        EXPECT_EQ(load.queued_requests, 0u);
        EXPECT_EQ(load.average_processing_time_ms, 0.0f);
        EXPECT_FALSE(load.is_overloaded);
    }
}

TEST_F(LoadBalancerTest, OverloadDetectionAndHandling) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    // Simulate high load on one expert by registering many concurrent requests
    std::size_t overloaded_expert = 0;
    std::vector<std::size_t> request_ids;

    // Register many requests without completion to simulate overload
    for (int i = 0; i < 25; ++i) {  // More than max_queue_size_per_expert=20
        std::size_t request_id = static_cast<std::size_t>(i + 200);
        request_ids.push_back(request_id);

        auto start_result = load_balancer->register_request_start(overloaded_expert, request_id);
        if (start_result.is_err()) {
            // Expected after queue fills up
            break;
        }
    }

    // Check that expert is marked as overloaded
    auto expert_loads = load_balancer->get_expert_loads();
    EXPECT_TRUE(expert_loads[overloaded_expert].is_overloaded)
        << "Expert should be marked as overloaded";

    // Clean up by completing requests
    for (auto request_id : request_ids) {
        load_balancer->register_request_completion(overloaded_expert, request_id, 10.0f);
    }
}

TEST_F(LoadBalancerTest, ExpertFailureHandling) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    std::size_t failed_expert = 2;

    // Register some requests on the expert that will fail
    for (int i = 0; i < 5; ++i) {
        std::size_t request_id = static_cast<std::size_t>(i + 300);
        load_balancer->register_request_start(failed_expert, request_id);
    }

    // Simulate expert failure
    auto failure_result = load_balancer->handle_expert_failure(failed_expert);
    EXPECT_TRUE(failure_result.is_ok()) << "Expert failure handling should succeed";

    // Verify that subsequent requests don't get routed to failed expert
    std::vector<std::size_t> candidates = {1, 2, 3};  // Include failed expert
    std::vector<float> weights = {0.33f, 0.33f, 0.34f};

    // Multiple selections should avoid the failed expert (with high probability)
    int failed_expert_selections = 0;
    for (int i = 0; i < 20; ++i) {
        auto expert_result = load_balancer->select_optimal_expert(candidates, weights);
        if (expert_result.is_ok() && std::move(expert_result).unwrap() == failed_expert) {
            failed_expert_selections++;
        }
    }

    // Failed expert should be selected much less frequently
    EXPECT_LT(failed_expert_selections, 5)
        << "Failed expert should be avoided in routing decisions";
}

TEST_F(LoadBalancerTest, AdaptiveRoutingAdjustments) {
    // Enable adaptive routing
    config_.enable_adaptive_routing = true;

    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    // Create imbalanced load by heavily using one expert
    std::size_t heavy_expert = 1;
    std::size_t light_expert = 3;

    // Generate heavy load on one expert
    for (int i = 0; i < 15; ++i) {
        std::size_t request_id = static_cast<std::size_t>(i + 400);
        load_balancer->register_request_start(heavy_expert, request_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        load_balancer->register_request_completion(heavy_expert, request_id, 20.0f);  // High
                                                                                      // latency
    }

    // Light load on another expert
    for (int i = 0; i < 3; ++i) {
        std::size_t request_id = static_cast<std::size_t>(i + 500);
        load_balancer->register_request_start(light_expert, request_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        load_balancer->register_request_completion(light_expert, request_id, 5.0f);  // Low latency
    }

    // Trigger rebalancing
    auto rebalance_result = load_balancer->trigger_expert_rebalancing();
    EXPECT_TRUE(rebalance_result.is_ok()) << "Rebalancing should succeed";

    // Test selection with equal routing weights - should prefer light expert
    std::vector<std::size_t> candidates = {heavy_expert, light_expert};
    std::vector<float> weights = {0.5f, 0.5f};  // Equal routing preference

    int light_expert_selections = 0;
    for (int i = 0; i < 20; ++i) {
        auto expert_result = load_balancer->select_optimal_expert(candidates, weights);
        if (expert_result.is_ok() && std::move(expert_result).unwrap() == light_expert) {
            light_expert_selections++;
        }
    }

    // Adaptive routing should favor the lighter expert
    EXPECT_GT(light_expert_selections, 10)
        << "Adaptive routing should prefer the less loaded expert";
}

TEST_F(LoadBalancerTest, ConcurrentLoadTracking) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    const int num_threads = 8;
    const int requests_per_thread = 10;
    std::atomic<int> successful_registrations{0};
    std::atomic<int> successful_completions{0};

    std::vector<std::thread> threads;

    // Launch concurrent threads making requests
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < requests_per_thread; ++i) {
                std::size_t expert_id = static_cast<std::size_t>((t + i) % config_.num_experts);
                std::size_t request_id = static_cast<std::size_t>(t * 1000 + i);

                auto start_result = load_balancer->register_request_start(expert_id, request_id);
                if (start_result.is_ok()) {
                    successful_registrations.fetch_add(1);

                    // Simulate some processing time
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));

                    auto completion_result =
                        load_balancer->register_request_completion(expert_id, request_id, 2.0f);
                    if (completion_result.is_ok()) {
                        successful_completions.fetch_add(1);
                    }
                }
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify thread safety - most operations should succeed
    int total_requests = num_threads * requests_per_thread;
    EXPECT_GE(successful_registrations.load(), static_cast<int>(total_requests * 0.9f))
        << "Most request registrations should succeed under concurrent load";
    EXPECT_GE(successful_completions.load(),
              static_cast<int>(successful_registrations.load() * 0.9f))
        << "Most completions should succeed for successful registrations";
}

TEST_F(LoadBalancerTest, ValidateLoadBalancingHealth) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    // Health check on newly created load balancer
    auto health_result = load_balancer->validate_load_balancing_health();
    EXPECT_TRUE(health_result.is_ok()) << "Newly created load balancer should be healthy";

    // Simulate some normal operation
    for (int i = 0; i < 10; ++i) {
        std::size_t expert_id = static_cast<std::size_t>(i % config_.num_experts);
        std::size_t request_id = static_cast<std::size_t>(i + 600);

        load_balancer->register_request_start(expert_id, request_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        load_balancer->register_request_completion(expert_id, request_id, 1.0f);
    }

    // Health check after normal operation
    health_result = load_balancer->validate_load_balancing_health();
    EXPECT_TRUE(health_result.is_ok())
        << "Load balancer should remain healthy after normal operation";
}

TEST_F(LoadBalancerTest, UtilizationVarianceTarget) {
    auto lb_result = LoadBalancer::create(config_);
    ASSERT_TRUE(lb_result.is_ok());
    auto load_balancer = std::move(lb_result).unwrap();

    // Simulate balanced load across all experts
    const int requests_per_expert = 20;
    for (std::size_t expert_id = 0; expert_id < config_.num_experts; ++expert_id) {
        for (int i = 0; i < requests_per_expert; ++i) {
            std::size_t request_id = expert_id * 1000 + static_cast<std::size_t>(i);

            load_balancer->register_request_start(expert_id, request_id);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            load_balancer->register_request_completion(expert_id, request_id, 5.0f);
        }
    }

    auto stats = load_balancer->get_load_balancing_stats();

    // Check utilization variance (roadmap target: <20%)
    EXPECT_LT(stats.overall_utilization_variance, 0.2f)
        << "Utilization variance should be under 20% target";

    // Check that all experts processed requests
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        EXPECT_GT(stats.expert_request_counts[i], 0u)
            << "Expert " << i << " should have processed some requests";
    }
}

}  // namespace engines::mixture_experts
