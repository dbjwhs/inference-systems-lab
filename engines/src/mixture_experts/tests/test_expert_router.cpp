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

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "../expert_router.hpp"

namespace engines::mixture_experts {

class ExpertRouterTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.num_experts = 8;
        config_.top_k_experts = 2;
        config_.hidden_dimension = 128;
        config_.entropy_regularization = 0.1f;
        config_.enable_gradient_computation = true;
        config_.learning_rate = 0.001f;
    }

    RouterConfig config_;
};

TEST_F(ExpertRouterTest, CreateRouterWithValidConfig) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok()) << "Should create router with valid config";

    auto router = std::move(router_result).unwrap();
    ASSERT_NE(router, nullptr);
}

TEST_F(ExpertRouterTest, CreateRouterWithInvalidConfig) {
    // Test with zero experts
    config_.num_experts = 0;

    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_err()) << "Should fail with zero experts";
}

TEST_F(ExpertRouterTest, CreateRouterWithInvalidTopK) {
    // Test with top_k > num_experts
    config_.top_k_experts = 10;  // More than num_experts=8

    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_err()) << "Should fail with top_k > num_experts";
}

TEST_F(ExpertRouterTest, SelectExpertsWithValidInput) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 1.0f);  // 256-dimensional feature vector

    auto experts_result = router->select_experts(features);
    ASSERT_TRUE(experts_result.is_ok()) << "Expert selection should succeed";

    auto selected_experts = std::move(experts_result).unwrap();
    EXPECT_EQ(selected_experts.size(), config_.top_k_experts)
        << "Should select exactly top_k experts";

    // Verify all selected experts are within valid range
    for (auto expert_id : selected_experts) {
        EXPECT_LT(expert_id, config_.num_experts) << "Expert ID should be within range";
    }

    // Verify no duplicate experts selected
    std::vector<std::size_t> sorted_experts = selected_experts;
    std::sort(sorted_experts.begin(), sorted_experts.end());
    auto unique_end = std::unique(sorted_experts.begin(), sorted_experts.end());
    EXPECT_EQ(std::distance(sorted_experts.begin(), unique_end),
              static_cast<long>(config_.top_k_experts))
        << "Should not select duplicate experts";
}

TEST_F(ExpertRouterTest, SelectExpertsWithEmptyInput) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> empty_features;  // Empty feature vector

    auto experts_result = router->select_experts(empty_features);
    ASSERT_TRUE(experts_result.is_err()) << "Should fail with empty features";
}

TEST_F(ExpertRouterTest, ComputeExpertWeightsWithValidInput) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 1.0f);

    // First select experts
    auto experts_result = router->select_experts(features);
    ASSERT_TRUE(experts_result.is_ok());
    auto selected_experts = std::move(experts_result).unwrap();

    // Then compute weights
    auto weights_result = router->compute_expert_weights(features, selected_experts);
    ASSERT_TRUE(weights_result.is_ok()) << "Weight computation should succeed";

    auto weights = std::move(weights_result).unwrap();
    EXPECT_EQ(weights.size(), selected_experts.size())
        << "Should have one weight per selected expert";

    // Verify weights are valid probabilities
    float total_weight = 0.0f;
    for (auto weight : weights) {
        EXPECT_GE(weight, 0.0f) << "Weights should be non-negative";
        EXPECT_LE(weight, 1.0f) << "Weights should not exceed 1.0";
        total_weight += weight;
    }

    // Weights should approximately sum to 1.0 (within numerical tolerance)
    EXPECT_NEAR(total_weight, 1.0f, 1e-5f) << "Weights should sum to approximately 1.0";
}

TEST_F(ExpertRouterTest, ComputeExpertWeightsWithMismatchedInput) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 1.0f);
    std::vector<std::size_t> invalid_experts = {0, 10};  // Expert 10 doesn't exist

    auto weights_result = router->compute_expert_weights(features, invalid_experts);
    ASSERT_TRUE(weights_result.is_err()) << "Should fail with invalid expert IDs";
}

TEST_F(ExpertRouterTest, RoutingConsistencyWithSameInput) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 0.5f);  // Fixed feature vector

    // Run routing multiple times with same input
    std::vector<std::vector<std::size_t>> all_selections;
    std::vector<std::vector<float>> all_weights;

    for (int i = 0; i < 5; ++i) {
        auto experts_result = router->select_experts(features);
        ASSERT_TRUE(experts_result.is_ok());
        auto selected_experts = std::move(experts_result).unwrap();

        auto weights_result = router->compute_expert_weights(features, selected_experts);
        ASSERT_TRUE(weights_result.is_ok());
        auto weights = std::move(weights_result).unwrap();

        all_selections.push_back(selected_experts);
        all_weights.push_back(weights);
    }

    // With deterministic routing (no noise), results should be consistent
    for (std::size_t i = 1; i < all_selections.size(); ++i) {
        EXPECT_EQ(all_selections[0], all_selections[i])
            << "Expert selection should be consistent for same input";

        for (std::size_t j = 0; j < all_weights[0].size(); ++j) {
            EXPECT_NEAR(all_weights[0][j], all_weights[i][j], 1e-6f)
                << "Expert weights should be consistent for same input";
        }
    }
}

TEST_F(ExpertRouterTest, RoutingDiversityWithDifferentInputs) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<std::vector<std::size_t>> all_selections;

    // Test with 10 different input patterns
    for (int pattern = 0; pattern < 10; ++pattern) {
        std::vector<float> features(256);
        for (std::size_t i = 0; i < features.size(); ++i) {
            features[i] = std::sin(static_cast<float>(pattern * i)) * 0.5f + 0.5f;
        }

        auto experts_result = router->select_experts(features);
        ASSERT_TRUE(experts_result.is_ok());
        all_selections.push_back(std::move(experts_result).unwrap());
    }

    // Count unique expert selections to verify diversity
    std::set<std::set<std::size_t>> unique_selections;
    for (const auto& selection : all_selections) {
        std::set<std::size_t> selection_set(selection.begin(), selection.end());
        unique_selections.insert(selection_set);
    }

    // Should have multiple different expert combinations
    EXPECT_GT(unique_selections.size(), 1u)
        << "Different inputs should lead to different expert selections";
}

TEST_F(ExpertRouterTest, LoadBalancingBehavior) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<std::size_t> expert_selection_counts(config_.num_experts, 0);

    // Run many routing decisions with varied inputs
    const int num_requests = 100;
    for (int i = 0; i < num_requests; ++i) {
        // Generate pseudo-random features
        std::vector<float> features(256);
        for (std::size_t j = 0; j < features.size(); ++j) {
            features[j] = std::sin(static_cast<float>(i * j)) * 0.3f +
                          std::cos(static_cast<float>(i + j)) * 0.3f + 0.5f;
        }

        auto experts_result = router->select_experts(features);
        ASSERT_TRUE(experts_result.is_ok());
        auto selected_experts = std::move(experts_result).unwrap();

        // Count selections for each expert
        for (auto expert_id : selected_experts) {
            expert_selection_counts[expert_id]++;
        }
    }

    // Calculate utilization variance
    float mean_utilization = static_cast<float>(num_requests * config_.top_k_experts) /
                             static_cast<float>(config_.num_experts);

    float variance = 0.0f;
    for (auto count : expert_selection_counts) {
        float deviation = static_cast<float>(count) - mean_utilization;
        variance += deviation * deviation;
    }
    variance /= static_cast<float>(config_.num_experts);

    float coefficient_of_variation = std::sqrt(variance) / mean_utilization;

    // Load balancing should keep utilization variance reasonable
    EXPECT_LT(coefficient_of_variation, 0.5f) << "Expert utilization should be reasonably balanced";

    // All experts should be selected at least once over many requests
    for (std::size_t i = 0; i < config_.num_experts; ++i) {
        EXPECT_GT(expert_selection_counts[i], 0u)
            << "Expert " << i << " should be selected at least once";
    }
}

TEST_F(ExpertRouterTest, ValidateRoutingHealthForHealthyRouter) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    auto health_result = router->validate_routing_health();
    EXPECT_TRUE(health_result.is_ok()) << "Newly created router should be healthy";
}

TEST_F(ExpertRouterTest, UpdateRoutingParametersWithValidGradients) {
    // Enable gradient computation for this test
    config_.enable_gradient_computation = true;
    config_.learning_rate = 0.01f;

    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 1.0f);
    std::vector<float> gradients(config_.hidden_dimension, 0.1f);

    std::vector<std::size_t> selected_experts = {0, 1};
    auto update_result = router->update_routing_parameters(features, selected_experts, 0.8f);
    EXPECT_TRUE(update_result.is_ok()) << "Parameter update should succeed with valid gradients";
}

TEST_F(ExpertRouterTest, UpdateRoutingParametersWithGradientsDisabled) {
    // Disable gradient computation
    config_.enable_gradient_computation = false;

    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 1.0f);
    std::vector<float> gradients(config_.hidden_dimension, 0.1f);

    std::vector<std::size_t> selected_experts = {0, 1};
    auto update_result = router->update_routing_parameters(features, selected_experts, 0.8f);
    EXPECT_TRUE(update_result.is_err()) << "Parameter update should fail when gradients disabled";
}

TEST_F(ExpertRouterTest, RoutingLatencyPerformance) {
    auto router_result = ExpertRouter::create(config_);
    ASSERT_TRUE(router_result.is_ok());
    auto router = std::move(router_result).unwrap();

    std::vector<float> features(256, 1.0f);

    // Measure routing latency over multiple calls
    std::vector<float> latencies;

    for (int i = 0; i < 50; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto experts_result = router->select_experts(features);
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_TRUE(experts_result.is_ok());

        float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
        latencies.push_back(latency_ms);
    }

    // Calculate median latency
    std::sort(latencies.begin(), latencies.end());
    float median_latency = latencies[latencies.size() / 2];

    // Routing should be very fast (target: < 5ms from roadmap)
    EXPECT_LT(median_latency, 5.0f)
        << "Routing latency should be under 5ms target (median: " << median_latency << "ms)";
}

}  // namespace engines::mixture_experts
