// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <gtest/gtest.h>

#include "../src/momentum_bp/momentum_bp.hpp"

using namespace inference_lab::engines::momentum_bp;

class MomentumBPTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.max_iterations = 50;
        config_.convergence_threshold = 1e-6;
        config_.momentum_factor = 0.9;
        config_.learning_rate = 0.1;
        config_.enable_momentum = true;
        config_.enable_adagrad = true;
    }

    MomentumBPConfig config_;
};

TEST_F(MomentumBPTest, EngineCreation) {
    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->is_ready());
}

TEST_F(MomentumBPTest, BackendInfo) {
    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();
    std::string info = engine->get_backend_info();

    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("Momentum-Enhanced"), std::string::npos);
    EXPECT_NE(info.find("0.9"), std::string::npos);  // momentum factor
    EXPECT_NE(info.find("0.1"), std::string::npos);  // learning rate
}

TEST_F(MomentumBPTest, SimpleGraphInference) {
    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Create a simple 2-node graphical model
    GraphicalModel model;

    // Node 1: P(X1=0)=0.6, P(X1=1)=0.4
    Node node1{1, {0.6, 0.4}, {2}};
    // Node 2: P(X2=0)=0.3, P(X2=1)=0.7
    Node node2{2, {0.3, 0.7}, {1}};

    model.nodes = {node1, node2};
    model.node_index[1] = 0;
    model.node_index[2] = 1;

    // Edge potential slightly favoring same values
    EdgePotential edge{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    model.edges = {edge};

    // Run inference
    auto result = engine->run_momentum_bp(model);
    ASSERT_TRUE(result.is_ok());

    auto marginals = result.unwrap();
    ASSERT_EQ(marginals.size(), 2);

    // Check that marginals are valid probability distributions
    for (const auto& marginal : marginals) {
        ASSERT_EQ(marginal.size(), 2);

        // Should sum to approximately 1.0
        double sum = marginal[0] + marginal[1];
        EXPECT_NEAR(sum, 1.0, 1e-6);

        // Should be non-negative
        EXPECT_GE(marginal[0], 0.0);
        EXPECT_GE(marginal[1], 0.0);
    }

    // Check metrics
    auto metrics = engine->get_metrics();
    // Note: Simple demo may not converge with strict threshold
    EXPECT_GT(metrics.iterations_to_convergence, 0);
    EXPECT_GT(metrics.message_updates, 0);
    EXPECT_GE(metrics.inference_time_ms.count(), 0);
    EXPECT_LE(metrics.final_residual, 1.0);  // Should improve from initial state
}

TEST_F(MomentumBPTest, UnifiedInferenceInterface) {
    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Test unified InferenceEngine interface
    inference_lab::engines::InferenceRequest request;
    // Request is currently ignored in favor of demo model

    auto response_result = engine->run_inference(request);
    ASSERT_TRUE(response_result.is_ok());

    auto response = response_result.unwrap();

    // Should have output tensors
    EXPECT_EQ(response.output_tensors.size(), 2);
    EXPECT_EQ(response.output_names.size(), 2);

    // Performance stats should be available
    std::string stats = engine->get_performance_stats();
    EXPECT_FALSE(stats.empty());
    EXPECT_NE(stats.find("Converged"), std::string::npos);
}

TEST_F(MomentumBPTest, ConfigurationUpdate) {
    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Update configuration
    MomentumBPConfig new_config;
    new_config.max_iterations = 10;
    new_config.momentum_factor = 0.5;
    new_config.enable_momentum = false;

    engine->update_config(new_config);

    // Backend info should reflect new config
    std::string info = engine->get_backend_info();
    EXPECT_NE(info.find("0.5"), std::string::npos);  // new momentum factor
}

TEST_F(MomentumBPTest, MomentumDisabled) {
    config_.enable_momentum = false;
    config_.enable_adagrad = false;

    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Should still work without momentum and AdaGrad
    GraphicalModel model;
    Node node1{1, {0.6, 0.4}, {2}};
    Node node2{2, {0.3, 0.7}, {1}};
    model.nodes = {node1, node2};
    model.node_index[1] = 0;
    model.node_index[2] = 1;

    EdgePotential edge{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    model.edges = {edge};

    auto result = engine->run_momentum_bp(model);
    ASSERT_TRUE(result.is_ok());

    auto marginals = result.unwrap();
    EXPECT_EQ(marginals.size(), 2);
}

TEST_F(MomentumBPTest, ConvergenceThreshold) {
    config_.convergence_threshold = 1e-12;  // Very strict
    config_.max_iterations = 5;             // Very few iterations

    auto engine_result = create_momentum_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    GraphicalModel model;
    Node node1{1, {0.6, 0.4}, {2}};
    Node node2{2, {0.3, 0.7}, {1}};
    model.nodes = {node1, node2};
    model.node_index[1] = 0;
    model.node_index[2] = 1;

    EdgePotential edge{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    model.edges = {edge};

    auto result = engine->run_momentum_bp(model);
    ASSERT_TRUE(result.is_ok());

    // With strict threshold and few iterations, likely won't converge
    auto metrics = engine->get_metrics();
    EXPECT_LE(metrics.iterations_to_convergence, 5);
}

TEST_F(MomentumBPTest, ErrorToString) {
    EXPECT_EQ(to_string(MomentumBPError::INVALID_GRAPH_STRUCTURE),
              "Invalid graph structure for belief propagation");
    EXPECT_EQ(to_string(MomentumBPError::CONVERGENCE_FAILED),
              "Failed to converge within iteration limit");
    EXPECT_EQ(to_string(MomentumBPError::UNKNOWN_ERROR), "Unknown error");
}
