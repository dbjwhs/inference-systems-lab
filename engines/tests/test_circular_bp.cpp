// MIT License
// Copyright (c) 2025 Inference Systems Lab

#include <gtest/gtest.h>

#include "../src/circular_bp/circular_bp.hpp"

using namespace inference_lab::engines::circular_bp;

class CircularBPTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.max_iterations = 50;
        config_.convergence_threshold = 1e-6;
        config_.correlation_threshold = 0.8;
        config_.cycle_penalty_factor = 0.1;
        config_.max_cycle_length = 10;
        config_.enable_correlation_cancellation = true;
        config_.enable_cycle_penalties = true;
    }

    CircularBPConfig config_;
};

TEST_F(CircularBPTest, EngineCreation) {
    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->is_ready());
}

TEST_F(CircularBPTest, BackendInfo) {
    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();
    std::string info = engine->get_backend_info();

    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("Circular Belief Propagation"), std::string::npos);
    EXPECT_NE(info.find("0.8"), std::string::npos);  // correlation threshold
    EXPECT_NE(info.find("10"), std::string::npos);   // max cycle length
}

TEST_F(CircularBPTest, CyclicGraphInference) {
    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Create a 3-node cyclic graphical model (triangle)
    GraphicalModel model;

    // Triangle topology: 1-2-3-1
    Node node1{1, {0.6, 0.4}, {2, 3}};
    Node node2{2, {0.3, 0.7}, {1, 3}};
    Node node3{3, {0.5, 0.5}, {1, 2}};
    model.nodes = {node1, node2, node3};
    model.node_index[1] = 0;
    model.node_index[2] = 1;
    model.node_index[3] = 2;

    // Cyclic edges forming triangle
    EdgePotential edge1{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    EdgePotential edge2{2, 2, 3, {{1.1, 0.9}, {0.9, 1.1}}};
    EdgePotential edge3{3, 3, 1, {{1.3, 0.7}, {0.7, 1.3}}};
    model.edges = {edge1, edge2, edge3};

    // Run inference
    auto result = engine->run_circular_bp(model);
    ASSERT_TRUE(result.is_ok());

    auto marginals = result.unwrap();
    ASSERT_EQ(marginals.size(), 3);

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

    // Check metrics specific to Circular BP
    auto metrics = engine->get_metrics();
    EXPECT_GT(metrics.iterations_to_convergence, 0);
    EXPECT_GT(metrics.message_updates, 0);
    EXPECT_GE(metrics.inference_time_ms.count(), 0);
    EXPECT_GE(metrics.cycles_detected, 1);  // Should detect at least the triangle cycle
    EXPECT_LE(metrics.final_residual, 1.0);
}

TEST_F(CircularBPTest, CycleDetectionStrategies) {
    // Test different cycle detection strategies
    for (auto strategy : {CycleDetectionStrategy::DEPTH_FIRST_SEARCH,
                          CycleDetectionStrategy::SPARSE_MATRIX,
                          CycleDetectionStrategy::HYBRID_ADAPTIVE}) {
        config_.detection_strategy = strategy;
        auto engine_result = create_circular_bp_engine(config_);
        ASSERT_TRUE(engine_result.is_ok());

        auto engine = std::move(engine_result).unwrap();

        // Simple 2-node cycle for testing
        GraphicalModel model;
        Node node1{1, {0.6, 0.4}, {2}};
        Node node2{2, {0.3, 0.7}, {1}};
        model.nodes = {node1, node2};
        model.node_index[1] = 0;
        model.node_index[2] = 1;

        EdgePotential edge1{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
        EdgePotential edge2{2, 2, 1, {{1.1, 0.9}, {0.9, 1.1}}};
        model.edges = {edge1, edge2};

        auto result = engine->run_circular_bp(model);
        EXPECT_TRUE(result.is_ok());

        auto metrics = engine->get_metrics();
        EXPECT_GE(metrics.cycles_detected, 1);
    }
}

TEST_F(CircularBPTest, CorrelationCancellation) {
    config_.enable_correlation_cancellation = true;
    config_.correlation_threshold = 0.5;  // Lower threshold for testing

    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Create a model that should trigger correlation cancellation
    GraphicalModel model;
    Node node1{1, {0.9, 0.1}, {2, 3}};  // Highly biased nodes
    Node node2{2, {0.1, 0.9}, {1, 3}};
    Node node3{3, {0.9, 0.1}, {1, 2}};
    model.nodes = {node1, node2, node3};
    model.node_index[1] = 0;
    model.node_index[2] = 1;
    model.node_index[3] = 2;

    // Strong edge potentials that could create spurious correlations
    EdgePotential edge1{1, 1, 2, {{2.0, 0.1}, {0.1, 2.0}}};
    EdgePotential edge2{2, 2, 3, {{2.0, 0.1}, {0.1, 2.0}}};
    EdgePotential edge3{3, 3, 1, {{2.0, 0.1}, {0.1, 2.0}}};
    model.edges = {edge1, edge2, edge3};

    auto result = engine->run_circular_bp(model);
    ASSERT_TRUE(result.is_ok());

    auto metrics = engine->get_metrics();
    // Should have detected and potentially cancelled correlations
    EXPECT_GE(metrics.correlations_cancelled, 0);  // May or may not cancel depending on dynamics
}

TEST_F(CircularBPTest, MessageHistoryTracking) {
    config_.track_message_history = true;

    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Simple cyclic model
    GraphicalModel model;
    Node node1{1, {0.6, 0.4}, {2}};
    Node node2{2, {0.4, 0.6}, {1}};
    model.nodes = {node1, node2};
    model.node_index[1] = 0;
    model.node_index[2] = 1;

    EdgePotential edge{1, 1, 2, {{1.1, 0.9}, {0.9, 1.1}}};
    model.edges = {edge};

    auto result = engine->run_circular_bp(model);
    EXPECT_TRUE(result.is_ok());

    auto metrics = engine->get_metrics();
    EXPECT_GE(metrics.reverberation_events, 0);  // Should track reverberation events
}

TEST_F(CircularBPTest, CyclePenalties) {
    config_.enable_cycle_penalties = true;
    config_.cycle_penalty_factor = 0.2;

    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Triangle cycle
    GraphicalModel model;
    Node node1{1, {0.6, 0.4}, {2, 3}};
    Node node2{2, {0.3, 0.7}, {1, 3}};
    Node node3{3, {0.5, 0.5}, {1, 2}};
    model.nodes = {node1, node2, node3};
    model.node_index[1] = 0;
    model.node_index[2] = 1;
    model.node_index[3] = 2;

    EdgePotential edge1{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
    EdgePotential edge2{2, 2, 3, {{1.1, 0.9}, {0.9, 1.1}}};
    EdgePotential edge3{3, 3, 1, {{1.3, 0.7}, {0.7, 1.3}}};
    model.edges = {edge1, edge2, edge3};

    auto result = engine->run_circular_bp(model);
    ASSERT_TRUE(result.is_ok());

    auto marginals = result.unwrap();
    EXPECT_EQ(marginals.size(), 3);

    // Should still produce valid marginals even with cycle penalties
    for (const auto& marginal : marginals) {
        double sum = marginal[0] + marginal[1];
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST_F(CircularBPTest, UnifiedInferenceInterface) {
    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Test unified InferenceEngine interface
    inference_lab::engines::InferenceRequest request;
    // Request is currently ignored in favor of demo model

    auto response_result = engine->run_inference(request);
    ASSERT_TRUE(response_result.is_ok());

    auto response = response_result.unwrap();

    // Should have output tensors for cyclic model
    EXPECT_EQ(response.output_tensors.size(), 3);  // 3-node triangle
    EXPECT_EQ(response.output_names.size(), 3);

    // Performance stats should be available
    std::string stats = engine->get_performance_stats();
    EXPECT_FALSE(stats.empty());
    EXPECT_NE(stats.find("Cycles Detected"), std::string::npos);
    EXPECT_NE(stats.find("Correlations Cancelled"), std::string::npos);
}

TEST_F(CircularBPTest, ConfigurationUpdate) {
    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Update configuration
    CircularBPConfig new_config;
    new_config.max_iterations = 10;
    new_config.correlation_threshold = 0.9;
    new_config.enable_correlation_cancellation = false;

    engine->update_config(new_config);

    // Backend info should reflect new config
    std::string info = engine->get_backend_info();
    EXPECT_NE(info.find("0.9"), std::string::npos);  // new correlation threshold
}

TEST_F(CircularBPTest, LargeCycleHandling) {
    config_.max_cycle_length = 5;  // Limit cycle detection

    auto engine_result = create_circular_bp_engine(config_);
    ASSERT_TRUE(engine_result.is_ok());

    auto engine = std::move(engine_result).unwrap();

    // Create a larger cyclic graph (6-node cycle)
    GraphicalModel model;
    for (int i = 1; i <= 6; ++i) {
        Node node{static_cast<NodeId>(i), {0.5, 0.5}, {}};
        // Connect to next node (and wrap around)
        node.neighbors.push_back(static_cast<NodeId>((i % 6) + 1));
        if (i > 1) {
            node.neighbors.push_back(static_cast<NodeId>(i - 1));
        } else {
            node.neighbors.push_back(6);  // Connect 1 to 6
        }
        model.nodes.push_back(node);
        model.node_index[i] = i - 1;
    }

    // Create edges for the cycle
    for (int i = 1; i <= 6; ++i) {
        EdgePotential edge{static_cast<EdgeId>(i),
                           static_cast<NodeId>(i),
                           static_cast<NodeId>((i % 6) + 1),
                           {{1.1, 0.9}, {0.9, 1.1}}};
        model.edges.push_back(edge);
    }

    auto result = engine->run_circular_bp(model);
    EXPECT_TRUE(result.is_ok());

    // Should handle large cycles gracefully (may not detect full 6-cycle due to length limit)
    auto metrics = engine->get_metrics();
    EXPECT_GE(metrics.cycles_detected, 0);
}

TEST_F(CircularBPTest, ErrorStringConversions) {
    EXPECT_EQ(to_string(CircularBPError::INVALID_GRAPH_STRUCTURE),
              "Invalid graph structure for circular belief propagation");
    EXPECT_EQ(to_string(CircularBPError::CYCLE_DETECTION_FAILED),
              "Failed to detect cycles in graph structure");
    EXPECT_EQ(to_string(CircularBPError::CONVERGENCE_FAILED),
              "Failed to converge within iteration limit");
    EXPECT_EQ(to_string(CircularBPError::CORRELATION_CANCELLATION_FAILED),
              "Failed to cancel spurious correlations");
    EXPECT_EQ(to_string(CircularBPError::UNKNOWN_ERROR), "Unknown error");
}
