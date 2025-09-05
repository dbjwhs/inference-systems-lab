#include <benchmark/benchmark.h>

#include "../src/circular_bp/circular_bp.hpp"

using namespace inference_lab::engines::circular_bp;

namespace inference_lab::engines::circular_bp {
namespace benchmark {

class CircularBPFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& state) override {
        // Configure Circular-BP engine
        config_.max_iterations = 50;
        config_.convergence_threshold = 1e-6;
        config_.correlation_threshold = 0.8;
        config_.cycle_penalty_factor = 0.1;
        config_.enable_correlation_cancellation = true;
        config_.enable_cycle_penalties = true;

        auto engine_result = create_circular_bp_engine(config_);
        if (engine_result.is_err()) {
            throw std::runtime_error("Failed to create CircularBP engine in benchmark setup");
        }
        engine_ = std::move(engine_result).unwrap();

        // Create test models
        triangle_model_ = create_triangle_model();
        chain_cycle_model_ = create_chain_cycle_model(5);
    }

    void TearDown(const ::benchmark::State& state) override { engine_.reset(); }

  protected:
    CircularBPConfig config_;
    std::unique_ptr<CircularBPEngine> engine_;
    GraphicalModel triangle_model_;
    GraphicalModel chain_cycle_model_;

    GraphicalModel create_triangle_model() {
        GraphicalModel model;

        // Create 3-node triangle (simplest cycle)
        Node node1{1, {0.6, 0.4}, {2, 3}};
        Node node2{2, {0.3, 0.7}, {1, 3}};
        Node node3{3, {0.5, 0.5}, {1, 2}};
        model.nodes = {node1, node2, node3};
        model.node_index[1] = 0;
        model.node_index[2] = 1;
        model.node_index[3] = 2;

        // Create cyclic edges
        EdgePotential edge1{1, 1, 2, {{1.2, 0.8}, {0.8, 1.2}}};
        EdgePotential edge2{2, 2, 3, {{1.1, 0.9}, {0.9, 1.1}}};
        EdgePotential edge3{3, 3, 1, {{1.3, 0.7}, {0.7, 1.3}}};
        model.edges = {edge1, edge2, edge3};

        return model;
    }

    GraphicalModel create_chain_cycle_model(size_t chain_length) {
        GraphicalModel model;

        // Create chain with cycle back to start
        for (size_t i = 1; i <= chain_length; ++i) {
            std::vector<NodeId> neighbors;
            if (i > 1)
                neighbors.push_back(i - 1);
            if (i < chain_length)
                neighbors.push_back(i + 1);
            if (i == chain_length)
                neighbors.push_back(1);  // Close the cycle
            if (i == 1)
                neighbors.push_back(chain_length);  // Other direction

            Node node{i, {0.6 - 0.1 * (i % 3), 0.4 + 0.1 * (i % 3)}, neighbors};
            model.nodes.push_back(node);
            model.node_index[i] = i - 1;
        }

        // Create edges for chain + cycle closure
        EdgeId edge_id = 1;
        for (size_t i = 1; i <= chain_length; ++i) {
            NodeId next = (i == chain_length) ? 1 : i + 1;
            EdgePotential edge{edge_id++, i, next, {{1.2, 0.8}, {0.8, 1.2}}};
            model.edges.push_back(edge);
        }

        return model;
    }
};

BENCHMARK_F(CircularBPFixture, TriangleCycleInference)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_circular_bp(triangle_model_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto marginals = result.unwrap();
            ::benchmark::DoNotOptimize(marginals);
        }
    }

    state.SetItemsProcessed(state.iterations());
    auto metrics = engine_->get_metrics();
    state.counters["iterations"] = metrics.iterations_to_convergence;
    state.counters["cycles_detected"] = metrics.cycles_detected;
    state.counters["correlations_cancelled"] = metrics.correlations_cancelled;
}

BENCHMARK_F(CircularBPFixture, ChainCycleInference)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_circular_bp(chain_cycle_model_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto marginals = result.unwrap();
            ::benchmark::DoNotOptimize(marginals);
        }
    }

    state.SetItemsProcessed(state.iterations());
    auto metrics = engine_->get_metrics();
    state.counters["iterations"] = metrics.iterations_to_convergence;
    state.counters["cycles_detected"] = metrics.cycles_detected;
    state.counters["message_updates"] = metrics.message_updates;
}

BENCHMARK_F(CircularBPFixture, CycleDetectionStrategies)(::benchmark::State& state) {
    // Benchmark different cycle detection strategies
    std::vector<CycleDetectionStrategy> strategies = {CycleDetectionStrategy::DEPTH_FIRST_SEARCH,
                                                      CycleDetectionStrategy::SPARSE_MATRIX,
                                                      CycleDetectionStrategy::HYBRID_ADAPTIVE};

    size_t strategy_index = 0;

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        // Cycle through strategies
        CircularBPConfig test_config = config_;
        test_config.detection_strategy = strategies[strategy_index % strategies.size()];

        auto test_engine_result = create_circular_bp_engine(test_config);
        if (test_engine_result.is_ok()) {
            auto test_engine = std::move(test_engine_result).unwrap();
            auto result = test_engine->run_circular_bp(chain_cycle_model_);
            ::benchmark::DoNotOptimize(result);
        }

        strategy_index++;
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(CircularBPFixture, CorrelationCancellationOverhead)(::benchmark::State& state) {
    // Compare with and without correlation cancellation
    CircularBPConfig no_cancel_config = config_;
    no_cancel_config.enable_correlation_cancellation = false;

    auto no_cancel_engine_result = create_circular_bp_engine(no_cancel_config);
    if (no_cancel_engine_result.is_err()) {
        state.SkipWithError("Failed to create CircularBP engine without correlation cancellation");
        return;
    }
    auto no_cancel_engine = std::move(no_cancel_engine_result).unwrap();

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        // Benchmark with correlation cancellation
        auto with_cancel_result = engine_->run_circular_bp(triangle_model_);
        ::benchmark::DoNotOptimize(with_cancel_result);

        // Benchmark without correlation cancellation
        auto without_cancel_result = no_cancel_engine->run_circular_bp(triangle_model_);
        ::benchmark::DoNotOptimize(without_cancel_result);
    }

    state.SetItemsProcessed(state.iterations() * 2);  // Two engines per iteration

    auto with_metrics = engine_->get_metrics();
    auto without_metrics = no_cancel_engine->get_metrics();
    state.counters["with_cancellation_time"] = with_metrics.inference_time_ms.count();
    state.counters["without_cancellation_time"] = without_metrics.inference_time_ms.count();
}

BENCHMARK_F(CircularBPFixture, ScalabilityTest)(::benchmark::State& state) {
    // Test scalability with increasing cycle sizes
    size_t cycle_size = state.range(0);
    auto large_cycle_model = create_chain_cycle_model(cycle_size);

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto result = engine_->run_circular_bp(large_cycle_model);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto marginals = result.unwrap();
            ::benchmark::DoNotOptimize(marginals);
        }
    }

    state.SetItemsProcessed(state.iterations());
    state.SetComplexityN(cycle_size);

    auto metrics = engine_->get_metrics();
    state.counters["cycles_detected"] = metrics.cycles_detected;
    state.counters["inference_time_ms"] = metrics.inference_time_ms.count();
}
BENCHMARK_REGISTER_F(CircularBPFixture, ScalabilityTest)
    ->Range(3, 16)  // Test cycle sizes from 3 to 16
    ->Complexity();

BENCHMARK_F(CircularBPFixture, MessageHistoryOverhead)(::benchmark::State& state) {
    // Compare with and without message history tracking
    CircularBPConfig no_history_config = config_;
    no_history_config.track_message_history = false;

    auto no_history_engine_result = create_circular_bp_engine(no_history_config);
    auto no_history_engine = std::move(no_history_engine_result).unwrap();

    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        // Benchmark with message history
        auto with_history_result = engine_->run_circular_bp(chain_cycle_model_);
        ::benchmark::DoNotOptimize(with_history_result);

        // Benchmark without message history
        auto without_history_result = no_history_engine->run_circular_bp(chain_cycle_model_);
        ::benchmark::DoNotOptimize(without_history_result);
    }

    state.SetItemsProcessed(state.iterations() * 2);

    auto with_metrics = engine_->get_metrics();
    auto without_metrics = no_history_engine->get_metrics();
    state.counters["reverberation_events"] = with_metrics.reverberation_events;
    state.counters["without_history_reverb_events"] = without_metrics.reverberation_events;
}

BENCHMARK_F(CircularBPFixture, EngineCreationCost)(::benchmark::State& state) {
    for (auto _ : state) {  // NOLINT(clang-analyzer-deadcode.DeadStores) - benchmark loop variable
        auto engine_result = create_circular_bp_engine(config_);
        ::benchmark::DoNotOptimize(engine_result);

        if (engine_result.is_ok()) {
            auto engine = std::move(engine_result).unwrap();
            ::benchmark::DoNotOptimize(engine);
        }
    }

    state.SetItemsProcessed(state.iterations());
}

}  // namespace benchmark
}  // namespace inference_lab::engines::circular_bp
