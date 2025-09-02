#include <benchmark/benchmark.h>

#include "../src/momentum_bp/momentum_bp.hpp"

using namespace inference_lab::engines::momentum_bp;

namespace inference_lab::engines::momentum_bp {
namespace benchmark {

class MomentumBPFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& state) override {
        // Configure Momentum-BP engine
        config_.max_iterations = 50;
        config_.convergence_threshold = 1e-6;
        config_.momentum_factor = 0.9;
        config_.learning_rate = 0.1;
        config_.enable_momentum = true;
        config_.enable_adagrad = true;

        auto engine_result = create_momentum_bp_engine(config_);
        if (engine_result.is_ok()) {
            engine_ = std::move(engine_result).unwrap();
        }

        // Create simple test graphical model
        test_model_ = create_test_model(2);  // 2-node model
    }

    void TearDown(const ::benchmark::State& state) override { engine_.reset(); }

  protected:
    MomentumBPConfig config_;
    std::unique_ptr<MomentumBPEngine> engine_;
    GraphicalModel test_model_;

  private:
    GraphicalModel create_test_model(size_t num_nodes) {
        GraphicalModel model;

        // Create nodes with random potentials
        for (size_t i = 1; i <= num_nodes; ++i) {
            Node node{i, {0.6, 0.4}, {}};
            if (i < num_nodes) {
                node.neighbors.push_back(i + 1);
            }
            if (i > 1) {
                node.neighbors.push_back(i - 1);
            }
            model.nodes.push_back(node);
            model.node_index[i] = i - 1;
        }

        // Create edges between adjacent nodes
        for (size_t i = 1; i < num_nodes; ++i) {
            EdgePotential edge{i, i, i + 1, {{1.2, 0.8}, {0.8, 1.2}}};
            model.edges.push_back(edge);
        }

        return model;
    }
};

BENCHMARK_F(MomentumBPFixture, SmallGraphInference)(::benchmark::State& state) {
    for (auto _ : state) {
        auto result = engine_->run_momentum_bp(test_model_);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto marginals = result.unwrap();
            ::benchmark::DoNotOptimize(marginals);
        }
    }

    state.SetItemsProcessed(state.iterations());
    auto metrics = engine_->get_metrics();
    state.counters["iterations"] = metrics.iterations_to_convergence;
    state.counters["message_updates"] = metrics.message_updates;
}

BENCHMARK_F(MomentumBPFixture, MediumGraphInference)(::benchmark::State& state) {
    // Create larger model for this benchmark
    auto large_model = create_test_model(5);  // 5-node chain

    for (auto _ : state) {
        auto result = engine_->run_momentum_bp(large_model);
        ::benchmark::DoNotOptimize(result);

        if (result.is_ok()) {
            auto marginals = result.unwrap();
            ::benchmark::DoNotOptimize(marginals);
        }
    }

    state.SetItemsProcessed(state.iterations());
    auto metrics = engine_->get_metrics();
    state.counters["iterations"] = metrics.iterations_to_convergence;
    state.counters["message_updates"] = metrics.message_updates;
}

BENCHMARK_F(MomentumBPFixture, MomentumVsStandardBP)(::benchmark::State& state) {
    // Compare momentum vs standard BP
    MomentumBPConfig standard_config = config_;
    standard_config.enable_momentum = false;
    standard_config.enable_adagrad = false;

    auto standard_engine_result = create_momentum_bp_engine(standard_config);
    auto standard_engine = std::move(standard_engine_result).unwrap();

    for (auto _ : state) {
        // Benchmark momentum-enhanced version
        auto momentum_result = engine_->run_momentum_bp(test_model_);
        ::benchmark::DoNotOptimize(momentum_result);

        // Benchmark standard version
        auto standard_result = standard_engine->run_momentum_bp(test_model_);
        ::benchmark::DoNotOptimize(standard_result);
    }

    state.SetItemsProcessed(state.iterations() * 2);  // Two engines per iteration

    auto momentum_metrics = engine_->get_metrics();
    auto standard_metrics = standard_engine->get_metrics();
    state.counters["momentum_iterations"] = momentum_metrics.iterations_to_convergence;
    state.counters["standard_iterations"] = standard_metrics.iterations_to_convergence;
}

BENCHMARK_F(MomentumBPFixture, EngineCreationCost)(::benchmark::State& state) {
    for (auto _ : state) {
        auto engine_result = create_momentum_bp_engine(config_);
        ::benchmark::DoNotOptimize(engine_result);

        if (engine_result.is_ok()) {
            auto engine = std::move(engine_result).unwrap();
            ::benchmark::DoNotOptimize(engine);
        }
    }

    state.SetItemsProcessed(state.iterations());
}

}  // namespace benchmark
}  // namespace inference_lab::engines::momentum_bp

BENCHMARK_MAIN();
