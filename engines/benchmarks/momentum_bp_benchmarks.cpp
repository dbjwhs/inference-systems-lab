#include <benchmark/benchmark.h>

#include "momentum_bp.hpp"

namespace inference_lab::momentum_bp {
namespace benchmark {

class MomentumBpFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& state) override {
        processor_ = std::make_unique<MomentumBp>();
        processor_->initialize();
    }

    void TearDown(const ::benchmark::State& state) override { processor_.reset(); }

  protected:
    std::unique_ptr<MomentumBp> processor_;
};

BENCHMARK_F(MomentumBpFixture, ProcessSmallInput)(::benchmark::State& state) {
    std::string input = "small test input";

    for (auto _ : state) {
        std::string result = processor_->process(input);
        ::benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(MomentumBpFixture, ProcessLargeInput)(::benchmark::State& state) {
    std::string input(1000, 'x');  // 1KB of data

    for (auto _ : state) {
        std::string result = processor_->process(input);
        ::benchmark::DoNotOptimize(result);
    }

    state.SetBytesProcessed(state.iterations() * input.size());
}

BENCHMARK_F(MomentumBpFixture, InitializationCost)(::benchmark::State& state) {
    for (auto _ : state) {
        auto processor = std::make_unique<MomentumBp>();
        ::benchmark::DoNotOptimize(processor);
        bool success = processor->initialize();
        ::benchmark::DoNotOptimize(success);
    }
}

}  // namespace benchmark
}  // namespace inference_lab::momentum_bp

BENCHMARK_MAIN();
