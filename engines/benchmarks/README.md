# Unified POC Benchmarking Suite - Phase 7A Implementation

This directory contains the comprehensive benchmarking infrastructure for comparing all implemented Proof-of-Concept (POC) inference techniques in the Inference Systems Laboratory.

## 🎯 Overview

The Unified POC Benchmarking Suite provides systematic performance comparison across three advanced inference techniques:

1. **Momentum-Enhanced Belief Propagation** - ML-optimized message passing with momentum and AdaGrad
2. **Circular Belief Propagation** - Cycle-aware BP with spurious correlation cancellation  
3. **Mamba State Space Models** - Linear-time sequence modeling with selective state spaces

## 📁 File Structure

```
engines/benchmarks/
├── README.md                           # This documentation
├── unified_inference_benchmarks.cpp    # Main C++ unified benchmark suite
├── momentum_bp_benchmarks.cpp          # Individual Momentum-BP benchmarks
├── circular_bp_benchmarks.cpp          # Individual Circular-BP benchmarks
├── mamba_ssm_benchmarks.cpp           # Individual Mamba-SSM benchmarks
└── placeholder.cpp                     # Legacy placeholder
```

## 🚀 Quick Start

### Build and Run Unified Benchmarks

```bash
# Build the unified benchmark executable
cd build
cmake .. && make -j$(nproc)

# Run comprehensive unified analysis
./engines/unified_inference_benchmarks

# Run with JSON output for analysis
./engines/unified_inference_benchmarks --benchmark_format=json > unified_results.json
```

### Python Integration

```bash
# Run comprehensive analysis with Python orchestrator
python3 tools/run_unified_benchmarks.py --comprehensive

# Save performance baseline
python3 tools/run_unified_benchmarks.py --save-baseline v1_0

# Compare against baseline
python3 tools/run_unified_benchmarks.py --compare-against v1_0

# Generate HTML report
python3 tools/run_unified_benchmarks.py --output-format html
```

## 📊 Benchmark Types

### 1. Performance Comparison Benchmarks

Test all three POC techniques on standardized datasets:

- **Small Binary** (4 nodes, 4 edges) - Basic connectivity testing
- **Medium Chain** (10 nodes, 9 edges) - Linear topology scaling
- **Large Grid** (25 nodes, 40 edges) - Complex structured problems
- **Dense Small** (6 nodes, 12 edges) - High connectivity stress test
- **Sparse Large** (50 nodes, 60 edges) - Scalability validation

### 2. Convergence Analysis

Measures:
- Convergence rate (% of successful convergences)
- Iterations to convergence
- Final accuracy/residual
- Numerical stability under various conditions

### 3. Memory and Scaling Analysis

Evaluates:
- Memory usage patterns
- Scaling characteristics with problem size
- Resource efficiency across techniques

### 4. Cross-Technique Comparative Analysis

Direct head-to-head comparison on identical datasets with:
- Statistical significance testing
- Performance regression detection
- Baseline comparison capabilities

## 🔬 Technical Implementation

### Unified Dataset Generation

The `UnifiedDatasetGenerator` creates consistent test datasets across all techniques:

```cpp
// Example: Create standardized test model
auto datasets = UnifiedDatasetGenerator::get_standard_datasets();
for (const auto& dataset : datasets) {
    auto momentum_model = UnifiedDatasetGenerator::create_momentum_bp_model(dataset);
    auto circular_model = UnifiedDatasetGenerator::create_circular_bp_model(dataset);  
    auto mamba_sequence = UnifiedDatasetGenerator::create_mamba_sequence_data(dataset);
}
```

### Performance Metrics Collection

All benchmarks collect comprehensive metrics:

```cpp
struct UnifiedMetrics {
    std::string technique_name;
    std::string dataset_name;
    double inference_time_ms;      // Execution time
    double memory_usage_mb;        // Memory consumption
    double convergence_iterations; // Iterations to converge
    double final_accuracy;         // Solution quality
    bool converged;                // Successful completion
};
```

### Google Benchmark Integration

Native integration with Google Benchmark framework:

```cpp
// Automated benchmarking with statistical analysis
BENCHMARK(BM_UnifiedComparison_SmallBinary)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_UnifiedComparison_MediumChain)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_StandaloneComparativeAnalysis)
    ->Unit(benchmark::kSecond)
    ->Iterations(1)
    ->MeasureProcessCPUTime();
```

## 📈 Output and Reporting

### Console Output

Real-time performance comparison with tabulated results:

```
=== UNIFIED POC COMPARATIVE ANALYSIS REPORT ===

Dataset: small_binary
Technique               | Time (ms) | Memory (MB) | Iters | Accuracy | Converged
------------------------|-----------|-------------|-------|----------|----------
Momentum-Enhanced BP    |      2.45 |        3.2 |    15 |    0.945 |       Yes
Circular BP             |      2.98 |        4.1 |    22 |    0.952 |       Yes  
Mamba SSM              |      1.89 |        2.8 |     1 |    0.934 |       Yes
```

### JSON Reports

Machine-readable output for automated analysis:

```json
{
  "summary": {
    "total_benchmarks_run": 15,
    "techniques_tested": 3,
    "datasets_tested": 5,
    "overall_convergence_rate": 0.87
  },
  "by_technique": {
    "Momentum-Enhanced BP": {
      "avg_time_ms": 12.45,
      "convergence_rate": 0.90,
      "sample_count": 5
    }
  }
}
```

### HTML Reports

Comprehensive web-based reports with:
- Executive summary with key metrics
- Interactive performance comparisons
- Trend analysis and insights
- Exportable results tables

### Performance Plots

Automated visualization generation:
- Box plots of inference time distributions
- Memory usage analysis by dataset
- Convergence rate comparisons
- Time vs accuracy scatter plots

## 🎯 Usage Patterns

### Development Workflow Integration

```bash
# Pre-commit performance validation
python3 tools/run_unified_benchmarks.py --quick

# Feature branch baseline establishment  
python3 tools/run_unified_benchmarks.py --save-baseline feature_xyz

# Pull request performance validation
python3 tools/run_unified_benchmarks.py --compare-against main_baseline
```

### Research and Analysis

```bash
# Comprehensive research analysis
python3 tools/run_unified_benchmarks.py --comprehensive --output-format both

# Performance trend analysis
python3 tools/run_unified_benchmarks.py --compare-against baseline_v1_0 --output-format html
```

### Production Validation

```bash
# Production readiness assessment
./engines/unified_inference_benchmarks --benchmark_min_time=10s --benchmark_repetitions=10
```

## 🔧 Configuration Options

### C++ Benchmark Configuration

Modify benchmark behavior through command-line flags:

```bash
# Run benchmarks for minimum 5 seconds each
./unified_inference_benchmarks --benchmark_min_time=5s

# Generate JSON output for analysis
./unified_inference_benchmarks --benchmark_format=json

# Run specific benchmark patterns
./unified_inference_benchmarks --benchmark_filter="SmallBinary"
```

### Python Suite Configuration

Control analysis scope and output through arguments:

```bash
# Quick subset for development
python3 tools/run_unified_benchmarks.py --quick

# Specify custom build directory
python3 tools/run_unified_benchmarks.py --build-dir release_build

# Generate only HTML reports
python3 tools/run_unified_benchmarks.py --output-format html
```

## 🧪 Extending the Suite

### Adding New Datasets

1. Extend `UnifiedDatasetGenerator::get_standard_datasets()`:

```cpp
return {
    // Existing datasets...
    {"my_custom_dataset", 20, 35, 0.25, "Custom dataset description"}
};
```

2. Implement dataset-specific model generation in the generator methods.

### Adding New Techniques

1. Create benchmark functions following the established pattern:

```cpp
static UnifiedMetrics benchmark_new_technique(const TestDataset& dataset) {
    // Implement technique-specific benchmarking
}
```

2. Integrate with the `UnifiedBenchmarkSuite::run_comparative_analysis()` method.

3. Add Google Benchmark registrations for automated testing.

## 📋 Best Practices

### Performance Testing

- **Warm-up iterations**: Allow JIT compilation and cache warming
- **Statistical significance**: Run multiple iterations for reliable results  
- **Environment control**: Minimize background processes during benchmarking
- **Baseline management**: Maintain versioned performance baselines

### Analysis and Reporting

- **Consistent datasets**: Use identical test data across all techniques
- **Fair comparisons**: Account for different algorithmic characteristics
- **Trend tracking**: Monitor performance changes over time
- **Documentation**: Record benchmark conditions and environmental factors

### Integration Workflow

- **Automated validation**: Integrate benchmarks into CI/CD pipelines
- **Regression detection**: Alert on significant performance degradations
- **Baseline updates**: Regular baseline refreshes as codebase evolves
- **Cross-platform validation**: Test on representative target hardware

## 🚨 Troubleshooting

### Common Issues

1. **Missing executable**: Ensure `unified_inference_benchmarks` is built
   ```bash
   cd build && make unified_inference_benchmarks
   ```

2. **Python dependencies**: Install required packages
   ```bash
   pip install matplotlib seaborn numpy
   ```

3. **Memory errors**: Reduce dataset sizes for memory-constrained environments

4. **Benchmark timeouts**: Adjust `--benchmark_min_time` for slower systems

### Performance Debugging

- Enable debug logging in C++ benchmarks
- Use profiling tools (perf, Instruments) for detailed analysis
- Compare individual technique benchmarks with unified results
- Check system resource usage during benchmark execution

## 📚 Related Documentation

- [POCS_ROAD_MAP.md](../../docs/reports/POCS_ROAD_MAP.md) - Overall POC implementation strategy
- [CLAUDE.md](../../CLAUDE.md) - Project context and development standards
- [tools/run_benchmarks.py](../../tools/run_benchmarks.py) - Core benchmarking infrastructure
- Individual technique documentation in respective `src/` directories

## 🔮 Future Enhancements

Planned improvements for the unified benchmarking suite:

1. **Distributed benchmarking** across multiple machines
2. **GPU-accelerated** performance comparisons  
3. **Real-time monitoring** integration with production systems
4. **Automated optimization** recommendations based on benchmark results
5. **Cross-language bindings** for broader ecosystem integration

This unified benchmarking suite represents the culmination of Phase 7A implementation, providing the infrastructure needed to evaluate and optimize advanced inference techniques systematically.
