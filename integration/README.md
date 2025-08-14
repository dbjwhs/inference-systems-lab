# integration Domain

This directory contains the integration components of the Inference Systems Laboratory.

## Structure

- `src/` - Implementation code
- `tests/` - Unit and integration tests  
- `benchmarks/` - Performance measurements
- `examples/` - Demonstration programs
- `docs/` - Design notes and findings

## Getting Started

Build this domain with:
```bash
cd build
make integration_tests integration_benchmarks
```

Run tests:
```bash
ctest -R integrationTests
```

