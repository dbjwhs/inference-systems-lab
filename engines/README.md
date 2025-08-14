# engines Domain

This directory contains the engines components of the Inference Systems Laboratory.

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
make engines_tests engines_benchmarks
```

Run tests:
```bash
ctest -R enginesTests
```

