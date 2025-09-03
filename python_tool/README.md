# Python Tools for Inference Systems Lab

This directory contains all Python tooling for the Inference Systems Lab project, isolated from the C++ codebase with its own virtual environment.

## Quick Setup

```bash
# From project root
cd python_tool
./setup_python.sh

# Activate environment (after setup)
source .venv/bin/activate
```

## Available Tools (31 total)

### 🔬 Benchmarking & Analysis
- **`run_unified_benchmarks.py`** - Unified POC benchmarking suite
- **`benchmark_inference.py`** - ML inference performance analysis  
- **`run_container_benchmarks.py`** - Container performance testing
- **`performance_profiler.py`** - System performance profiling
- **`ml_inference_load_test.py`** - Load testing for ML inference

### 🧠 ML Operations
- **`model_manager.py`** - ML model lifecycle management
- **`convert_model.py`** - Model format conversion (PyTorch→ONNX→TensorRT)
- **`validate_model.py`** - Model correctness and accuracy validation

### 🔧 Development Tools
- **`check_format.py`** - Code formatting validation/fixing
- **`check_static_analysis.py`** - Clang-tidy automation
- **`check_coverage.py`** - Test coverage verification
- **`check_documentation.py`** - Documentation completeness
- **`check_commented_code.py`** - Detect commented-out code
- **`check_eof_newline.py`** - POSIX EOF newline compliance
- **`fix_static_analysis_by_file.py`** - Incremental static analysis fixes

### 🧪 Testing Infrastructure  
- **`run_tests.py`** - Test execution orchestration
- **`run_comprehensive_tests.py`** - Complete testing pipeline
- **`test_ml_dependencies.py`** - ML framework testing
- **`test_python_bindings.py`** - Python bindings validation
- **`test_unified_benchmark_integration.py`** - Python-C++ integration tests

### 📊 Project Management
- **`new_module.py`** - Generate module scaffolding
- **`install_hooks.py`** - Pre-commit hook management
- **`coverage_tracker.py`** - Coverage tracking and visualization
- **`notification_system.py`** - Development notifications
- **`log_manager.py`** - Logging system management

## Usage Examples

```bash
# Activate environment first
source .venv/bin/activate

# Run unified benchmarks
python3 run_unified_benchmarks.py --comprehensive

# Check code formatting
python3 check_format.py --fix

# Validate ML model
python3 validate_model.py --model-path ../models/model.onnx

# Run comprehensive tests
python3 run_comprehensive_tests.py --quick

# Generate coverage report
python3 check_coverage.py --html --threshold 85.0
```

## Dependencies

### Core Dependencies
- **numpy** - Numerical computing
- **matplotlib** - Visualization and plotting  
- **seaborn** - Statistical data visualization
- **pandas** - Data analysis and manipulation
- **psutil** - System and process monitoring

### Development Tools
- **pytest** - Testing framework with plugins
- **black** - Code formatting
- **mypy** - Static type checking
- **pylint** - Code analysis and linting
- **ipython** - Enhanced interactive Python

### Optional ML Dependencies
- **torch** - PyTorch for deep learning
- **onnx** - ONNX model format support
- **onnxruntime** - ONNX model execution

## Architecture

```
python_tool/
├── setup_python.sh          # Environment setup script
├── requirements-dev.txt      # Dependency specifications
├── pyproject.toml           # Tool configurations
├── .venv/                   # Virtual environment (created by setup)
├── README.md               # This documentation
├── PYTHON_SETUP.md         # Detailed setup guide
└── *.py                    # All Python tools (31 files)
```

## Integration with C++ Project

- **Isolated Environment**: Python tools have their own dependencies and virtual environment
- **C++ Interoperability**: Python tools can execute C++ binaries and parse their output
- **Build Integration**: Python tools work alongside the CMake/C++ build system
- **Quality Assurance**: Python tools help maintain C++ code quality (formatting, analysis, testing)

## Why This Structure?

1. **Separation of Concerns**: Python tooling is isolated from C++ codebase
2. **Dependency Management**: Clean dependency isolation with virtual environment
3. **Easy Discovery**: All Python tools in one place with consistent interface
4. **Scalable**: Easy to add new tools and maintain existing ones
5. **Cross-Platform**: Works consistently across different development environments

## Contributing

When adding new Python tools:
1. Add the tool to this directory
2. Update dependencies in `requirements-dev.txt` if needed
3. Follow existing naming conventions
4. Add documentation to this README
5. Ensure the tool works within the virtual environment