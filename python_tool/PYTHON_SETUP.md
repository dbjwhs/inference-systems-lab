# Python Development Environment Setup

This project uses Python for tooling, benchmarking, and ML operations. We use **uv** for fast virtual environment and dependency management.

## Quick Setup

### 1. Install uv (if not already installed)
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
# or
brew install uv
```

### 2. Automated Setup (Recommended)
```bash
# Run the setup script
./setup_venv.sh

# Activate the environment
source .venv/bin/activate
```

### 3. Manual Setup with uv
```bash
# Create virtual environment with uv
uv venv .venv --python python3

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies with uv (modern approach)
uv pip install -e ".[dev]"  # Uses pyproject.toml

# Or fallback to requirements file
uv pip install -r requirements-dev.txt
```

## Usage

### Daily Development
```bash
# Always activate before running Python tools
source .venv/bin/activate

# Run any Python tool
python3 tools/run_unified_benchmarks.py
python3 tools/model_manager.py --help

# Install new packages with uv (much faster!)
uv pip install new-package

# When done
deactivate
```

### Available Tools
The project includes 31+ Python tools for:
- **Benchmarking**: `run_unified_benchmarks.py`, `benchmark_inference.py`
- **ML Operations**: `model_manager.py`, `convert_model.py`, `validate_model.py`
- **Development**: `check_format.py`, `run_tests.py`, `check_coverage.py`
- **Quality Assurance**: `check_static_analysis.py`, `run_comprehensive_tests.py`

## Dependencies

### Core Dependencies
- **numpy**: Numerical computing
- **matplotlib**: Visualization for benchmarks
- **pandas**: Data analysis
- **pytest**: Testing framework

### Development Tools
- **black**: Code formatting
- **mypy**: Type checking
- **pylint**: Code analysis

### Optional ML Dependencies
- **torch**: PyTorch for ML models
- **onnx**: ONNX model format
- **onnxruntime**: ONNX model execution

## Troubleshooting

### matplotlib Warning
If you see `⚠️ Matplotlib not available, skipping visualization generation`:
```bash
source .venv/bin/activate
uv pip install matplotlib seaborn
```

### Missing Dependencies
```bash
# Install optional ML dependencies (modern approach)
source .venv/bin/activate
uv pip install -e ".[ml]"  # Installs torch, onnx, onnxruntime

# Install all optional dependencies
uv pip install -e ".[all]"

# Or install specific packages as needed
uv pip install package-name
```

### Environment Issues
```bash
# Reset virtual environment
rm -rf .venv
./setup_venv.sh
```

## Integration with Build System

The virtual environment is separate from the C++ build system but complements it:
- **C++ builds**: Use `cmake`, `make`, `ctest` as usual
- **Python tools**: Use within activated virtual environment
- **Integration**: Python tools can call C++ executables and parse their output

## Why uv + Virtual Environments?

### uv Advantages
1. **Speed**: 10-100x faster than pip for package installation
2. **Better Dependency Resolution**: More robust conflict resolution
3. **Improved Caching**: Faster subsequent installs
4. **Modern Architecture**: Written in Rust for performance

### Virtual Environment Benefits  
1. **Dependency Isolation**: Prevents conflicts with system Python packages
2. **Reproducible Builds**: Exact package versions across machines  
3. **Clean Development**: No pollution of system Python installation
4. **Team Consistency**: All developers use same dependency versions

The combination of uv + virtual environment ensures that all 31+ Python tools work consistently and install quickly across different development machines and CI/CD environments.