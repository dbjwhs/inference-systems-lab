# Developer Tools - RELOCATED

⚠️ **IMPORTANT**: The Python development tools have been moved to improve organization and dependency management.

## New Location

All Python development tools are now located in:
```
📁 python_tool/
```

## Quick Migration

```bash
# Navigate to the new directory
cd python_tool

# Set up virtual environment (one-time setup)
./setup_python.sh

# Activate virtual environment for daily use
source .venv/bin/activate

# All tools now available without path prefix:
python3 check_format.py --help
python3 model_manager.py --help
python3 run_unified_benchmarks.py --help
```

## Benefits of the New Structure

- **Virtual Environment**: Isolated Python dependencies prevent conflicts
- **Faster Package Management**: Using `uv` package manager (10-100x faster than pip)
- **Better Organization**: Clear separation of Python tools from C++ build system
- **Simplified Usage**: Tools available directly without path prefixes

## Complete Documentation

See the comprehensive documentation in:
- [`python_tool/README.md`](../python_tool/README.md) - Main documentation
- [`python_tool/PYTHON_SETUP.md`](../python_tool/PYTHON_SETUP.md) - Environment setup
- [`python_tool/DEVELOPMENT.md`](../python_tool/DEVELOPMENT.md) - Development guide

---

**Archive Date**: 2024-09-03  
**Migration**: All functionality preserved in `python_tool/` directory
