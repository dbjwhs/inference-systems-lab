#!/bin/bash
# Python Virtual Environment Setup for Inference Systems Lab
# 
# This script creates and configures a Python virtual environment using uv
# for fast dependency resolution and installation of Python tools.

set -e  # Exit on error

PYTHON_TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$PYTHON_TOOL_DIR/.." && pwd)"
VENV_DIR="$PYTHON_TOOL_DIR/.venv"

echo "🔧 Setting up Python virtual environment for Inference Systems Lab"
echo "Project root: $PROJECT_ROOT"
echo "Python tools: $PYTHON_TOOL_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    echo "📥 Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or: pip install uv"
    echo "   or: brew install uv"
    exit 1
fi

uv_version=$(uv --version)
echo "🚀 Using uv: $uv_version"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "📊 Using Python: $python_version"

# Create virtual environment with uv
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python python3
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install development requirements with uv
if [ -f "$PYTHON_TOOL_DIR/requirements-dev.txt" ]; then
    echo "📚 Installing development requirements with uv (from requirements-dev.txt)..."
    uv pip install -r "$PYTHON_TOOL_DIR/requirements-dev.txt"
else
    echo "⚠️  No requirements-dev.txt found, installing basic dependencies..."
    uv pip install numpy pytest matplotlib seaborn pandas psutil black mypy pylint
fi

echo ""
echo "✅ Virtual environment setup complete with uv!"
echo ""
echo "To activate the environment:"
echo "  cd python_tool"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "uv advantages:"
echo "  ⚡ 10-100x faster than pip"
echo "  🔒 Better dependency resolution"
echo "  📦 Improved caching and reproducibility"
echo ""
echo "The environment is now ready for:"
echo "  - All Python tools in python_tool/ directory"
echo "  - ML benchmarking and analysis"
echo "  - Development and testing"
