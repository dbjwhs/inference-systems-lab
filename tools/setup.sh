#!/bin/bash

# Inference Systems Lab Setup Script
# Sets up the development environment and builds the project

set -e

echo "🔬 Setting up Inference Systems Laboratory..."

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is required but not installed"
        exit 1
    else
        echo "✅ $1 found"
    fi
}

echo "Checking required tools..."
check_tool cmake
check_tool make
check_tool git

# Check compiler
if command -v g++ &> /dev/null; then
    echo "✅ g++ found ($(g++ --version | head -n1))"
elif command -v clang++ &> /dev/null; then
    echo "✅ clang++ found ($(clang++ --version | head -n1))"
else
    echo "❌ No C++ compiler found (g++ or clang++ required)"
    exit 1
fi

# Navigate to project root (parent of tools directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CLANG_TIDY=OFF \
    -DENABLE_SANITIZERS=OFF

# Build the project
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Available targets:"
echo "  make test          - Run all tests"
echo "  make benchmarks    - Build all benchmarks"
echo "  make format        - Format source code"
echo "  make docs          - Generate documentation"
echo ""
echo "Quick start:"
echo "  cd build"
echo "  ctest              - Run tests"
echo "  make run_experiments - Run research experiments"
echo ""