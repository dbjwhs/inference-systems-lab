#!/bin/bash

# Inference Systems Lab Setup Script
# Sets up the development environment and builds the project

set -e

# Parse command line arguments
SANITIZER_TYPE="none"
BUILD_TYPE="Release"
ENABLE_TESTS=ON

while [[ $# -gt 0 ]]; do
    case $1 in
        --sanitizer)
            SANITIZER_TYPE="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --no-tests)
            ENABLE_TESTS=OFF
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sanitizer TYPE    Enable sanitizer (none, address, thread, memory, undefined, address+undefined)"
            echo "  --debug             Build in Debug mode (default: Release)"
            echo "  --release           Build in Release mode"
            echo "  --no-tests          Disable building tests"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                  # Standard release build"
            echo "  $0 --debug                          # Debug build"
            echo "  $0 --sanitizer address              # Release build with AddressSanitizer"
            echo "  $0 --debug --sanitizer address+undefined  # Debug build with AddressSanitizer and UBSan"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🔬 Setting up Inference Systems Laboratory..."
echo "Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Sanitizer: $SANITIZER_TYPE"
echo "  Tests: $ENABLE_TESTS"
echo ""

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
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DSANITIZER_TYPE=$SANITIZER_TYPE"
    "-DENABLE_CLANG_TIDY=OFF"
)

# Add test configuration
if [[ "$ENABLE_TESTS" == "OFF" ]]; then
    CMAKE_ARGS+=("-DBUILD_TESTING=OFF")
fi

# Show configuration
echo "CMake arguments: ${CMAKE_ARGS[*]}"
echo ""

cmake .. "${CMAKE_ARGS[@]}"

# Build the project
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Build configuration:"
echo "  Type: $BUILD_TYPE"
echo "  Sanitizer: $SANITIZER_TYPE"
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
if [[ "$SANITIZER_TYPE" != "none" ]]; then
    echo "⚠️  Sanitizer Notes:"
    echo "  - Expect slower execution (~2-20x depending on sanitizer)"
    echo "  - Higher memory usage (~2-3x for AddressSanitizer)"
    echo "  - Run tests to detect memory/undefined behavior issues"
    echo "  - Set ASAN_OPTIONS=\"abort_on_error=1\" for immediate failure"
    echo ""
fi
