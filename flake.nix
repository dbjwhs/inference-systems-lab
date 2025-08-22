{
  description = "Inference Systems Lab - C++17 ML/AI Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;  # Allow PyTorch and other ML packages
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # C++ Compiler and Build Tools
            clang_17
            cmake
            ninja
            gnumake
            
            # Debugging and Analysis
            lldb  # lldb works on macOS, gdb/valgrind don't on ARM
            
            # Code Quality Tools
            clang-tools_17  # clang-format, clang-tidy
            
            # Testing
            gtest
            gbenchmark
            
            # Coverage
            gcovr
            lcov
            
            # Dependencies
            capnproto
            
            # Python-C++ Bindings
            python3Packages.pybind11
            
            # ML Dependencies (cross-platform)
            onnxruntime
            opencv4
            
            # Python for tooling scripts and ML development
            python3
            python3Packages.pip
            python3Packages.numpy
            python3Packages.onnx
            python3Packages.opencv4  # Python OpenCV bindings
            python3Packages.torch-bin  # PyTorch with CPU support (requires allowUnfree = true)
            
            # Development utilities
            git
            ripgrep
            fd
            curl
            wget
          ] ++ lib.optionals stdenv.isLinux [
            # Linux-only tools
            gdb
            valgrind
            
            # CUDA/GPU Dependencies (Linux only) - commented out for now, enable when needed
            # cudaPackages.cudatoolkit
            # cudaPackages.cudnn
            # cudaPackages.tensorrt
          ] ++ lib.optionals stdenv.isDarwin [
            # macOS-specific tools (Metal and Accelerate are included with clang on macOS)
          ];

          shellHook = ''
            echo "🚀 Inference Systems Lab Development Environment"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Compiler: $(clang++ --version | head -n1)"
            echo "CMake:    $(cmake --version | head -n1)"
            echo "Python:   $(python3 --version)"
            
            # Show ML dependencies
            echo ""
            echo "🤖 ML Dependencies:"
            echo "ONNX Runtime: Available"
            echo "OpenCV:       Available"
            if command -v python3 >/dev/null; then
              echo "NumPy:        $(python3 -c "import numpy; print(f'v{numpy.__version__}')" 2>/dev/null || echo 'Available')"
              echo "ONNX:         $(python3 -c "import onnx; print(f'v{onnx.__version__}')" 2>/dev/null || echo 'Available')"
              echo "PyTorch:      $(python3 -c "import torch; print(f'v{torch.__version__}')" 2>/dev/null || echo 'Available')"
            fi
            
            # Platform-specific info
            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
              echo "Platform:     Linux (CUDA available when enabled)"
            else
              echo "Platform:     macOS (Metal/Accelerate available)"
            fi
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Quick commands:"
            echo "  cmake -B build -DCMAKE_BUILD_TYPE=Debug"
            echo "  cmake --build build -j"
            echo "  ctest --test-dir build"
            echo ""
            echo "🧠 ML Development:"
            echo "  python3 tools/test_ml_dependencies.py              # Test ML libraries"
            echo "  python3 tools/test_python_bindings.py              # Test C++/Python bindings"
            echo "  python3 -c \"import torch; print(f'PyTorch {torch.__version__} ready!')\""
            echo "  python3 -c \"import numpy; print('NumPy ready!')\""
            echo "  python3 -c \"import onnx; print('ONNX ready!')\""
            echo "  python3 -c \"import cv2; print('OpenCV ready!')\""
            echo ""
            
            # Set up development environment
            export INFERENCE_LAB_ROOT="$PWD"
            export PATH="$PWD/tools:$PATH"
            
            # ML environment variables
            export PYTHONPATH="$PWD:$PYTHONPATH"
            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
              export CUDA_ROOT="$(dirname $(dirname $(which nvcc)))" 2>/dev/null || true
            fi
            
            # Ensure compile_commands.json is generated for clang-tidy
            export CMAKE_EXPORT_COMPILE_COMMANDS=ON
          '';
        };
      });
}
