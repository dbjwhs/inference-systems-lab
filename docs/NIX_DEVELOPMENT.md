# Nix Development Environment

This project uses Nix for reproducible, cross-platform development environments. Nix ensures everyone has the exact same tools and dependencies, eliminating "works on my machine" issues.

## Quick Start

### 1. Install Nix

```bash
# Install Nix package manager (macOS/Linux)
curl -L https://nixos.org/nix/install | sh

# Follow the installer instructions, then restart your terminal
```

### 2. Enter Development Environment

```bash
# Enter the Nix development shell
nix develop

# You're now in a shell with all dependencies!
# Build the project
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

That's it! No Docker, no complex setup, just `nix develop` and you're ready.

## What's Included

The Nix environment provides:

- **C++ Toolchain**: Clang 17, CMake, Ninja
- **Debugging**: LLDB (macOS), GDB/Valgrind (Linux)
- **Quality Tools**: clang-format, clang-tidy
- **Testing**: GoogleTest, Google Benchmark
- **Coverage**: gcovr, lcov
- **Dependencies**: Cap'n Proto, Python 3
- **ML Libraries**: ONNX Runtime, OpenCV, NumPy, ONNX
- **Utilities**: ripgrep, fd, git, curl, wget

## Common Commands

```bash
# Enter development environment
nix develop

# Run a single command in Nix environment
nix develop -c cmake --build build

# Run tests
nix develop -c ctest --test-dir build

# Use project tools
nix develop -c python3 tools/check_coverage.py
```

## Advantages Over Docker

1. **Faster**: No virtualization overhead, native performance
2. **Lighter**: No container images, just cached packages
3. **Reproducible**: Exact versions pinned in flake.lock
4. **Cross-platform**: Same environment on macOS/Linux
5. **Composable**: Easy to add/remove dependencies

## Customizing the Environment

Edit `flake.nix` to add dependencies:

```nix
buildInputs = with pkgs; [
  # Add your packages here
  opencv
  boost
  # etc...
];
```

Then reload:
```bash
nix develop --recreate-lock-file
```

## Troubleshooting

### "experimental features" error
Enable flakes:
```bash
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

### Package not found
Search for packages:
```bash
nix search nixpkgs packagename
```

### Clean rebuild
```bash
nix develop --rebuild
```

## CI/CD Integration

GitHub Actions can use the same Nix environment:

```yaml
- uses: cachix/install-nix-action@v20
- run: nix develop -c cmake --build build
- run: nix develop -c ctest --test-dir build
```

## ML Development Ready

The environment includes ML libraries out of the box:

```bash
# Test all ML libraries at once
nix develop -c python3 test_ml_dependencies.py

# Or test individual libraries
nix develop -c python3 -c "import numpy; print('NumPy ready!')"
nix develop -c python3 -c "import onnx; print('ONNX ready!')"
nix develop -c python3 -c "import cv2; print('OpenCV ready!')"
nix develop -c python3 -c "import torch; print(f'PyTorch {torch.__version__} ready!')"

# C++ development with ONNX Runtime and OpenCV
nix develop -c cmake -B build -DCMAKE_BUILD_TYPE=Debug
nix develop -c cmake --build build -j
```

### GPU Development (Linux)

For CUDA/TensorRT development, uncomment in `flake.nix`:

```nix
# CUDA/GPU Dependencies (Linux only)
cudaPackages.cudatoolkit
cudaPackages.cudnn
cudaPackages.tensorrt
```

### Additional ML Dependencies (Optional)

```nix
# Add when needed (large downloads)
# Note: PyTorch is already included by default
python3Packages.tensorflow-bin  # TensorFlow
```

## Resources

- [Nix Pills](https://nixos.org/guides/nix-pills/) - Learn Nix concepts
- [nixpkgs](https://search.nixos.org/packages) - Search for packages
- [Nix Flakes](https://nixos.wiki/wiki/Flakes) - Flakes documentation