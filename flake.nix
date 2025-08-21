{
  description = "Inference Systems Lab - C++17 ML/AI Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
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
            
            # Python for tooling scripts
            python3
            python3Packages.pip
            
            # Development utilities
            git
            ripgrep
            fd
          ] ++ lib.optionals stdenv.isLinux [
            # Linux-only tools
            gdb
            valgrind
          ];

          shellHook = ''
            echo "🚀 Inference Systems Lab Development Environment"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Compiler: $(clang++ --version | head -n1)"
            echo "CMake:    $(cmake --version | head -n1)"
            echo "Python:   $(python3 --version)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Quick commands:"
            echo "  cmake -B build -DCMAKE_BUILD_TYPE=Debug"
            echo "  cmake --build build -j"
            echo "  ctest --test-dir build"
            echo ""
            
            # Set up development environment
            export INFERENCE_LAB_ROOT="$PWD"
            export PATH="$PWD/tools:$PATH"
            
            # Ensure compile_commands.json is generated for clang-tidy
            export CMAKE_EXPORT_COMPILE_COMMANDS=ON
          '';
        };
      });
}