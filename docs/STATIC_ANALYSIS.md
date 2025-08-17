# Static Analysis Standards

This document describes the static analysis standards and tools for the Inference Systems Lab project.

## Overview

Static analysis helps detect potential bugs, code smells, and enforce coding standards before runtime. This project uses [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) to perform comprehensive static analysis on C++ code, complementing the dynamic testing and formatting tools.

## Configuration

The project's static analysis rules are defined in `.clang-tidy` at the project root. This configuration enables comprehensive checks while disabling rules that conflict with the project's coding style.

### Enabled Check Categories

- **clang-analyzer-\***: Static analysis from Clang Static Analyzer
- **bugprone-\***: Checks for bug-prone code patterns
- **cert-\***: SEI CERT C++ Coding Standard compliance
- **cppcoreguidelines-\***: C++ Core Guidelines compliance
- **google-\***: Google C++ Style Guide compliance
- **hicpp-\***: High Integrity C++ compliance
- **llvm-\***: LLVM coding standards
- **misc-\***: Miscellaneous static analysis checks
- **modernize-\***: Modern C++ feature recommendations
- **performance-\***: Performance optimization suggestions
- **portability-\***: Cross-platform portability checks
- **readability-\***: Code readability improvements
- **concurrency-\***: Thread safety and concurrency issues

### Key Configuration Features

- **Naming Conventions**: Enforces project naming standards (CamelCase classes, lower_case functions, member_ suffix)
- **Critical Errors**: Use-after-move, dangling handles, and CERT violations treated as errors
- **Header Filtering**: Analyzes only project headers, excludes system/third-party code
- **Template Parameters**: Supports modern descriptive naming (ElementType vs T)
- **Function Limits**: Reasonable thresholds for function complexity (150 lines, 100 statements)

### Disabled Checks

Certain checks are disabled to align with project preferences:
- `modernize-use-trailing-return-type`: Traditional function syntax preferred
- `readability-magic-numbers`: Allow reasonable numeric literals
- `llvm-header-guard`: Use pragma once instead
- `readability-function-cognitive-complexity`: Use simpler metrics

## Tools and Usage

### Automatic Static Analysis Script

The project includes `tools/check_static_analysis.py` for comprehensive analysis:

```bash
# Check all files
python tools/check_static_analysis.py --check

# Fix issues automatically with backup
python tools/check_static_analysis.py --fix --backup

# Check specific files
python tools/check_static_analysis.py --check --filter "common/src/*"

# Show only errors
python tools/check_static_analysis.py --check --severity error

# Generate suppression suggestions
python tools/check_static_analysis.py --generate-suppressions
```

### Manual clang-tidy Usage

For direct clang-tidy usage:

```bash
# Check single file
clang-tidy filename.cpp -- -std=c++17 -I.

# With compilation database
clang-tidy filename.cpp -p build/

# Apply fixes
clang-tidy --fix filename.cpp -p build/

# List available checks
clang-tidy --list-checks
```

## Integration with Development Workflow

### Before Committing

Include static analysis in your development workflow:

```bash
# Quick analysis check
python tools/check_static_analysis.py --check --severity error

# Fix critical issues
python tools/check_static_analysis.py --fix --backup

# Review changes
git diff

# Commit fixes separately
git add -A && git commit -m "Fix static analysis issues"
```

### Build System Integration

The static analysis tool integrates with the CMake build system:

```bash
# Generate compilation database
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Use compilation database for accurate analysis
python tools/check_static_analysis.py --check
```

### IDE Integration

#### VS Code

Install the "clang-tidy" extension and configure in `.vscode/settings.json`:

```json
{
    "clang-tidy.executable": "/opt/homebrew/Cellar/llvm/20.1.8/bin/clang-tidy",
    "clang-tidy.compilerArgs": ["-std=c++17"],
    "clang-tidy.buildPath": "build/",
    "clang-tidy.lintOnSave": true
}
```

#### CLion

1. Go to Settings → Tools → External Tools
2. Add clang-tidy with:
   - Program: `/opt/homebrew/Cellar/llvm/20.1.8/bin/clang-tidy`
   - Arguments: `$FilePath$ -p build/`
   - Working directory: `$ProjectFileDir$`

#### Vim/Neovim

With ALE (Asynchronous Lint Engine):

```vim
let g:ale_linters = {
\   'cpp': ['clang-tidy'],
\}
let g:ale_cpp_clangtidy_executable = '/opt/homebrew/Cellar/llvm/20.1.8/bin/clang-tidy'
let g:ale_cpp_clangtidy_options = '-p build/'
```

## Analysis Categories and Examples

### Bug Detection

**Use-after-move detection:**
```cpp
// Detected by bugprone-use-after-move
std::string text = "hello";
std::string moved = std::move(text);
auto length = text.length();  // WARNING: use after move
```

**Dangling pointer detection:**
```cpp
// Detected by bugprone-dangling-handle
std::string_view get_view() {
    std::string temp = "temporary";
    return temp;  // WARNING: dangling reference
}
```

### Modern C++ Recommendations

**Auto usage suggestions:**
```cpp
// Before (flagged by modernize-use-auto)
std::vector<int>::iterator it = vec.begin();

// After
auto it = vec.begin();
```

**Range-based for loops:**
```cpp
// Before (flagged by modernize-loop-convert)
for (size_t i = 0; i < container.size(); ++i) {
    process(container[i]);
}

// After
for (const auto& item : container) {
    process(item);
}
```

### Performance Issues

**Unnecessary copies:**
```cpp
// Detected by performance-unnecessary-copy-initialization
void process(const std::string& input) {
    std::string copy = input;  // WARNING: unnecessary copy
    use_readonly(copy);
}

// Fixed
void process(const std::string& input) {
    use_readonly(input);  // Direct usage
}
```

**Move semantics:**
```cpp
// Detected by performance-move-const-arg
void func(const std::string& str) {
    return std::move(str);  // WARNING: moving const object
}
```

### Readability and Style

**Naming conventions:**
```cpp
// Detected by readability-identifier-naming
class my_class {  // WARNING: should be MyClass
    int m_Value;  // WARNING: should be value_
};

// Fixed
class MyClass {
    int value_;
};
```

**Function complexity:**
```cpp
// Detected by readability-function-size
void complex_function() {
    // ... 200 lines of code ...  // WARNING: function too long
}
```

## Suppression and Configuration Management

### Inline Suppressions

For specific cases where warnings should be ignored:

```cpp
// NOLINTNEXTLINE(readability-magic-numbers)
constexpr int BUFFER_SIZE = 4096;

// NOLINT(modernize-use-trailing-return-type)
const std::string& get_name() const { return name_; }
```

### File-level Suppressions

For entire files:

```cpp
// file.hpp
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
// ... file content with many numeric constants ...
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
```

### Configuration Updates

To disable checks project-wide, update `.clang-tidy`:

```yaml
Checks: >
  ...,
  -check-to-disable,
  ...
```

### Suppression Generation

Use the automated suppression generator:

```bash
python tools/check_static_analysis.py --generate-suppressions
```

This analyzes common issues and suggests configuration changes.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Static Analysis
on: [push, pull_request]

jobs:
  static-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install clang-tidy
        run: sudo apt-get install clang-tidy
      - name: Generate compilation database
        run: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      - name: Run static analysis
        run: python tools/check_static_analysis.py --check --severity error
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/sh
# Run static analysis before commit
python tools/check_static_analysis.py --check --severity error
if [ $? -ne 0 ]; then
    echo "Static analysis errors found. Run: python tools/check_static_analysis.py --fix"
    exit 1
fi
```

## Performance Considerations

### Analysis Speed

- **Compilation database**: Significantly speeds up analysis by providing accurate compilation flags
- **Incremental analysis**: Use `--filter` to analyze only changed files
- **Parallel execution**: clang-tidy automatically uses multiple cores
- **Header filtering**: Configured to exclude system headers for faster analysis

### Resource Usage

```bash
# Monitor analysis progress
python tools/check_static_analysis.py --check

# Limit to specific severity for faster feedback
python tools/check_static_analysis.py --check --severity error

# Analyze changed files only
git diff --name-only | grep -E '\.(cpp|hpp)$' | xargs -I {} python tools/check_static_analysis.py --check --filter "{}"
```

## Troubleshooting

### Common Issues

1. **"clang-tidy not found"**
   ```bash
   # Install via Homebrew (macOS)
   brew install llvm
   
   # Or specify custom path
   python tools/check_static_analysis.py --check --clang-tidy-path /path/to/clang-tidy
   ```

2. **"No compilation database"**
   ```bash
   # Generate compilation database
   cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   ```

3. **Too many warnings**
   ```bash
   # Start with errors only
   python tools/check_static_analysis.py --check --severity error
   
   # Generate suppressions for common issues
   python tools/check_static_analysis.py --generate-suppressions
   ```

4. **False positives**
   ```bash
   # Use inline suppressions
   // NOLINTNEXTLINE(check-name)
   
   # Or update .clang-tidy configuration
   ```

### Configuration Debugging

```bash
# Verify configuration is valid
clang-tidy --list-checks

# Test specific checks
clang-tidy --checks=bugprone-* file.cpp

# Dump effective configuration
clang-tidy --dump-config
```

## Best Practices

1. **Start with errors**: Begin by fixing error-level issues before warnings
2. **Incremental adoption**: Enable check categories gradually
3. **Consistent suppressions**: Use inline suppressions sparingly and document reasons
4. **Regular updates**: Keep clang-tidy updated for new checks and improvements
5. **Team alignment**: Ensure all team members use the same configuration
6. **Documentation**: Document any project-specific suppression decisions

## Integration with Other Tools

### Formatting Integration

Static analysis and formatting work together:

```bash
# Format first, then analyze
python tools/check_format.py --fix
python tools/check_static_analysis.py --check
```

### Coverage Integration

Combine with coverage analysis:

```bash
# Build with coverage and analysis
python tools/check_coverage.py --threshold 80.0
python tools/check_static_analysis.py --check --severity error
```

### Benchmark Integration

Ensure performance changes don't introduce issues:

```bash
# Run static analysis before benchmarking
python tools/check_static_analysis.py --check
python tools/run_benchmarks.py --compare-against baseline
```

## References

- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [CERT C++ Coding Standard](https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=88046682)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- [Project DEVELOPMENT.md](../DEVELOPMENT.md) - Additional coding standards
