# Code Formatting Standards

This document describes the code formatting standards and tools for the Inference Systems Lab project.

## Overview

Consistent code formatting improves readability, reduces merge conflicts, and makes the codebase more maintainable. This project uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to automatically enforce C++ formatting standards.

## Configuration

The project's formatting rules are defined in `.clang-format` at the project root. This configuration is based on the Google C++ Style Guide with customizations for modern C++17+ development.

### Key Formatting Rules

- **Base Style**: Google C++ Style Guide
- **Indentation**: 4 spaces (no tabs)
- **Line Length**: 100 characters maximum
- **Braces**: Attached style (`{` on same line)
- **Pointer/Reference Alignment**: Left-aligned (`int* ptr`)
- **Template Declarations**: Always break long template declarations
- **Include Sorting**: Automatic with categorization
- **Modern C++**: Support for C++17+ features

### Template Parameter Naming

Following the modern C++17+ conventions established in DEVELOPMENT.md:

```cpp
// Preferred: Descriptive names with concepts
template<std::copyable ElementType, std::invocable<ElementType> TransformType>
auto transform_elements(const std::vector<ElementType>& elements, TransformType func);

// Acceptable: Traditional naming for simple cases
template<typename T>
class Result;
```

## Tools and Usage

### Automatic Formatting Script

The project includes `tools/check_format.py` for checking and fixing formatting issues:

```bash
# Check formatting (CI/CD mode)
python tools/check_format.py --check

# Fix formatting with backup
python tools/check_format.py --fix --backup

# Check specific files
python tools/check_format.py --check --filter "common/src/*"

# Show formatting differences
python tools/check_format.py --check --show-diffs
```

### Manual clang-format Usage

For direct clang-format usage:

```bash
# Check if file needs formatting
clang-format --dry-run --Werror filename.cpp

# Format file in-place
clang-format -i filename.cpp

# Show formatted output without modifying file
clang-format filename.cpp
```

## Integration with Development Workflow

### Before Committing

Always check formatting before committing:

```bash
# Quick format check
python tools/check_format.py --check

# Fix any issues
python tools/check_format.py --fix --backup

# Review changes
git diff

# Commit formatting changes
git add -A && git commit -m "Apply clang-format"
```

### Editor Integration

#### VS Code

Install the "C/C++" extension and add to `.vscode/settings.json`:

```json
{
    "C_Cpp.clang_format_style": "file",
    "C_Cpp.clang_format_fallbackStyle": "Google",
    "editor.formatOnSave": true,
    "files.associations": {
        "*.hpp": "cpp"
    }
}
```

#### CLion/IntelliJ

1. Go to Settings → Editor → Code Style → C/C++
2. Set Scheme to "Project"
3. Import clang-format configuration
4. Enable "Format on file save"

#### Vim/Neovim

Add to your config:

```vim
" Format with clang-format
autocmd FileType cpp,c nnoremap <buffer><Leader>cf :%!clang-format<CR>
autocmd FileType cpp,c vnoremap <buffer><Leader>cf :!clang-format<CR>
```

#### Emacs

```elisp
(require 'clang-format)
(global-set-key [C-M-tab] 'clang-format-region)
(add-hook 'c++-mode-hook 
          (lambda () (add-hook 'before-save-hook 'clang-format-buffer nil t)))
```

## File Types and Exclusions

### Included File Types

The formatting tools automatically process these file extensions:
- `.cpp`, `.cc`, `.cxx` - C++ source files
- `.hpp`, `.h`, `.hxx` - C++ header files
- `.c` - C source files

### Excluded Files and Directories

The following are automatically excluded from formatting:
- Build directories: `build/`, `cmake-build-*/`
- Generated files: `*.pb.h`, `*.pb.cc`, `*.capnp.h`, `*.capnp.c++`
- Third-party code: `third_party/`, `external/`, `_deps/`
- CMake artifacts: `CMakeFiles/`

### Custom Exclusions

To exclude additional files or directories:

```bash
# Exclude specific patterns
python tools/check_format.py --check --exclude "tests,examples,legacy"

# Include only specific patterns
python tools/check_format.py --check --filter "common/src/*"
```

## Formatting Examples

### Before and After

**Before formatting:**
```cpp
template<typename T,typename E>
class Result{
public:
  constexpr bool is_ok( )const noexcept{
        return std::holds_alternative<detail::ValueWrapper<T>>(data_);
    }
private:
    std::variant<detail::ValueWrapper<T>,detail::ErrorWrapper<E>>data_;
};
```

**After formatting:**
```cpp
template<typename T, typename E>
class Result {
public:
    constexpr bool is_ok() const noexcept {
        return std::holds_alternative<detail::ValueWrapper<T>>(data_);
    }

private:
    std::variant<detail::ValueWrapper<T>, detail::ErrorWrapper<E>> data_;
};
```

### Include Ordering

The configuration automatically sorts and groups includes:

```cpp
// System headers
#include <algorithm>
#include <memory>
#include <vector>

// C++ standard library
#include <iostream>
#include <string>

// Third-party libraries
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

// Project headers
#include "common/src/result.hpp"
#include "logging.hpp"
```

## Troubleshooting

### Common Issues

1. **"clang-format not found"**
   ```bash
   # Install via Homebrew (macOS)
   brew install clang-format
   
   # Or specify custom path
   python tools/check_format.py --check --clang-format-path /path/to/clang-format
   ```

2. **Too many formatting violations**
   ```bash
   # Fix all at once with backup
   python tools/check_format.py --fix --backup
   
   # Or fix incrementally by directory
   python tools/check_format.py --fix --filter "common/src/*" --backup
   ```

3. **Merge conflicts from formatting**
   ```bash
   # Reformat after resolving conflicts
   python tools/check_format.py --fix
   git add -A && git commit -m "Resolve formatting after merge"
   ```

### Configuration Debugging

To debug clang-format configuration:

```bash
# Check configuration is valid
clang-format --dump-config

# Test specific settings
clang-format --style="{BasedOnStyle: Google, IndentWidth: 4}" --dump-config
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Format Check
on: [push, pull_request]

jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install clang-format
        run: sudo apt-get install clang-format
      - name: Check formatting
        run: python tools/check_format.py --check
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/sh
# Check code formatting before commit
python tools/check_format.py --check
if [ $? -ne 0 ]; then
    echo "Code formatting issues found. Run: python tools/check_format.py --fix"
    exit 1
fi
```

## Best Practices

1. **Format early and often** - Don't let formatting violations accumulate
2. **Use editor integration** - Automatic formatting on save prevents issues
3. **Separate formatting commits** - Keep formatting changes separate from functional changes
4. **Review formatting changes** - Ensure automatic formatting doesn't break code logic
5. **Update configuration carefully** - Formatting changes affect the entire codebase

## References

- [clang-format Documentation](https://clang.llvm.org/docs/ClangFormat.html)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [C++17 Modern Features](https://en.cppreference.com/w/cpp/17)
- [Project DEVELOPMENT.md](../DEVELOPMENT.md) - Additional coding standards