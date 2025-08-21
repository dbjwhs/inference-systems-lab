# Pre-commit Hooks

This document describes the pre-commit hook system for the Inference Systems Lab project, which provides automated code quality checks before commits to maintain consistency and catch issues early in the development workflow.

## Overview

Pre-commit hooks are automated scripts that run before each Git commit to validate code quality, formatting, and adherence to project standards. The hooks integrate seamlessly with the existing development tooling to provide comprehensive quality assurance without disrupting the development workflow.

## Features

- **Comprehensive Quality Checks**: Integration with formatting, static analysis, and validation tools
- **Performance Optimized**: Only analyzes staged C++ files for fast feedback
- **Developer Friendly**: Clear error messages with actionable guidance and bypass options
- **Non-disruptive**: Skips checks when no C++ files are modified
- **Configurable**: Supports different severity levels and selective checking
- **Backup Support**: Automatic backup and restoration of existing hooks

## Installation

### Automatic Installation

Use the provided installation script for seamless setup:

```bash
# Install pre-commit hooks
python3 tools/install_hooks.py --install

# Check installation status
python3 tools/install_hooks.py --status

# Test hooks with current staged changes
python3 tools/install_hooks.py --test
```

### Manual Installation

If you prefer manual setup, the pre-commit hook will be installed at:
```
.git/hooks/pre-commit
```

The hook integrates with existing project tools and requires no additional configuration.

## Hook Workflow

When you run `git commit`, the pre-commit hook automatically:

1. **File Detection**: Identifies staged C++ files (`.cpp`, `.hpp`, `.cc`, `.cxx`, `.hxx`, `.h`, `.c`)
2. **Code Formatting**: Runs `clang-format` checks using project `.clang-format` configuration
3. **Static Analysis**: Performs `clang-tidy` analysis focusing on error-level issues for performance
4. **Basic Validation**: Checks for common issues (merge conflicts, debug statements, etc.)
5. **Result Reporting**: Provides clear feedback with actionable guidance

### Staged Files Only

The hooks analyze **only staged files** for optimal performance:
- Fast feedback on changes you're about to commit
- No interference with work-in-progress files
- Efficient use of development time

## Quality Checks

### Code Formatting

Uses `clang-format` with the project's `.clang-format` configuration:

```bash
# What the hook runs internally
python3 tools/check_format.py --check --filter-from-file <staged_files> --quiet
```

**Common Issues Detected:**
- Inconsistent indentation (4 spaces required)
- Line length violations (100 character limit)
- Incorrect pointer/reference alignment (left-aligned: `int* ptr`)
- Missing spaces around operators
- Inconsistent brace placement

**Resolution:**
```bash
# Fix formatting automatically
python3 tools/check_format.py --fix

# Stage the formatted changes
git add -u

# Commit again
git commit
```

### Static Analysis

Uses `clang-tidy` focusing on error-level issues for pre-commit performance:

```bash
# What the hook runs internally
python3 tools/check_static_analysis.py --check --severity error --filter-from-file <staged_files> --quiet
```

**Critical Issues Detected:**
- Use-after-move violations
- Dangling pointer/reference issues
- CERT security standard violations
- Clang static analyzer findings

**Resolution:**
```bash
# Fix issues automatically
python3 tools/check_static_analysis.py --fix --backup

# Review changes
git diff

# Stage the fixes
git add -u

# Commit again
git commit
```

### Basic Validation

Performs additional checks for common development issues:

**Debug Statements**: Warns about debug output (`std::cout`, `printf`, `std::cerr`)
```cpp
// Detected by hook
std::cout << "Debug: " << value << std::endl;  // Consider removing
```

**TODO/FIXME Tracking**: Suggests linking to issue numbers
```cpp
// Better practice
// TODO(#123): Implement error handling for edge case
// FIXME(#456): Performance bottleneck in loop
```

**Merge Conflict Markers**: Prevents committing unresolved conflicts
```cpp
// Blocked by hook
<<<<<<< HEAD
old_code();
=======
new_code();
>>>>>>> branch
```

**Trailing Whitespace**: Warns about whitespace issues (handled by Git)

## Usage Examples

### Normal Workflow

```bash
# Make your changes
vim common/src/result.hpp

# Stage changes
git add common/src/result.hpp

# Commit (hooks run automatically)
git commit -m "Improve error handling in Result type"

# If hooks pass: commit succeeds
# If hooks fail: fix issues and try again
```

### Hook Failure Handling

```bash
# Hook detects formatting issues
git commit -m "Update feature"
# Error: Code formatting check failed
# To fix formatting issues automatically, run:
#   python3 tools/check_format.py --fix

# Fix the issues
python3 tools/check_format.py --fix

# Stage the fixes
git add -u

# Commit again
git commit -m "Update feature"
# Success: All pre-commit checks passed!
```

### Emergency Bypass

For urgent commits when quality checks must be bypassed:

```bash
# Bypass all pre-commit checks (use sparingly)
git commit --no-verify -m "Emergency fix for production issue"
```

**Note**: Bypassed commits should be followed up with quality fixes in subsequent commits.

## Hook Management

### Installation Commands

```bash
# Install hooks (fails if hook already exists)
python3 tools/install_hooks.py --install

# Force reinstall (overwrites existing hook)
python3 tools/install_hooks.py --install --force

# Check current status
python3 tools/install_hooks.py --status

# Test hooks without committing
python3 tools/install_hooks.py --test
```

### Uninstallation

```bash
# Remove pre-commit hooks
python3 tools/install_hooks.py --uninstall

# This will:
# 1. Remove the current hook
# 2. Restore any backup hook if it existed
```

### Hook Status Information

```bash
$ python3 tools/install_hooks.py --status
Pre-commit Hook Status:
========================================
📁 Project root: /path/to/project
📁 Hooks directory: /path/to/project/.git/hooks
✅ Pre-commit hook: /path/to/project/.git/hooks/pre-commit
✅ Hook is executable

Required Tools:
  ✅ check_format.py
  ✅ check_static_analysis.py
```

## Integration with Development Tools

### IDE Integration

The pre-commit hooks complement IDE-based quality checking:

**VS Code**: Install format-on-save and clang-tidy extensions
**CLion**: Enable built-in formatting and inspection tools
**Vim/Neovim**: Configure ALE or similar linting plugins

This provides immediate feedback during development, with pre-commit hooks as the final quality gate.

### CI/CD Integration

Pre-commit hooks work alongside CI/CD pipelines:

```yaml
# .github/workflows/quality.yml
- name: Install hooks and test
  run: |
    python3 tools/install_hooks.py --install
    python3 tools/install_hooks.py --test
```

The same quality standards are enforced locally and in CI/CD.

### Team Workflow

**New Team Members**:
```bash
# After cloning the repository
python3 tools/install_hooks.py --install
```

**Existing Projects**:
```bash
# Update to latest hook version
python3 tools/install_hooks.py --install --force
```

## Performance Considerations

### Optimization Strategies

1. **Staged Files Only**: Hooks analyze only files being committed
2. **Error-Level Analysis**: Static analysis focuses on critical issues for speed
3. **Incremental Checking**: No full project analysis during commits
4. **Tool Efficiency**: Uses existing compiled tools (clang-format, clang-tidy)

### Typical Performance

```bash
# Small changes (1-3 files): < 5 seconds
# Medium changes (4-10 files): 5-15 seconds  
# Large changes (10+ files): 15-30 seconds
```

### Performance Tuning

For very large commits, consider:

```bash
# Split large commits into smaller logical units
git add specific_files
git commit -m "Part 1: Core functionality"

git add more_files  
git commit -m "Part 2: Tests and documentation"
```

## Troubleshooting

### Common Issues

**1. Hook Not Running**
```bash
# Check if hook is installed and executable
python3 tools/install_hooks.py --status

# Reinstall if needed
python3 tools/install_hooks.py --install --force
```

**2. Tool Not Found Errors**
```bash
# Verify all tools are available
ls -la tools/check_format.py tools/check_static_analysis.py

# Check PATH and Python environment
which python3
python3 --version
```

**3. Performance Issues**
```bash
# Test hook performance
time python3 tools/install_hooks.py --test

# Check individual tool performance
time python3 tools/check_format.py --check --filter "specific_file.cpp"
time python3 tools/check_static_analysis.py --check --severity error --filter "specific_file.cpp"
```

**4. False Positives**
```bash
# For persistent formatting issues
python3 tools/check_format.py --fix

# For static analysis false positives, add suppressions
// NOLINTNEXTLINE(check-name)
problematic_code();
```

### Hook Debugging

Enable verbose output for debugging:

```bash
# Modify the hook script temporarily for debugging
# Add 'set -x' after 'set -e' in .git/hooks/pre-commit

# Or run tools manually with verbose output
python3 tools/check_format.py --check --filter "problem_file.cpp"
python3 tools/check_static_analysis.py --check --filter "problem_file.cpp"
```

### Recovery Procedures

**Corrupted Hook Installation**:
```bash
# Remove and reinstall
python3 tools/install_hooks.py --uninstall
python3 tools/install_hooks.py --install
```

**Emergency Commit with Failed Hooks**:
```bash
# Bypass hooks for urgent fixes
git commit --no-verify -m "Emergency: Critical production fix"

# Follow up with quality fixes
python3 tools/check_format.py --fix
python3 tools/check_static_analysis.py --fix
git add -u
git commit -m "Code quality fixes following emergency commit"
```

## Best Practices

### Development Workflow

1. **Install hooks early**: Set up hooks immediately after cloning
2. **Regular updates**: Keep hooks updated with latest tool versions
3. **Small commits**: Make focused commits for faster hook execution
4. **Fix issues promptly**: Address quality issues as they're detected

### Quality Standards

1. **Format first**: Run formatting before static analysis for cleaner results
2. **Address errors**: Focus on error-level issues before warnings
3. **Consistent style**: Follow project formatting and naming conventions
4. **Document suppressions**: Explain any static analysis suppressions

### Team Collaboration

1. **Shared standards**: Ensure all team members use the same hook configuration
2. **Tool versions**: Use consistent tool versions across the team
3. **Issue tracking**: Link TODOs and FIXMEs to issue tracking system
4. **Code reviews**: Use hooks as first line of defense, not replacement for reviews

## Configuration

### Hook Customization

The hook behavior can be modified by editing the installation script:

```python
# In tools/install_hooks.py, modify create_pre_commit_hook()

# Example: Change static analysis severity
if ! check_static_analysis "${staged_files[@]}"; then
    # Change --severity error to --severity warning for more checks
```

### Tool Configuration

Individual tools have their own configuration:

- **Formatting**: `.clang-format` in project root
- **Static Analysis**: `.clang-tidy` in project root
- **Coverage**: Settings in CMakeLists.txt

### Environment Variables

```bash
# Skip specific checks (for debugging)
export SKIP_FORMAT_CHECK=1
export SKIP_STATIC_ANALYSIS=1

# Custom tool paths
export CLANG_FORMAT_PATH=/custom/path/clang-format
export CLANG_TIDY_PATH=/custom/path/clang-tidy
```

## Integration with Other Tools

### Compatibility

**Git Flow**: Works with all Git workflows (feature branches, merge requests, etc.)
**GitHub**: Integrates with GitHub Actions and pull request workflows
**GitLab**: Compatible with GitLab CI/CD pipelines
**Bitbucket**: Works with Bitbucket Pipelines

### Complementary Tools

**Pre-push hooks**: Can be added for additional remote validation
**Commit message hooks**: Can validate commit message formatting
**IDE plugins**: Provide real-time feedback during development
**Documentation tools**: Can validate documentation updates

## Future Enhancements

### Planned Features

1. **Coverage Integration**: Optional coverage checks for new code
2. **Benchmark Validation**: Performance regression detection for critical paths
3. **Documentation Checks**: Validation of API documentation completeness
4. **License Header Validation**: Ensure proper license headers in source files

### Configuration Improvements

1. **Per-directory rules**: Different quality standards for different modules
2. **File-type specific checks**: Specialized validation for different file types
3. **Incremental analysis**: Even faster checks for large codebases
4. **Custom check plugins**: Framework for project-specific quality checks

## References

- [Git Hooks Documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
- [clang-format Documentation](https://clang.llvm.org/docs/ClangFormat.html)
- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [Project DEVELOPMENT.md](DEVELOPMENT.md) - Coding standards and guidelines
- [Project FORMATTING.md](FORMATTING.md) - Code formatting standards
- [Project STATIC_ANALYSIS.md](STATIC_ANALYSIS.md) - Static analysis configuration
