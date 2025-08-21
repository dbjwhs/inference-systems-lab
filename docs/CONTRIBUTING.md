# Contributing to inference-systems-lab

## Coding Standards

### Language Requirements

#### C++ Code
- **C++17 or greater syntax required** - Use modern features like structured bindings, if constexpr, std::optional, std::variant
- Prefer concepts and templates for generic programming
- Use `std::span` and `std::string_view` for non-owning references
- Follow RAII principles strictly

#### Python Code
- **Strong typing required** - Use type hints for all functions
- Minimum Python 3.8 for full typing support
- Use `mypy` in strict mode for type checking
- Prefer `dataclasses` or `pydantic` for data structures

### Testing Requirements

**Every piece of code must have tests, no exceptions.**

#### Test Coverage Expectations
- Unit tests for all public APIs (minimum 80% coverage)
- Property-based tests for algorithmic components
- Integration tests for component interactions
- Performance regression tests for critical paths
- Even single-file examples need a test section

#### Test Structure
```cpp
// For each source file src/foo.cpp
// Create corresponding tests/test_foo.cpp with:
// - Normal case tests
// - Edge case tests
// - Error condition tests
// - Performance benchmarks
```

### Error Handling

#### C++ Error Handling
- Prefer `Result<T, E>` or `std::expected` (C++23) over exceptions
- Use `std::optional` for values that might not exist
- Document all error conditions in comments
- Never use raw error codes without strong typing

#### Python Error Handling
- Use type-safe error returns with `Union[Result, Error]`
- Document exceptions in docstrings
- Prefer explicit error handling over try/catch when possible

### Performance Considerations

- **Measure first, optimize second** - All optimizations need benchmarks
- Use `perf` or `vtune` data to justify complex optimizations
- Document Big-O complexity for all algorithms
- Consider cache-friendliness in data structure design
- Prefer stack allocation over heap when possible

### Code Organization

#### File Structure
- Headers: `.hpp` extension (not `.h`)
- Implementation: `.cpp` extension
- Keep headers minimal - forward declare when possible
- One class per file for major components
- Group related utilities in single files

#### Naming Conventions
- Classes: `PascalCase`
- Functions/Methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Template parameters: `PascalCase` or single capital letters
- Namespaces: `lower_snake_case`

### Documentation Requirements

- Every public API needs documentation
- Include examples in documentation
- Document performance characteristics
- Document thread safety guarantees
- Use Doxygen-style comments for C++
- Use Google-style docstrings for Python

### Build and Dependencies

- All code must build with CMake
- Specify exact version requirements for dependencies
- Prefer header-only libraries when reasonable
- Minimize external dependencies
- Document build requirements in module README

### Commit Guidelines

- Write clear commit messages explaining "why" not just "what"
- Reference issue numbers when applicable
- Keep commits focused on single changes
- Run tests before committing
- Update relevant documentation with code changes

## Development Workflow

1. **Before starting any new component:**
   - Check existing patterns in `common/`
   - Review relevant module README
   - Ensure test infrastructure is ready

2. **While coding:**
   - Write tests alongside implementation
   - Run benchmarks for performance-critical sections
   - Document design decisions

3. **Before committing:**
   - Run full test suite
   - Check performance benchmarks
   - Update documentation
   - Verify code follows these guidelines

## Module-Specific Requirements

### engines/
- Inference algorithms must be deterministic
- Support both forward and backward chaining
- Include formal correctness proofs in documentation

### distributed/
- All network communication must be encrypted
- Implement proper timeout and retry logic
- Test with simulated network failures

### performance/
- Benchmarks must be reproducible
- Include warm-up runs
- Report statistical significance

## Questions or Exceptions

If you need to deviate from these guidelines, document why in the code with a comment starting with `// DEVIATION:` explaining the reasoning.
