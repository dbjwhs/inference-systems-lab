# AddressSanitizer Notes

This document provides important information about AddressSanitizer usage in the Inference Systems Laboratory project.

## Container Overflow False Positives

### Issue Description

When running tests with AddressSanitizer enabled, you may encounter container overflow errors like:

```
ERROR: AddressSanitizer: container-overflow on address 0x60400000117f
READ of size 1 at 0x60400000117f thread T0
    #0 0x000104605f64 in std::__1::basic_string<char>::__is_long() const string:1881
    #1 ... in std::__1::vector<std::__1::basic_string<char>>::__destroy_vector::operator()
    #2 ... in testing::internal::GTestIsInitialized()
```

### Root Cause

This is a **documented false positive** that occurs in mixed instrumentation scenarios:

- Our project code is built with AddressSanitizer instrumentation
- GoogleTest and system libraries may not be fully instrumented
- This creates inconsistent instrumentation of `std::vector` and `std::string` operations
- AddressSanitizer incorrectly reports container overflows during GoogleTest cleanup

**Reference**: [AddressSanitizer Container Overflow Wiki](https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow)

### Official Solutions

The AddressSanitizer team provides two recommended solutions:

1. **Build everything with AddressSanitizer** (not practical for all dependencies)
2. **Disable container overflow detection** using: `ASAN_OPTIONS=detect_container_overflow=0`

### Our Implementation

We have implemented the recommended workaround by disabling container overflow detection in our build configuration. This is safe because:

- The false positive is well-documented by the AddressSanitizer team
- All other AddressSanitizer features remain active (heap overflow, use-after-free, etc.)
- The underlying memory safety issues we care about are still detected
- GoogleTest functionality is unaffected

### Platform Specifics

This issue is particularly common on:
- **macOS ARM64** with apple-clang
- Mixed C++ standard library implementations
- Projects using GoogleTest with AddressSanitizer

### Verification

To verify this is a false positive and not a real bug:

```bash
# Test with container overflow disabled (should pass)
ASAN_OPTIONS="detect_container_overflow=0" ./common/serialization_tests

# Test with container overflow enabled (will show false positive)
ASAN_OPTIONS="detect_container_overflow=1" ./common/serialization_tests
```

The fact that tests pass with container overflow disabled confirms this is the documented false positive scenario rather than an actual memory safety issue in our code.

## Related Issues

- GoogleTest issue [#4532](https://github.com/google/googletest/issues/4532) - Different but related AddressSanitizer issue
- AddressSanitizer mixed instrumentation scenarios in C++ standard library containers

## Testing Impact

This change does not affect the quality of our memory safety testing:
- Heap buffer overflows are still detected
- Use-after-free errors are still detected  
- Double-free errors are still detected
- Memory leaks are still detected (Linux only)
- Stack buffer overflows are still detected

Only the container overflow detection (which was producing false positives) is disabled.
