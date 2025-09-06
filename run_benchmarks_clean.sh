#!/bin/bash
# Clean benchmark runner that filters macOS-specific warnings
#
# PROBLEM SOLVED:
# ✅ Identified: macOS-specific Google Benchmark warnings on Apple Silicon Macs
# ✅ Root Cause: Google Benchmark trying to use Intel-style hw.cpufrequency sysctl that doesn't exist on Apple Silicon  
# ✅ Solution: Created run_benchmarks_clean.sh wrapper that filters harmless warnings
# ✅ Impact: Benchmarks work perfectly, warnings are cosmetic only
#
# This script filters out these known harmless warnings:
# - "Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory"
# - "***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect."
# - "This does not affect benchmark measurements, only the metadata output."
#
# Usage: ./run_benchmarks_clean.sh <benchmark_executable> [args...]
# Example: ./run_benchmarks_clean.sh ./build/engines/engines_benchmarks

# Filter out known macOS warnings but preserve actual errors
"$@" 2> >(grep -v -E "(Unable to determine clock rate|Failed to set thread affinity|This does not affect benchmark measurements)" >&2)
