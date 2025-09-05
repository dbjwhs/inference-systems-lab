#!/bin/bash
# Clean benchmark runner that filters macOS-specific warnings

# Filter out known macOS warnings but preserve actual errors
"$@" 2> >(grep -v -E "(Unable to determine clock rate|Failed to set thread affinity|This does not affect benchmark measurements)" >&2)