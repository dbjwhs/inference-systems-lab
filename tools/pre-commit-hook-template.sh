#!/bin/bash
# Pre-commit hook for Inference Systems Lab
# Runs code quality checks on staged files before commit
# 
# This hook integrates:
# - Code formatting (clang-format)
# - Static analysis (clang-tidy) 
# - Basic validation checks
# - Build verification
#
# To bypass these checks (emergency commits only):
# git commit --no-verify

set -e

PROJECT_ROOT="/Users/dbjones/ng/dbjwhs/inference-systems-lab"
TOOLS_DIR="$PROJECT_ROOT/tools"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if tools exist
check_tools() {
    local missing_tools=()
    
    if [[ ! -f "$TOOLS_DIR/check_format.py" ]]; then
        missing_tools+=("check_format.py")
    fi
    
    if [[ ! -f "$TOOLS_DIR/check_static_analysis.py" ]]; then
        missing_tools+=("check_static_analysis.py")
    fi
    
    if [[ ! -f "$TOOLS_DIR/check_eof_newline.py" ]]; then
        missing_tools+=("check_eof_newline.py")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please ensure all development tools are available."
        exit 1
    fi
}

# Get list of staged C++ files
get_staged_cpp_files() {
    git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|hpp|cc|cxx|hxx|h|c)$' || true
}

# Run formatting check on staged files
check_formatting() {
    local staged_files=("$@")
    
    if [[ ${#staged_files[@]} -eq 0 ]]; then
        log_info "No C++ files to check formatting"
        return 0
    fi
    
    log_info "Checking code formatting on ${#staged_files[@]} files..."
    
    # Create temporary file list for filtering
    local temp_filter_file=$(mktemp)
    printf '%s\n' "${staged_files[@]}" > "$temp_filter_file"
    
    # Check formatting using include filter for staged files only
    if python3 "$TOOLS_DIR/check_format.py" --check --filter-from-file "$temp_filter_file" --quiet; then
        log_success "Code formatting check passed"
        rm -f "$temp_filter_file"
        return 0
    else
        log_error "Code formatting check failed"
        log_info "To fix formatting issues automatically, run:"
        log_info "  python3 tools/check_format.py --fix"
        log_info "Then stage the changes and commit again."
        log_info "To bypass this check (not recommended): git commit --no-verify"
        rm -f "$temp_filter_file"
        return 1
    fi
}

# Run static analysis on staged files (errors only for performance)
check_static_analysis() {
    local staged_files=("$@")
    
    if [[ ${#staged_files[@]} -eq 0 ]]; then
        log_info "No C++ files for static analysis"
        return 0
    fi
    
    log_info "Running static analysis (errors only) on ${#staged_files[@]} files..."
    
    # Create temporary file list for filtering
    local temp_filter_file=$(mktemp)
    printf '%s\n' "${staged_files[@]}" > "$temp_filter_file"
    
    # Run static analysis with error severity only for pre-commit performance
    if python3 "$TOOLS_DIR/check_static_analysis.py" --check --severity error --filter-from-file "$temp_filter_file" --quiet; then
        log_success "Static analysis check passed"
        rm -f "$temp_filter_file"
        return 0
    else
        log_error "Static analysis found critical issues"
        log_info "To see all issues, run:"
        log_info "  python3 tools/check_static_analysis.py --check"
        log_info "To fix issues automatically, run:"
        log_info "  python3 tools/check_static_analysis.py --fix --backup"
        log_info "To bypass this check (not recommended): git commit --no-verify"
        rm -f "$temp_filter_file"
        return 1
    fi
}

# Check EOF newlines on all staged files (not just C++)
check_eof_newlines() {
    log_info "Checking end-of-file newlines on staged files..."
    
    # Get all staged files (not just C++)
    local staged_files=($(git diff --cached --name-only --diff-filter=ACM))
    
    if [[ ${#staged_files[@]} -eq 0 ]]; then
        log_info "No staged files to check for EOF newlines"
        return 0
    fi
    
    # Create temporary file list for filtering
    local temp_filter_file=$(mktemp)
    printf '%s\n' "${staged_files[@]}" > "$temp_filter_file"
    
    # Check EOF newlines
    if python3 "$TOOLS_DIR/check_eof_newline.py" --check --filter-from-file "$temp_filter_file" --quiet; then
        log_success "EOF newline check passed"
        rm -f "$temp_filter_file"
        return 0
    else
        log_error "Found files missing EOF newlines"
        log_info "To fix automatically, run:"
        log_info "  python3 tools/check_eof_newline.py --fix --backup"
        log_info "Then stage the changes and commit again."
        log_info "To bypass this check (not recommended): git commit --no-verify"
        rm -f "$temp_filter_file"
        return 1
    fi
}

# Basic validation checks
check_basic_validation() {
    log_info "Running basic validation checks..."
    
    # Check for common issues
    local issues_found=false
    
    # Check for debug print statements
    if git diff --cached | grep -E '(std::cout|printf|std::cerr)' >/dev/null 2>&1; then
        log_warning "Found debug print statements in staged changes"
        log_info "Consider removing debug output before committing"
        log_info "If intentional, you can proceed or use git commit --no-verify"
    fi
    
    # Check for TODO/FIXME without issue numbers
    if git diff --cached | grep -iE '(TODO|FIXME)' | grep -v '#[0-9]' >/dev/null 2>&1; then
        log_warning "Found TODO/FIXME without issue references"
        log_info "Consider linking TODOs to GitHub issues for better tracking"
    fi
    
    # Check for merge conflict markers
    if git diff --cached | grep -E '^(<<<<<<<|=======|>>>>>>>)' >/dev/null 2>&1; then
        log_error "Found merge conflict markers in staged changes"
        issues_found=true
    fi
    
    # Check for trailing whitespace
    if git diff --cached --check >/dev/null 2>&1; then
        log_success "No trailing whitespace found"
    else
        log_warning "Found trailing whitespace in staged changes"
        log_info "Git will automatically highlight these in the diff"
    fi
    
    if [[ "$issues_found" == "true" ]]; then
        log_error "Basic validation failed - please fix issues before committing"
        return 1
    fi
    
    log_success "Basic validation passed"
    return 0
}

# Build verification check
check_build() {
    log_info "Running build verification..."
    
    # Check if build directory exists
    if [[ ! -d "$PROJECT_ROOT/build" ]]; then
        log_error "Build directory not found at $PROJECT_ROOT/build"
        log_info "Please run: mkdir build && cd build && cmake .. && make"
        return 1
    fi
    
    # Run make in build directory
    cd "$PROJECT_ROOT/build"
    
    if make -j4 >/dev/null 2>&1; then
        log_success "Build verification passed"
        cd "$PROJECT_ROOT"
        return 0
    else
        log_error "Build failed - cannot commit code that doesn't compile"
        log_info "Please fix build errors before committing"
        log_info "Run 'make' in build directory to see detailed errors"
        log_info "To bypass this check (emergency only): git commit --no-verify"
        cd "$PROJECT_ROOT"
        return 1
    fi
}

# Main pre-commit hook logic
main() {
    log_info "Running Inference Systems Lab pre-commit checks..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check if required tools are available
    check_tools
    
    # Get staged C++ files
    staged_cpp_files=($(get_staged_cpp_files))
    
    if [[ ${#staged_cpp_files[@]} -eq 0 ]]; then
        log_info "No C++ files in this commit, skipping quality checks"
        log_success "Pre-commit checks completed"
        exit 0
    fi
    
    log_info "Found ${#staged_cpp_files[@]} staged C++ files"
    
    # Track overall success
    local overall_success=true
    
    # Run formatting check
    if ! check_formatting "${staged_cpp_files[@]}"; then
        overall_success=false
    fi
    
    # Run static analysis (only if formatting passed to avoid noise)
    if [[ "$overall_success" == "true" ]]; then
        if ! check_static_analysis "${staged_cpp_files[@]}"; then
            overall_success=false
        fi
    else
        log_info "Skipping static analysis due to formatting issues"
    fi
    
    # Run EOF newline check
    if ! check_eof_newlines; then
        overall_success=false
    fi
    
    # Run basic validation
    if ! check_basic_validation; then
        overall_success=false
    fi
    
    # Run build verification (only if other checks passed to avoid noise)
    if [[ "$overall_success" == "true" ]]; then
        if ! check_build; then
            overall_success=false
        fi
    else
        log_info "Skipping build verification due to previous check failures"
    fi
    
    # Final result
    if [[ "$overall_success" == "true" ]]; then
        log_success "All pre-commit checks passed!"
        exit 0
    else
        log_error "Pre-commit checks failed"
        log_info ""
        log_info "To bypass these checks (emergency only):"
        log_info "  git commit --no-verify"
        log_info ""
        exit 1
    fi
}

# Run main function
main "$@"