#!/bin/bash
# Post-commit hook for Inference Systems Lab
# Handles documentation generation after successful commits
# 
# This hook automatically generates documentation when header files
# are modified, avoiding the chicken-and-egg problem of generating
# files that need to be committed.
#
# Workflow:
# 1. Pre-commit creates marker file if headers changed
# 2. Post-commit detects marker and generates docs
# 3. Stages generated docs and prompts user for follow-up commit

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
TOOLS_DIR="$PROJECT_ROOT/tools"
MARKER_FILE="$PROJECT_ROOT/.git/hooks/needs_docs_update"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[POST-COMMIT]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[POST-COMMIT]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[POST-COMMIT]${NC} $1"
}

log_error() {
    echo -e "${RED}[POST-COMMIT]${NC} $1"
}

# Generate documentation and handle staging
update_documentation() {
    log_info "Generating updated documentation..."
    
    cd "$PROJECT_ROOT"
    
    # Generate documentation with coverage check
    if python3 "$TOOLS_DIR/check_documentation.py" --generate --copy --stage --coverage-threshold 70.0 2>/dev/null; then
        log_success "Documentation generated and copied to docs/ directory"
        
        # Check if files were staged
        if git diff --cached --quiet; then
            log_info "No documentation changes to commit"
        else
            log_info "📝 Documentation files have been updated and staged"
            log_info "💡 Consider running: git commit -m \"Update API documentation\""
        fi
    else
        log_warning "Documentation generated with warnings (see full output with --verbose)"
    fi
    
    # Check if documentation files were created/modified
    local docs_modified=false
    
    # Check for new or modified files in docs output directory
    if [[ -d "$PROJECT_ROOT/build/docs" ]]; then
        # Look for any files that might need to be tracked
        # Note: We don't automatically stage build artifacts, but we inform the user
        log_info "Documentation updated in build/docs/"
        docs_modified=true
    fi
    
    # Check if any documentation source files were generated (shouldn't happen, but just in case)
    if git status --porcelain | grep -E '\.(md|rst|html)$' >/dev/null 2>&1; then
        log_warning "Documentation source files were modified:"
        git status --porcelain | grep -E '\.(md|rst|html)$' | sed 's/^/  /'
        
        log_info "To include these changes:"
        log_info "  git add ."  
        log_info "  git commit -m 'Update documentation'"
        
        docs_modified=true
    fi
    
    if [[ "$docs_modified" == "true" ]]; then
        log_info ""
        log_info "📖 Documentation has been updated!"
        log_info "🌐 View documentation at: build/docs/html/index.html"
        log_info ""
        
        # Check if we're in a CI environment
        if [[ -n "$CI" || -n "$GITHUB_ACTIONS" ]]; then
            log_info "CI environment detected - documentation updated automatically"
        else
            log_info "💡 To open documentation:"
            log_info "  open build/docs/html/index.html     # macOS"
            log_info "  xdg-open build/docs/html/index.html # Linux" 
        fi
    else
        log_info "Documentation is up to date"
    fi
}

# Check if we need to update documentation
check_needs_update() {
    if [[ -f "$MARKER_FILE" ]]; then
        log_info "Documentation update requested by pre-commit hook"
        return 0
    fi
    
    # Also check if the last commit modified header files directly
    local modified_headers=($(git diff --name-only HEAD~1 HEAD | grep -E '\.(hpp|h)$' || true))
    
    if [[ ${#modified_headers[@]} -gt 0 ]]; then
        log_info "Header files were modified in last commit: ${modified_headers[*]}"
        return 0
    fi
    
    return 1
}

# Clean up marker file
cleanup() {
    if [[ -f "$MARKER_FILE" ]]; then
        rm -f "$MARKER_FILE"
    fi
}

# Main post-commit logic
main() {
    # Only proceed if documentation update is needed
    if check_needs_update; then
        update_documentation
    fi
    
    # Always clean up marker file
    cleanup
    
    # Check for any additional helpful information
    local last_commit_msg=$(git log -1 --pretty=format:"%s")
    if [[ "$last_commit_msg" == *"doc"* || "$last_commit_msg" == *"README"* ]]; then
        log_info "💡 Tip: Since you updated documentation, consider running:"
        log_info "  python3 tools/check_documentation.py --generate --check"
        log_info "  to ensure everything is properly documented"
    fi
}

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Run main function (with error handling)
if main; then
    exit 0
else
    log_error "Post-commit documentation update failed"
    cleanup
    exit 0  # Don't fail the commit for documentation issues
fi
