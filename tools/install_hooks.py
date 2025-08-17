#!/usr/bin/env python3
"""
install_hooks.py - Pre-commit hook installation and management for the Inference Systems Lab

This script installs and manages Git pre-commit hooks that integrate all project
quality tools including formatting, static analysis, and basic validation checks.
The hooks are designed to catch issues early in the development workflow while
maintaining developer productivity.

Features:
- Automatic installation of Git pre-commit hooks
- Integration with existing quality tools (format, static analysis)
- Configurable hook behavior and bypass options
- Selective checking of only modified files for performance
- Comprehensive error reporting and developer guidance
- Backup and restoration of existing hooks

Usage:
    python tools/install_hooks.py [options]
    
Examples:
    python tools/install_hooks.py --install           # Install pre-commit hooks
    python tools/install_hooks.py --uninstall         # Remove pre-commit hooks
    python tools/install_hooks.py --status            # Check hook status
    python tools/install_hooks.py --test              # Test hooks on current changes
"""

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class PreCommitHookManager:
    """Manages installation and configuration of Git pre-commit hooks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.hooks_dir = project_root / ".git" / "hooks"
        self.tools_dir = project_root / "tools"
        self.pre_commit_hook = self.hooks_dir / "pre-commit"
        self.backup_hook = self.hooks_dir / "pre-commit.backup"
        
    def is_git_repository(self) -> bool:
        """Check if the current directory is a Git repository."""
        return (self.project_root / ".git").exists()
    
    def has_existing_hook(self) -> bool:
        """Check if a pre-commit hook already exists."""
        return self.pre_commit_hook.exists()
    
    def backup_existing_hook(self) -> bool:
        """Backup existing pre-commit hook if it exists."""
        if self.has_existing_hook():
            try:
                shutil.copy2(self.pre_commit_hook, self.backup_hook)
                print(f"Backed up existing hook to {self.backup_hook}")
                return True
            except Exception as e:
                print(f"Error backing up existing hook: {e}")
                return False
        return True
    
    def restore_backup_hook(self) -> bool:
        """Restore backed up pre-commit hook."""
        if self.backup_hook.exists():
            try:
                shutil.copy2(self.backup_hook, self.pre_commit_hook)
                self.backup_hook.unlink()
                print(f"Restored backup hook from {self.backup_hook}")
                return True
            except Exception as e:
                print(f"Error restoring backup hook: {e}")
                return False
        return False
    
    def create_pre_commit_hook(self) -> str:
        """Create the pre-commit hook script content."""
        hook_content = f'''#!/bin/bash
# Pre-commit hook for Inference Systems Lab
# Runs code quality checks on staged files before commit
# 
# This hook integrates:
# - Code formatting (clang-format)
# - Static analysis (clang-tidy) 
# - Basic validation checks
#
# To bypass these checks (emergency commits only):
# git commit --no-verify

set -e

PROJECT_ROOT="{self.project_root}"
TOOLS_DIR="$PROJECT_ROOT/tools"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Logging functions
log_info() {{
    echo -e "${{BLUE}}[INFO]${{NC}} $1"
}}

log_success() {{
    echo -e "${{GREEN}}[SUCCESS]${{NC}} $1"
}}

log_warning() {{
    echo -e "${{YELLOW}}[WARNING]${{NC}} $1"
}}

log_error() {{
    echo -e "${{RED}}[ERROR]${{NC}} $1"
}}

# Check if tools exist
check_tools() {{
    local missing_tools=()
    
    if [[ ! -f "$TOOLS_DIR/check_format.py" ]]; then
        missing_tools+=("check_format.py")
    fi
    
    if [[ ! -f "$TOOLS_DIR/check_static_analysis.py" ]]; then
        missing_tools+=("check_static_analysis.py")
    fi
    
    if [[ ${{#missing_tools[@]}} -gt 0 ]]; then
        log_error "Missing required tools: ${{missing_tools[*]}}"
        log_error "Please ensure all development tools are available."
        exit 1
    fi
}}

# Get list of staged C++ files
get_staged_cpp_files() {{
    git diff --cached --name-only --diff-filter=ACM | grep -E '\\.(cpp|hpp|cc|cxx|hxx|h|c)$' || true
}}

# Run formatting check on staged files
check_formatting() {{
    local staged_files=("$@")
    
    if [[ ${{#staged_files[@]}} -eq 0 ]]; then
        log_info "No C++ files to check formatting"
        return 0
    fi
    
    log_info "Checking code formatting on ${{#staged_files[@]}} files..."
    
    # Create temporary file list for filtering
    local temp_filter_file=$(mktemp)
    printf '%s\\n' "${{staged_files[@]}}" > "$temp_filter_file"
    
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
}}

# Run static analysis on staged files (errors only for performance)
check_static_analysis() {{
    local staged_files=("$@")
    
    if [[ ${{#staged_files[@]}} -eq 0 ]]; then
        log_info "No C++ files for static analysis"
        return 0
    fi
    
    log_info "Running static analysis (errors only) on ${{#staged_files[@]}} files..."
    
    # Create temporary file list for filtering
    local temp_filter_file=$(mktemp)
    printf '%s\\n' "${{staged_files[@]}}" > "$temp_filter_file"
    
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
}}

# Basic validation checks
check_basic_validation() {{
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
}}

# Main pre-commit hook logic
main() {{
    log_info "Running Inference Systems Lab pre-commit checks..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check if required tools are available
    check_tools
    
    # Get staged C++ files
    staged_cpp_files=($(get_staged_cpp_files))
    
    if [[ ${{#staged_cpp_files[@]}} -eq 0 ]]; then
        log_info "No C++ files in this commit, skipping quality checks"
        log_success "Pre-commit checks completed"
        exit 0
    fi
    
    log_info "Found ${{#staged_cpp_files[@]}} staged C++ files"
    
    # Track overall success
    local overall_success=true
    
    # Run formatting check
    if ! check_formatting "${{staged_cpp_files[@]}}"; then
        overall_success=false
    fi
    
    # Run static analysis (only if formatting passed to avoid noise)
    if [[ "$overall_success" == "true" ]]; then
        if ! check_static_analysis "${{staged_cpp_files[@]}}"; then
            overall_success=false
        fi
    else
        log_info "Skipping static analysis due to formatting issues"
    fi
    
    # Run basic validation
    if ! check_basic_validation; then
        overall_success=false
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
}}

# Run main function
main "$@"
'''
        return hook_content
    
    def install_hook(self, force: bool = False) -> bool:
        """Install the pre-commit hook."""
        if not self.is_git_repository():
            print("Error: Not in a Git repository")
            return False
        
        if self.has_existing_hook() and not force:
            print("Pre-commit hook already exists. Use --force to overwrite.")
            return False
        
        # Backup existing hook if it exists
        if not self.backup_existing_hook():
            return False
        
        try:
            # Create hooks directory if it doesn't exist
            self.hooks_dir.mkdir(parents=True, exist_ok=True)
            
            # Write the hook script
            hook_content = self.create_pre_commit_hook()
            self.pre_commit_hook.write_text(hook_content)
            
            # Make the hook executable
            current_mode = self.pre_commit_hook.stat().st_mode
            self.pre_commit_hook.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            
            print(f"✅ Pre-commit hook installed successfully at {self.pre_commit_hook}")
            print("The hook will run automatically on 'git commit'")
            print("To bypass the hook: git commit --no-verify")
            return True
            
        except Exception as e:
            print(f"Error installing pre-commit hook: {e}")
            return False
    
    def uninstall_hook(self) -> bool:
        """Uninstall the pre-commit hook."""
        if not self.has_existing_hook():
            print("No pre-commit hook found to uninstall")
            return True
        
        try:
            # Remove the current hook
            self.pre_commit_hook.unlink()
            
            # Restore backup if it exists
            if self.backup_hook.exists():
                if not self.restore_backup_hook():
                    print("Removed hook but failed to restore backup")
                    return False
            
            print("✅ Pre-commit hook uninstalled successfully")
            return True
            
        except Exception as e:
            print(f"Error uninstalling pre-commit hook: {e}")
            return False
    
    def get_status(self) -> None:
        """Display the current status of pre-commit hooks."""
        print("Pre-commit Hook Status:")
        print("=" * 40)
        
        if not self.is_git_repository():
            print("❌ Not in a Git repository")
            return
        
        print(f"📁 Project root: {self.project_root}")
        print(f"📁 Hooks directory: {self.hooks_dir}")
        
        if self.has_existing_hook():
            print(f"✅ Pre-commit hook: {self.pre_commit_hook}")
            
            # Check if it's executable
            if os.access(self.pre_commit_hook, os.X_OK):
                print("✅ Hook is executable")
            else:
                print("❌ Hook is not executable")
        else:
            print("❌ No pre-commit hook installed")
        
        if self.backup_hook.exists():
            print(f"📄 Backup hook: {self.backup_hook}")
        
        # Check if required tools exist
        tools_status = []
        required_tools = ["check_format.py", "check_static_analysis.py"]
        
        for tool in required_tools:
            tool_path = self.tools_dir / tool
            if tool_path.exists():
                tools_status.append(f"✅ {tool}")
            else:
                tools_status.append(f"❌ {tool}")
        
        print("\nRequired Tools:")
        for status in tools_status:
            print(f"  {status}")
    
    def test_hook(self) -> bool:
        """Test the pre-commit hook on current staged changes."""
        if not self.has_existing_hook():
            print("No pre-commit hook installed")
            return False
        
        print("Testing pre-commit hook...")
        try:
            # Run the hook script directly
            result = subprocess.run([str(self.pre_commit_hook)], 
                                  cwd=self.project_root,
                                  capture_output=True, 
                                  text=True)
            
            print("Hook output:")
            print(result.stdout)
            if result.stderr:
                print("Hook errors:")
                print(result.stderr)
            
            if result.returncode == 0:
                print("✅ Pre-commit hook test passed")
                return True
            else:
                print(f"❌ Pre-commit hook test failed with exit code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"Error testing pre-commit hook: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-commit hook installation and management for the Inference Systems Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --install                    # Install pre-commit hooks
  %(prog)s --install --force            # Force reinstall hooks
  %(prog)s --uninstall                  # Remove pre-commit hooks
  %(prog)s --status                     # Check hook status
  %(prog)s --test                       # Test hooks on current changes
        """
    )
    
    # Action options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install",
                      action="store_true",
                      help="Install pre-commit hooks")
    group.add_argument("--uninstall",
                      action="store_true",
                      help="Uninstall pre-commit hooks")
    group.add_argument("--status",
                      action="store_true",
                      help="Show pre-commit hook status")
    group.add_argument("--test",
                      action="store_true",
                      help="Test pre-commit hook on current staged changes")
    
    # Installation options
    parser.add_argument("--force",
                       action="store_true",
                       help="Force installation even if hook exists")
    
    args = parser.parse_args()
    
    # Determine project root (script is in tools/ subdirectory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    if not (project_root / "CMakeLists.txt").exists():
        print(f"Error: Project root not found. Expected CMakeLists.txt at {project_root}")
        sys.exit(1)
    
    manager = PreCommitHookManager(project_root)
    
    if args.install:
        success = manager.install_hook(force=args.force)
        sys.exit(0 if success else 1)
    elif args.uninstall:
        success = manager.uninstall_hook()
        sys.exit(0 if success else 1)
    elif args.status:
        manager.get_status()
        sys.exit(0)
    elif args.test:
        success = manager.test_hook()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()