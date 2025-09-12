#!/usr/bin/env python3
"""
Check and fix MIT license headers in C++ source files.

This script ensures all C++ source files (.cpp, .hpp) have consistent
MIT license headers. It can check files, fix them, and integrate with
pre-commit hooks.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess


EXPECTED_HEADER = """// MIT License
// Copyright (c) 2025 dbjwhs
//
// This software is provided "as is" without warranty of any kind, express or implied.
// The authors are not liable for any damages arising from the use of this software."""

# Directories to exclude from header checks
EXCLUDE_DIRS = {
    'build',
    'build-debug',
    'build-release', 
    'build-sanitizer',
    'cmake-build-debug',
    'cmake-build-release',
    '.git',
    'third_party',
    'external',
    'vendor',
    'generated',
    '.vscode',
    '.idea',
    'docs',
    '__pycache__',
    '_deps',
}

# Files to exclude (these might be from external sources or generated)
EXCLUDE_FILES = {
    'json.hpp',  # Common single-header libraries
    'catch.hpp',
    'doctest.h',
}


def get_git_root() -> Path:
    """Get the root directory of the git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return Path.cwd()


def should_skip_file(file_path: Path) -> bool:
    """Check if a file should be skipped based on exclusion rules."""
    # Check if file is in excluded directory
    for part in file_path.parts:
        if part in EXCLUDE_DIRS:
            return True
        # Also check if path starts with build
        if part.startswith('build'):
            return True
    
    # Check if file is in exclude list
    if file_path.name in EXCLUDE_FILES:
        return True
    
    # Skip generated Cap'n Proto headers
    if file_path.name.endswith('.capnp.h'):
        return True
    
    # Skip CMake generated files
    if 'CMakeFiles' in str(file_path):
        return True
    
    # Skip files in python_tool directory (these are Python files)
    if 'python_tool' in file_path.parts:
        return True
    
    # Skip files in tools directory (archived)
    if 'tools' in file_path.parts:
        return True
    
    return False


def check_header(content: str) -> bool:
    """Check if content starts with the expected header."""
    # Normalize line endings and compare
    expected_lines = EXPECTED_HEADER.strip().split('\n')
    content_lines = content.strip().split('\n')
    
    # If file has fewer lines than header, it's missing the header
    if len(content_lines) < len(expected_lines):
        return False
    
    # Check if the first lines match the expected header
    for i, expected_line in enumerate(expected_lines):
        if i >= len(content_lines):
            return False
        if content_lines[i].strip() != expected_line.strip():
            return False
    
    return True


def add_header(content: str) -> str:
    """Add or replace header in content."""
    # Check if there's already a copyright or license header (to replace it)
    lines = content.split('\n')
    
    # Find where the actual code starts (after any existing headers/comments)
    code_start = 0
    in_comment_block = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check for comment blocks
        if stripped.startswith('/*'):
            in_comment_block = True
        if in_comment_block:
            if '*/' in line:
                in_comment_block = False
                code_start = i + 1
            continue
        
        # Skip single-line comments at the beginning
        if stripped.startswith('//'):
            # Look for copyright, license, or author mentions
            lower_line = stripped.lower()
            if any(keyword in lower_line for keyword in ['copyright', 'license', 'author', 'mit']):
                continue
        else:
            # Found first non-comment line
            code_start = i
            break
    
    # Skip empty lines at the beginning after removing old header
    while code_start < len(lines) and not lines[code_start].strip():
        code_start += 1
    
    # Reconstruct the file with new header
    new_content = EXPECTED_HEADER + '\n\n'
    if code_start < len(lines):
        new_content += '\n'.join(lines[code_start:])
    
    # Ensure file ends with newline
    if not new_content.endswith('\n'):
        new_content += '\n'
    
    return new_content


def process_file(file_path: Path, fix: bool = False, backup: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Process a single file.
    
    Returns:
        Tuple of (has_correct_header, error_message)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"Error reading {file_path}: {e}"
    
    has_correct_header = check_header(content)
    
    if not has_correct_header and fix:
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            backup_path.write_text(content, encoding='utf-8')
        
        # Add/fix header
        new_content = add_header(content)
        
        try:
            file_path.write_text(new_content, encoding='utf-8')
            return True, None
        except Exception as e:
            return False, f"Error writing {file_path}: {e}"
    
    return has_correct_header, None


def find_cpp_files(root_dir: Path, staged_only: bool = False) -> List[Path]:
    """Find all C++ files to process."""
    files = []
    
    if staged_only:
        # Get staged files from git
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
                capture_output=True,
                text=True,
                check=True,
                cwd=root_dir
            )
            for line in result.stdout.strip().split('\n'):
                if line:
                    file_path = root_dir / line
                    if file_path.suffix in ['.cpp', '.hpp', '.h', '.cc', '.cxx']:
                        if not should_skip_file(file_path):
                            files.append(file_path)
        except subprocess.CalledProcessError:
            pass
    else:
        # Find all C++ files in the repository
        for ext in ['*.cpp', '*.hpp', '*.h', '*.cc', '*.cxx']:
            for file_path in root_dir.rglob(ext):
                if not should_skip_file(file_path):
                    files.append(file_path)
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='Check and fix MIT license headers in C++ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                  # Check all files
  %(prog)s --fix                    # Fix all files
  %(prog)s --fix --backup           # Fix with backups
  %(prog)s --staged                 # Check staged files only (for pre-commit)
  %(prog)s --fix --staged           # Fix staged files only
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check files for correct headers (default mode)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Fix files with incorrect headers'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .bak files when fixing'
    )
    parser.add_argument(
        '--staged',
        action='store_true',
        help='Only process staged files (for pre-commit hook)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress informational output'
    )
    
    args = parser.parse_args()
    
    # Default to check mode if neither check nor fix specified
    if not args.fix:
        args.check = True
    
    # Find repository root
    root_dir = get_git_root()
    
    # Find files to process
    files = find_cpp_files(root_dir, args.staged)
    
    if not files:
        if not args.quiet:
            print("[INFO] No C++ files to process")
        return 0
    
    # Process files
    files_with_issues = []
    files_fixed = []
    errors = []
    
    for file_path in files:
        has_correct_header, error = process_file(file_path, args.fix, args.backup)
        
        if error:
            errors.append(error)
        elif not has_correct_header:
            if args.fix:
                files_fixed.append(file_path)
            else:
                files_with_issues.append(file_path)
    
    # Report results
    if errors:
        print("[ERROR] Encountered errors:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    
    if args.fix:
        if files_fixed:
            if not args.quiet:
                print(f"[SUCCESS] Fixed headers in {len(files_fixed)} file(s):")
                for file_path in files_fixed:
                    print(f"  {file_path.relative_to(root_dir)}")
        else:
            if not args.quiet:
                print("[INFO] All files already have correct headers")
    else:
        if files_with_issues:
            print(f"[WARNING] {len(files_with_issues)} file(s) have incorrect headers:", file=sys.stderr)
            for file_path in files_with_issues:
                print(f"  {file_path.relative_to(root_dir)}", file=sys.stderr)
            print("\nRun with --fix to correct these headers", file=sys.stderr)
            return 1
        else:
            if not args.quiet:
                print("[SUCCESS] All files have correct headers")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
