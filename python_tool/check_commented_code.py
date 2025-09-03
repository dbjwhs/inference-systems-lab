#!/usr/bin/env python3
"""
Pre-commit hook to detect commented-out code patterns.

This script detects common patterns of commented code that violate
the project's coding standards. It helps enforce the policy that
no commented code should be committed.

Usage:
    python3 python_tool/check_commented_code.py [--fix] [--staged]
    
Exit codes:
    0: No commented code found or all issues fixed
    1: Commented code patterns detected
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

class CommentedCodeDetector:
    """Detects commented-out code patterns in source files."""
    
    def __init__(self):
        """Initialize the detector with patterns to match."""
        # Patterns that likely indicate commented-out code
        self.cpp_patterns = [
            # Commented function calls
            r'^\s*//\s*\w+\s*\(',
            # Commented LOG statements  
            r'^\s*//\s*LOG_\w+(_PRINT)?\s*\(',
            # Commented assignments
            r'^\s*//\s*\w+\s*=',
            # Commented control structures
            r'^\s*//\s*(if|for|while|switch)\s*\(',
            # Commented return statements
            r'^\s*//\s*return\s',
            # Commented include statements
            r'^\s*//\s*#include',
            # Commented class/struct definitions
            r'^\s*//\s*(class|struct|enum)\s+\w+',
            # Commented method definitions
            r'^\s*//\s*\w+::\w+\s*\(',
            # Multiple commented lines that look like code blocks
            r'^\s*//.*\{.*$',  # Opening braces
            r'^\s*//.*\}.*$',  # Closing braces
        ]
        
        # Exceptions - comments that are legitimate documentation
        self.exceptions = [
            r'^\s*//\s*Copyright',
            r'^\s*//\s*TODO',
            r'^\s*//\s*FIXME', 
            r'^\s*//\s*NOTE',
            r'^\s*//\s*HACK', 
            r'^\s*//\s*BUG',
            r'^\s*//\s*WARNING',
            r'^\s*//\s*Example:',
            r'^\s*//\s*Usage:',
            r'^\s*//\s*\w+\s+\w+\s*$',  # Short descriptive comments
            r'^\s*//\s*\d+\.',  # Numbered lists
            r'^\s*//\s*-\s',    # Bullet points
            r'^\s*//\s*[A-Z][a-z]+.*[.!?]$',  # Sentences ending with punctuation
            r'^\s*//.*\*+.*$',  # Comment decorations with asterisks
        ]
    
    def is_legitimate_comment(self, line: str) -> bool:
        """Check if a commented line is legitimate documentation."""
        for pattern in self.exceptions:
            if re.match(pattern, line):
                return True
        return False
    
    def detect_in_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """
        Detect commented code in a single file.
        
        Returns:
            List of (line_number, line_content, pattern_matched) tuples
        """
        if not file_path.exists() or not file_path.is_file():
            return []
        
        # Only check C++ source files
        if file_path.suffix not in ['.cpp', '.hpp', '.cc', '.h', '.cxx', '.hxx']:
            return []
        
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Skip if it's a legitimate comment
                if self.is_legitimate_comment(line):
                    continue
                
                # Check against commented code patterns
                for pattern in self.cpp_patterns:
                    if re.match(pattern, line):
                        violations.append((line_num, line.strip(), pattern))
                        break  # Only report once per line
                        
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
        
        return violations
    
    def scan_project(self, paths: List[Path]) -> Dict[Path, List[Tuple[int, str, str]]]:
        """Scan multiple paths for commented code."""
        all_violations = {}
        
        for path in paths:
            if path.is_file():
                violations = self.detect_in_file(path)
                if violations:
                    all_violations[path] = violations
            elif path.is_dir():
                # Recursively scan directory
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        violations = self.detect_in_file(file_path)
                        if violations:
                            all_violations[file_path] = violations
        
        return all_violations

def get_staged_files() -> List[Path]:
    """Get list of staged C++ files for pre-commit hook."""
    import subprocess
    
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=AM'],
            capture_output=True, text=True, check=True
        )
        
        staged_files = []
        for file_path in result.stdout.strip().split('\n'):
            if file_path and Path(file_path).suffix in ['.cpp', '.hpp', '.cc', '.h', '.cxx', '.hxx']:
                staged_files.append(Path(file_path))
        
        return staged_files
        
    except subprocess.CalledProcessError:
        return []

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Detect commented-out code')
    parser.add_argument('--staged', action='store_true', 
                       help='Only check staged files (for pre-commit)')
    parser.add_argument('--fix', action='store_true',
                       help='Interactive mode to review and remove violations')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output (for pre-commit hooks)')
    parser.add_argument('paths', nargs='*', default=['.'],
                       help='Paths to check (default: current directory)')
    
    args = parser.parse_args()
    
    # Get files to check
    if args.staged:
        check_paths = get_staged_files()
        if not check_paths:
            print("No staged C++ files to check")
            return 0
    else:
        check_paths = [Path(p) for p in args.paths]
    
    # Scan for violations
    detector = CommentedCodeDetector()
    violations = detector.scan_project(check_paths)
    
    if not violations:
        if not args.quiet:
            print("✅ No commented code patterns detected")
        return 0
    
    # Report violations (unless quiet)
    if not args.quiet:
        total_violations = sum(len(v) for v in violations.values())
        print(f"❌ Found {total_violations} potential commented code violations:")
        print()
        
        for file_path, file_violations in violations.items():
            print(f"📁 {file_path}")
            for line_num, line_content, pattern in file_violations:
                print(f"   Line {line_num:3d}: {line_content}")
                if args.fix:
                    print(f"   Pattern : {pattern}")
            print()
    
    # Interactive fix mode
    if args.fix:
        if not args.quiet:
            print("🔧 Fix mode enabled. Please review each violation manually:")
            print()
            print("Recommended actions:")
            print("1. Remove the commented code entirely")
            print("2. Replace with appropriate LOG_DEBUG_PRINT() calls")
            print("3. Use conditional compilation (#ifdef DEBUG)")
            print("4. If legitimately documentation, ignore this warning")
            print()
            print("Remember: Git preserves deleted code history - no need to keep commented code!")
        return 0  # Don't fail in fix mode, just inform
    
    # Provide guidance (unless quiet)
    if not args.quiet:
        print("💡 How to fix:")
        print("   • Remove commented code entirely - git preserves the history")
        print("   • Use LOG_DEBUG_PRINT() for development diagnostics")
        print("   • Use #ifdef DEBUG for debug-only code")
        print("   • Run with --fix for interactive review")
        print()
        print("📖 See CLAUDE.md 'Code Cleanliness Standards' for full policy")
    
    return 1  # Fail pre-commit if violations found

if __name__ == '__main__':
    sys.exit(main())
