#!/usr/bin/env python3
"""
check_format.py - Code formatting verification and enforcement for the Inference Systems Lab

This script checks and optionally fixes code formatting using clang-format
to ensure consistent code style across the project. It supports both
verification mode for CI/CD and automatic fixing for development.

Features:
- Automatic discovery of C++ source files
- clang-format integration with project configuration
- Verification mode for CI/CD pipelines
- Automatic formatting with backup options
- Detailed reporting of formatting violations
- Integration with git for tracking changes

Usage:
    python tools/check_format.py [options]
    
Examples:
    python tools/check_format.py --check
    python tools/check_format.py --fix --backup
    python tools/check_format.py --check --filter "common/src/*"
    python tools/check_format.py --fix --exclude "build,cmake-build-debug"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple
import shutil
import fnmatch
import tempfile


class FormatViolation:
    """Represents a formatting violation found in a file."""
    
    def __init__(self, file_path: str, line_number: int, message: str):
        self.file_path = file_path
        self.line_number = line_number
        self.message = message
        
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}: {self.message}"


class CodeFormatter:
    """Main class for checking and fixing code formatting."""
    
    def __init__(self, project_root: Path, clang_format_path: str = "clang-format"):
        self.project_root = project_root
        self.clang_format_path = clang_format_path
        self.cpp_extensions = {'.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.h', '.c'}
        
    def check_clang_format_available(self) -> bool:
        """Check if clang-format is available."""
        try:
            result = subprocess.run([self.clang_format_path, '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def get_clang_format_version(self) -> Optional[str]:
        """Get clang-format version string."""
        try:
            result = subprocess.run([self.clang_format_path, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        return None
    
    def discover_source_files(self, include_patterns: Optional[List[str]] = None,
                            exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover C++ source files in the project."""
        source_files = []
        
        # Default exclude patterns
        default_excludes = [
            'build', 'cmake-build-debug', 'cmake-build-release', '_deps', 'CMakeFiles',
            '*.pb.h', '*.pb.cc',  # Protocol buffer generated files
            '*.capnp.h', '*.capnp.c++',  # Cap'n Proto generated files
            'third_party', 'external'
        ]
        
        exclude_patterns = (exclude_patterns or []) + default_excludes
        include_patterns = include_patterns or ['*']
        
        # Walk through project directory
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories early
            dirs[:] = [d for d in dirs if d not in exclude_patterns]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check if file has C++ extension
                if file_path.suffix not in self.cpp_extensions:
                    continue
                    
                # Convert to relative path for pattern matching
                rel_path = file_path.relative_to(self.project_root)
                
                # Check exclude patterns for generated files
                if any(fnmatch.fnmatch(file, pattern)
                      for pattern in exclude_patterns if '*' in pattern):
                    continue
                
                # Check include patterns
                if any(fnmatch.fnmatch(str(rel_path), pattern) or
                      fnmatch.fnmatch(str(file_path), pattern)
                      for pattern in include_patterns):
                    source_files.append(file_path)
        
        return sorted(source_files)
    
    def check_file_formatting(self, file_path: Path) -> List[FormatViolation]:
        """Check if a file has formatting violations."""
        violations = []
        
        try:
            # Run clang-format in dry-run mode to check for violations
            result = subprocess.run([
                self.clang_format_path, 
                '--dry-run', 
                '--Werror',
                str(file_path)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse clang-format output for violations
            if result.returncode != 0 and result.stderr:
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'code should be clang-formatted' in line:
                        # Parse format: file:line:col: warning: message
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            try:
                                file_part = parts[0]
                                line_number = int(parts[1])
                                message = parts[3].strip() if len(parts) > 3 else "formatting violation"
                                
                                violations.append(FormatViolation(
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=line_number,
                                    message=message
                                ))
                            except (ValueError, IndexError):
                                # Skip lines we can't parse
                                continue
                                
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
            
        return violations
    
    def format_file(self, file_path: Path, create_backup: bool = False) -> bool:
        """Format a single file using clang-format."""
        try:
            if create_backup:
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                shutil.copy2(file_path, backup_path)
                print(f"Created backup: {backup_path}")
            
            # Run clang-format to fix the file
            result = subprocess.run([
                self.clang_format_path,
                '-i',  # In-place formatting
                str(file_path)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                return True
            else:
                print(f"Error formatting {file_path}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error formatting {file_path}: {e}")
            return False
    
    def get_formatted_diff(self, file_path: Path) -> Optional[str]:
        """Get the diff between current file and formatted version."""
        try:
            # Get formatted content
            result = subprocess.run([
                self.clang_format_path,
                str(file_path)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                return None
                
            formatted_content = result.stdout
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # If content is the same, no diff
            if original_content == formatted_content:
                return None
            
            # Create diff using unified format
            with tempfile.NamedTemporaryFile(mode='w', suffix='.orig', delete=False) as orig_file:
                orig_file.write(original_content)
                orig_file.flush()
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.fmt', delete=False) as fmt_file:
                    fmt_file.write(formatted_content)
                    fmt_file.flush()
                    
                    try:
                        diff_result = subprocess.run([
                            'diff', '-u', 
                            '--label', f'a/{file_path.relative_to(self.project_root)}',
                            '--label', f'b/{file_path.relative_to(self.project_root)}',
                            orig_file.name, fmt_file.name
                        ], capture_output=True, text=True)
                        
                        return diff_result.stdout if diff_result.stdout else None
                    finally:
                        os.unlink(orig_file.name)
                        os.unlink(fmt_file.name)
                        
        except Exception as e:
            print(f"Error getting diff for {file_path}: {e}")
            return None
    
    def check_all_files(self, files: List[Path], show_diffs: bool = False) -> Tuple[List[FormatViolation], int]:
        """Check formatting for all specified files."""
        all_violations = []
        files_with_violations = 0
        
        print(f"Checking formatting for {len(files)} files...")
        
        for file_path in files:
            violations = self.check_file_formatting(file_path)
            if violations:
                files_with_violations += 1
                all_violations.extend(violations)
                
                if show_diffs:
                    diff = self.get_formatted_diff(file_path)
                    if diff:
                        print(f"\nFormatting diff for {file_path.relative_to(self.project_root)}:")
                        print(diff)
        
        return all_violations, files_with_violations
    
    def format_all_files(self, files: List[Path], create_backup: bool = False) -> Tuple[int, int]:
        """Format all specified files."""
        formatted_count = 0
        error_count = 0
        
        print(f"Formatting {len(files)} files...")
        
        for file_path in files:
            print(f"Formatting {file_path.relative_to(self.project_root)}...")
            
            if self.format_file(file_path, create_backup):
                formatted_count += 1
            else:
                error_count += 1
        
        return formatted_count, error_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Code formatting verification and enforcement for the Inference Systems Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                                   # Check all files
  %(prog)s --fix --backup                           # Fix all files with backup
  %(prog)s --check --filter "common/src/*"          # Check specific files
  %(prog)s --fix --exclude "build,third_party"      # Fix excluding certain dirs
  %(prog)s --check --show-diffs                     # Show formatting diffs
        """
    )
    
    # Action options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",
                      action="store_true",
                      help="Check formatting without making changes")
    group.add_argument("--fix",
                      action="store_true", 
                      help="Fix formatting issues automatically")
    
    # File selection options
    parser.add_argument("--filter",
                       help="Include only files matching this pattern (supports wildcards)")
    parser.add_argument("--filter-from-file",
                       type=Path,
                       help="Include only files listed in the specified file (one per line)")
    parser.add_argument("--exclude",
                       help="Exclude files/directories matching these patterns (comma-separated)")
    
    # Formatting options
    parser.add_argument("--backup",
                       action="store_true",
                       help="Create backup files before formatting (use with --fix)")
    parser.add_argument("--show-diffs",
                       action="store_true",
                       help="Show formatting diffs (use with --check)")
    
    # Tool options
    parser.add_argument("--clang-format-path",
                       default="clang-format",
                       help="Path to clang-format executable (default: clang-format)")
    parser.add_argument("--quiet",
                       action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Determine project root (script is in tools/ subdirectory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    if not (project_root / "CMakeLists.txt").exists():
        print(f"Error: Project root not found. Expected CMakeLists.txt at {project_root}")
        sys.exit(1)
    
    formatter = CodeFormatter(project_root, args.clang_format_path)
    
    # Check if clang-format is available
    if not formatter.check_clang_format_available():
        print(f"Error: clang-format not found at '{args.clang_format_path}'")
        print("Please install clang-format or specify correct path with --clang-format-path")
        sys.exit(1)
    
    if not args.quiet:
        version = formatter.get_clang_format_version()
        print(f"Using clang-format: {version}")
        print(f"Project root: {project_root}")
        
        config_file = project_root / ".clang-format"
        if config_file.exists():
            print(f"Configuration: {config_file}")
        else:
            print("Warning: No .clang-format configuration found")
        print()
    
    # Parse file patterns
    include_patterns = [args.filter] if args.filter else None
    exclude_patterns = args.exclude.split(',') if args.exclude else None
    
    # Handle file list input
    if args.filter_from_file:
        try:
            with open(args.filter_from_file, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]
            # Convert to absolute paths and filter existing files
            source_files = []
            for file_path in file_list:
                abs_path = project_root / file_path if not Path(file_path).is_absolute() else Path(file_path)
                if abs_path.exists() and abs_path.suffix in formatter.cpp_extensions:
                    source_files.append(abs_path)
        except Exception as e:
            print(f"Error reading file list from {args.filter_from_file}: {e}")
            sys.exit(1)
    else:
        # Discover source files
        source_files = formatter.discover_source_files(include_patterns, exclude_patterns)
    
    if not source_files:
        print("No source files found matching criteria")
        sys.exit(0)
    
    if not args.quiet:
        print(f"Found {len(source_files)} source files")
        if args.filter:
            print(f"Include pattern: {args.filter}")
        if args.filter_from_file:
            print(f"Files from: {args.filter_from_file}")
        if args.exclude:
            print(f"Exclude patterns: {args.exclude}")
        if len(source_files) <= 10:
            for f in source_files:
                print(f"  {f.relative_to(project_root)}")
        print()
    
    # Execute action
    if args.check:
        violations, files_with_violations = formatter.check_all_files(source_files, args.show_diffs)
        
        if violations:
            print(f"\n❌ Found {len(violations)} formatting violations in {files_with_violations} files:")
            for violation in violations[:20]:  # Show first 20 violations
                print(f"  {violation}")
            
            if len(violations) > 20:
                print(f"  ... and {len(violations) - 20} more violations")
            
            print(f"\nTo fix these issues, run: python tools/check_format.py --fix")
            if not args.backup:
                print("Add --backup to create backup files before formatting")
            
            sys.exit(1)
        else:
            print(f"✅ All {len(source_files)} files are properly formatted")
            sys.exit(0)
    
    elif args.fix:
        formatted_count, error_count = formatter.format_all_files(source_files, args.backup)
        
        if error_count > 0:
            print(f"\n❌ Formatted {formatted_count} files, {error_count} errors")
            sys.exit(1)
        else:
            print(f"\n✅ Successfully formatted {formatted_count} files")
            
            if args.backup:
                print("Backup files created with .bak extension")
            
            print("\nRecommended next steps:")
            print("1. Review the changes: git diff")
            print("2. Run tests to ensure functionality: python tools/run_tests.py")
            print("3. Commit the formatting changes: git add -A && git commit -m 'Apply clang-format'")
            
            sys.exit(0)


if __name__ == "__main__":
    main()
