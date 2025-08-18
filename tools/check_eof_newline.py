#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
check_eof_newline.py - End-of-file newline verification and enforcement for the Inference Systems Lab

This script checks and optionally fixes files that are missing a newline character at the end.
Having a newline at EOF is a POSIX standard requirement and many tools expect it for proper
file handling. This tool supports both verification mode for CI/CD and automatic fixing.

Features:
- Automatic discovery of source files (C++, Python, and other text files)
- EOF newline verification with detailed reporting
- Automatic fixing with backup options
- Integration with existing development workflow
- Support for multiple file types and patterns
- Performance optimized for large codebases

Usage:
    python tools/check_eof_newline.py [options]
    
Examples:
    python tools/check_eof_newline.py --check              # Check all files
    python tools/check_eof_newline.py --fix --backup      # Fix with backup
    python tools/check_eof_newline.py --check --filter "*.py"  # Check Python files only
    python tools/check_eof_newline.py --check --filter-from-file staged_files.txt
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple
import fnmatch


class EOFNewlineChecker:
    """Checks and fixes missing newlines at end of files."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        # File extensions that should have EOF newlines
        self.text_extensions = {
            '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.h', '.c',  # C++
            '.py', '.pyx', '.pyi',  # Python
            '.js', '.ts', '.jsx', '.tsx',  # JavaScript/TypeScript
            '.css', '.scss', '.sass', '.less',  # Stylesheets
            '.html', '.htm', '.xml', '.svg',  # Markup
            '.json', '.yaml', '.yml', '.toml',  # Data formats
            '.md', '.rst', '.txt',  # Documentation
            '.sh', '.bash', '.zsh', '.fish',  # Shell scripts
            '.cmake', '.cmake.in',  # CMake
            '.proto', '.capnp',  # Protocol definitions
            '.sql', '.sqlite',  # Database
            '.conf', '.cfg', '.ini',  # Configuration
            '.gitignore', '.gitattributes',  # Git files
        }
        
        # Files that should be checked regardless of extension
        self.special_files = {
            'Makefile', 'CMakeLists.txt', 'Dockerfile', 'Vagrantfile',
            'Jenkinsfile', 'Pipfile', 'requirements.txt', 'setup.py',
            'pyproject.toml', 'package.json', 'tsconfig.json'
        }
    
    def should_check_file(self, file_path: Path) -> bool:
        """Determine if a file should be checked for EOF newline."""
        # Check by extension
        if file_path.suffix.lower() in self.text_extensions:
            return True
            
        # Check special files by name
        if file_path.name in self.special_files:
            return True
            
        # Check if it's a text file (basic heuristic)
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes to check if it's text
                sample = f.read(1024)
                if not sample:
                    return True  # Empty files are valid and should be checked
                    
                # Simple text detection - no null bytes in sample
                return b'\x00' not in sample
        except (OSError, IOError):
            return False
    
    def has_eof_newline(self, file_path: Path) -> bool:
        """Check if a file ends with a newline character."""
        try:
            with open(file_path, 'rb') as f:
                # Check if file is empty first
                f.seek(0, 2)  # Seek to end
                if f.tell() == 0:  # File is empty
                    return True  # Empty files are valid and don't need newlines
                
                # File has content, check last character
                f.seek(-1, 2)  # Seek to last character
                last_char = f.read(1)
                return last_char in (b'\n', b'\r\n', b'\r')
        except (OSError, IOError):
            return False
    
    def fix_eof_newline(self, file_path: Path, create_backup: bool = False) -> bool:
        """Add newline to end of file if missing."""
        try:
            # Create backup if requested
            if create_backup:
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                shutil.copy2(file_path, backup_path)
            
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check if file is empty
            if not content:
                return True  # Empty files don't need newlines
            
            # Check if already ends with newline
            if content.endswith((b'\n', b'\r\n', b'\r')):
                return True  # Already has newline
            
            # Add newline (use Unix style \n)
            with open(file_path, 'wb') as f:
                f.write(content + b'\n')
            
            return True
            
        except (OSError, IOError) as e:
            print(f"Error fixing {file_path}: {e}")
            return False
    
    def discover_files(self, include_patterns: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover files that should be checked for EOF newlines."""
        files = []
        
        # Default exclude patterns for directories and files we don't want to check
        default_excludes = [
            'build', 'cmake-build-*', '_deps', 'CMakeFiles',
            '.git', '.svn', '.hg', '.bzr',
            '__pycache__', '*.pyc', '*.pyo', '*.pyd',
            '*.so', '*.dll', '*.dylib', '*.exe',
            '*.o', '*.obj', '*.a', '*.lib',
            '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.ico',
            '*.pdf', '*.zip', '*.tar', '*.gz', '*.bz2',
            'node_modules', '.venv', 'venv', '.env',
            '.idea', '.vscode', '*.swp', '*.swo', '*~'
        ]
        
        exclude_patterns = (exclude_patterns or []) + default_excludes
        include_patterns = include_patterns or ['*']
        
        # Walk through project directory
        for root, dirs, filenames in os.walk(self.project_root):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns
            )]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Skip excluded files
                if any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns):
                    continue
                
                # Check include patterns
                if not any(fnmatch.fnmatch(filename, pattern) or 
                          fnmatch.fnmatch(str(file_path.relative_to(self.project_root)), pattern)
                          for pattern in include_patterns):
                    continue
                
                # Check if this file should be checked
                if self.should_check_file(file_path):
                    files.append(file_path)
        
        return sorted(files)
    
    def check_files(self, files: List[Path], show_details: bool = False) -> Tuple[List[Path], List[Path]]:
        """Check files for EOF newlines and return lists of compliant and non-compliant files."""
        compliant_files = []
        non_compliant_files = []
        
        for file_path in files:
            if self.has_eof_newline(file_path):
                compliant_files.append(file_path)
                if show_details:
                    print(f"✅ {file_path.relative_to(self.project_root)}")
            else:
                non_compliant_files.append(file_path)
                if show_details:
                    print(f"❌ {file_path.relative_to(self.project_root)} - Missing EOF newline")
        
        return compliant_files, non_compliant_files
    
    def fix_files(self, files: List[Path], create_backup: bool = False) -> Tuple[int, int]:
        """Fix EOF newlines in files and return counts of successful and failed fixes."""
        success_count = 0
        error_count = 0
        
        for file_path in files:
            if self.fix_eof_newline(file_path, create_backup):
                success_count += 1
                print(f"✅ Fixed {file_path.relative_to(self.project_root)}")
            else:
                error_count += 1
                print(f"❌ Failed to fix {file_path.relative_to(self.project_root)}")
        
        return success_count, error_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="End-of-file newline verification and enforcement for the Inference Systems Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                                    # Check all text files
  %(prog)s --fix --backup                            # Fix all files with backup
  %(prog)s --check --filter "*.py"                   # Check Python files only
  %(prog)s --check --filter "common/src/*"           # Check specific directory
  %(prog)s --fix --filter-from-file staged_files.txt # Fix files from list
        """
    )
    
    # Action options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",
                      action="store_true",
                      help="Check files for EOF newlines without fixing")
    group.add_argument("--fix",
                      action="store_true",
                      help="Fix missing EOF newlines automatically")
    
    # File selection options
    parser.add_argument("--filter",
                       help="Include only files matching this pattern (supports wildcards)")
    parser.add_argument("--filter-from-file",
                       type=Path,
                       help="Include only files listed in the specified file (one per line)")
    parser.add_argument("--exclude",
                       help="Exclude files/directories matching these patterns (comma-separated)")
    
    # Fix options
    parser.add_argument("--backup",
                       action="store_true",
                       help="Create backup files before fixing (use with --fix)")
    
    # Output options
    parser.add_argument("--show-details",
                       action="store_true",
                       help="Show detailed results for each file")
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
    
    checker = EOFNewlineChecker(project_root)
    
    if not args.quiet:
        print("End-of-file newline checker for Inference Systems Lab")
        print(f"Project root: {project_root}")
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
            files_to_check = []
            for file_path in file_list:
                abs_path = project_root / file_path if not Path(file_path).is_absolute() else Path(file_path)
                if abs_path.exists() and checker.should_check_file(abs_path):
                    files_to_check.append(abs_path)
        except Exception as e:
            print(f"Error reading file list from {args.filter_from_file}: {e}")
            sys.exit(1)
    else:
        # Discover files
        files_to_check = checker.discover_files(include_patterns, exclude_patterns)
    
    if not files_to_check:
        print("No files found matching criteria")
        sys.exit(0)
    
    if not args.quiet:
        print(f"Found {len(files_to_check)} files to check")
        if args.filter:
            print(f"Include pattern: {args.filter}")
        if args.filter_from_file:
            print(f"Files from: {args.filter_from_file}")
        if args.exclude:
            print(f"Exclude patterns: {args.exclude}")
        print()
    
    # Execute action
    if args.check:
        compliant_files, non_compliant_files = checker.check_files(files_to_check, args.show_details)
        
        if not args.quiet:
            print(f"\nResults:")
            print(f"✅ Files with EOF newline: {len(compliant_files)}")
            print(f"❌ Files missing EOF newline: {len(non_compliant_files)}")
        
        if non_compliant_files:
            if not args.show_details and not args.quiet:
                print(f"\nFiles missing EOF newline:")
                for file_path in non_compliant_files[:10]:  # Show first 10
                    print(f"  {file_path.relative_to(project_root)}")
                if len(non_compliant_files) > 10:
                    print(f"  ... and {len(non_compliant_files) - 10} more")
            
            print(f"\nTo fix all files: python tools/check_eof_newline.py --fix")
            if not args.backup:
                print("Add --backup to create backup files before fixing")
            
            sys.exit(1)
        else:
            if not args.quiet:
                print(f"\n✅ All {len(files_to_check)} files have proper EOF newlines")
            sys.exit(0)
    
    elif args.fix:
        # Check which files need fixing
        _, files_to_fix = checker.check_files(files_to_check)
        
        if not files_to_fix:
            print(f"✅ All {len(files_to_check)} files already have EOF newlines")
            sys.exit(0)
        
        if not args.quiet:
            print(f"Fixing EOF newlines in {len(files_to_fix)} files...")
            if args.backup:
                print("Creating backup files with .bak extension")
            print()
        
        success_count, error_count = checker.fix_files(files_to_fix, args.backup)
        
        if not args.quiet:
            print(f"\nResults:")
            print(f"✅ Successfully fixed: {success_count}")
            if error_count > 0:
                print(f"❌ Failed to fix: {error_count}")
        
        if error_count > 0:
            print(f"\n❌ Some files could not be fixed")
            sys.exit(1)
        else:
            print(f"\n✅ Successfully fixed EOF newlines in {success_count} files")
            
            if not args.quiet:
                print("\nRecommended next steps:")
                print("1. Review the changes: git diff")
                print("2. Stage the changes: git add -u")
                print("3. Commit the fixes: git commit -m 'Fix EOF newlines in source files'")
            
            sys.exit(0)


if __name__ == "__main__":
    main()
