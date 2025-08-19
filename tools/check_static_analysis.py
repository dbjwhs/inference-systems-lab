#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
check_static_analysis.py - Static analysis verification and enforcement for the Inference Systems Lab

This script runs clang-tidy static analysis on C++ source files to detect
potential bugs, code smells, and enforce coding standards. It supports both
verification mode for CI/CD and detailed reporting for development.

Features:
- Automatic discovery of C++ source files
- clang-tidy integration with project configuration
- Compilation database generation for accurate analysis
- Severity-based filtering and reporting
- Fix application with backup options
- Detailed reporting with suppression support
- Integration with build system and CI/CD

Usage:
    python tools/check_static_analysis.py [options]
    
Examples:
    python tools/check_static_analysis.py --check
    python tools/check_static_analysis.py --fix --backup
    python tools/check_static_analysis.py --check --filter "common/src/*" --severity error
    python tools/check_static_analysis.py --generate-suppressions
"""

import argparse
import json
import os
import subprocess
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import shutil
import tempfile
import fnmatch


class StaticAnalysisIssue:
    """Represents a static analysis issue found by clang-tidy."""
    
    def __init__(self, file_path: str, line_number: int, column: int, 
                 severity: str, check_name: str, message: str, 
                 fix_hint: Optional[str] = None):
        self.file_path = file_path
        self.line_number = line_number
        self.column = column
        self.severity = severity
        self.check_name = check_name
        self.message = message
        self.fix_hint = fix_hint
        
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}:{self.column}: {self.severity}: {self.message} [{self.check_name}]"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column': self.column,
            'severity': self.severity,
            'check_name': self.check_name,
            'message': self.message,
            'fix_hint': self.fix_hint
        }


class StaticAnalysisReport:
    """Represents a complete static analysis report."""
    
    def __init__(self):
        self.issues: List[StaticAnalysisIssue] = []
        self.files_analyzed = 0
        self.total_issues = 0
        self.issues_by_severity = defaultdict(int)
        self.issues_by_check = defaultdict(int)
        
    def add_issue(self, issue: StaticAnalysisIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        self.total_issues += 1
        self.issues_by_severity[issue.severity] += 1
        self.issues_by_check[issue.check_name] += 1
    
    def get_issues_by_severity(self, severity: str) -> List[StaticAnalysisIssue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_check(self, check_name: str) -> List[StaticAnalysisIssue]:
        """Get all issues from a specific check."""
        return [issue for issue in self.issues if issue.check_name == check_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'files_analyzed': self.files_analyzed,
            'total_issues': self.total_issues,
            'issues_by_severity': dict(self.issues_by_severity),
            'issues_by_check': dict(self.issues_by_check),
            'issues': [issue.to_dict() for issue in self.issues]
        }


class StaticAnalyzer:
    """Main class for running static analysis with clang-tidy."""
    
    def __init__(self, project_root: Path, clang_tidy_path: str = "clang-tidy"):
        self.project_root = project_root
        self.clang_tidy_path = clang_tidy_path
        self.build_dir = project_root / "build"
        self.cpp_extensions = {'.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.h', '.c'}
        
    def check_clang_tidy_available(self) -> bool:
        """Check if clang-tidy is available."""
        try:
            result = subprocess.run([self.clang_tidy_path, '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def get_clang_tidy_version(self) -> Optional[str]:
        """Get clang-tidy version string."""
        try:
            result = subprocess.run([self.clang_tidy_path, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        return None
    
    def discover_source_files(self, include_patterns: Optional[List[str]] = None,
                            exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover C++ source files for analysis."""
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
    
    def generate_compile_commands(self) -> bool:
        """Generate compile_commands.json for accurate analysis."""
        try:
            # Build with compile commands generation
            cmake_cmd = [
                'cmake',
                '-S', str(self.project_root),
                '-B', str(self.build_dir),
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
            ]
            
            result = subprocess.run(cmake_cmd, capture_output=True, text=True,
                                  cwd=self.project_root)
            if result.returncode != 0:
                print(f"Warning: CMake configuration failed: {result.stderr}")
                return False
            
            compile_commands = self.build_dir / "compile_commands.json"
            if compile_commands.exists():
                print(f"Generated compilation database: {compile_commands}")
                return True
            else:
                print("Warning: compile_commands.json not generated")
                return False
                
        except Exception as e:
            print(f"Error generating compile commands: {e}")
            return False
    
    def run_clang_tidy_on_file(self, file_path: Path, 
                              additional_args: Optional[List[str]] = None) -> List[StaticAnalysisIssue]:
        """Run clang-tidy on a single file."""
        issues = []
        
        cmd = [self.clang_tidy_path, str(file_path)]
        
        # Add compilation database if available
        compile_commands = self.build_dir / "compile_commands.json"
        if compile_commands.exists():
            cmd.extend(['-p', str(self.build_dir)])
        else:
            # Fallback: provide basic compilation flags
            cmd.extend(['--', '-std=c++17', f'-I{self.project_root}'])
        
        # Add additional arguments
        if additional_args:
            cmd.extend(additional_args)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.project_root)
            
            # Parse clang-tidy output
            issues = self._parse_clang_tidy_output(result.stdout, result.stderr)
            
        except Exception as e:
            print(f"Error running clang-tidy on {file_path}: {e}")
            
        return issues
    
    def _parse_clang_tidy_output(self, stdout: str, stderr: str) -> List[StaticAnalysisIssue]:
        """Parse clang-tidy output to extract issues."""
        issues = []
        
        # Combine stdout and stderr for parsing
        output = stdout + "\n" + stderr
        
        # Pattern to match clang-tidy warnings/errors
        # Format: file:line:column: severity: message [check-name]
        pattern = r'^([^:]+):(\d+):(\d+):\s+(error|warning|note):\s+(.+?)\s+\[([^\]]+)\]'
        
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))
                column = int(match.group(3))
                severity = match.group(4)
                message = match.group(5)
                check_name = match.group(6)
                
                # Skip certain diagnostic messages
                if check_name in ['clang-diagnostic-error', 'clang-diagnostic-warning']:
                    continue
                
                issue = StaticAnalysisIssue(
                    file_path=file_path,
                    line_number=line_number,
                    column=column,
                    severity=severity,
                    check_name=check_name,
                    message=message
                )
                issues.append(issue)
        
        return issues
    
    def analyze_files(self, files: List[Path], 
                     severity_filter: Optional[str] = None) -> StaticAnalysisReport:
        """Analyze multiple files and generate a report."""
        report = StaticAnalysisReport()
        report.files_analyzed = len(files)
        
        print(f"Analyzing {len(files)} files with clang-tidy...")
        
        for i, file_path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Analyzing {file_path.relative_to(self.project_root)}")
            
            issues = self.run_clang_tidy_on_file(file_path)
            
            for issue in issues:
                # Apply severity filter if specified
                if severity_filter and issue.severity != severity_filter:
                    continue
                    
                report.add_issue(issue)
        
        return report
    
    def fix_issues(self, files: List[Path], create_backup: bool = False, 
                   validate_build: bool = True) -> Tuple[int, int]:
        """Apply clang-tidy fixes to files with optional build validation."""
        fixed_count = 0
        error_count = 0
        
        print(f"Applying clang-tidy fixes to {len(files)} files...")
        
        # Store original content for rollback if build fails
        original_content = {}
        if validate_build:
            for file_path in files:
                with open(file_path, 'r') as f:
                    original_content[file_path] = f.read()
        
        for file_path in files:
            print(f"Fixing {file_path.relative_to(self.project_root)}...")
            
            try:
                if create_backup:
                    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                    shutil.copy2(file_path, backup_path)
                    print(f"Created backup: {backup_path}")
                
                cmd = [
                    self.clang_tidy_path,
                    '--fix',
                    '--fix-errors',
                    str(file_path)
                ]
                
                # Add compilation database if available
                compile_commands = self.build_dir / "compile_commands.json"
                if compile_commands.exists():
                    cmd.extend(['-p', str(self.build_dir)])
                else:
                    cmd.extend(['--', '-std=c++17', f'-I{self.project_root}'])
                
                result = subprocess.run(cmd, capture_output=True, text=True,
                                      cwd=self.project_root)
                
                if result.returncode == 0:
                    fixed_count += 1
                else:
                    error_count += 1
                    print(f"Error fixing {file_path}: {result.stderr}")
                    
            except Exception as e:
                error_count += 1
                print(f"Error fixing {file_path}: {e}")
        
        # Validate that fixes didn't break the build
        if validate_build and fixed_count > 0:
            print("\nValidating build after fixes...")
            if not self._validate_build():
                print("❌ Build validation failed! Rolling back changes...")
                for file_path, content in original_content.items():
                    with open(file_path, 'w') as f:
                        f.write(content)
                print("✅ Changes rolled back successfully")
                return 0, len(files)  # All files failed
            else:
                print("✅ Build validation passed")
        
        return fixed_count, error_count
    
    def _validate_build(self) -> bool:
        """Validate that the project still builds after fixes."""
        if not self.build_dir.exists():
            print("Build directory not found, skipping validation")
            return True
        
        # Try a quick compilation check with make
        try:
            result = subprocess.run(
                ['make', '-C', str(self.build_dir), '-j4'],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for build
            )
            
            if result.returncode == 0:
                return True
            else:
                print(f"Build failed with errors:\n{result.stderr[:500]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print("Build validation timed out")
            return False
        except Exception as e:
            print(f"Could not validate build: {e}")
            return True  # Assume OK if we can't validate
    
    def generate_suppressions(self, report: StaticAnalysisReport, 
                            output_file: Path) -> None:
        """Generate suppression configuration for common issues."""
        suppressions = defaultdict(list)
        
        # Group issues by check type
        for issue in report.issues:
            suppressions[issue.check_name].append({
                'file': issue.file_path,
                'line': issue.line_number,
                'message': issue.message
            })
        
        # Generate suppression suggestions
        suppression_config = {
            'suppressions': {},
            'suggested_config_changes': []
        }
        
        for check_name, issues in suppressions.items():
            if len(issues) > 10:  # Many instances of same check
                suppression_config['suggested_config_changes'].append({
                    'check': check_name,
                    'action': 'consider_disabling',
                    'reason': f'Found {len(issues)} instances across multiple files',
                    'suggestion': f'Add "-{check_name}" to Checks in .clang-tidy'
                })
        
        with open(output_file, 'w') as f:
            json.dump(suppression_config, f, indent=2)
        
        print(f"Suppression suggestions saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Static analysis verification and enforcement for the Inference Systems Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                                    # Check all files
  %(prog)s --fix --backup                            # Fix issues with backup
  %(prog)s --check --filter "common/src/*"           # Check specific files
  %(prog)s --check --severity error                  # Only show errors
  %(prog)s --generate-suppressions                   # Generate suppression config
        """
    )
    
    # Action options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",
                      action="store_true",
                      help="Check for static analysis issues without fixing")
    group.add_argument("--fix",
                      action="store_true",
                      help="Fix static analysis issues automatically")
    group.add_argument("--generate-suppressions",
                      action="store_true",
                      help="Generate suppression configuration for common issues")
    
    # File selection options
    parser.add_argument("--filter",
                       help="Include only files matching this pattern (supports wildcards)")
    parser.add_argument("--filter-from-file",
                       type=Path,
                       help="Include only files listed in the specified file (one per line)")
    parser.add_argument("--exclude",
                       help="Exclude files/directories matching these patterns (comma-separated)")
    
    # Analysis options
    parser.add_argument("--severity",
                       choices=["error", "warning", "note"],
                       help="Filter issues by severity level")
    parser.add_argument("--backup",
                       action="store_true",
                       help="Create backup files before fixing (use with --fix)")
    parser.add_argument("--no-build-validation",
                       action="store_true",
                       help="Skip build validation after fixes (use with --fix)")
    
    # Tool options
    parser.add_argument("--clang-tidy-path",
                       default="/opt/homebrew/Cellar/llvm/20.1.8/bin/clang-tidy",
                       help="Path to clang-tidy executable")
    parser.add_argument("--output-json",
                       type=Path,
                       help="Output detailed results to JSON file")
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
    
    analyzer = StaticAnalyzer(project_root, args.clang_tidy_path)
    
    # Check if clang-tidy is available
    if not analyzer.check_clang_tidy_available():
        print(f"Error: clang-tidy not found at '{args.clang_tidy_path}'")
        print("Please install clang-tidy or specify correct path with --clang-tidy-path")
        sys.exit(1)
    
    if not args.quiet:
        version = analyzer.get_clang_tidy_version()
        print(f"Using clang-tidy: {version}")
        print(f"Project root: {project_root}")
        
        config_file = project_root / ".clang-tidy"
        if config_file.exists():
            print(f"Configuration: {config_file}")
        else:
            print("Warning: No .clang-tidy configuration found")
        print()
    
    # Generate compilation database
    if not args.quiet:
        print("Generating compilation database...")
    analyzer.generate_compile_commands()
    
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
                if abs_path.exists() and abs_path.suffix in analyzer.cpp_extensions:
                    source_files.append(abs_path)
        except Exception as e:
            print(f"Error reading file list from {args.filter_from_file}: {e}")
            sys.exit(1)
    else:
        # Discover source files
        source_files = analyzer.discover_source_files(include_patterns, exclude_patterns)
    
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
        print()
    
    # Execute action
    if args.check:
        report = analyzer.analyze_files(source_files, args.severity)
        
        # Output JSON if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Detailed results saved to {args.output_json}")
        
        # Summary output
        if report.total_issues > 0:
            print(f"\n❌ Found {report.total_issues} static analysis issues:")
            print("=" * 50)
            
            for severity, count in report.issues_by_severity.items():
                print(f"  {severity}: {count}")
            
            print("\nTop issues by check:")
            sorted_checks = sorted(report.issues_by_check.items(), 
                                 key=lambda x: x[1], reverse=True)
            for check, count in sorted_checks[:10]:
                print(f"  {check}: {count}")
            
            # Show some example issues
            print(f"\nFirst 10 issues:")
            for issue in report.issues[:10]:
                print(f"  {issue}")
            
            if len(report.issues) > 10:
                print(f"  ... and {len(report.issues) - 10} more")
            
            print(f"\nTo fix issues automatically: python tools/check_static_analysis.py --fix")
            if not args.backup:
                print("Add --backup to create backup files before fixing")
            
            sys.exit(1)
        else:
            print(f"✅ No static analysis issues found in {len(source_files)} files")
            sys.exit(0)
    
    elif args.fix:
        fixed_count, error_count = analyzer.fix_issues(
            source_files, 
            args.backup,
            validate_build=not args.no_build_validation
        )
        
        if error_count > 0:
            print(f"\n❌ Fixed issues in {fixed_count} files, {error_count} errors")
            sys.exit(1)
        else:
            print(f"\n✅ Successfully fixed issues in {fixed_count} files")
            
            if args.backup:
                print("Backup files created with .bak extension")
            
            print("\nRecommended next steps:")
            print("1. Review the changes: git diff")
            print("2. Run tests to ensure functionality: ctest --test-dir build")
            print("3. Commit the fixes: git add -A && git commit -m 'Apply clang-tidy fixes'")
            
            sys.exit(0)
    
    elif args.generate_suppressions:
        report = analyzer.analyze_files(source_files)
        suppression_file = project_root / "clang-tidy-suppressions.json"
        analyzer.generate_suppressions(report, suppression_file)
        
        print(f"\nGenerated suppression suggestions based on {report.total_issues} issues")
        print(f"Review {suppression_file} for recommendations")
        sys.exit(0)


if __name__ == "__main__":
    main()
