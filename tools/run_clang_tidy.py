#!/usr/bin/env python3
"""
Automated clang-tidy runner script for C++ projects.

This script:
1. Finds all .cpp and .hpp files in the project
2. Runs clang-tidy with automatic fixes on each file
3. Creates a git commit with the applied fixes

Usage:
    python3 run_clang_tidy.py [--dry-run] [--config CONFIG_FILE]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import time


class ClangTidyRunner:
    def __init__(self, project_root: Path, clang_tidy_path: str = "/opt/homebrew/Cellar/llvm/20.1.8/bin/clang-tidy"):
        self.project_root = project_root
        self.clang_tidy_path = clang_tidy_path
        self.cpp_extensions = {'.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.h'}
        self.build_dirs = {'build', 'cmake-build-debug', '_deps', 'CMakeFiles'}
        
    def find_cpp_files(self) -> List[Path]:
        """Find all C++ source and header files, excluding build directories."""
        cpp_files = []
        
        for file_path in self.project_root.rglob('*'):
            # Skip build directories and their contents
            if any(build_dir in file_path.parts for build_dir in self.build_dirs):
                continue
                
            # Skip hidden directories and files
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            # Check if it's a C++ file
            if file_path.suffix in self.cpp_extensions and file_path.is_file():
                cpp_files.append(file_path)
        
        return sorted(cpp_files)
    
    def find_clang_tidy_config(self) -> Optional[Path]:
        """Find .clang-tidy configuration file."""
        # Look for .clang-tidy in project root and parent directories
        current_dir = self.project_root
        while current_dir != current_dir.parent:
            config_file = current_dir / '.clang-tidy'
            if config_file.exists():
                return config_file
            current_dir = current_dir.parent
        
        # Check if there's a .clang-tidy in any subdirectory (like benchmark deps)
        for config_file in self.project_root.rglob('.clang-tidy'):
            # Skip build directories
            if any(build_dir in config_file.parts for build_dir in self.build_dirs):
                continue
            return config_file
        
        return None
    
    def create_default_clang_tidy_config(self) -> Path:
        """Create a default .clang-tidy configuration file."""
        config_content = """---
Checks: 'clang-analyzer-*,bugprone-*,cert-*,cppcoreguidelines-*,google-*,hicpp-*,llvm-*,misc-*,modernize-*,performance-*,portability-*,readability-*,-modernize-use-trailing-return-type,-readability-braces-around-statements,-hicpp-braces-around-statements,-google-readability-braces-around-statements'
WarningsAsErrors: ''
HeaderFilterRegex: '.*'
AnalyzeTemporaryDtors: false
FormatStyle: none
"""
        config_file = self.project_root / '.clang-tidy'
        config_file.write_text(config_content)
        print(f"Created default .clang-tidy config at {config_file}")
        return config_file
    
    def run_clang_tidy_on_file(self, file_path: Path, config_file: Optional[Path] = None, dry_run: bool = False) -> bool:
        """Run clang-tidy on a single file with automatic fixes."""
        cmd = [
            self.clang_tidy_path,
            str(file_path),
            '--fix',
            '--fix-errors'
        ]
        
        if config_file:
            cmd.extend(['--config-file', str(config_file)])
        
        try:
            print(f"Processing {file_path.relative_to(self.project_root)}...")
            
            if dry_run:
                print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
                return True
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per file
            )
            
            # clang-tidy returns 0 for success, non-zero for warnings/errors
            # We consider both success and warnings as successful runs
            if result.returncode in [0, 1]:
                if result.stdout.strip():
                    print(f"  Applied fixes")
                return True
            else:
                print(f"  Error processing file: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  Timeout processing {file_path}")
            return False
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            return False
    
    def check_git_status(self) -> bool:
        """Check if we're in a git repository and it's clean."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  cwd=self.project_root, 
                                  capture_output=True)
            if result.returncode != 0:
                print("Error: Not in a git repository")
                return False
            
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  cwd=self.project_root, 
                                  capture_output=True, 
                                  text=True)
            
            if result.stdout.strip():
                print("Warning: Git working directory is not clean")
                print("Uncommitted changes:")
                print(result.stdout)
                response = input("Continue anyway? (y/N): ")
                return response.lower().startswith('y')
            
            return True
            
        except Exception as e:
            print(f"Error checking git status: {e}")
            return False
    
    def create_git_commit(self, dry_run: bool = False) -> bool:
        """Create a git commit with clang-tidy fixes."""
        try:
            # Check if there are any changes to commit
            result = subprocess.run(['git', 'diff', '--name-only'], 
                                  cwd=self.project_root, 
                                  capture_output=True, 
                                  text=True)
            
            if not result.stdout.strip():
                print("No changes to commit")
                return True
            
            changed_files = result.stdout.strip().split('\n')
            print(f"Files modified by clang-tidy: {len(changed_files)}")
            for file in changed_files:
                print(f"  {file}")
            
            if dry_run:
                print("[DRY RUN] Would create commit with clang-tidy fixes")
                return True
            
            # Add all modified files
            subprocess.run(['git', 'add', '-u'], cwd=self.project_root, check=True)
            
            # Create commit
            commit_message = """Apply clang-tidy automatic fixes

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
            
            subprocess.run(['git', 'commit', '-m', commit_message], 
                         cwd=self.project_root, check=True)
            
            print("Successfully created commit with clang-tidy fixes")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating git commit: {e}")
            return False
    
    def run(self, config_file: Optional[Path] = None, dry_run: bool = False) -> bool:
        """Run clang-tidy on all C++ files and create a commit."""
        print(f"Starting clang-tidy run on {self.project_root}")
        
        # Check git status first
        if not dry_run and not self.check_git_status():
            return False
        
        # Find C++ files
        cpp_files = self.find_cpp_files()
        if not cpp_files:
            print("No C++ files found")
            return True
        
        print(f"Found {len(cpp_files)} C++ files to process")
        
        # Find or create clang-tidy config
        if not config_file:
            config_file = self.find_clang_tidy_config()
            if not config_file:
                if not dry_run:
                    config_file = self.create_default_clang_tidy_config()
                else:
                    print("[DRY RUN] Would create default .clang-tidy config")
        
        if config_file:
            print(f"Using clang-tidy config: {config_file}")
        
        # Process each file
        successful_files = 0
        failed_files = 0
        
        start_time = time.time()
        
        for i, file_path in enumerate(cpp_files, 1):
            print(f"[{i}/{len(cpp_files)}] ", end="")
            if self.run_clang_tidy_on_file(file_path, config_file, dry_run):
                successful_files += 1
            else:
                failed_files += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\nCompleted processing {len(cpp_files)} files in {elapsed_time:.1f}s")
        print(f"  Successful: {successful_files}")
        print(f"  Failed: {failed_files}")
        
        # Create git commit if changes were made
        if successful_files > 0 and not dry_run:
            return self.create_git_commit(dry_run)
        elif dry_run:
            print("[DRY RUN] Would check for changes and create commit if needed")
        
        return failed_files == 0


def main():
    parser = argparse.ArgumentParser(description="Run clang-tidy on all C++ files and commit fixes")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--config', type=Path, 
                       help='Path to .clang-tidy configuration file')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).parent.parent,
                       help='Project root directory (default: parent of tools directory)')
    
    args = parser.parse_args()
    
    # Validate project root
    if not args.project_root.exists() or not args.project_root.is_dir():
        print(f"Error: Project root '{args.project_root}' does not exist or is not a directory")
        sys.exit(1)
    
    # Validate config file if provided
    if args.config and not args.config.exists():
        print(f"Error: Config file '{args.config}' does not exist")
        sys.exit(1)
    
    # Check if clang-tidy is available
    clang_tidy_path = "/opt/homebrew/Cellar/llvm/20.1.8/bin/clang-tidy"
    if not Path(clang_tidy_path).exists():
        # Try to find clang-tidy in PATH
        try:
            result = subprocess.run(['which', 'clang-tidy'], capture_output=True, text=True)
            if result.returncode == 0:
                clang_tidy_path = result.stdout.strip()
            else:
                print("Error: clang-tidy not found. Please install LLVM toolchain:")
                print("  brew install llvm")
                sys.exit(1)
        except Exception:
            print("Error: clang-tidy not found")
            sys.exit(1)
    
    # Run clang-tidy
    runner = ClangTidyRunner(args.project_root, clang_tidy_path)
    success = runner.run(args.config, args.dry_run)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()