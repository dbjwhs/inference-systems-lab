#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
fix_static_analysis_by_file.py - Systematic file-by-file static analysis fixing

This script helps tackle static analysis issues systematically by focusing on one file
at a time, starting with quick wins and building momentum toward larger files.

Usage:
    python tools/fix_static_analysis_by_file.py --list-files              # Show files by complexity
    python tools/fix_static_analysis_by_file.py --fix-file <filename>     # Fix specific file
    python tools/fix_static_analysis_by_file.py --next-easy               # Fix next easiest file
    python tools/fix_static_analysis_by_file.py --phase 1                 # Show phase 1 files
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import os

class StaticAnalysisFileFixer:
    """Manages systematic file-by-file static analysis fixing."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report_file = project_root / "static_analysis_report.json"
        
        # Issue difficulty mapping (minutes per issue)
        self.difficulty_map = {
            'readability-identifier-naming': 2,
            'modernize-use-trailing-return-type': 2,
            'misc-use-anonymous-namespace': 2,
            'readability-convert-member-functions-to-static': 2,
            'misc-include-cleaner': 2,
            'google-build-using-namespace': 2,
            'cppcoreguidelines-avoid-non-const-global-variables': 10,
            'misc-const-correctness': 10,
            'cppcoreguidelines-pro-type-member-init': 10,
            'cppcoreguidelines-missing-std-forward': 10,
            'performance-enum-size': 10,
            'bugprone-empty-catch': 30,
            'bugprone-branch-clone': 30,
        }
        
        # Phase definitions based on analysis
        self.phases = {
            1: {
                'name': 'Quick Wins',
                'description': 'Files with ≤10 issues, build momentum',
                'max_issues': 10,
                'estimated_hours': 2
            },
            2: {
                'name': 'Medium Files', 
                'description': 'Files with 11-50 issues, manageable scope',
                'max_issues': 50,
                'estimated_hours': 8
            },
            3: {
                'name': 'Large Headers',
                'description': 'Header files 51+ issues, maximize impact',
                'max_issues': float('inf'),
                'estimated_hours': 20,
                'filter': 'headers'
            },
            4: {
                'name': 'Large Implementation',
                'description': 'Implementation files 51+ issues, most complex',
                'max_issues': float('inf'),
                'estimated_hours': 65,
                'filter': 'implementation'
            }
        }
    
    def load_analysis_report(self) -> Dict:
        """Load the static analysis JSON report."""
        if not self.report_file.exists():
            print(f"Error: Report file {self.report_file} not found.")
            print("Run: python tools/check_static_analysis.py --check --output-json static_analysis_report.json")
            sys.exit(1)
            
        with open(self.report_file) as f:
            return json.load(f)
    
    def categorize_files(self) -> Dict[int, List[Tuple[str, int, float]]]:
        """Categorize files by phase based on issue count and type."""
        report = self.load_analysis_report()
        files_by_phase = {1: [], 2: [], 3: [], 4: []}
        
        # Group issues by file
        files_data = {}
        for issue in report.get('issues', []):
            file_path = issue.get('file_path', '')
            if file_path not in files_data:
                files_data[file_path] = []
            files_data[file_path].append(issue)
        
        for file_path, issues in files_data.items():
            issue_count = len(issues)
            
            # Calculate estimated effort
            effort_minutes = 0
            for issue in issues:
                check_name = issue.get('check_name', 'unknown')
                difficulty = self.difficulty_map.get(check_name, 15)  # Default 15 min
                effort_minutes += difficulty
            
            effort_hours = effort_minutes / 60.0
            
            # Determine phase
            is_header = file_path.endswith(('.hpp', '.h'))
            
            if issue_count <= 10:
                phase = 1
            elif issue_count <= 50:
                phase = 2
            elif is_header:
                phase = 3
            else:
                phase = 4
            
            files_by_phase[phase].append((file_path, issue_count, effort_hours))
        
        # Sort each phase by effort (easiest first)
        for phase in files_by_phase:
            files_by_phase[phase].sort(key=lambda x: x[2])  # Sort by effort hours
        
        return files_by_phase
    
    def list_files_by_phase(self, phase: int = None):
        """List files categorized by fixing phases."""
        files_by_phase = self.categorize_files()
        
        if phase:
            phases_to_show = [phase]
        else:
            phases_to_show = [1, 2, 3, 4]
        
        total_issues = 0
        total_effort = 0.0
        
        for p in phases_to_show:
            phase_info = self.phases[p]
            files = files_by_phase[p]
            
            if not files:
                continue
                
            phase_issues = sum(f[1] for f in files)
            phase_effort = sum(f[2] for f in files)
            
            print(f"\n🔧 Phase {p}: {phase_info['name']}")
            print(f"   {phase_info['description']}")
            print(f"   📊 {len(files)} files, {phase_issues} issues, ~{phase_effort:.1f} hours")
            print("   " + "="*60)
            
            for i, (file_path, issues, effort) in enumerate(files, 1):
                rel_path = Path(file_path).relative_to(self.project_root)
                status = "🟢" if effort < 1 else "🟡" if effort < 5 else "🔴"
                print(f"   {i:2d}. {status} {rel_path}")
                print(f"       {issues:3d} issues, ~{effort:.1f}h")
            
            total_issues += phase_issues
            total_effort += phase_effort
        
        print(f"\n📈 Summary: {total_issues} total issues, ~{total_effort:.1f} total hours")
    
    def get_next_easy_file(self) -> str:
        """Get the next easiest file to fix."""
        files_by_phase = self.categorize_files()
        
        # Find the easiest file across all phases
        all_files = []
        for phase_files in files_by_phase.values():
            all_files.extend(phase_files)
        
        if not all_files:
            return None
            
        # Sort by effort and return the easiest
        all_files.sort(key=lambda x: x[2])
        return all_files[0][0]
    
    def fix_file(self, file_path: str, backup: bool = True, test_build: bool = True):
        """Fix static analysis issues in a specific file."""
        rel_path = Path(file_path).relative_to(self.project_root)
        print(f"🔧 Fixing static analysis issues in: {rel_path}")
        
        # Run static analysis fix on specific file
        cmd = [
            'python3', 'tools/check_static_analysis.py',
            '--fix',
            '--filter', str(rel_path),
            '--quiet'
        ]
        
        if backup:
            cmd.append('--backup')
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Static analysis fixes applied successfully")
                
                # Test build if requested
                if test_build:
                    print("🔨 Testing build...")
                    build_result = subprocess.run(['make'], cwd=self.project_root / 'build', 
                                                 capture_output=True, text=True)
                    
                    if build_result.returncode == 0:
                        print("✅ Build successful")
                    else:
                        print("❌ Build failed! You may need to fix compilation errors manually.")
                        print("Build output:")
                        print(build_result.stderr)
                        return False
                
                # Show remaining issues
                check_cmd = [
                    'python3', 'tools/check_static_analysis.py',
                    '--check',
                    '--filter', str(rel_path),
                    '--quiet'
                ]
                
                check_result = subprocess.run(check_cmd, cwd=self.project_root, capture_output=True, text=True)
                
                if "Found 0 static analysis issues" in check_result.stderr:
                    print("🎉 All issues in this file have been resolved!")
                else:
                    # Extract issue count from output
                    for line in check_result.stderr.split('\n'):
                        if 'Found' in line and 'issues' in line:
                            print(f"📊 {line}")
                            break
                
                return True
                
            else:
                print("❌ Error during static analysis fix:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running static analysis fix: {e}")
            return False
    
    def show_file_details(self, file_path: str):
        """Show detailed analysis of issues in a specific file."""
        report = self.load_analysis_report()
        
        # Find issues for this file
        issues = []
        for issue in report.get('issues', []):
            if issue.get('file_path') == file_path:
                issues.append(issue)
        
        if not issues:
            rel_path = Path(file_path).relative_to(self.project_root)
            print(f"❌ No analysis data found for {rel_path}")
            return
        
        rel_path = Path(file_path).relative_to(self.project_root)
        print(f"\n📋 Analysis for: {rel_path}")
        print(f"📊 Total issues: {len(issues)}")
        
        # Group by check type
        by_check = {}
        total_effort = 0
        
        for issue in issues:
            check = issue.get('check_name', 'unknown')
            if check not in by_check:
                by_check[check] = []
            by_check[check].append(issue)
            
            # Calculate effort
            difficulty = self.difficulty_map.get(check, 15)
            total_effort += difficulty
        
        print(f"⏱️  Estimated effort: {total_effort/60:.1f} hours")
        print("\n🔍 Issues by type:")
        
        for check, check_issues in sorted(by_check.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(check_issues)
            difficulty = self.difficulty_map.get(check, 15)
            effort = (count * difficulty) / 60.0
            
            print(f"   • {check}: {count} issues (~{effort:.1f}h)")
            
            # Show first few examples
            for i, issue in enumerate(check_issues[:3]):
                line = issue.get('line_number', 'unknown')
                msg = issue.get('message', 'No message')[:60]
                print(f"     {line:4}: {msg}...")
            
            if len(check_issues) > 3:
                print(f"     ... and {len(check_issues) - 3} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Systematic file-by-file static analysis fixing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-files                    # Show all files by complexity
  %(prog)s --phase 1                       # Show phase 1 (quick wins) files  
  %(prog)s --next-easy                     # Show next easiest file to fix
  %(prog)s --fix-file utils.cpp            # Fix specific file
  %(prog)s --details utils.cpp             # Show detailed analysis of file
        """
    )
    
    # Action options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-files", action="store_true",
                      help="List all files categorized by fixing phases")
    group.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                      help="Show files in specific phase")
    group.add_argument("--next-easy", action="store_true",
                      help="Show the next easiest file to fix")
    group.add_argument("--fix-file", 
                      help="Fix static analysis issues in specific file")
    group.add_argument("--details",
                      help="Show detailed analysis of specific file")
    
    # Options
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup files when fixing")
    parser.add_argument("--no-build", action="store_true",
                       help="Don't test build after fixing files")
    
    args = parser.parse_args()
    
    # Determine project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    fixer = StaticAnalysisFileFixer(project_root)
    
    if args.list_files:
        fixer.list_files_by_phase()
    elif args.phase:
        fixer.list_files_by_phase(args.phase)
    elif args.next_easy:
        next_file = fixer.get_next_easy_file()
        if next_file:
            rel_path = Path(next_file).relative_to(project_root)
            print(f"🎯 Next easiest file to fix: {rel_path}")
            fixer.show_file_details(next_file)
        else:
            print("🎉 No files with static analysis issues found!")
    elif args.fix_file:
        file_path = project_root / args.fix_file
        if not file_path.exists():
            print(f"❌ File not found: {args.fix_file}")
            sys.exit(1)
        fixer.fix_file(str(file_path), backup=not args.no_backup, test_build=not args.no_build)
    elif args.details:
        file_path = project_root / args.details
        if not file_path.exists():
            print(f"❌ File not found: {args.details}")
            sys.exit(1)
        fixer.show_file_details(str(file_path))


if __name__ == "__main__":
    main()
