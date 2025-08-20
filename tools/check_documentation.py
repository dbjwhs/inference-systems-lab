#!/usr/bin/env python3
"""
check_documentation.py - Comprehensive Documentation Generation and Validation

This tool provides automated documentation generation using Doxygen, validates
documentation coverage, and integrates with CI/CD pipelines for continuous
documentation updates.

Key Features:
- Automated Doxygen documentation generation with error handling
- Documentation coverage analysis for undocumented public APIs
- Integration with existing build system and quality gates
- CI/CD friendly with structured output and exit codes
- Performance optimized for large codebases

Usage Examples:
    python3 tools/check_documentation.py --generate          # Generate docs only
    python3 tools/check_documentation.py --check            # Validate coverage
    python3 tools/check_documentation.py --generate --check  # Full workflow
    python3 tools/check_documentation.py --clean            # Clean generated docs
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

class DocumentationResult(NamedTuple):
    """Result of documentation generation or analysis"""
    success: bool
    warnings: int
    errors: int
    coverage_percent: float
    generated_files: int
    build_time_seconds: float
    output_dir: Optional[str] = None

class UndocumentedItem(NamedTuple):
    """Represents an undocumented API item"""
    file_path: str
    line_number: int
    item_type: str  # 'class', 'function', 'method', 'enum', etc.
    item_name: str
    context: str

class DocumentationChecker:
    """Main class for documentation generation and validation"""
    
    def __init__(self, project_root: str, build_dir: str = "build"):
        self.project_root = Path(project_root).resolve()
        self.build_dir = self.project_root / build_dir
        self.docs_output_dir = self.build_dir / "docs"
        self.doxygen_config = self.project_root / "docs" / "Doxyfile.in"
        self.processed_doxyfile = self.build_dir / "Doxyfile"
        
        # Documentation coverage thresholds
        self.coverage_threshold = 80.0
        self.warning_threshold = 90.0
        
        # File patterns for documentation analysis
        self.header_patterns = ["*.hpp", "*.h"]
        self.source_patterns = ["*.cpp", "*.cc", "*.cxx"]
        self.exclude_patterns = [
            "build*", "test_build*", ".git", 
            "*test*", "*benchmark*", "*/third_party/*"
        ]
    
    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check if all required tools are available"""
        issues = []
        
        # Check for Doxygen
        try:
            result = subprocess.run(['doxygen', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                issues.append("Doxygen not found or not working")
        except FileNotFoundError:
            issues.append("Doxygen not installed (required for documentation generation)")
        
        # Check for dot (Graphviz) for diagrams
        try:
            subprocess.run(['dot', '-V'], capture_output=True)
        except FileNotFoundError:
            issues.append("Graphviz/dot not found (optional, for diagrams)")
        
        # Check for CMake (for configuration processing)
        try:
            subprocess.run(['cmake', '--version'], capture_output=True)
        except FileNotFoundError:
            issues.append("CMake not found (required for configuration processing)")
        
        # Check if Doxygen config exists
        if not self.doxygen_config.exists():
            issues.append(f"Doxygen configuration not found: {self.doxygen_config}")
        
        return len(issues) == 0, issues
    
    def generate_documentation(self, clean_first: bool = False) -> DocumentationResult:
        """Generate Doxygen documentation"""
        start_time = time.time()
        
        if clean_first:
            self.clean_documentation()
        
        # Ensure build directory exists
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure project with CMake to process Doxyfile.in
        print("🔧 Configuring CMake to process Doxygen configuration...")
        cmake_result = subprocess.run([
            'cmake', 
            f'-S{self.project_root}', 
            f'-B{self.build_dir}',
            '-DCMAKE_BUILD_TYPE=Release'
        ], capture_output=True, text=True)
        
        if cmake_result.returncode != 0:
            print(f"❌ CMake configuration failed:")
            print(cmake_result.stderr)
            return DocumentationResult(
                success=False, warnings=0, errors=1, coverage_percent=0.0,
                generated_files=0, build_time_seconds=time.time() - start_time
            )
        
        # Generate documentation using CMake target
        print("📖 Generating Doxygen documentation...")
        docs_result = subprocess.run([
            'cmake', '--build', str(self.build_dir), '--target', 'docs'
        ], capture_output=True, text=True)
        
        build_time = time.time() - start_time
        
        # Parse Doxygen output for warnings and errors
        warnings = self._count_doxygen_warnings(docs_result.stderr)
        errors = 1 if docs_result.returncode != 0 else 0
        
        if docs_result.returncode == 0:
            generated_files = self._count_generated_files()
            print(f"✅ Documentation generated successfully in {build_time:.1f}s")
            print(f"📁 Output directory: {self.docs_output_dir}")
            print(f"📄 Generated {generated_files} documentation files")
            if warnings > 0:
                print(f"⚠️  {warnings} warnings found")
        else:
            print(f"❌ Documentation generation failed:")
            print(docs_result.stderr)
            generated_files = 0
        
        return DocumentationResult(
            success=docs_result.returncode == 0,
            warnings=warnings,
            errors=errors,
            coverage_percent=0.0,  # Will be calculated separately
            generated_files=generated_files,
            build_time_seconds=build_time,
            output_dir=str(self.docs_output_dir) if docs_result.returncode == 0 else None
        )
    
    def check_coverage(self) -> Tuple[float, List[UndocumentedItem]]:
        """Analyze documentation coverage of public APIs"""
        print("📊 Analyzing documentation coverage...")
        
        undocumented_items = []
        total_items = 0
        documented_items = 0
        
        # Find all header files in common and engines directories
        header_files = []
        for directory in ['common/src', 'engines/src']:
            dir_path = self.project_root / directory
            if dir_path.exists():
                header_files.extend(dir_path.rglob('*.hpp'))
                header_files.extend(dir_path.rglob('*.h'))
        
        # Analyze each header file
        for header_file in header_files:
            if self._should_skip_file(header_file):
                continue
                
            items, documented = self._analyze_file_documentation(header_file)
            total_items += len(items)
            documented_items += documented
            
            # Add undocumented items to the list
            for item in items:
                if not self._is_documented(header_file, item):
                    undocumented_items.append(UndocumentedItem(
                        file_path=str(header_file.relative_to(self.project_root)),
                        line_number=item.get('line', 0),
                        item_type=item['type'],
                        item_name=item['name'],
                        context=item.get('context', '')
                    ))
        
        coverage_percent = (documented_items / total_items * 100) if total_items > 0 else 100.0
        
        print(f"📈 Documentation Coverage: {coverage_percent:.1f}% ({documented_items}/{total_items})")
        if len(undocumented_items) > 0:
            print(f"⚠️  Found {len(undocumented_items)} undocumented public API items")
        
        return coverage_percent, undocumented_items
    
    def clean_documentation(self) -> bool:
        """Remove generated documentation files"""
        print("🧹 Cleaning generated documentation...")
        
        if self.docs_output_dir.exists():
            import shutil
            shutil.rmtree(self.docs_output_dir)
            print(f"✅ Cleaned documentation directory: {self.docs_output_dir}")
            return True
        else:
            print("ℹ️  No documentation to clean")
            return False
    
    def validate_links(self) -> Tuple[bool, List[str]]:
        """Validate internal documentation links"""
        print("🔗 Validating documentation links...")
        
        broken_links = []
        
        if not (self.docs_output_dir / "html").exists():
            return False, ["Documentation not generated yet"]
        
        # Simple validation - check if key files exist
        key_files = ["index.html", "annotated.html", "files.html", "hierarchy.html"]
        for file_name in key_files:
            file_path = self.docs_output_dir / "html" / file_name
            if not file_path.exists():
                broken_links.append(f"Missing key documentation file: {file_name}")
        
        if len(broken_links) == 0:
            print("✅ Documentation links validation passed")
        else:
            print(f"❌ Found {len(broken_links)} link issues")
        
        return len(broken_links) == 0, broken_links
    
    def generate_coverage_report(self, undocumented_items: List[UndocumentedItem], 
                               output_file: Optional[str] = None) -> str:
        """Generate detailed coverage report"""
        report_lines = [
            "# Documentation Coverage Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Undocumented Items**: {len(undocumented_items)}",
            ""
        ]
        
        if undocumented_items:
            report_lines.extend([
                "## Undocumented Items",
                "",
                "| File | Line | Type | Name |",
                "|------|------|------|------|"
            ])
            
            for item in sorted(undocumented_items, key=lambda x: (x.file_path, x.line_number)):
                report_lines.append(
                    f"| {item.file_path} | {item.line_number} | {item.item_type} | `{item.item_name}` |"
                )
        else:
            report_lines.append("🎉 **All public APIs are documented!**")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            print(f"📄 Coverage report saved to: {output_file}")
        
        return report_content
    
    def _count_doxygen_warnings(self, stderr_output: str) -> int:
        """Count warnings in Doxygen output"""
        if not stderr_output:
            return 0
        
        # Doxygen warnings typically contain "warning:" keyword
        warning_pattern = re.compile(r'warning:', re.IGNORECASE)
        return len(warning_pattern.findall(stderr_output))
    
    def _count_generated_files(self) -> int:
        """Count generated documentation files"""
        if not self.docs_output_dir.exists():
            return 0
        
        count = 0
        for file_path in self.docs_output_dir.rglob('*'):
            if file_path.is_file():
                count += 1
        
        return count
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped for documentation analysis"""
        file_str = str(file_path)
        
        for pattern in self.exclude_patterns:
            if pattern.replace('*', '') in file_str:
                return True
        
        return False
    
    def _analyze_file_documentation(self, file_path: Path) -> Tuple[List[Dict], int]:
        """Analyze documentation in a single file"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return [], 0
        
        items = []
        documented_count = 0
        
        # Simple regex patterns for C++ constructs
        patterns = {
            'class': r'class\s+(\w+)',
            'struct': r'struct\s+(\w+)', 
            'function': r'^\s*(?:inline\s+)?(?:static\s+)?(?:virtual\s+)?(?:constexpr\s+)?(?:\w+(?:<[^>]*>)?\s+)+(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*(?:=\s*(?:default|delete))?\s*[;{]',
            'enum': r'enum\s+(?:class\s+)?(\w+)',
            'namespace': r'namespace\s+(\w+)'
        }
        
        lines = content.split('\n')
        in_doc_comment = False
        last_doc_line = -10  # Track recent documentation comments
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Track documentation comments
            if '/**' in stripped_line or stripped_line.startswith('/**'):
                in_doc_comment = True
                last_doc_line = i
            elif '*/' in stripped_line and in_doc_comment:
                in_doc_comment = False
                last_doc_line = i
            elif in_doc_comment or stripped_line.startswith('///') or stripped_line.startswith('//!'):
                last_doc_line = i
            
            # Check for documented items
            for item_type, pattern in patterns.items():
                matches = re.finditer(pattern, line, re.MULTILINE)
                for match in matches:
                    item_name = match.group(1)
                    
                    # Skip private/implementation details
                    if item_name.startswith('_') or 'private:' in line:
                        continue
                    
                    items.append({
                        'type': item_type,
                        'name': item_name,
                        'line': i,
                        'context': line.strip()
                    })
                    
                    # Check if documented (has documentation within 5 lines above)
                    if i - last_doc_line <= 5:
                        documented_count += 1
        
        return items, documented_count
    
    def _is_documented(self, file_path: Path, item: Dict) -> bool:
        """Check if a specific item is documented"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return False
        
        lines = content.split('\n')
        item_line = item.get('line', 1) - 1
        
        # Check 5 lines before the item for documentation
        start_line = max(0, item_line - 5)
        for i in range(start_line, item_line):
            if i < len(lines):
                line = lines[i].strip()
                if ('/**' in line or line.startswith('///') or 
                    line.startswith('//!') or line.startswith('*')):
                    return True
        
        return False

def create_documentation_workflow(checker: DocumentationChecker, 
                                generate: bool = True, 
                                check_coverage: bool = True,
                                clean_first: bool = False) -> bool:
    """Complete documentation workflow"""
    print("📚 Starting documentation workflow...")
    
    # Check prerequisites
    prereqs_ok, issues = checker.check_prerequisites()
    if not prereqs_ok:
        print("❌ Prerequisites not met:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    success = True
    
    # Generate documentation
    if generate:
        result = checker.generate_documentation(clean_first=clean_first)
        success &= result.success
        
        if result.success and result.warnings > checker.warning_threshold:
            print(f"⚠️  High warning count: {result.warnings}")
    
    # Check coverage
    if check_coverage:
        coverage, undocumented = checker.check_coverage()
        
        if coverage < checker.coverage_threshold:
            print(f"❌ Documentation coverage below threshold: {coverage:.1f}% < {checker.coverage_threshold}%")
            success = False
        else:
            print(f"✅ Documentation coverage meets threshold: {coverage:.1f}%")
        
        # Generate coverage report
        if undocumented:
            report_file = checker.build_dir / "documentation_coverage_report.md"
            checker.generate_coverage_report(undocumented, str(report_file))
    
    # Validate links
    if generate:
        links_ok, broken_links = checker.validate_links()
        success &= links_ok
        
        if broken_links:
            for link in broken_links:
                print(f"   ❌ {link}")
    
    return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive documentation generation and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--generate', action='store_true',
                       help='Generate Doxygen documentation')
    parser.add_argument('--check', action='store_true', 
                       help='Check documentation coverage')
    parser.add_argument('--clean', action='store_true',
                       help='Clean generated documentation')
    parser.add_argument('--build-dir', default='build',
                       help='Build directory (default: build)')
    parser.add_argument('--coverage-threshold', type=float, default=80.0,
                       help='Documentation coverage threshold (default: 80.0)')
    parser.add_argument('--json-output', 
                       help='Output results as JSON to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Default to generate and check if no specific action
    if not any([args.generate, args.check, args.clean]):
        args.generate = True
        args.check = True
    
    # Setup checker
    project_root = Path(__file__).parent.parent
    checker = DocumentationChecker(str(project_root), args.build_dir)
    checker.coverage_threshold = args.coverage_threshold
    
    if not args.quiet:
        print("📚 Inference Systems Laboratory Documentation Tool")
        print(f"🏗️  Project root: {project_root}")
        print(f"📁 Build directory: {checker.build_dir}")
    
    success = True
    
    # Handle clean operation
    if args.clean:
        checker.clean_documentation()
        if not (args.generate or args.check):
            return 0
    
    # Run main workflow
    if args.generate or args.check:
        success = create_documentation_workflow(
            checker, 
            generate=args.generate,
            check_coverage=args.check,
            clean_first=args.clean
        )
    
    # Output results
    if success:
        print("✅ Documentation workflow completed successfully")
        return 0
    else:
        print("❌ Documentation workflow failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())