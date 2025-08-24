#!/usr/bin/env python3
"""
Coverage Tracking and Trend Analysis Tool
Tracks test coverage over time and generates reports
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class CoverageTracker:
    """Tracks coverage metrics and trends over time"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build"
        self.coverage_dir = project_root / "coverage"
        self.coverage_dir.mkdir(exist_ok=True)
        self.history_file = self.coverage_dir / "coverage_history.json"
        
    def run_coverage(self) -> Dict[str, any]:
        """Run coverage analysis and parse results"""
        try:
            # Run coverage summary
            result = subprocess.run(
                ["make", "coverage-summary"],
                cwd=self.build_dir,
                capture_output=True,
                text=True
            )
            
            # Parse the output
            coverage_data = self._parse_coverage_output(result.stdout)
            coverage_data['timestamp'] = datetime.now().isoformat()
            coverage_data['commit'] = self._get_git_commit()
            
            return coverage_data
            
        except Exception as e:
            print(f"Error running coverage: {e}")
            return {}
    
    def _parse_coverage_output(self, output: str) -> Dict[str, any]:
        """Parse gcovr coverage output"""
        data = {
            'files': {},
            'total': {'lines': 0, 'executed': 0, 'coverage': 0.0}
        }
        
        # Parse individual file coverage
        file_pattern = r'^([\w/\._-]+\.(?:cpp|hpp))\s+(\d+)\s+(\d+)\s+(\d+)%'
        for match in re.finditer(file_pattern, output, re.MULTILINE):
            filepath, lines, executed, coverage = match.groups()
            data['files'][filepath] = {
                'lines': int(lines),
                'executed': int(executed),
                'coverage': int(coverage)
            }
        
        # Parse total coverage
        total_pattern = r'^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%'
        match = re.search(total_pattern, output, re.MULTILINE)
        if match:
            lines, executed, coverage = match.groups()
            data['total'] = {
                'lines': int(lines),
                'executed': int(executed),
                'coverage': int(coverage)
            }
        
        return data
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()[:8]
        except:
            return "unknown"
    
    def save_coverage_data(self, data: Dict[str, any]):
        """Save coverage data to history file"""
        history = self.load_history()
        history.append(data)
        
        # Keep last 100 entries
        if len(history) > 100:
            history = history[-100:]
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_history(self) -> List[Dict[str, any]]:
        """Load coverage history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def generate_report(self) -> str:
        """Generate coverage report with trends"""
        history = self.load_history()
        if not history:
            return "No coverage history available"
        
        current = history[-1] if history else None
        previous = history[-2] if len(history) > 1 else None
        
        report = []
        report.append("=" * 80)
        report.append("COVERAGE REPORT - Inference Systems Laboratory")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Commit: {current.get('commit', 'unknown')}")
        report.append("")
        
        # Current coverage
        total = current.get('total', {})
        report.append(f"Overall Coverage: {total.get('coverage', 0)}%")
        report.append(f"Lines Covered: {total.get('executed', 0)}/{total.get('lines', 0)}")
        report.append("")
        
        # Trend analysis
        if previous:
            prev_total = previous.get('total', {})
            coverage_delta = total.get('coverage', 0) - prev_total.get('coverage', 0)
            trend = "↑" if coverage_delta > 0 else "↓" if coverage_delta < 0 else "→"
            report.append(f"Trend: {trend} {abs(coverage_delta):.1f}% from previous run")
            report.append("")
        
        # File breakdown
        report.append("FILE COVERAGE BREAKDOWN")
        report.append("-" * 80)
        report.append(f"{'File':<50} {'Lines':>8} {'Exec':>8} {'Cover':>8}")
        report.append("-" * 80)
        
        files = current.get('files', {})
        # Sort by coverage percentage
        sorted_files = sorted(files.items(), key=lambda x: x[1]['coverage'])
        
        # Show worst covered files
        report.append("\nLeast Covered Files:")
        for filepath, stats in sorted_files[:10]:
            filename = Path(filepath).name
            report.append(f"{filename:<50} {stats['lines']:>8} {stats['executed']:>8} {stats['coverage']:>7}%")
        
        # Show best covered files
        report.append("\nBest Covered Files:")
        for filepath, stats in sorted_files[-10:]:
            filename = Path(filepath).name
            report.append(f"{filename:<50} {stats['lines']:>8} {stats['executed']:>8} {stats['coverage']:>7}%")
        
        # Coverage gaps
        report.append("\n" + "=" * 80)
        report.append("COVERAGE GAPS - Files Below 70%")
        report.append("-" * 80)
        
        gaps = [(f, s) for f, s in files.items() if s['coverage'] < 70]
        if gaps:
            for filepath, stats in sorted(gaps, key=lambda x: x[1]['coverage']):
                report.append(f"  {filepath}: {stats['coverage']}% ({stats['lines'] - stats['executed']} lines uncovered)")
        else:
            report.append("  No files below 70% coverage!")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        
        if total.get('coverage', 0) >= 80:
            report.append("✅ Excellent! Maintaining 80%+ coverage")
        else:
            report.append(f"⚠️  Coverage below 80% target ({total.get('coverage', 0)}%)")
        
        # Find files that need attention
        zero_coverage = [f for f, s in files.items() if s['coverage'] == 0]
        if zero_coverage:
            report.append(f"\n❌ {len(zero_coverage)} files with 0% coverage:")
            for f in zero_coverage[:5]:
                report.append(f"   - {f}")
        
        low_coverage = [f for f, s in files.items() if 0 < s['coverage'] < 50]
        if low_coverage:
            report.append(f"\n⚠️  {len(low_coverage)} files with <50% coverage:")
            for f in low_coverage[:5]:
                report.append(f"   - {f}: {files[f]['coverage']}%")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def generate_html_report(self):
        """Generate HTML coverage trend report"""
        history = self.load_history()
        if not history:
            return
        
        html = []
        html.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Coverage Trends - Inference Systems Lab</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .metric-label { color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
        th { background: #4CAF50; color: white; padding: 10px; text-align: left; }
        td { padding: 8px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .good { color: #4CAF50; }
        .warning { color: #FF9800; }
        .bad { color: #F44336; }
        .trend-chart { margin: 20px 0; padding: 20px; background: white; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>Coverage Trends - Inference Systems Laboratory</h1>
""")
        
        current = history[-1]
        total = current.get('total', {})
        
        # Summary metrics
        html.append('<div class="metrics">')
        html.append(f'<div class="metric"><div class="metric-value">{total.get("coverage", 0)}%</div><div class="metric-label">Overall Coverage</div></div>')
        html.append(f'<div class="metric"><div class="metric-value">{total.get("executed", 0)}</div><div class="metric-label">Lines Covered</div></div>')
        html.append(f'<div class="metric"><div class="metric-value">{total.get("lines", 0)}</div><div class="metric-label">Total Lines</div></div>')
        html.append('</div>')
        
        # Trend chart (simple ASCII for now)
        html.append('<div class="trend-chart">')
        html.append('<h2>Coverage Trend (Last 10 Runs)</h2>')
        html.append('<pre>')
        
        recent = history[-10:] if len(history) > 10 else history
        for entry in recent:
            cov = entry.get('total', {}).get('coverage', 0)
            bar = '█' * int(cov / 2)
            html.append(f'{entry.get("timestamp", "")[:10]} [{cov:3}%] {bar}')
        
        html.append('</pre>')
        html.append('</div>')
        
        # File coverage table
        html.append('<h2>File Coverage Details</h2>')
        html.append('<table>')
        html.append('<tr><th>File</th><th>Lines</th><th>Covered</th><th>Coverage</th></tr>')
        
        files = current.get('files', {})
        for filepath, stats in sorted(files.items(), key=lambda x: x[1]['coverage'], reverse=True):
            coverage = stats['coverage']
            css_class = 'good' if coverage >= 80 else 'warning' if coverage >= 60 else 'bad'
            html.append(f'<tr>')
            html.append(f'<td>{Path(filepath).name}</td>')
            html.append(f'<td>{stats["lines"]}</td>')
            html.append(f'<td>{stats["executed"]}</td>')
            html.append(f'<td class="{css_class}">{coverage}%</td>')
            html.append(f'</tr>')
        
        html.append('</table>')
        html.append('</body></html>')
        
        # Save HTML report
        html_file = self.coverage_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w') as f:
            f.write('\n'.join(html))
        
        print(f"HTML report saved to: {html_file}")
        return html_file


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Coverage tracking and reporting tool')
    parser.add_argument('--run', action='store_true', help='Run coverage analysis')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--history', action='store_true', help='Show coverage history')
    parser.add_argument('--baseline', action='store_true', help='Save current as baseline')
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    tracker = CoverageTracker(project_root)
    
    if args.run:
        print("Running coverage analysis...")
        data = tracker.run_coverage()
        if data:
            tracker.save_coverage_data(data)
            print(f"Coverage: {data.get('total', {}).get('coverage', 0)}%")
    
    if args.report:
        report = tracker.generate_report()
        print(report)
        
        # Save to file
        report_file = tracker.coverage_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")
    
    if args.html:
        html_file = tracker.generate_html_report()
        print(f"HTML report generated: {html_file}")
    
    if args.history:
        history = tracker.load_history()
        print(f"Coverage History ({len(history)} entries):")
        for entry in history[-10:]:
            print(f"  {entry.get('timestamp', 'unknown')[:19]}: {entry.get('total', {}).get('coverage', 0)}%")
    
    if args.baseline:
        data = tracker.run_coverage()
        if data:
            baseline_file = tracker.coverage_dir / "coverage_baseline.json"
            with open(baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Baseline saved to: {baseline_file}")
    
    if not any([args.run, args.report, args.html, args.history, args.baseline]):
        # Default: run and report
        print("Running coverage analysis and generating report...")
        data = tracker.run_coverage()
        if data:
            tracker.save_coverage_data(data)
            report = tracker.generate_report()
            print(report)


if __name__ == '__main__':
    main()
