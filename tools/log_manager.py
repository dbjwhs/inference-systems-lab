#!/usr/bin/env python3
"""
Log Management System for Inference Systems Laboratory

This module provides persistent log storage, compression, and analysis
for comprehensive test runs. It enables historical analysis, trend detection,
and long-term performance monitoring.

Key Features:
- Persistent log storage with configurable retention
- Automatic compression of older logs
- Log analysis and trend detection
- Historical performance comparisons
- Search and filtering capabilities
- Export to various formats (JSON, CSV, HTML)

Usage:
    from tools.log_manager import LogManager
    
    log_mgr = LogManager()
    log_mgr.store_test_run(results, logs, artifacts)
    historical_data = log_mgr.get_historical_trends(days=30)
"""

import os
import gzip
import json
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Add project root to path
project_root = Path(__file__).parent.parent


@dataclass
class TestRunRecord:
    """Record of a complete test run with all metadata"""
    run_id: str
    timestamp: datetime
    duration_seconds: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    configurations: List[str]
    platform_info: str
    git_commit: str
    git_branch: str
    pass_rate: float
    failed_test_names: List[str]
    log_file_path: str
    artifacts_path: str
    
    def __post_init__(self):
        # Ensure pass_rate is calculated
        if self.total_tests > 0:
            self.pass_rate = self.passed_tests / self.total_tests


class LogManager:
    """
    Manages persistent storage and analysis of test run logs.
    
    Provides comprehensive logging with automatic retention policies,
    compression, indexing, and historical analysis capabilities.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize log manager.
        
        Args:
            base_dir: Base directory for log storage (defaults to logs/)
        """
        self.project_root = Path(__file__).parent.parent
        
        if base_dir is None:
            base_dir = self.project_root / "logs"
        
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "test_runs.db"
        
        # Create directory structure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "runs").mkdir(exist_ok=True)
        (self.base_dir / "archives").mkdir(exist_ok=True)
        (self.base_dir / "exports").mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"LogManager initialized with base_dir: {self.base_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the log manager"""
        logger = logging.getLogger("LogManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = self.base_dir / "log_manager.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_database(self) -> None:
        """Initialize SQLite database for test run metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    skipped_tests INTEGER DEFAULT 0,
                    configurations TEXT NOT NULL,  -- JSON array
                    platform_info TEXT NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    pass_rate REAL NOT NULL,
                    failed_test_names TEXT,  -- JSON array
                    log_file_path TEXT NOT NULL,
                    artifacts_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for common queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON test_runs (timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_pass_rate 
                ON test_runs (pass_rate)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_git_branch 
                ON test_runs (git_branch)
            ''')
            
            conn.commit()
    
    def store_test_run(self, results: Dict[str, Any], logs: Dict[str, str], 
                      artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Store a complete test run with logs and artifacts.
        
        Args:
            results: Test results dictionary
            logs: Dictionary of log files {name: content}
            artifacts: Optional artifacts {name: file_path}
            
        Returns:
            run_id: Unique identifier for this test run
        """
        # Generate unique run ID
        timestamp = datetime.now()
        run_id = timestamp.strftime("run_%Y%m%d_%H%M%S")
        
        # Create run directory
        run_dir = self.base_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Store logs
        log_files = {}
        for log_name, log_content in logs.items():
            log_file = run_dir / f"{log_name}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(log_content)
            log_files[log_name] = str(log_file)
        
        # Store artifacts
        artifact_files = {}
        if artifacts:
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            for artifact_name, source_path in artifacts.items():
                dest_path = artifacts_dir / artifact_name
                shutil.copy2(source_path, dest_path)
                artifact_files[artifact_name] = str(dest_path)
        
        # Get git information
        git_commit, git_branch = self._get_git_info()
        
        # Create test run record
        record = TestRunRecord(
            run_id=run_id,
            timestamp=timestamp,
            duration_seconds=results.get('duration_seconds', 0.0),
            total_tests=results.get('total_tests', 0),
            passed_tests=results.get('passed_tests', 0),
            failed_tests=results.get('failed_tests', 0),
            skipped_tests=results.get('skipped_tests', 0),
            configurations=results.get('configurations', []),
            platform_info=results.get('platform', ''),
            git_commit=git_commit,
            git_branch=git_branch,
            pass_rate=0.0,  # Will be calculated in __post_init__
            failed_test_names=results.get('failed_test_names', []),
            log_file_path=str(run_dir),
            artifacts_path=str(run_dir / "artifacts") if artifacts else ""
        )
        
        # Store in database
        self._store_record_in_db(record)
        
        # Store summary JSON
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(record), f, indent=2, default=str)
        
        self.logger.info(f"Stored test run {run_id}: {record.pass_rate:.1%} pass rate")
        
        # Cleanup old logs based on retention policy
        self._cleanup_old_logs()
        
        return run_id
    
    def _store_record_in_db(self, record: TestRunRecord) -> None:
        """Store test run record in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO test_runs (
                    run_id, timestamp, duration_seconds, total_tests, passed_tests,
                    failed_tests, skipped_tests, configurations, platform_info,
                    git_commit, git_branch, pass_rate, failed_test_names,
                    log_file_path, artifacts_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.run_id,
                record.timestamp.isoformat(),
                record.duration_seconds,
                record.total_tests,
                record.passed_tests,
                record.failed_tests,
                record.skipped_tests,
                json.dumps(record.configurations),
                record.platform_info,
                record.git_commit,
                record.git_branch,
                record.pass_rate,
                json.dumps(record.failed_test_names),
                record.log_file_path,
                record.artifacts_path
            ))
            conn.commit()
    
    def get_historical_trends(self, days: int = 30, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get historical test results for trend analysis.
        
        Args:
            days: Number of days to look back
            branch: Optional git branch filter
            
        Returns:
            List of test run records
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = '''
                SELECT * FROM test_runs 
                WHERE timestamp >= ? 
            '''
            params = [cutoff_date.isoformat()]
            
            if branch:
                query += ' AND git_branch = ?'
                params.append(branch)
            
            query += ' ORDER BY timestamp DESC'
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                record = dict(row)
                # Parse JSON fields
                record['configurations'] = json.loads(record['configurations'])
                record['failed_test_names'] = json.loads(record['failed_test_names'])
                results.append(record)
            
            return results
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate performance summary for recent test runs.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance summary dictionary
        """
        trends = self.get_historical_trends(days)
        
        if not trends:
            return {"error": "No test data found for the specified period"}
        
        # Calculate statistics
        pass_rates = [run['pass_rate'] for run in trends]
        durations = [run['duration_seconds'] for run in trends]
        
        summary = {
            "period_days": days,
            "total_runs": len(trends),
            "average_pass_rate": sum(pass_rates) / len(pass_rates),
            "min_pass_rate": min(pass_rates),
            "max_pass_rate": max(pass_rates),
            "average_duration_minutes": sum(durations) / len(durations) / 60,
            "latest_run": trends[0] if trends else None,
            "trend_direction": self._calculate_trend_direction(trends),
            "most_common_failures": self._get_common_failures(trends),
        }
        
        return summary
    
    def _calculate_trend_direction(self, trends: List[Dict[str, Any]]) -> str:
        """Calculate whether test results are improving or degrading"""
        if len(trends) < 2:
            return "insufficient_data"
        
        # Compare recent vs older results
        recent_runs = trends[:len(trends)//3] if len(trends) >= 6 else trends[:2]
        older_runs = trends[-len(trends)//3:] if len(trends) >= 6 else trends[-2:]
        
        recent_avg = sum(run['pass_rate'] for run in recent_runs) / len(recent_runs)
        older_avg = sum(run['pass_rate'] for run in older_runs) / len(older_runs)
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:  # 5% improvement
            return "improving"
        elif diff < -0.05:  # 5% degradation
            return "degrading"
        else:
            return "stable"
    
    def _get_common_failures(self, trends: List[Dict[str, Any]], limit: int = 5) -> List[Tuple[str, int]]:
        """Get most commonly failing tests"""
        failure_counts = {}
        
        for run in trends:
            for test_name in run['failed_test_names']:
                failure_counts[test_name] = failure_counts.get(test_name, 0) + 1
        
        # Sort by frequency and return top N
        return sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _get_git_info(self) -> Tuple[str, str]:
        """Get current git commit and branch information"""
        try:
            import subprocess
            
            # Get commit hash
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                text=True
            ).strip()[:8]  # Short hash
            
            # Get branch name
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                text=True
            ).strip()
            
            return commit, branch
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown", "unknown"
    
    def _cleanup_old_logs(self, retention_days: int = 30, compress_after_days: int = 7) -> None:
        """
        Clean up old logs according to retention policy.
        
        Args:
            retention_days: Delete logs older than this many days
            compress_after_days: Compress logs older than this many days
        """
        now = datetime.now()
        compress_cutoff = now - timedelta(days=compress_after_days)
        delete_cutoff = now - timedelta(days=retention_days)
        
        runs_dir = self.base_dir / "runs"
        archives_dir = self.base_dir / "archives"
        
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # Parse timestamp from directory name
            try:
                run_timestamp = datetime.strptime(run_dir.name, "run_%Y%m%d_%H%M%S")
            except ValueError:
                continue
            
            if run_timestamp < delete_cutoff:
                # Delete old runs
                shutil.rmtree(run_dir)
                self.logger.info(f"Deleted old test run: {run_dir.name}")
                
            elif run_timestamp < compress_cutoff:
                # Compress logs
                archive_path = archives_dir / f"{run_dir.name}.tar.gz"
                if not archive_path.exists():
                    shutil.make_archive(
                        archive_path.with_suffix(''),
                        'gztar',
                        runs_dir,
                        run_dir.name
                    )
                    shutil.rmtree(run_dir)
                    self.logger.info(f"Compressed test run: {run_dir.name}")
    
    def export_trends_csv(self, output_file: str, days: int = 30) -> None:
        """Export historical trends to CSV file"""
        trends = self.get_historical_trends(days)
        
        import csv
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'run_id', 'timestamp', 'duration_minutes', 'total_tests',
                'passed_tests', 'failed_tests', 'pass_rate', 'git_branch',
                'git_commit', 'platform_info'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for run in trends:
                writer.writerow({
                    'run_id': run['run_id'],
                    'timestamp': run['timestamp'],
                    'duration_minutes': run['duration_seconds'] / 60,
                    'total_tests': run['total_tests'],
                    'passed_tests': run['passed_tests'],
                    'failed_tests': run['failed_tests'],
                    'pass_rate': run['pass_rate'],
                    'git_branch': run['git_branch'],
                    'git_commit': run['git_commit'],
                    'platform_info': run['platform_info']
                })
        
        self.logger.info(f"Exported {len(trends)} test runs to {output_file}")


def main():
    """Test the log manager"""
    log_mgr = LogManager()
    
    # Test data
    test_results = {
        'total_tests': 16,
        'passed_tests': 12,
        'failed_tests': 4,
        'duration_seconds': 1250.5,
        'configurations': ['release', 'debug', 'asan'],
        'platform': 'macOS-15.6.1-arm64',
        'failed_test_names': [
            'ErrorHandlingTest.GPUMemoryExhaustion',
            'ErrorHandlingTest.ModelLoadingFailure',
            'MemoryManagementTest.ResourceExhaustionHandling',
            'IntegrationTestSuite.ComprehensiveIntegrationTest'
        ]
    }
    
    test_logs = {
        'comprehensive_test': 'This is a sample comprehensive test log...',
        'integration_test': 'Sample integration test log...',
        'memory_test': 'Sample memory test log...'
    }
    
    # Store test run
    run_id = log_mgr.store_test_run(test_results, test_logs)
    print(f"Stored test run: {run_id}")
    
    # Get trends
    trends = log_mgr.get_historical_trends(days=7)
    print(f"Found {len(trends)} recent test runs")
    
    # Get performance summary
    summary = log_mgr.get_performance_summary(days=7)
    print("Performance Summary:")
    for key, value in summary.items():
        if key != 'latest_run':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
