#!/usr/bin/env python3
"""
Notification System for Inference Systems Laboratory

This module provides comprehensive notification capabilities for test results,
supporting local macOS notifications, remote push notifications via Pushover,
email, Slack, and persistent logging for historical analysis.

Key Features:
- macOS native notifications with sound and priority
- Pushover push notifications to all devices
- Email notifications with HTML reports
- Slack webhook integration
- Persistent log storage with retention policies
- Configurable triggers and thresholds
- Rich notification formatting with emojis and context

Usage:
    from tools.notification_system import NotificationManager
    
    notifier = NotificationManager()
    notifier.notify_test_start(4)  # 4 configurations
    notifier.notify_test_complete(results_summary)
"""

import os
import sys
import json
import yaml
import logging
import smtplib
import requests
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResults:
    """Container for test result data"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int = 0
    duration_seconds: float = 0.0
    configurations: List[str] = None
    failed_test_names: List[str] = None
    platform: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.configurations is None:
            self.configurations = []
        if self.failed_test_names is None:
            self.failed_test_names = []
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as a percentage (0.0 to 1.0)"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def success_emoji(self) -> str:
        """Get appropriate emoji based on pass rate"""
        if self.pass_rate >= 0.95:
            return "✅"
        elif self.pass_rate >= 0.80:
            return "⚠️"
        elif self.pass_rate >= 0.50:
            return "❌"
        else:
            return "🚨"
    
    @property
    def duration_formatted(self) -> str:
        """Format duration in human-readable format"""
        if self.duration_seconds < 60:
            return f"{self.duration_seconds:.1f}s"
        elif self.duration_seconds < 3600:
            minutes = self.duration_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = self.duration_seconds / 3600
            return f"{hours:.1f}h"


class NotificationManager:
    """
    Comprehensive notification manager for test results.
    
    Supports multiple notification channels with intelligent filtering,
    persistent logging, and rich formatting.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize notification manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.project_root = Path(__file__).parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config" / "notifications.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Track previous results for improvement detection
        self.history_file = self.project_root / "logs" / "test_history.json"
        
        self.logger.info("NotificationManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            print("Using default configuration")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            print("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is missing"""
        return {
            "notifications": {
                "local": {"enabled": True, "sound": True},
                "pushover": {"enabled": False},
                "email": {"enabled": False},
                "slack": {"enabled": False}
            },
            "triggers": {
                "notify_on_success": True,
                "notify_on_failure": True,
                "min_duration_threshold": 300
            },
            "logging": {
                "log_directory": "logs/comprehensive_tests",
                "retention_days": 30
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the notification system"""
        log_dir = self.project_root / self.config["logging"]["log_directory"]
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger("NotificationManager")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = log_dir / f"notifications_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def notify_test_start(self, config_count: int, estimated_duration: Optional[int] = None) -> None:
        """
        Notify that comprehensive testing has started.
        
        Args:
            config_count: Number of build configurations to test
            estimated_duration: Estimated duration in minutes
        """
        if not self.config["triggers"].get("notify_on_start", False):
            return
        
        duration_text = f", ~{estimated_duration}min" if estimated_duration else ""
        
        title = "🧪 Tests Starting"
        message = f"Testing {config_count} configurations{duration_text}"
        
        self.logger.info(f"Test start: {config_count} configurations")
        
        self._send_local_notification(title, message, priority="normal")
        self._send_pushover_notification(
            title, message, 
            priority=self.config["notifications"]["pushover"].get("priority_normal", 0),
            sound=self.config["notifications"]["pushover"].get("sound_progress", "mechanical")
        )
    
    def notify_config_complete(self, config_name: str, success_rate: float, 
                             current_config: int, total_configs: int) -> None:
        """
        Notify that a build configuration has completed.
        
        Args:
            config_name: Name of completed configuration
            success_rate: Success rate (0.0 to 1.0)
            current_config: Current configuration number
            total_configs: Total number of configurations
        """
        if not self.config["triggers"].get("notify_on_config_complete", False):
            return
        
        emoji = "✅" if success_rate >= 0.9 else "⚠️" if success_rate >= 0.5 else "❌"
        
        title = f"⏳ Progress: {current_config}/{total_configs}"
        message = f"{emoji} {config_name}: {success_rate:.1%} passed"
        
        self.logger.info(f"Config complete: {config_name} - {success_rate:.1%}")
        
        # Only local notifications for progress (avoid spam)
        self._send_local_notification(title, message, priority="low")
    
    def notify_test_complete(self, results: TestResults) -> None:
        """
        Notify that all comprehensive testing has completed.
        
        Args:
            results: Test results summary
        """
        # Check duration threshold
        min_duration = self.config["triggers"].get("min_duration_threshold", 300)
        if results.duration_seconds < min_duration:
            self.logger.info(f"Skipping notification - duration {results.duration_seconds}s < {min_duration}s threshold")
            return
        
        # Determine notification type
        should_notify = self._should_notify_for_results(results)
        if not should_notify:
            self.logger.info("Skipping notification based on triggers")
            return
        
        # Generate notification content
        title, message = self._format_completion_notification(results)
        priority_level = self._get_priority_for_results(results)
        
        self.logger.info(f"Test complete: {results.pass_rate:.1%} pass rate, {results.duration_formatted}")
        
        # Send to all enabled channels
        self._send_local_notification(title, message, priority=priority_level)
        self._send_pushover_notification(title, message, priority=priority_level, results=results)
        self._send_email_notification(title, message, results)
        self._send_slack_notification(title, message, results)
        
        # Store in history for trend analysis
        self._store_results_history(results)
    
    def notify_critical_failure(self, error_message: str, context: str = "") -> None:
        """
        Send critical failure notification (high priority).
        
        Args:
            error_message: Description of the critical error
            context: Additional context about what was happening
        """
        title = "🚨 Critical Test Failure"
        message = f"{error_message}"
        if context:
            message += f"\n\nContext: {context}"
        
        self.logger.error(f"Critical failure: {error_message}")
        
        # Send high-priority notifications to all channels
        self._send_local_notification(title, message, priority="critical")
        self._send_pushover_notification(
            title, message,
            priority=self.config["notifications"]["pushover"].get("priority_critical", 2),
            sound=self.config["notifications"]["pushover"].get("sound_failure", "siren")
        )
    
    def _should_notify_for_results(self, results: TestResults) -> bool:
        """Determine if we should send notifications for these results"""
        triggers = self.config["triggers"]
        
        if results.pass_rate == 1.0 and triggers.get("notify_on_success", True):
            return True
        elif results.pass_rate < 1.0 and triggers.get("notify_on_failure", True):
            return True
        
        # Check for improvement/regression
        if self._is_improvement(results) and triggers.get("notify_on_improvement", True):
            return True
        elif self._is_regression(results) and triggers.get("notify_on_regression", True):
            return True
        
        return False
    
    def _get_priority_for_results(self, results: TestResults) -> str:
        """Get priority level based on results"""
        if results.pass_rate >= self.config["triggers"].get("success_threshold", 0.95):
            return "normal"
        elif results.pass_rate >= self.config["triggers"].get("warning_threshold", 0.80):
            return "normal"
        elif results.pass_rate >= self.config["triggers"].get("failure_threshold", 0.50):
            return "high"
        else:
            return "critical"
    
    def _format_completion_notification(self, results: TestResults) -> tuple[str, str]:
        """Format notification title and message for test completion"""
        emoji = results.success_emoji
        
        title = f"{emoji} Tests Complete"
        
        message = f"""Test Results Summary:

{results.success_emoji} {results.passed_tests}/{results.total_tests} tests passing ({results.pass_rate:.1%})
⏱️ Duration: {results.duration_formatted}
🖥️ Platform: {results.platform}
📊 Configurations: {len(results.configurations)}"""
        
        if results.failed_tests > 0:
            message += f"\n\n❌ Failed Tests:"
            for test_name in results.failed_test_names[:5]:  # Show first 5 failures
                message += f"\n  • {test_name}"
            if len(results.failed_test_names) > 5:
                message += f"\n  • ... and {len(results.failed_test_names) - 5} more"
        
        # Add improvement/regression context
        if self._is_improvement(results):
            message += f"\n\n📈 Improvement detected!"
        elif self._is_regression(results):
            message += f"\n\n📉 Regression detected"
        
        return title, message
    
    def _send_local_notification(self, title: str, message: str, priority: str = "normal") -> None:
        """Send macOS native notification using osascript with terminal-notifier fallback"""
        if not self.config["notifications"]["local"].get("enabled", False):
            return
        
        # First try osascript (native macOS notifications)
        try:
            # Escape quotes in message
            escaped_title = title.replace('"', '\\"')
            escaped_message = message.replace('"', '\\"')
            
            # Build osascript command
            script_parts = [
                f'display notification "{escaped_message}"',
                f'with title "{escaped_title}"'
            ]
            
            # Add sound if enabled
            if self.config["notifications"]["local"].get("sound", True):
                sound_name = self.config["notifications"]["local"].get("sound_name", "Glass")
                script_parts.append(f'sound name "{sound_name}"')
            
            script = " ".join(script_parts)
            
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.debug("Local notification sent successfully via osascript")
                return
            else:
                self.logger.warning(f"osascript notification failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("osascript notification timed out")
        except Exception as e:
            self.logger.warning(f"osascript notification failed: {e}")
        
        # Fallback to terminal-notifier if osascript fails
        try:
            cmd = ['terminal-notifier', '-title', title, '-message', message]
            
            # Add sound if enabled
            if self.config["notifications"]["local"].get("sound", True):
                sound_name = self.config["notifications"]["local"].get("sound_name", "Glass")
                cmd.extend(['-sound', sound_name])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.logger.debug("Local notification sent successfully via terminal-notifier")
            else:
                self.logger.warning(f"terminal-notifier notification failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("terminal-notifier notification timed out")
        except FileNotFoundError:
            self.logger.warning("terminal-notifier not found - install with: brew install terminal-notifier")
        except Exception as e:
            self.logger.warning(f"terminal-notifier notification failed: {e}")
    
    def _send_pushover_notification(self, title: str, message: str, 
                                  priority: int = 0, sound: str = None, 
                                  results: TestResults = None) -> None:
        """Send notification via Pushover API"""
        pushover_config = self.config["notifications"]["pushover"]
        if not pushover_config.get("enabled", False):
            return
        
        # Load credentials from environment variables FIRST (secure)
        # Fall back to config file if not in environment
        api_token = os.environ.get('PUSHOVER_API_TOKEN') or pushover_config.get("api_token")
        user_key = os.environ.get('PUSHOVER_USER_KEY') or pushover_config.get("user_key")
        
        if not api_token or not user_key:
            self.logger.warning("Pushover enabled but missing api_token or user_key")
            self.logger.warning("Set PUSHOVER_API_TOKEN and PUSHOVER_USER_KEY environment variables")
            self.logger.warning("Or add them to config/notifications.yaml (less secure)")
            return
        
        try:
            # Prepare payload
            payload = {
                "token": api_token,
                "user": user_key,
                "title": title,
                "message": message,
                "priority": priority,
                "timestamp": int(datetime.now().timestamp())
            }
            
            # Add optional parameters
            if sound:
                payload["sound"] = sound
            
            device = pushover_config.get("device")
            if device:
                payload["device"] = device
            
            # Add URL for detailed report if available
            if results:
                # TODO: Add URL to detailed HTML report
                pass
            
            # Send request
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.debug("Pushover notification sent successfully")
            else:
                self.logger.warning(f"Pushover notification failed: {response.status_code} - {response.text}")
                
        except requests.RequestException as e:
            self.logger.warning(f"Failed to send Pushover notification: {e}")
    
    def _send_email_notification(self, title: str, message: str, results: TestResults = None) -> None:
        """Send email notification with HTML report"""
        # TODO: Phase 2 implementation
        pass
    
    def _send_slack_notification(self, title: str, message: str, results: TestResults = None) -> None:
        """Send Slack webhook notification"""
        # TODO: Phase 2 implementation
        pass
    
    def _is_improvement(self, results: TestResults) -> bool:
        """Check if results show improvement over previous run"""
        # TODO: Compare with stored history
        return False
    
    def _is_regression(self, results: TestResults) -> bool:
        """Check if results show regression from previous run"""
        # TODO: Compare with stored history
        return False
    
    def _store_results_history(self, results: TestResults) -> None:
        """Store results in history file for trend analysis"""
        history_file = self.project_root / "logs" / "test_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        
        # Add current results
        history.append({
            "timestamp": results.timestamp.isoformat(),
            "total_tests": results.total_tests,
            "passed_tests": results.passed_tests,
            "failed_tests": results.failed_tests,
            "pass_rate": results.pass_rate,
            "duration_seconds": results.duration_seconds,
            "configurations": results.configurations,
            "platform": results.platform,
            "failed_test_names": results.failed_test_names
        })
        
        # Keep only recent history (based on retention policy)
        retention_days = self.config["logging"].get("retention_days", 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        history = [
            entry for entry in history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]
        
        # Save updated history
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            self.logger.debug(f"Stored results in history: {len(history)} entries")
        except Exception as e:
            self.logger.warning(f"Failed to store results history: {e}")


def main():
    """Test the notification system"""
    notifier = NotificationManager()
    
    # Test notifications
    print("Testing notification system...")
    
    # Test local notification
    notifier._send_local_notification(
        "🧪 Test Notification", 
        "This is a test notification from the Inference Systems Lab",
        priority="normal"
    )
    
    print("Test notification sent!")
    print("Check your macOS notifications to see if it worked.")
    print("\nTo setup Pushover notifications:")
    print("1. Sign up at https://pushover.net")
    print("2. Install Pushover app on your devices ($5)")
    print("3. Create an application at https://pushover.net/apps/build")
    print("4. Update config/notifications.yaml with your keys")
    print("5. Set pushover.enabled = true")


if __name__ == "__main__":
    main()