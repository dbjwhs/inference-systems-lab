#!/usr/bin/env python3
"""
Test script for the notification system.

This script demonstrates and tests all notification capabilities:
- Local macOS notifications
- Pushover integration (if configured)
- Persistent logging
- Test result formatting
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.notification_system import NotificationManager, TestResults
    from tools.log_manager import LogManager
    notifications_available = True
except ImportError as e:
    print(f"Failed to import notification system: {e}")
    notifications_available = False
    sys.exit(1)


def test_local_notifications():
    """Test local macOS notifications"""
    print("🧪 Testing local macOS notifications...")
    
    notifier = NotificationManager()
    
    # Test different types of notifications
    print("  - Sending test start notification...")
    notifier._send_local_notification(
        "🧪 Test Started", 
        "Testing notification system for Inference Systems Lab"
    )
    time.sleep(2)
    
    print("  - Sending success notification...")
    notifier._send_local_notification(
        "✅ Tests Passed", 
        "12/16 tests passed (75% success rate)"
    )
    time.sleep(2)
    
    print("  - Sending failure notification...")
    notifier._send_local_notification(
        "❌ Tests Failed", 
        "4/16 tests failed - see details for more info"
    )
    
    print("✅ Local notification test complete!")


def test_notification_manager():
    """Test the full notification manager"""
    print("\n🔔 Testing full notification manager...")
    
    notifier = NotificationManager()
    
    # Test start notification
    print("  - Testing start notification...")
    notifier.notify_test_start(4, estimated_duration=25)
    time.sleep(2)
    
    # Test completion notification
    print("  - Testing completion notification...")
    test_results = TestResults(
        total_tests=16,
        passed_tests=12,
        failed_tests=4,
        skipped_tests=0,
        duration_seconds=1567.5,
        configurations=["release", "debug", "asan", "ubsan"],
        failed_test_names=[
            "ErrorHandlingTest.GPUMemoryExhaustion",
            "ErrorHandlingTest.ModelLoadingFailure", 
            "MemoryManagementTest.ResourceExhaustionHandling",
            "IntegrationTestSuite.ComprehensiveIntegrationTest"
        ],
        platform="macOS-15.6.1-arm64",
        timestamp=datetime.now()
    )
    
    notifier.notify_test_complete(test_results)
    time.sleep(2)
    
    # Test critical failure
    print("  - Testing critical failure notification...")
    notifier.notify_critical_failure(
        "Build system crashed during compilation",
        "This happened while building the AddressSanitizer configuration"
    )
    
    print("✅ Notification manager test complete!")


def test_log_manager():
    """Test the persistent log manager"""
    print("\n📁 Testing log manager...")
    
    log_mgr = LogManager()
    
    # Test storing a test run
    print("  - Storing test run...")
    results = {
        'total_tests': 16,
        'passed_tests': 12,
        'failed_tests': 4,
        'duration_seconds': 1567.5,
        'configurations': ['release', 'debug', 'asan'],
        'platform': 'macOS-15.6.1-arm64',
        'failed_test_names': [
            'ErrorHandlingTest.GPUMemoryExhaustion',
            'MemoryManagementTest.ResourceExhaustionHandling'
        ]
    }
    
    logs = {
        'comprehensive_test': '''
Test Results Summary:
- 16 total tests
- 12 passed (75%)
- 4 failed
Duration: 26.1 minutes

Failed tests:
1. ErrorHandlingTest.GPUMemoryExhaustion
2. MemoryManagementTest.ResourceExhaustionHandling
        ''',
        'integration_test': 'Integration test log content here...',
        'memory_test': 'Memory safety test log content here...'
    }
    
    run_id = log_mgr.store_test_run(results, logs)
    print(f"    ✅ Stored as run ID: {run_id}")
    
    # Test getting historical trends
    print("  - Getting historical trends...")
    trends = log_mgr.get_historical_trends(days=7)
    print(f"    📊 Found {len(trends)} recent test runs")
    
    # Test performance summary
    print("  - Getting performance summary...")
    summary = log_mgr.get_performance_summary(days=7)
    print(f"    📈 Average pass rate: {summary.get('average_pass_rate', 0):.1%}")
    
    print("✅ Log manager test complete!")


def test_pushover_setup():
    """Guide user through Pushover setup"""
    print("\n📱 Pushover Setup Guide:")
    print("=" * 40)
    
    config_file = project_root / "config" / "notifications.yaml"
    print(f"1. Sign up at https://pushover.net (free)")
    print(f"2. Install Pushover app on your devices ($5 one-time)")
    print(f"3. Create an application at https://pushover.net/apps/build")
    print(f"4. Get your User Key from https://pushover.net")
    print(f"5. Edit {config_file}")
    print(f"   - Set pushover.enabled = true")
    print(f"   - Set pushover.api_token = 'your_app_token'")
    print(f"   - Set pushover.user_key = 'your_user_key'")
    print(f"6. Re-run this test to verify Pushover works")
    
    # Check if Pushover is configured
    try:
        notifier = NotificationManager()
        pushover_config = notifier.config.get("notifications", {}).get("pushover", {})
        
        if pushover_config.get("enabled") and pushover_config.get("api_token") and pushover_config.get("user_key"):
            print(f"\n✅ Pushover appears to be configured!")
            print(f"   Testing Pushover notification...")
            
            notifier._send_pushover_notification(
                "🧪 Test from Inference Lab",
                "If you see this, Pushover is working correctly!",
                priority=0,
                sound="intermission"
            )
            print(f"   Check your devices for the notification!")
        else:
            print(f"\n⚠️  Pushover not configured yet - follow steps above")
            
    except Exception as e:
        print(f"\n❌ Error testing Pushover: {e}")


def main():
    """Run all notification tests"""
    print("🔔 NOTIFICATION SYSTEM TEST")
    print("=" * 50)
    
    if not notifications_available:
        print("❌ Notification system not available")
        return 1
    
    try:
        # Test local notifications (should always work on macOS)
        test_local_notifications()
        
        # Test full notification manager
        test_notification_manager()
        
        # Test persistent logging
        test_log_manager()
        
        # Guide through Pushover setup
        test_pushover_setup()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS COMPLETE!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Check your macOS notifications to see if they worked")
        print("2. Setup Pushover if you want remote notifications")
        print("3. Run comprehensive tests with notifications enabled:")
        print("   python3 tools/run_comprehensive_tests.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
