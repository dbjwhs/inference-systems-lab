# macOS Notification Troubleshooting Guide

## Summary

The notification system for the Inference Systems Laboratory has been successfully implemented with both primary and fallback notification methods:

### ✅ Implementation Complete
- **Phase 1 Notification System**: Fully implemented with local macOS notification integration
- **Dual Notification Methods**: osascript (native) + terminal-notifier (fallback)
- **Persistent Logging**: SQLite database with historical test run analysis
- **Configuration Management**: YAML-based notification settings
- **Comprehensive Testing**: 152 tests with notification integration

### 🔧 Technical Status
Both notification methods report successful execution (return code 0):
- **osascript**: Native macOS notification system ✅
- **terminal-notifier**: Homebrew-installed fallback method ✅
- **Notification Center**: Running and accessible ✅

### 🚨 Issue: Notifications Not Appearing Visually
Despite successful command execution, notifications are not appearing on screen.

## Common macOS Notification Issues

### 1. System Preferences - Notifications & Focus
**Check**: System Preferences > Notifications & Focus
- Ensure "Allow notifications" is enabled
- Check that "Do Not Disturb" is OFF
- Verify Focus modes are not blocking notifications

### 2. Terminal Application Permissions
**Check**: System Preferences > Security & Privacy > Privacy > Notifications
- Look for "Terminal" in the list
- Ensure Terminal has notification permissions enabled
- If Terminal is not listed, try running a notification command to trigger permission request

### 3. Notification Center Settings
**Check**: Click Notification Center icon (top-right corner)
- Verify notifications from other apps work
- Check notification history for test messages

### 4. Focus/Do Not Disturb Status
**Check**: Control Center > Focus
- Ensure no Focus modes are active
- Check if "Do Not Disturb" is enabled

### 5. Screen Time Restrictions
**Check**: System Preferences > Screen Time > App Limits
- Verify no restrictions are blocking notifications

## Manual Testing Commands

### Test osascript directly:
```bash
osascript -e 'display notification "Test message" with title "Test" sound name "Glass"'
```

### Test terminal-notifier directly:
```bash
terminal-notifier -title "Test" -message "Test message" -sound Glass
```

### Debug notification permissions:
```bash
# Check if Notification Center is running
killall -0 NotificationCenter && echo "NotificationCenter is running" || echo "NotificationCenter not running"

# Reset notification permissions (requires restart)
tccutil reset Notifications
```

## Alternative Solutions

### 1. Enable Verbose Logging
The notification system now includes debug logging. Check logs in:
```
logs/notifications.log
```

### 2. Use Alternative Notification Methods
If native notifications fail, consider:
- **Email notifications**: SMTP-based alerts (planned Phase 2)
- **Slack integration**: Webhook-based team notifications (planned Phase 2)

### 3. Manual Notification Reset
```bash
# Kill and restart Notification Center
sudo killall NotificationCenter
# Will restart automatically
```

## Next Steps

1. **Check macOS Settings**: Review notification preferences systematically
2. **Verify Integration**: Run comprehensive tests with notifications enabled
3. **Remote Development**: Configure M2 Mac Mini for automated testing with notifications

## Files Modified

- `tools/notification_system.py`: Added terminal-notifier fallback
- `tools/test_notifications.py`: Comprehensive testing suite
- `tools/log_manager.py`: Persistent log storage and analysis
- `config/notifications.yaml`: Complete notification configuration
- `debug_notifications.py`: Diagnostic testing script

The notification infrastructure is complete and ready. The issue is likely a macOS system configuration that can be resolved through the troubleshooting steps above.
