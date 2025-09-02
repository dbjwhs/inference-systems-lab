# Pushover Setup Guide for Inference Systems Lab

## Quick Setup Steps

### 1. Get Your User Key
1. Go to https://pushover.net
2. Log in to your account  
3. Your **User Key** is displayed on the main page (30 characters, like `u1234567890abcdefghijklmnopqrs`)
4. This is NOT your email - it's a unique 30-character identifier

### 2. Create an Application Token
1. Go to https://pushover.net/apps/build
2. Create a new application:
   - **Name**: "Inference Systems Lab" (or any name you prefer)
   - **Description**: "Test notifications for ML inference system"
   - **URL**: (optional, can leave blank)
   - **Icon**: (optional)
3. Click "Create Application"
4. You'll get an **API Token** (30 characters, like `a1234567890abcdefghijklmnopqrs`)

### 3. Update Your .env File
Edit `/Users/dbjones/ng/dbjwhs/inference-systems-lab/.env`:

```bash
# Your Pushover API token from the application you created
PUSHOVER_API_TOKEN=a1234567890abcdefghijklmnopqrs

# Your Pushover user key from your account page (NOT your email)
PUSHOVER_USER_KEY=u1234567890abcdefghijklmnopqrs
```

### 4. Test Your Setup
```bash
python3 test_pushover_secure.py
```

## Security Notes

✅ **DO**: 
- Keep credentials in `.env` file (gitignored)
- Use environment variables to load credentials
- Verify `.env` is in `.gitignore`

❌ **DON'T**:
- Put credentials directly in code
- Put credentials in `notifications.yaml`
- Commit `.env` to git
- Share your API tokens publicly

## Credential Format

Both keys should look like:
- **30 characters long**
- **Alphanumeric** (letters and numbers only)
- **Start with a letter** (usually 'a' for API tokens, 'u' for user keys)

Example format:
- API Token: `azGDORePK8gMaC0QOYAMyEEuzJnyUi`
- User Key: `uQiRzpo4DXghDmr9QzzfQu27cmVRsG`

## Common Issues

### "Invalid token" error
- You're using your email instead of the User Key
- You haven't created an application yet
- You're using the wrong token (swapped user key and API token)

### No notifications received
- Check the Pushover app is installed and logged in
- Verify device name in config (leave blank for all devices)
- Check notification settings in the app

## Your Current Issue

Based on the error, it looks like:
1. The API token `urf74o9e6i6uz91szcq4xjdh3p534g` is not recognized by Pushover
2. The user key `httjidb5ob@pomail.net` looks like an email, not a Pushover user key

**Action Required**:
1. Go to https://pushover.net and get your actual 30-character User Key
2. Go to https://pushover.net/apps/build and create an app to get an API Token
3. Update `.env` with the correct credentials
4. Run the test again
