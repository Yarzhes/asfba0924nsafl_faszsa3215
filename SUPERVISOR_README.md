# Ultra Signals Supervisor

This supervisor system provides robust error handling and automatic restart capabilities for the Ultra Signals Trading Helper.

## 🚀 Quick Start

### Windows (Recommended)
```powershell
# Run with PowerShell (better error handling)
.\run_supervisor.ps1

# Or with Command Prompt
run_supervisor.bat
```

### Linux/Mac
```bash
# Run directly with Python
python supervisor.py --max-restarts 10 --backoff-base 60
```

## 🔧 Features

### ✅ Automatic Restart
- **Exponential Backoff**: Starts with 60s, doubles each restart (max 1 hour)
- **Configurable Limits**: Set maximum restart attempts (default: 10)
- **Graceful Shutdown**: Handles Ctrl+C properly

### ✅ Error Capture
- **Last 100 Lines**: Saves the last 100 lines of terminal output when errors occur
- **Detailed Reports**: Creates timestamped error reports in `logs/` directory
- **Restart History**: Maintains JSON file with all restart events

### ✅ Monitoring
- **Real-time Output**: Shows all program output with timestamps
- **Process Tracking**: Monitors PID and exit codes
- **Health Checks**: Detects when processes become unresponsive

## 📁 File Structure

```
logs/
├── error_20250830_143022.txt    # Error reports with last 100 lines
├── error_20250830_150145.txt    # Another error report
└── restart_history.json         # Complete restart history
```

## ⚙️ Configuration Options

### Command Line Arguments
```bash
python supervisor.py [OPTIONS]

Options:
  --config CONFIG          Configuration file (default: settings.yaml)
  --max-restarts N         Maximum restart attempts (default: 10)
  --backoff-base SECONDS   Base backoff time in seconds (default: 60)
  --backoff-max SECONDS    Maximum backoff time in seconds (default: 3600)
```

### Example Configurations

**Conservative (Long backoff, few restarts):**
```bash
python supervisor.py --max-restarts 5 --backoff-base 120 --backoff-max 7200
```

**Aggressive (Short backoff, many restarts):**
```bash
python supervisor.py --max-restarts 20 --backoff-base 30 --backoff-max 1800
```

**Overnight Run (Very conservative):**
```bash
python supervisor.py --max-restarts 3 --backoff-base 300 --backoff-max 3600
```

## 📊 Error Reports

When the program crashes, the supervisor creates detailed error reports:

### Error Report Format
```
# Ultra Signals Error Report
# Timestamp: 2025-08-30 14:30:22
# Restart Count: 3/10
# Total Runtime: 7200.5 seconds

## Error Details
Error Type: ProcessExitError
Error Message: Process exit code: 1

## Last 100 Lines of Output
==================================================
[2025-08-30 14:29:15] Starting Canary Test Harness...
[2025-08-30 14:29:16] Settings loaded successfully.
[2025-08-30 14:29:17] Canary profile activated...
[2025-08-30 14:30:20] ERROR: WebSocket connection failed
==================================================

## Restart History
1. 2025-08-30T12:00:00 - Exit code: 1
2. 2025-08-30T12:02:00 - Exit code: 1
3. 2025-08-30T14:30:22 - Exit code: 1
```

## 🔍 Troubleshooting

### Common Issues

**1. Program exits immediately**
- Check error reports in `logs/` directory
- Verify Python environment and dependencies
- Check if `canary_harness.py` exists and is executable

**2. Too many restarts**
- Increase `--backoff-base` for longer delays
- Check error reports for recurring issues
- Consider reducing `--max-restarts` if issues persist

**3. No output captured**
- Ensure `PYTHONUNBUFFERED=1` environment variable is set
- Check if process is generating output to stdout/stderr

### Debug Mode
For debugging, you can run with verbose output:
```bash
python supervisor.py --max-restarts 1 --backoff-base 10
```

## 🛑 Stopping the Supervisor

### Graceful Shutdown
- Press `Ctrl+C` to stop gracefully
- The supervisor will wait for the current process to finish
- All error reports will be saved

### Force Stop
- Press `Ctrl+C` twice quickly
- Or kill the supervisor process directly

## 📈 Monitoring

### Real-time Status
The supervisor shows real-time status:
```
🎯 Ultra Signals Supervisor Starting...
📁 Logs directory: C:\Users\Almir\Projects\Trading Helper\logs
⚙️  Config: settings.yaml
🚀 Starting process: python canary_harness.py
📊 Max restarts: 10, Backoff base: 60s
👀 Monitoring process PID: 12345
[12345] Starting Canary Test Harness...
[12345] Settings loaded successfully.
```

### Restart Information
When restarts occur:
```
⚠️  Process exited with code 1
🔄 Restarting in 60s (attempt 1/10)
📋 Error report saved to logs/error_20250830_143022.txt
```

## 🔧 Integration

### With Windows Task Scheduler
1. Create a scheduled task
2. Set action: `powershell.exe -ExecutionPolicy Bypass -File "C:\path\to\run_supervisor.ps1"`
3. Set to run at startup or specific times

### With systemd (Linux)
Create `/etc/systemd/system/ultra-signals.service`:
```ini
[Unit]
Description=Ultra Signals Trading Helper
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/trading/helper
ExecStart=/usr/bin/python3 supervisor.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

## 📝 Log Analysis

### Check Recent Errors
```bash
# List recent error reports
ls -la logs/error_*.txt | tail -5

# View latest error
cat logs/error_$(ls logs/error_*.txt | tail -1)
```

### Analyze Restart History
```bash
# View restart history
cat logs/restart_history.json | python -m json.tool
```

## 🎯 Best Practices

1. **Start Conservative**: Use longer backoff times initially
2. **Monitor Logs**: Check error reports regularly
3. **Adjust Settings**: Fine-tune based on your environment
4. **Backup Logs**: Archive old error reports periodically
5. **Test First**: Run with `--max-restarts 1` to test setup

## 🆘 Support

If you encounter issues:
1. Check the error reports in `logs/` directory
2. Review the restart history
3. Try running with debug settings
4. Check system resources and network connectivity
