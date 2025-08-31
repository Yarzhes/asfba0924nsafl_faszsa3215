# Ultra Signals Trading Helper Supervisor
# PowerShell script for running the supervisor with proper error handling

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ultra Signals Trading Helper Supervisor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "✓ Created logs directory" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting supervisor with automatic restart capability..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Configuration:" -ForegroundColor White
Write-Host "- Max restarts: 10" -ForegroundColor Gray
Write-Host "- Base backoff: 60 seconds" -ForegroundColor Gray
Write-Host "- Max backoff: 1 hour" -ForegroundColor Gray
Write-Host "- Output capture: Last 100 lines" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the supervisor" -ForegroundColor Yellow
Write-Host ""

# Set environment variables
$env:PYTHONUNBUFFERED = "1"

# Run the supervisor
try {
    python supervisor.py --max-restarts 10 --backoff-base 60 --backoff-max 3600
} catch {
    Write-Host ""
    Write-Host "❌ Supervisor encountered an error: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "Supervisor has stopped." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
