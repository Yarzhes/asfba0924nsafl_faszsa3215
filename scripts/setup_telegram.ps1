# Telegram Configuration Setup Script
# Sets environment variables for Telegram integration

Write-Host "Setting up Telegram Bot Integration..." -ForegroundColor Green

# Set Telegram environment variables
$env:TELEGRAM_BOT_TOKEN = "8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs"
$env:TELEGRAM_CHAT_ID = "7072100094"

Write-Host "Telegram Bot Token: Set (8360503431:***)" -ForegroundColor Green
Write-Host "Telegram Chat ID: $env:TELEGRAM_CHAT_ID" -ForegroundColor Green

# Verify environment variables are set
if ($env:TELEGRAM_BOT_TOKEN -and $env:TELEGRAM_CHAT_ID) {
    Write-Host "Telegram environment variables configured successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Bot Details:" -ForegroundColor Yellow
    Write-Host "   Token: 8360503431:***" -ForegroundColor White
    Write-Host "   Chat ID: $env:TELEGRAM_CHAT_ID" -ForegroundColor White
    Write-Host ""
    Write-Host "You can now run shadow tests with Telegram notifications:" -ForegroundColor Cyan
    Write-Host "   python scripts/run_shadow_test.py --duration 120" -ForegroundColor White
} else {
    Write-Host "Failed to set Telegram environment variables" -ForegroundColor Red
    exit 1
}

# Optional: Persist for the session
Write-Host ""
Write-Host "To make these environment variables persistent across sessions:" -ForegroundColor Yellow
Write-Host "   Run this script each time you open a new terminal," -ForegroundColor White
Write-Host "   or add them to your PowerShell profile." -ForegroundColor White
