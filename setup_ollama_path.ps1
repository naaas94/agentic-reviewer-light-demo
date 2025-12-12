# Set Ollama Models Directory for Windows
# Run this script to permanently set OLLAMA_MODELS environment variable

$modelsPath = "D:\Apps\Ollama\models"

# Set for current user (permanent)
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $modelsPath, "User")

# Set for current session
$env:OLLAMA_MODELS = $modelsPath

Write-Host "OLLAMA_MODELS has been set to: $modelsPath" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: You need to restart Ollama for this to take effect!" -ForegroundColor Yellow
Write-Host "1. Close Ollama completely (check system tray)"
Write-Host "2. Restart Ollama from Start menu"
Write-Host "3. Then run: python run_demo.py"
Write-Host ""

# Verify
$verify = [System.Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
if ($verify -eq $modelsPath) {
    Write-Host "✓ Environment variable set successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to set environment variable" -ForegroundColor Red
}

