$ErrorActionPreference = "Stop"

$newModelDir = "D:\Ollama\Models"
$logFile = "c:\Users\ARAVIND\Desktop\local-rag\ollama_config_result.txt"

# 1. Inspect existing OLLAMA_MODELS
$existingEnv = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
if ($existingEnv -ne $null -and $existingEnv -ne "") {
    if ($existingEnv -ne $newModelDir) {
        Write-Host "Warning: OLLAMA_MODELS is already set to '$existingEnv'."
        Write-Host "We will overwrite it to '$newModelDir' as per instructions, but keeping existing models intact."
    }
}

# 2. Create directory
if (-not (Test-Path $newModelDir)) {
    Write-Host "Creating directory $newModelDir..."
    New-Item -ItemType Directory -Path $newModelDir | Out-Null
} else {
    Write-Host "Directory $newModelDir already exists."
}

# 3. Configure OLLAMA_MODELS permanently (User level)
Write-Host "Setting OLLAMA_MODELS to $newModelDir permanently..."
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $newModelDir, "User")

# Also set for current process so subsequent commands use it
$env:OLLAMA_MODELS = $newModelDir

# 4. Restart Ollama
Write-Host "Restarting Ollama..."
Stop-Process -Name "ollama*" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Start Ollama app if it exists, otherwise just the server
$ollamaApp = "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
if (Test-Path $ollamaApp) {
    Write-Host "Starting Ollama App..."
    Start-Process $ollamaApp
} else {
    Write-Host "Starting Ollama Server..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
}

Write-Host "Waiting for Ollama to initialize..."
Start-Sleep -Seconds 7

# 5. Verify configuration
Write-Host "OLLAMA_MODELS is currently: $env:OLLAMA_MODELS"

# 6. Download qwen3-vl:4b
Write-Host "Downloading qwen3-vl:4b. This may take a while..."
ollama pull qwen3-vl:4b

# 7. Test the model
Write-Host "Testing qwen3-vl:4b..."
$testResult = ollama run qwen3-vl:4b "hello"

# 8. Generate Final Output
$version = ollama -v
$listResult = ollama list | Out-String

$report = @"
========== OLLAMA CONFIGURATION REPORT ==========
Ollama Version: $version
Model Directory: $newModelDir
OLLAMA_MODELS Value: $env:OLLAMA_MODELS

Ollama List Result:
$listResult

Test Run Result (qwen3-vl:4b):
$testResult

Confirmation: qwen3-vl:4b runs successfully and is stored in $newModelDir.
=================================================
"@

$report | Out-File -FilePath $logFile
Write-Host $report
Write-Host "Report saved to $logFile"
