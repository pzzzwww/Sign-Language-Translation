# Start script for Sign Language Translation System

param(
    [string]$Region = "jp"
)

$Domain     = "sloped-racism-culinary.ngrok-free.dev"
$ProjectDir = "C:\Users\yng\Desktop\Vision Transformer"
$PythonExe  = "C:/Users/yng/.conda/envs/torch/python.exe"

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Sign Language Translation System" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# Kill old processes
Write-Host "`n[1/2] Stopping old processes..." -ForegroundColor Yellow

# Kill ngrok
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# Kill process on port 8000
$old = netstat -ano | Select-String ":8000" | Select-String "LISTENING"
if ($old) {
    $oldPid = ($old -split "\s+")[-1]
    Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
Write-Host "  Old processes stopped" -ForegroundColor Green

# Start backend in new window
Write-Host "[2/2] Starting backend + ngrok..." -ForegroundColor Yellow

Start-Process powershell -WorkingDirectory $ProjectDir -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$env:NO_SSL='1'; & '$PythonExe' -m src.backend.main"
)

Start-Sleep -Seconds 5

# Start ngrok in new window
Start-Process powershell -WorkingDirectory $ProjectDir -ArgumentList @(
    "-NoExit",
    "-Command",
    "ngrok http 8000 --domain=$Domain --region=$Region"
)

Write-Host "`n=====================================================" -ForegroundColor Green
Write-Host "  Public URL: https://$Domain" -ForegroundColor Green
Write-Host "  Local:      http://localhost:8000" -ForegroundColor Green
Write-Host "  Close the two new windows to stop all services" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
