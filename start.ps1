# start.ps1 — Launch API + Frontend (Windows PowerShell)

$Root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Uvicorn  = Join-Path $Root ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $Uvicorn)) {
    Write-Error "virtualenv not found. Run 'poetry install' first."
    exit 1
}

Write-Host "Starting API on http://localhost:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$Root\backend'; & '$Uvicorn' api:app --reload --port 8000" `
    -WindowStyle Normal

Write-Host "Starting Frontend on http://localhost:5173 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$Root\frontend'; npm run dev" `
    -WindowStyle Normal

Write-Host ""
Write-Host "Both services launched in separate windows."
