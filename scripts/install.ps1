# AgentBridge Quick Installer for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  🚀 Installing AgentBridge Local Gateway   " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Found uv package manager, installing with uv tool..." -ForegroundColor Green
    try {
        uv tool install agentbridge --upgrade
    } catch {
        uv pip install "git+https://github.com/DewellNaa/agentbridge.git"
    }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Installing via python -m pip..." -ForegroundColor Green
    python -m pip install --upgrade "git+https://github.com/DewellNaa/agentbridge.git"
} else {
    Write-Error "Neither uv nor python was found in PATH. Please install Python 3.10+ first."
}

Write-Host ""
Write-Host "✅ AgentBridge installed successfully!" -ForegroundColor Green
Write-Host "Run 'agentbridge --help' or 'agentbridge start' to get started." -ForegroundColor Yellow
