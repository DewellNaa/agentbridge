#!/usr/bin/env bash
# AgentBridge Quick Installer for Linux and macOS
set -euo pipefail

echo "============================================="
echo "  🚀 Installing AgentBridge Local Gateway   "
echo "============================================="

if command -v uv &> /dev/null; then
    echo "Found uv package manager, installing with uv tool..."
    uv tool install agentbridge --upgrade || uv pip install "git+https://github.com/DewellNaa/agentbridge.git"
elif command -v pip3 &> /dev/null; then
    echo "Installing via pip3..."
    pip3 install --upgrade "git+https://github.com/DewellNaa/agentbridge.git"
elif command -v pip &> /dev/null; then
    echo "Installing via pip..."
    pip install --upgrade "git+https://github.com/DewellNaa/agentbridge.git"
else
    echo "Error: Neither uv nor pip was found. Please install Python 3.10+ first."
    exit 1
fi

echo ""
echo "✅ AgentBridge installed successfully!"
echo "Run 'agentbridge --help' or 'agentbridge start' to get started."
