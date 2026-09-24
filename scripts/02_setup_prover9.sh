#!/bin/bash
# Install Prover9 for formal theorem verification

set -e

echo "Installing Prover9..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew not found. Install from https://brew.sh"
        exit 1
    fi
    brew install prover9
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! command -v apt-get &> /dev/null; then
        echo "Error: apt-get not found. Install Prover9 manually from:"
        echo "https://www.cs.unm.edu/~mccune/prover9/"
        exit 1
    fi
    sudo apt-get update
    sudo apt-get install -y prover9
else
    echo "Unsupported OS: $OSTYPE"
    echo "Install Prover9 from: https://www.cs.unm.edu/~mccune/prover9/"
    exit 1
fi

# Verify installation
if command -v prover9 &> /dev/null; then
    echo "✓ Prover9 installed successfully"
    prover9 --version 2>/dev/null || echo "  (version check available after restart)"
else
    echo "✗ Prover9 installation failed"
    exit 1
fi
