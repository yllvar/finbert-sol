#!/bin/bash

# Sub-Repo Setup Script - Alternative to Keep-List Strategy
# Converts lakehouse to separate Git repository for multiple project access

set -euo pipefail

# Configuration
PROJECT_DIR="/Users/apple/finbert-sol/hyper-deeplearn"
DATA_REPO_DIR="/Users/apple/finbert-sol/lakehouse_data_repo"

echo "🔧 Setting up Sub-Repo Pattern for data sharing..."
echo "📁 Project Directory: $PROJECT_DIR"
echo "📊 Data Repository: $DATA_REPO_DIR"

# Safety checks
if [[ ! -d "$PROJECT_DIR/lakehouse" ]]; then
    echo "❌ Error: lakehouse directory not found"
    exit 1
fi

# Create separate data repository
echo "📦 Creating data repository..."
mkdir -p "$DATA_REPO_DIR"

# Move lakehouse to data repo
echo "🚚 Moving lakehouse to data repository..."
mv "$PROJECT_DIR/lakehouse" "$DATA_REPO_DIR/"

# Initialize Git repo in data directory
echo "🔧 Initializing Git repository..."
cd "$DATA_REPO_DIR"
git init
git add .
git commit -m "Initial commit: SOL historical data (24GB)"

# Create symbolic link back to project
echo "🔗 Creating symbolic link in project..."
ln -s "$DATA_REPO_DIR/lakehouse" "$PROJECT_DIR/lakehouse"

echo "✅ Sub-Repo setup completed!"
echo "📊 Lakehouse now available as: $DATA_REPO_DIR/lakehouse"
echo "🔗 Linked in project as: $PROJECT_DIR/lakehouse"
echo ""
echo "🔄 Usage for new projects:"
echo "  ln -s $DATA_REPO_DIR/lakehouse /path/to/new/project/lakehouse"
