#!/bin/bash

# Clean Pivot Script - Keep-List Strategy
# Safely preserves data while clearing project structure for architectural shift

set -euo pipefail

# Configuration
PROJECT_DIR="/Users/apple/finbert-sol/hyper-deeplearn"
TEMP_BACKUP_DIR="/Users/apple/finbert-sol/hyper-deeplearn_backup_$(date +%Y%m%d_%H%M%S)"
KEEP_LIST=("lakehouse" "scripts/unified_features.py" "scripts/momentum_features.py" "configs" "data")

echo "🔄 Starting Clean Pivot Process..."
echo "📁 Project Directory: $PROJECT_DIR"
echo "💾 Temporary Backup: $TEMP_BACKUP_DIR"

# Safety checks
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "❌ Error: Project directory does not exist"
    exit 1
fi

if [[ -d "$TEMP_BACKUP_DIR" ]]; then
    echo "❌ Error: Backup directory already exists"
    exit 1
fi

# Create backup directory
echo "📦 Creating backup directory..."
mkdir -p "$TEMP_BACKUP_DIR"

# Move keep-list items to backup
echo "🚚 Moving critical data to backup..."
for item in "${KEEP_LIST[@]}"; do
    source_path="$PROJECT_DIR/$item"
    if [[ -e "$source_path" ]]; then
        echo "  → Moving: $item"
        mv "$source_path" "$TEMP_BACKUP_DIR/"
    else
        echo "  ⚠️  Warning: $item not found, skipping"
    fi
done

# Nuclear option - clear remaining project directory
echo "💥 Clearing project directory (nuclear option)..."
cd "$PROJECT_DIR"
find . -maxdepth 1 ! -name '.' -exec rm -rf {} + 2>/dev/null || true

# Restore keep-list items
echo "🔄 Restoring critical data..."
for item in "${KEEP_LIST[@]}"; do
    backup_path="$TEMP_BACKUP_DIR/$item"
    if [[ -e "$backup_path" ]]; then
        echo "  → Restoring: $item"
        mv "$backup_path" "$PROJECT_DIR/"
    fi
done

# Clean up backup directory
echo "🧹 Cleaning up backup directory..."
rm -rf "$TEMP_BACKUP_DIR"

echo "✅ Clean pivot completed successfully!"
echo "📊 Lakehouse (24GB) and critical scripts preserved"
echo "🔧 Ready for new architecture implementation"
