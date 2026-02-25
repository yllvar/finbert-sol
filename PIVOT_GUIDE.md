# Clean Pivot Documentation

## Overview
Two strategies for architectural shifts without re-downloading 30GB of historical data.

## Strategy 1: Keep-List Strategy (Recommended)

### Purpose
Clean slate approach while preserving critical data and feature processing recipes.

### What Gets Preserved
- **`lakehouse/`** (24GB of SOL historical data)
- **`scripts/unified_features.py`** (feature processing recipe)
- **`scripts/momentum_features.py`** (momentum calculations)
- **`configs/`** (configuration files)
- **`data/`** (metadata and trade dumps)

### Usage
```bash
./clean_pivot.sh
```

### Process Flow
1. **Isolate**: Move keep-list items to temporary backup
2. **Nuclear**: `rm -rf *` clears project directory
3. **Restore**: Move keep-list items back
4. **Cleanup**: Remove temporary backup

### Safety Features
- Atomic operations with rollback capability
- Preserves feature processing "recipes"
- Zero data loss risk

## Strategy 2: Sub-Repo Pattern

### Purpose
For frequent architectural shifts or multiple projects sharing same data.

### Benefits
- Single copy of 24GB data
- Multiple "Logic Folders" can reference it
- Git version control for data changes
- Disk space efficient

### Usage
```bash
./setup_subrepo.sh
```

### Process Flow
1. **Extract**: Move `lakehouse/` to separate repository
2. **Initialize**: Git repository for data
3. **Link**: Create symbolic link back to project
4. **Share**: Link from any new project

## Critical Components to Preserve

### Feature Processing Scripts
- `scripts/unified_features.py` - Explains parquet file feature meanings
- `scripts/momentum_features.py` - Momentum calculation logic

### Data Structures
- `lakehouse/consolidated/SOL/` - Monthly organized features (2023-2026)
- `lakehouse/enhanced/l2_enhanced/SOL/` - Daily enhanced features
- `training_data/` - BTC candlestick data
- `test_data/` - Sample trades and test datasets

## Safety Warnings

### Sharp Edges
- **Feature Recipes**: Deleting feature scripts loses meaning of parquet data
- **Dependencies**: New architecture must maintain feature compatibility
- **Symbolic Links**: Sub-repo approach requires link maintenance

### Best Practices
1. Always backup before major changes
2. Test pivot on copy first
3. Document any feature schema changes
4. Validate data integrity after pivot

## Quick Start Commands

```bash
# Strategy 1: Clean Pivot
cd /Users/apple/finbert-sol
./clean_pivot.sh

# Strategy 2: Sub-Repo Setup
cd /Users/apple/finbert-sol  
./setup_subrepo.sh
```

## Recovery

If something goes wrong during pivot:
1. Check backup directory: `ls /Users/apple/finbert-sol/hyper-deeplearn_backup_*`
2. Manual restore: Move items back from backup
3. Verify data integrity: Check parquet files are readable

## Post-Pivot Checklist

- [ ] Lakehouse data accessible
- [ ] Feature scripts intact
- [ ] Configurations preserved
- [ ] New architecture can read parquet files
- [ ] Test with sample data
- [ ] Backup old architecture if needed
