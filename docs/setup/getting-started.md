# Getting Started

## 🎯 Quick Start Guide

Welcome to the SOL/USDC Perpetual Trading System! This guide will get you up and running in 30 minutes.

---

## 🚀 5-Minute Setup

### 1. Clone and Setup
```bash
cd /Users/apple/finbert-sol

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch>=2.0.0 transformers>=4.30.0 xgboost>=1.7.0
pip install pandas>=2.0.0 numpy>=1.24.0 pyarrow>=12.0.0
pip install backtrader>=1.9.76.123 requests>=2.31.0 scikit-learn>=1.3.0
```

### 2. Configure Environment
```bash
# Create environment file
cat > .env << EOF
# API Configuration
HYPERLIQUID_API_URL=https://api.hyperliquid.xyz
PAPER_TRADING=true

# Data Configuration
DATA_ROOT=/Users/apple/finbert-sol/lakehouse
MODEL_ROOT=/Users/apple/finbert-sol/models
LOG_ROOT=/Users/apple/finbert-sol/logs
EOF
```

### 3. Validate Setup
```bash
# Check environment
python -c "import torch, pandas, pyarrow; print('✅ Core packages working')"

# Check data
python -c "import pyarrow.parquet as pq; print('✅ Data access working')"
```

---

## 📊 Verify Your Data

### Check SOL Lakehouse Data
```python
# Quick data validation
import pyarrow.parquet as pq
from pathlib import Path

def quick_data_check():
    lakehouse_path = Path("/Users/apple/finbert-sol/lakehouse")
    
    # Check consolidated data
    consolidated = lakehouse_path / "consolidated" / "SOL" / "year=2024" / "month=01"
    if consolidated.exists():
        table = pq.read_table(str(consolidated / "features.parquet"))
        print(f"✅ Found SOL data: {table.num_rows} rows, {len(table.schema)} columns")
        print(f"Sample columns: {list(table.schema.names)[:5]}")
    else:
        print("❌ No SOL data found")

quick_data_check()
```

---

## 🎯 First Steps

### Option 1: Explore Your Data
```python
# notebooks/01_data_exploration.ipynb
import pyarrow.parquet as pq
import pandas as pd

# Load sample data
table = pq.read_table("/Users/apple/finbert-sol/lakehouse/consolidated/SOL/year=2024/month=01/features.parquet")
df = table.to_pandas()

# Basic exploration
print(f"Data shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Sample features: {list(df.columns)[:10]}")

# Plot some features
df[['mid_price', 'spread']].plot(figsize=(12, 6))
```

### Option 2: Test ML Pipeline
```python
# notebooks/02_ml_baseline.ipynb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = load_sol_data()  # Your data loading function
features = df[['mid_price', 'spread', 'depth_imbalance_5', 'ofi_ratio']]
labels = (df['sol_ret_1h'].shift(-1) > 0).astype(int)

# Simple baseline model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features.fillna(0))

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled[:-100], labels[:-100])

# Test accuracy
accuracy = model.score(X_scaled[-100:], labels[-100:])
print(f"Baseline accuracy: {accuracy:.3f}")
```

### Option 3: Paper Trading Test
```python
# notebooks/03_paper_trading_test.ipynb
import requests

# Test Hyperliquid API
response = requests.get("https://api.hyperliquid.xyz/v1/info/perps/trades?symbol=SOL-USD-PERP&limit=10")
if response.status_code == 200:
    trades = response.json()
    print(f"✅ API working: {len(trades)} trades retrieved")
    print(f"Latest trade: {trades[0]}")
else:
    print(f"❌ API error: {response.status_code}")
```

---

## 📋 Implementation Path

### 🏃‍♂️ Fast Track (2 weeks)
If you want to get trading quickly:

1. **Week 1**: Use your existing data + simple XGBoost model
2. **Week 2**: Basic Backtrader strategy + paper trading

### 🎓 Academic Track (8 weeks)
Follow the complete [implementation plan](../IMPLEMENTATION_PLAN.md):

1. **Phase 1** (Week 1-2): Foundation
2. **Phase 2** (Week 3-4): Machine Learning  
3. **Phase 3** (Week 5-6): Trading Strategy
4. **Phase 4** (Week 7-8): Production

---

## 🎯 What You'll Build

### Core Components
- **Data Pipeline**: Load your 24GB SOL lakehouse
- **ML Models**: XGBoost + FinBERT sentiment analysis
- **Trading Strategy**: Backtrader implementation
- **Paper Trading**: Hyperliquid integration
- **Monitoring**: Performance tracking and alerts

### Expected Performance
Based on the "Generating Alpha" paper:
- **XGBoost Accuracy**: ~63% (baseline)
- **Sharpe Ratio**: 1.5-2.0 (target)
- **Max Drawdown**: <15% (risk limit)
- **Win Rate**: >45% (minimum)

---

## 🛠️ Development Workflow

### Daily Development
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run tests
python -m pytest tests/

# 3. Start development
jupyter notebook notebooks/
```

### Project Structure
```
/Users/apple/finbert-sol/
├── docs/           # This documentation
├── src/            # Your source code
├── notebooks/      # Jupyter notebooks
├── tests/          # Unit tests
├── lakehouse/      # Your SOL data (24GB)
├── configs/        # Configuration files
└── data/           # Additional data
```

---

## 🚨 Common First Steps

### Fix Missing Feature Scripts
The clean pivot may have removed your feature processing scripts. Recreate them:

```python
# src/features/unified_features.py
def calculate_unified_features(df):
    """Recreate your unified features"""
    features = df.copy()
    
    # Add your feature calculations
    features['price_momentum'] = df['mid_price'].pct_change()
    features['volume_ratio'] = df['bid_volume_5'] / (df['ask_volume_5'] + 1e-8)
    
    return features
```

### Test Data Access
```python
# Verify you can read your parquet files
import pyarrow.parquet as pq

try:
    table = pq.read_table("/Users/apple/finbert-sol/lakehouse/consolidated/SOL/year=2024/month=01/features.parquet")
    print(f"✅ Data access working: {table.num_rows} rows")
except Exception as e:
    print(f"❌ Data access failed: {e}")
```

---

## 📚 Resources

### Documentation
- [Full Implementation Plan](../IMPLEMENTATION_PLAN.md)
- [Phase 1: Foundation](../phases/phase-1-foundation.md)
- [Environment Setup](environment.md)

### Reference Materials
- [Paper Summary](../reference/paper-summary.md)
- [Data Schema](../reference/data-schema.md)
- [Feature Reference](../reference/features.md)

### Troubleshooting
- [Common Issues](../appendix/troubleshooting.md)
- [Best Practices](../appendix/best-practices.md)

---

## 🎯 Success Metrics

### Week 1 Success
- [ ] Environment setup complete
- [ ] Can load SOL data without errors
- [ ] Basic ML baseline working
- [ ] API connection tested

### Week 2 Success  
- [ ] XGBoost model trained
- [ ] Backtrader strategy implemented
- [ ] Paper trading test successful
- [ ] Performance metrics calculated

---

## 🚀 Ready to Start?

Choose your path:

1. **Fast Track**: Jump to [Phase 1](../phases/phase-1-foundation.md) and start building
2. **Academic Track**: Read the [full plan](../IMPLEMENTATION_PLAN.md) for comprehensive approach
3. **Explore First**: Open `notebooks/01_data_exploration.ipynb` to understand your data

---

**Happy Trading! 🚀**

---

*Need help? Check the [troubleshooting guide](../appendix/troubleshooting.md) or [best practices](../appendix/best-practices.md).*
