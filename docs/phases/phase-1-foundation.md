# Phase 1: Foundation (Week 1-2)

## 🎯 Objectives
Build the data pipeline foundation and validate existing infrastructure.

## 📋 Prerequisites
- Python 3.8+ installed
- Access to existing SOL lakehouse data
- Hyperliquid API access (testnet recommended)

---

## 🛠️ Step 1.1: Environment Setup

### Core Dependencies
```bash
pip install torch>=2.0.0 transformers>=4.30.0 xgboost>=1.7.0
pip install pandas>=2.0.0 numpy>=1.24.0 pyarrow>=12.0.0
pip install backtrader>=1.9.76.123 requests>=2.31.0
pip install scikit-learn>=1.3.0
```

### Project Structure
```
/Users/apple/finbert-sol/
├── data/                    # Existing data
├── lakehouse/              # SOL historical data
├── configs/                # Configuration files
├── src/                    # Source code
│   ├── data/              # Data pipeline
│   ├── models/            # ML models
│   ├── strategies/        # Trading strategies
│   └── utils/             # Utilities
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

---

## 📊 Step 1.2: Data Validation

### Load Existing Data
```python
import pyarrow.parquet as pq
import pandas as pd

def load_sol_data(year=2024, month=1):
    """Load SOL parquet data for specific year/month"""
    file_path = f"/Users/apple/finbert-sol/lakehouse/consolidated/SOL/year={year}/month={month:02d}/features.parquet"
    table = pq.read_table(file_path)
    return table.to_pandas()

# Validate data
df = load_sol_data(2024, 1)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Data Quality Checks
```python
def validate_data_quality(df):
    """Perform basic data quality validation"""
    checks = {
        'null_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    return checks

# Expected features (70+ columns)
expected_features = [
    'best_bid', 'best_ask', 'mid_price', 'spread', 'depth_imbalance_5',
    'ofi_ratio', 'roc_1h', 'sma_diff_1h', 'golden_cross', 'bb_width_1h',
    'btc_close', 'sol_ret_1h', 'btc_ret_1h', 'rel_strength_1h'
]

missing_features = [f for f in expected_features if f not in df.columns]
```

---

## 🔌 Step 1.3: Hyperliquid API Integration

### Basic API Client
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class HyperliquidAPI:
    def __init__(self, base_url="https://api.hyperliquid.xyz"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_perp_trades(self, symbol="SOL-USD-PERP", limit=1000):
        """Fetch recent perpetual trades"""
        endpoint = f"{self.base_url}/info/perps/trades"
        params = {"symbol": symbol, "limit": limit}
        
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        df['timestamp'] = pd.to_datetime(df['time'])
        return df.set_index('timestamp')
    
    def get_funding_rates(self, symbol="SOL-USD-PERP"):
        """Fetch funding rate history"""
        endpoint = f"{self.base_url}/info/perps/funding"
        params = {"symbol": symbol}
        
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    
    def get_order_book(self, symbol="SOL-USD-PERP"):
        """Fetch current order book"""
        endpoint = f"{self.base_url}/info/perps/orderbook"
        params = {"symbol": symbol}
        
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

# Usage
api = HyperliquidAPI()
trades = api.get_perp_trades()
funding = api.get_funding_rates()
```

### Convert to OHLCV Format
```python
def trades_to_ohlcv(trades_df, freq='1H'):
    """Convert raw trades to OHLCV bars"""
    ohlcv = trades_df['price'].resample(freq).ohlc()
    volume = trades_df['size'].resample(freq).sum()
    
    result = pd.concat([ohlcv, volume], axis=1)
    result.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Add returns
    result['returns'] = result['close'].pct_change()
    
    return result.dropna()
```

---

## 📈 Step 1.4: Feature Gap Analysis

### Compare with Paper Features
```python
def compare_features(existing_df, paper_features):
    """Compare existing features with paper requirements"""
    
    # Paper's 10 core features
    paper_core = {
        'EMA_50': lambda df: df['close'].ewm(span=50).mean(),
        'EMA_200': lambda df: df['close'].ewm(span=200).mean(),
        'MACD': lambda df: calculate_macd(df['close']),
        'RSI_14': lambda df: calculate_rsi(df['close'], 14),
        'Bollinger_Upper': lambda df: calculate_bollinger_upper(df['close']),
        'Bollinger_Lower': lambda df: calculate_bollinger_lower(df['close']),
        'ATR': lambda df: calculate_atr(df),
        'Volatility': lambda df: df['close'].rolling(20).std(),
        'EMA_Ratio': lambda df: df['EMA_50'] / df['EMA_200'],
        'MACD_Histogram': lambda df: df['MACD'] - df['MACD_Signal']
    }
    
    # Check which features exist
    existing_features = set(existing_df.columns)
    missing_features = []
    
    for feature, calc_func in paper_core.items():
        if feature.lower() not in [c.lower() for c in existing_features]:
            missing_features.append(feature)
    
    return missing_features

# Technical indicator calculations
def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal).mean()
    return macd

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_upper(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return sma + (std * num_std)

def calculate_bollinger_lower(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return sma - (std * num_std)

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()
```

---

## ✅ Phase 1 Deliverables

### Required Files
- [ ] `src/data/data_loader.py` - Data loading utilities
- [ ] `src/data/hyperliquid_api.py` - API client implementation
- [ ] `src/utils/technical_indicators.py` - Feature calculations
- [ ] `notebooks/01_data_exploration.ipynb` - Data validation notebook

### Success Criteria
- [ ] Load existing parquet files without errors
- [ ] API client fetches current SOL/USDC data
- [ ] Feature completeness > 95%
- [ ] Data quality validation passes
- [ ] OHLCV conversion working correctly

### Validation Commands
```bash
# Test data loading
python -c "from src.data.data_loader import load_sol_data; df = load_sol_data(2024, 1); print(df.shape)"

# Test API connection
python -c "from src.data.hyperliquid_api import HyperliquidAPI; api = HyperliquidAPI(); print(api.get_perp_trades().shape)"

# Run data validation
python tests/test_data_quality.py
```

---

## 🚨 Common Issues

### Data Loading Problems
- **Memory Issues**: Use chunked loading for large datasets
- **Schema Changes**: Validate column names match expectations
- **Time Zones**: Ensure consistent UTC timestamp handling

### API Connection Issues
- **Rate Limits**: Implement exponential backoff
- **Authentication**: Use testnet for development
- **Data Format**: Handle API response format changes

### Feature Calculation Issues
- **Missing Data**: Forward fill or interpolate gaps
- **Lookahead Bias**: Ensure calculations use only historical data
- **Computational Efficiency**: Vectorize operations where possible

---

## 📚 Next Steps

After completing Phase 1:
1. Review data quality reports
2. Validate API integration
3. Proceed to [Phase 2: Machine Learning](phase-2-ml-pipeline.md)

---

*Phase 1 typically takes 1-2 weeks to complete thoroughly.*
