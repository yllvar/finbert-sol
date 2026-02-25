# 🦅 FINBERT-SOL: Hybrid AI Trading System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Strategy](https://img.shields.io/badge/Strategy-Generating%20Alpha-green.svg)](docs/reference/paper-summary.md)

A sophisticated, hybrid AI-driven trading system for SOL/USDC perpetuals on Hyperliquid, inspired by the methodology described in the **"Generating Alpha"** research paper.

---

## 🌟 Overview

**FINBERT-SOL** combines traditional technical analysis with modern machine learning and NLP to create a robust trading strategy. By integrating real-time market data, sentiment analysis via FinBERT, and XGBoost-based directional predictions, the system aims to identify high-probability trading opportunities while strictly managing risk.

### Core Mission
- **Sentiment Integration**: Leveraging FinBERT to quantify market sentiment from news and social signals.
- **Predictive Modeling**: Using XGBoost to predict price direction based on 70+ order flow and technical features.
- **Regime Awareness**: Automatically detecting market regimes (Bull/Bear) to adapt trading behavior.
- **Hyperliquid Execution**: Direct integration with Hyperliquid for low-latency perpetual trading.

---

## 🏗️ System Architecture

The project follows a modular pipeline architecture designed for scalability and reliability:

1.  **Data Layer**: Ingests historical parquet data and live Hyperliquid order flow/funding rates.
2.  **Feature Engine**: Computes 70+ indicators, including order book imbalance, toxicity, and technical signals.
3.  **ML Pipeline**:
    *   **FinBERT**: Extracts sentiment scores from text data.
    *   **XGBoost**: Predicts next-period return direction.
    *   **Regime Detector**: Classifies current market state.
4.  **Risk Engine**: Validates every signal against ATR-based position sizing and global drawdown limits.
5.  **Execution Engine**: Handles order placement, monitoring, and state management on Hyperliquid.

---

## 🛠️ Tech Stack

- **Deep Learning**: `torch`, `transformers` (FinBERT)
- **Machine Learning**: `xgboost`, `scikit-learn`
- **Data Processing**: `pandas`, `numpy`, `pyarrow`
- **Trading/Backtesting**: `backtrader`
- **Exchange API**: `hyperliquid`
- **Automation**: `schedule`

---

## 📁 Repository Structure

```text
├── config/             # System and API configurations
├── data/               # Local lakehouse for historical parquet files
├── docs/               # Advanced documentation (Phase-based)
├── models/             # Serialized model weights and checkpoints
├── notebooks/          # Research and EDA notebooks
├── scripts/            # Entry points and demonstration scripts
├── src/
│   ├── data/           # API clients and data loaders
│   ├── features/       # Feature engineering and transformation
│   ├── models/         # Implementation of FinBERT and XGBoost
│   ├── production/     # Trading bot, execution engine, and monitoring
│   ├── risk/           # Risk management and position sizing
│   └── strategies/     # Market regime detection and signal logic
└── tests/              # Unit and integration test suite
```

---

## 🚀 Getting Started

### 1. Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run System Demo
To see the entire pipeline in action (Data -> ML -> Mock Execution):
```bash
python scripts/run_system_demo.py
```

### 3. Comprehensive Documentation
For detailed setup, methodology, and phase-by-phase implementation:
👉 **[View Full Documentation](docs/README.md)**

---

## 📊 Project Status

| Phase | Component | Status | Documentation |
|-------|-----------|--------|---------------|
| Phase 1 | Foundation | ✅ Done | [Foundation](docs/phases/phase-1-foundation.md) |
| Phase 2 | ML Pipeline | ✅ Done | [ML Pipeline](docs/phases/phase-2-ml-pipeline.md) |
| Phase 3 | Strategy | ✅ Done | [Trading Strategy](docs/phases/phase-3-trading-strategy.md) |
| Phase 4 | Production | 🚧 In Progress | [Production](docs/phases/phase-4-production.md) |

---

## 📖 Key References
- **Target Paper**: "Generating Alpha" (Methodology Replication)
- **Exchange**: [Hyperliquid](https://hyperliquid.xyz)
- **Model**: [FinBERT](https://huggingface.co/ProsusAI/finbert)

---

*Last updated: 2026-02-25*
