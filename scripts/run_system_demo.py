import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from src.data.data_loader import load_sol_data
from src.features.feature_engineer import FeatureEngineer
from src.models.xgboost_trader import XGBoostTrader
from src.models.finbert_sentiment import FinBERTSentiment
from src.strategies.regime_detector import MarketRegimeDetector
from src.backtesting.backtest_engine import BacktestEngine
from src.risk.risk_manager import RiskManager
from src.production.execution_engine import TradingExecutionEngine
from src.production.hyperliquid_trader import HyperliquidPaperTrader

def run_system_demo():
    print("============================================================")
    print("🚀 FINBERT-SOL: SYSTEM-WIDE DEMO RUN")
    print("============================================================")
    
    # 1. DATA LOADING
    print("\n[Step 1/6] Loading historical data...")
    try:
        # Using 2026-02 data as it's the most recent available
        df = load_sol_data(2026, 2)
        # Ensure timestamp is index for both
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        print(f"✅ Loaded {len(df)} rows of data from Feb 2026")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. FEATURE ENGINEERING
    print("\n[Step 2/6] Running Feature Engineering...")
    fe = FeatureEngineer()
    processed_df = fe.prepare_features(df)
    labels = fe.create_labels(df, horizon='1h')
    
    # Align and clean
    # Since both now have DatetimeIndex, this should work
    valid_mask = labels.notna()
    X = processed_df.loc[valid_mask.index[valid_mask]]
    y = labels.loc[valid_mask.index[valid_mask]]
    
    # Split into train/test
    # We'll use 80% for "previous history" and 20% for our "demo dry run"
    split_idx = int(len(X) * 0.8)
    X_train, X_live = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_live = y.iloc[:split_idx], y.iloc[split_idx:]
    
    X_train_scaled = fe.scale_features(X_train, fit=True)
    X_live_scaled = fe.scale_features(X_live, fit=False)
    
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Demo samples: {len(X_live)}")

    # 3. MODEL TRAINING (Simulated History)
    print("\n[Step 3/6] Training XGBoost Baseline...")
    model = XGBoostTrader()
    # Training (using small estimators for speed in demo)
    model.params['n_estimators'] = 50 
    model.train_model(X_train_scaled, y_train)
    print("✅ Model trained successfully")

    # 4. COMPONENTS INITIALIZATION
    print("\n[Step 4/6] Initializing Production Components...")
    # Mock news for sentiment
    sentiment = None # We'll mock the score in the loop for demo
    regime_detector = MarketRegimeDetector(window=20)
    risk_manager = RiskManager()
    trader = HyperliquidPaperTrader()
    trader.setup_paper_account()
    
    print("✅ Components ready")

    # 5. LIVE SIMULATION LOOP
    print("\n[Step 5/6] Starting Multi-Step Trading Simulation...")
    print(f"{'Time':<20} | {'Price':<8} | {'Pred':<5} | {'Regime':<8} | {'Action':<10}")
    print("-" * 65)
    
    portfolio_value = 100000.0
    trades = 0
    
    # Run loop over the 'live' slice
    for i in range(min(48, len(X_live_scaled))): # Simulate 48 hours
        timestamp = X_live_scaled.index[i]
        current_features = X_live_scaled.iloc[[i]]
        current_price = df.loc[timestamp, 'close'] if 'close' in df.columns else 100.0
        
        # Prediction
        pred, prob = model.predict(current_features)
        
        # Regime (calculating on the fly from window)
        if i > 20:
            window_returns = df.loc[:timestamp, 'close'].pct_change().tail(21).dropna()
            regime = regime_detector.calculate_regime(window_returns).iloc[-1]
        else:
            regime = 1 # Start bullish
            
        # Mock sentiment (neutral-bullish)
        sentiment_score = 0.5 
        
        action = "HOLD"
        
        # Trading logic
        if pred == 1 and regime == 1 and sentiment_score > -0.7:
            # Check Risk
            risk_ok, _ = risk_manager.check_risk_limits(portfolio_value, 10, 0)
            if risk_ok:
                action = "BUY"
                trades += 1
                trader.place_order("SOL-USD-PERP", "buy", "market", 10)
        
        print(f"{str(timestamp)[:19]:<20} | {current_price:8.2f} | {pred:<5} | {'Bull' if regime==1 else 'Bear':<8} | {action:<10}")

    # 6. SUMMARY
    print("\n[Step 6/6] Simulation Summary")
    print("============================================================")
    print(f"Total Period: {len(X_live_scaled)} hours")
    print(f"Total Trades Executed: {trades}")
    print(f"Final Balance (Mock): ${portfolio_value:,.2f}")
    print("✅ System-wide dry run complete!")
    print("============================================================")

if __name__ == "__main__":
    run_system_demo()
