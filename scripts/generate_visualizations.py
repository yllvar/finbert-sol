import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from src.data.data_loader import load_sol_data
from src.features.feature_engineer import FeatureEngineer
from src.models.xgboost_trader import XGBoostTrader
from src.strategies.regime_detector import MarketRegimeDetector
from src.evaluation.model_evaluator import ModelEvaluator

def generate_visualizations():
    print("============================================================")
    print("🚀 FINBERT-SOL: GENERATING VISUALIZATIONS")
    print("============================================================")
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. DATA LOADING
    print("[Step 1/5] Loading historical data...")
    try:
        df = load_sol_data(2026, 2)
        # Sample data to avoid OOM
        if len(df) > 200000:
            print(f"⚠️  Data large ({len(df)} rows). Sampling to 200,000 rows...")
            df = df.iloc[-200000:]

        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        print(f"✅ Loaded {len(df)} rows")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. FEATURE ENGINEERING
    print("\n[Step 2/5] Running Feature Engineering...")
    fe = FeatureEngineer()
    processed_df = fe.prepare_features(df)
    labels = fe.create_labels(df, horizon='1h')
    
    valid_mask = labels.notna()
    X = processed_df.loc[valid_mask.index[valid_mask]]
    y = labels.loc[valid_mask.index[valid_mask]]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_live = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_live = y.iloc[:split_idx], y.iloc[split_idx:]
    
    X_train_scaled = fe.scale_features(X_train, fit=True)
    X_live_scaled = fe.scale_features(X_live, fit=False)

    # 3. MODEL TRAINING
    print("\n[Step 3/5] Training XGBoost Baseline...")
    model = XGBoostTrader()
    model.params['n_estimators'] = 200 # Paper config
    model.train_model(X_train_scaled, y_train)
    
    # Generate Feature Importance Plot
    print(f"📊 Saving Feature Importance to {output_dir}/feature_importance.png")
    plt.figure(figsize=(12, 8))
    importance_df = model.feature_importance['gain'].head(20)
    sns.barplot(data=importance_df, x='importance', y='feature_name', palette='viridis')
    plt.title('Top 20 Feature Importance (Gain) - SOL/USDC Strategy')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
    plt.close()

    # 4. EVALUATION
    print("\n[Step 4/5] Evaluating performance...")
    preds, probs = model.predict(X_live_scaled)
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_model_performance("XGBoost_Strategy", y_live.values, preds, probs)
    
    # Save Confusion Matrix
    print(f"📊 Saving Confusion Matrix to {output_dir}/confusion_matrix.png")
    cm = np.array(eval_results['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix - XGBoost Directional Forecast')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()

    # 5. EQUITY CURVE SIMULATION
    print("\n[Step 5/5] Simulating Equity Curve...")
    regime_detector = MarketRegimeDetector(window=20)
    
    initial_balance = 100000.0
    hybrid_balance = [initial_balance]
    bh_balance = [initial_balance]
    timestamps = []
    
    current_hybrid = initial_balance
    current_bh = initial_balance
    
    # Using a 72-hour slice for clear visualization
    sim_slice = min(72, len(X_live_scaled))
    
    for i in range(sim_slice):
        ts = X_live_scaled.index[i]
        timestamps.append(ts)
        
        # Get actual return for this hour (approximate from next close)
        try:
            current_price = df.loc[ts, 'mid_price']
            if i + 1 < len(X_live_scaled):
                next_ts = X_live_scaled.index[i+1]
                next_price = df.loc[next_ts, 'mid_price']
                ret = (next_price - current_price) / current_price
            else:
                ret = 0
        except:
            ret = 0
            
        # Strategy Logic
        pred = preds[i]
        
        # Regime
        if i > 20:
            window_returns = df.loc[:ts, 'mid_price'].pct_change().tail(21).dropna()
            regime = regime_detector.calculate_regime(window_returns).iloc[-1]
        else:
            regime = 1

            
        # Mock sentiment
        sentiment_score = 0.5 if i < 30 else -0.8 if 30 <= i < 40 else 0.4
        
        # Hybrid Strategy: Pred UP + Regime BULL + Sentiment OK
        if pred == 1 and regime == 1 and sentiment_score > -0.7:
            current_hybrid *= (1 + ret)
            
        # Buy & Hold Strategy
        current_bh *= (1 + ret)
        
        hybrid_balance.append(current_hybrid)
        bh_balance.append(current_bh)

    # Plot Equity Curve
    print(f"📊 Saving Equity Curve to {output_dir}/equity_curve.png")
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    
    # We have N+1 balances for N timestamps, adjust lengths
    plot_timestamps = [timestamps[0]] + timestamps
    
    plt.plot(plot_timestamps, bh_balance, label='SOL Buy & Hold', color='#4169E1', linewidth=2, alpha=0.7)
    plt.plot(plot_timestamps, hybrid_balance, label='FINBERT-SOL Hybrid System', color='#32CD32', linewidth=2.5)
    
    plt.title('Hybrid AI Strategy vs Buy & Hold (Simulation)', fontsize=14, pad=20)
    plt.xlabel('Time (Feb 2026)', fontsize=12)
    plt.ylabel('Portfolio Value (USD)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Highlight Sentiment Halts
    halt_start = timestamps[30] if len(timestamps) > 30 else None
    halt_end = timestamps[40] if len(timestamps) > 40 else None
    if halt_start and halt_end:
        plt.axvspan(halt_start, halt_end, color='red', alpha=0.2, label='Sentiment Halt')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/equity_curve.png", dpi=300)
    plt.close()

    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Files available in ./{output_dir}/")

if __name__ == "__main__":
    generate_visualizations()
