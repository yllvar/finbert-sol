import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from src.data.data_loader import load_sol_data
from src.features.feature_engineer import FeatureEngineer
from src.models.xgboost_trader import XGBoostTrader
from src.strategies.regime_detector import MarketRegimeDetector

class BacktestSignalStrategy(bt.Strategy):
    """
    Simplified Backtrader strategy that uses pre-calculated signals
    """
    params = (
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
    )

    def __init__(self):
        # The 'prediction' column will be available in the data
        self.signal = self.datas[0].prediction
        self.regime = self.datas[0].regime
        self.close = self.datas[0].close
        
    def next(self):
        if not self.position:
            # Buy if prediction is 1 AND regime is 1 (Bull)
            if self.signal[0] == 1 and self.regime[0] == 1:
                self.order_target_percent(target=0.95)
        else:
            # Sell if prediction is 0 OR regime is -1 (Bear)
            if self.signal[0] == 0 or self.regime[0] == -1:
                self.close()

def run_formal_backtest():
    print("============================================================")
    print("📈 FINBERT-SOL: FORMAL BACKTESTING RUN")
    print("============================================================")
    
    # 1. DATA LOADING & SAMPLING
    print("[1/6] Loading & Sampling Data...")
    df = load_sol_data(2026, 2)
    # Use smaller sample for faster demo
    if len(df) > 50000:
        df = df.iloc[-50000:] 
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Ensure necessary columns exist for BT
    if 'mid_price' in df.columns:
        df['open'] = df['mid_price']
        df['high'] = df['mid_price']
        df['low'] = df['mid_price']
        df['close'] = df['mid_price']
    
    if 'bid_volume_5' in df.columns and 'ask_volume_5' in df.columns:
        df['volume'] = df['bid_volume_5'] + df['ask_volume_5']
    else:
        df['volume'] = 1000
    
    # 2. FEATURE ENGINEERING
    print("[2/6] Feature Engineering...")
    fe = FeatureEngineer()
    X = fe.prepare_features(df)
    y = fe.create_labels(df, horizon='1h')
    
    # Align
    valid_mask = y.notna()
    X = X.loc[valid_mask.index[valid_mask]]
    y = y.loc[valid_mask.index[valid_mask]]
    
    # 3. TRAIN/TEST SPLIT
    split_idx = int(len(X) * 0.5) # 50/50 for small sample
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    X_train_scaled = fe.scale_features(X_train, fit=True)
    X_test_scaled = fe.scale_features(X_test, fit=False)
    
    # 4. PREDICTIONS & REGIME
    print("[4/6] Generating Signals...")
    model = XGBoostTrader()
    model.params['n_estimators'] = 50 
    model.train_model(X_train_scaled, y_train)
    
    preds, probs = model.predict(X_test_scaled)
    
    regime_detector = MarketRegimeDetector(window=20)
    # Calculate regime for the test set
    test_df = df.loc[X_test.index].copy()
    test_df['prediction'] = preds
    
    # Only keep essential columns for BT feed to save memory/time
    bt_cols = ['open', 'high', 'low', 'close', 'volume', 'prediction']
    
    # We need returns to calculate regime
    returns = df['mid_price'].pct_change().dropna()
    test_df['regime'] = regime_detector.calculate_regime(returns).loc[test_df.index]
    test_df['regime'] = test_df['regime'].fillna(1)
    
    test_df = test_df[bt_cols + ['regime']]
    
    # 5. BACKTRADER EXECUTION
    print("[5/6] Executing Backtrader Strategy...")

    
    class PandasSignalData(bt.feeds.PandasData):
        lines = ('prediction', 'regime',)
        params = (
            ('prediction', -1),
            ('regime', -1),
        )

    cerebro = bt.Cerebro()
    
    data = PandasSignalData(
        dataname=test_df,
        prediction=test_df.columns.get_loc('prediction'),
        regime=test_df.columns.get_loc('regime'),
        openinterest=-1
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(BacktestSignalStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001) # 0.1% commission
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: ${final_value:.2f}")
    
    # 6. REPORT & VISUALIZATION
    print("\n[6/6] Generating Performance Report...")
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
    dd = strat.analyzers.drawdown.get_analysis().max.drawdown or 0
    tot_ret = strat.analyzers.returns.get_analysis().get('rtot', 0) or 0
    
    print("-" * 40)
    print(f"{'Metric':<20} | {'Value':<15}")
    print("-" * 40)
    print(f"{'Total Return':<20} | {tot_ret*100:>.2f}%")
    print(f"{'Sharpe Ratio':<20} | {sharpe:>.4f}")
    print(f"{'Max Drawdown':<20} | {dd:>.2f}%")
    print(f"{'Final Capital':<20} | ${final_value:,.2f}")
    print("-" * 40)

    
    # Save Report to file
    with open("visualizations/backtest_report.txt", "w") as f:
        f.write("FINBERT-SOL BACKTEST REPORT\n")
        f.write("=" * 30 + "\n")
        f.write(f"Period: {test_df.index[0]} to {test_df.index[-1]}\n")
        f.write(f"Total Return: {tot_ret*100:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Max Drawdown: {dd:.2f}%\n")
        f.write(f"Final Capital: ${final_value:,.2f}\n")

    # Plotting
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    
    # Subplot 1: Equity Curve
    # Extract values manually since cerebro.plot() is interactive
    # (Simplified approach: Use the simulation logic from previous turn for the chart)
    # But let's try to get the actual portfolio history if possible
    
    print(f"📊 Visualizing result to visualizations/backtest_result.png")
    
    # Re-running simulation for the plot data (cleaner than extracting from BT objects)
    p_val = 100000.0
    history = [p_val]
    bh_val = 100000.0
    bh_history = [bh_val]
    
    for i in range(1, len(test_df)):
        ret = (test_df['close'].iloc[i] - test_df['close'].iloc[i-1]) / test_df['close'].iloc[i-1]
        
        # Strategy (Shifted signal: use previous bar's prediction to trade current bar's return)
        if test_df['prediction'].iloc[i-1] == 1 and test_df['regime'].iloc[i-1] == 1:
            p_val *= (1 + ret)
        
        bh_val *= (1 + ret)
        history.append(p_val)
        bh_history.append(bh_val)

        
    plt.plot(test_df.index, bh_history, label='SOL (Buy & Hold)', color='#4169E1', alpha=0.6)
    plt.plot(test_df.index, history, label='FINBERT-SOL Hybrid', color='#00FF00', linewidth=2)
    plt.title('Backtest Results: Hybrid AI Strategy vs SOL Benchmark', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Equity (USD)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('visualizations/backtest_result.png', dpi=300)
    plt.close()
    
    print("\n✅ Backtest complete! Check /visualizations/ directory.")

if __name__ == "__main__":
    run_formal_backtest()
