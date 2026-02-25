import os
import sys
import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from src.strategies.regime_detector import MarketRegimeDetector
from src.strategies.hybrid_ai_strategy import HybridAIStrategy
from src.backtesting.backtest_engine import BacktestEngine
from src.risk.risk_manager import RiskManager

class TestPhase3(unittest.TestCase):
    def setUp(self):
        # Generate mock price data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        close = np.cumsum(np.random.normal(0, 1, 200)) + 100
        self.df = pd.DataFrame({
            'open': close * (1 + np.random.normal(0, 0.001, 200)),
            'high': close * (1 + abs(np.random.normal(0, 0.005, 200))),
            'low': close * (1 - abs(np.random.normal(0, 0.005, 200))),
            'close': close,
            'volume': np.random.uniform(1000, 5000, 200)
        }, index=dates)

    def test_regime_detector(self):
        detector = MarketRegimeDetector(window=20)
        returns = self.df['close'].pct_change().dropna()
        regime = detector.calculate_regime(returns)
        
        self.assertEqual(len(regime), len(returns))
        self.assertTrue(all(r in [1, -1] for r in regime))
        print("✅ MarketRegimeDetector verified")

    def test_risk_manager(self):
        rm = RiskManager()
        # Mock values: current_value, position_size, last_trade_pnl
        passed, checks = rm.check_risk_limits(100000, 0.1, 100)
        self.assertTrue(passed)
        
        # Test drawdown limit
        passed, checks = rm.check_risk_limits(80000, 0.1, -20000)
        self.assertFalse(passed)
        self.assertTrue(checks['max_drawdown_exceeded'])
        print("✅ RiskManager verified")

    def test_backtest_engine_setup(self):
        engine = BacktestEngine(initial_cash=100000)
        engine.setup_cerebro()
        engine.add_data(self.df)
        
        # Mock model and feature engineer
        class MockModel:
            def predict(self, X): return 1, 0.8
        
        class MockFE:
            feature_columns = ['f1', 'f2']
            
        engine.add_strategy(HybridAIStrategy, model=MockModel(), feature_engineer=MockFE())
        results = engine.run_backtest()
        
        self.assertIsNotNone(results)
        metrics = engine.get_performance_metrics()
        self.assertIn('total_return', metrics)
        print(f"✅ BacktestEngine verified. Final Value: ${metrics['final_value']:.2f}")

if __name__ == '__main__':
    print("============================================================")
    print("PHASE 3 VALIDATION TESTS")
    print("============================================================")
    unittest.main()
