import os
import sys
import json
import unittest
from datetime import datetime

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from src.production.hyperliquid_trader import HyperliquidPaperTrader
from src.production.execution_engine import TradingExecutionEngine
from src.production.monitoring import PerformanceMonitor
from src.production.model_retention import ModelRetentionPipeline
from src.production.trading_bot import TradingBot
from src.risk.risk_manager import RiskManager

class TestPhase4(unittest.TestCase):
    def setUp(self):
        # Load config
        with open('config/production_config.json', 'r') as f:
            self.config = json.load(f)
            
        self.trader = HyperliquidPaperTrader()
        self.rm = RiskManager()
        self.rm.max_position_size = 100 # Allow larger units for testing
        self.monitor = PerformanceMonitor()
        self.retention = ModelRetentionPipeline()
        
        # Mock strategy
        class MockStrategy:
            def generate_signals(self, market_data):
                # Return a buy signal for testing
                symbol = list(market_data.keys())[0]
                return {symbol: {'action': 'BUY', 'quantity': 10}}
        
        self.strategy = MockStrategy()
        self.engine = TradingExecutionEngine(self.trader, self.strategy, self.rm, self.config)

    def test_paper_trader_setup(self):
        result = self.trader.setup_paper_account()
        self.assertIsNotNone(self.trader.account_id)
        print("✅ HyperliquidPaperTrader verified")

    def test_execution_engine(self):
        # Initial log count
        initial_logs = len(self.engine.execution_log)
        self.engine.execute_daily_strategy()
        # Should have at least an order log and performance log
        self.assertGreater(len(self.engine.execution_log), initial_logs)
        print("✅ TradingExecutionEngine verified")

    def test_monitoring_alert(self):
        # Force a bad return to trigger alert
        bad_metrics = {'daily_return': -0.06}
        alerts = self.monitor.check_alert_conditions(bad_metrics)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['type'], 'daily_loss')
        
        # Check if alert was logged
        with open(self.monitor.alert_file, 'r') as f:
            log = json.loads(f.readline())
            self.assertEqual(log['type'], 'daily_loss')
        print("✅ PerformanceMonitor verified")

if __name__ == '__main__':
    print("============================================================")
    print("PHASE 4 VALIDATION TESTS")
    print("============================================================")
    unittest.main()
