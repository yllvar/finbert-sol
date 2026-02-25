import backtrader as bt
import pandas as pd
from datetime import datetime

class BacktestEngine:
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cerebro = None
        self.results = None
        
    def setup_cerebro(self):
        """Initialize Backtrader engine"""
        self.cerebro = bt.Cerebro()
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Set initial cash
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Add commission (0.1% per trade)
        self.cerebro.broker.setcommission(commission=0.001)
        
    def add_data(self, data_frame, from_date=None, to_date=None):
        """Add price data to cerebro"""
        # Convert to Backtrader format
        data = bt.feeds.PandasData(
            dataname=data_frame,
            fromdate=from_date,
            todate=to_date,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(data)
        
    def add_strategy(self, strategy_class, **kwargs):
        """Add trading strategy"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
        
    def run_backtest(self):
        """Execute backtest"""
        print(f"Starting Portfolio Value: ${self.cerebro.broker.getvalue():.2f}")
        
        # Run backtest
        self.results = self.cerebro.run()
        
        print(f"Final Portfolio Value: ${self.cerebro.broker.getvalue():.2f}")
        
        return self.results
    
    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.results:
            return None
            
        strategy = self.results[0]
        
        # Get analyzer results
        sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
        returns_analyzer = strategy.analyzers.returns.get_analysis()
        trades_analyzer = strategy.analyzers.trades.get_analysis()
        
        # Calculate metrics
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        metrics = {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_analyzer.get('sharperatio', None),
            'max_drawdown': drawdown_analyzer.get('max', {}).get('drawdown', None),
            'max_drawdown_pct': (drawdown_analyzer.get('max', {}).get('drawdown', 0) * 100) if drawdown_analyzer.get('max', {}).get('drawdown', None) else None,
            'total_trades': trades_analyzer.get('total', {}).get('closed', 0),
            'won_trades': trades_analyzer.get('won', {}).get('total', 0),
            'lost_trades': trades_analyzer.get('lost', {}).get('total', 0),
            'win_rate': trades_analyzer.get('won', {}).get('total', 0) / max(trades_analyzer.get('total', {}).get('closed', 1), 1),
        }
        
        return metrics
    
    def plot_results(self, title="Backtest Results", iplot=False):
        """Plot backtest results"""
        self.cerebro.plot(style='candlestick', title=title, figsize=(15, 8), iplot=iplot)
