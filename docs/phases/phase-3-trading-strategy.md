# Phase 3: Trading Strategy (Week 5-6)

## 🎯 Objectives
Implement the paper's trading strategy using Backtrader, including market regime detection and volatility-based position sizing.

## 📋 Prerequisites
- Complete Phase 2: Trained ML models
- Validated feature pipeline
- Working sentiment analysis

---

## 📈 Step 3.1: Market Regime Detection

### Regime Classification (Paper Methodology)
```python
import pandas as pd
import numpy as np

class MarketRegimeDetector:
    def __init__(self, window=20):
        self.window = window
        
    def calculate_regime(self, returns):
        """
        Calculate market regime based on rolling window average of returns
        Bullish (1) if positive, Bearish (-1) if negative
        """
        rolling_mean = returns.rolling(window=self.window).mean()
        regime = np.where(rolling_mean > 0, 1, -1)
        
        return pd.Series(regime, index=returns.index, name='regime')
    
    def calculate_per_asset_regime(self, returns_dict):
        """Calculate regime per asset (paper's approach)"""
        regimes = {}
        
        for asset, returns in returns_dict.items():
            regimes[asset] = self.calculate_regime(returns)
        
        return regimes
    
    def get_regime_signal(self, current_regime):
        """Get trading signal based on regime"""
        # Paper: Regime works as on-off switch for long-sided trades
        return current_regime == 1  # Only trade in bullish regimes
    
    def plot_regime_analysis(self, returns, regime):
        """Plot returns with regime overlay"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        ax1.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
        ax1.set_title('Cumulative Returns')
        ax1.legend()
        
        # Plot regime
        ax2.plot(regime.index, regime, label='Market Regime', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(regime.index, regime, 0, 
                        where=(regime > 0), alpha=0.3, color='green', label='Bullish')
        ax2.fill_between(regime.index, regime, 0, 
                        where=(regime < 0), alpha=0.3, color='red', label='Bearish')
        ax2.set_title('Market Regime Detection')
        ax2.legend()
        ax2.set_ylabel('Regime')
        
        plt.tight_layout()
        plt.show()
```

---

## 🎯 Step 3.2: Backtrader Strategy Implementation

### Paper-Based Trading Strategy
```python
import backtrader as bt
from datetime import datetime
import numpy as np

class HybridAIStrategy(bt.Strategy):
    params = (
        # ML model parameters
        ('model', None),
        ('feature_engineer', None),
        ('sentiment_analyzer', None),
        
        # Position sizing parameters (paper's volatility-based sizing)
        ('position_size_pct', 0.02),  # 2% base position
        ('volatility_window', 20),    # 20-period volatility
        ('max_position_pct', 0.95),   # Max 95% of capital
        
        # Risk management
        ('sentiment_threshold', -0.70),  # Paper's sentiment threshold
        ('stop_loss_pct', 0.05),         # 5% stop loss
        ('take_profit_pct', 0.10),       # 10% take profit
        
        # Regime parameters
        ('regime_window', 20),           # Regime detection window
        ('enable_regime_filter', True),  # Enable regime-based filtering
    )
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_volume = self.datas[0].volume
        
        # ML components
        self.model = self.p.model
        self.feature_engineer = self.p.feature_engineer
        self.sentiment_analyzer = self.p.sentiment_analyzer
        
        # Tracking variables
        self.current_position = None
        self.entry_price = None
        self.regime_detector = MarketRegimeDetector(window=self.p.regime_window)
        self.current_regime = 1  # Default to bullish
        
        # Performance tracking
        self.trade_history = []
        self.portfolio_values = []
        
    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_price = self.data_close[0]
        
        # Calculate market regime
        if len(self.data_close) >= self.p.regime_window:
            recent_returns = pd.Series([self.data_close[-i] / self.data_close[-i-1] - 1 
                                       for i in range(1, self.p.regime_window)])
            self.current_regime = self.regime_detector.calculate_regime(recent_returns).iloc[-1]
        
        # Regime filter (paper's approach)
        if self.p.enable_regime_filter and not self.regime_detector.get_regime_signal(self.current_regime):
            return  # Don't trade in bearish regimes
        
        # Get ML prediction
        try:
            features = self._prepare_features()
            if features is None:
                return
                
            prediction, probability = self.model.predict(features)
            
            # Get sentiment filter
            sentiment_allowed = True
            if self.sentiment_analyzer:
                sentiment_score = self._get_sentiment_score(current_date)
                sentiment_allowed = sentiment_score >= self.p.sentiment_threshold
            
            # Trading logic (paper's long-only approach)
            if not self.position and prediction == 1 and sentiment_allowed:
                # Open long position
                self._open_long_position(current_price, probability[0])
                
            elif self.position and prediction == 0:
                # Close position based on signal
                self.close()
                
        except Exception as e:
            print(f"Error in trading logic: {e}")
    
    def _prepare_features(self):
        """Prepare features for ML prediction"""
        try:
            # Get recent data for feature calculation
            if len(self.data_close) < 50:  # Need enough history
                return None
            
            # Create feature dictionary similar to training data
            recent_data = {
                'close': self.data_close.get(size=50),
                'volume': self.data_volume.get(size=50),
                'high': self.datas[0].high.get(size=50),
                'low': self.datas[0].low.get(size=50),
            }
            
            # This would need to be implemented to match your feature engineering
            # For now, return dummy features
            features = pd.DataFrame(np.random.randn(1, 20))  # Replace with actual features
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def _get_sentiment_score(self, date):
        """Get sentiment score for current date"""
        # This would integrate with your sentiment analyzer
        # For now, return neutral sentiment
        return 0.0
    
    def _open_long_position(self, price, probability):
        """Open long position with volatility-based sizing"""
        # Calculate position size (paper's volatility-based sizing)
        volatility = self._calculate_volatility()
        
        # Volatility-based cash sizing (paper formula)
        if volatility > 0:
            risk_adjusted_size = self.p.position_size_pct / volatility
            position_size = min(risk_adjusted_size, self.p.max_position_pct)
        else:
            position_size = self.p.position_size_pct
        
        # Calculate actual position value
        available_cash = self.broker.getcash()
        position_value = available_cash * position_size
        
        # Execute trade
        size = position_value / price
        self.buy(size=size)
        
        # Track position
        self.current_position = {
            'size': size,
            'entry_price': price,
            'entry_date': self.datas[0].datetime.date(0),
            'probability': probability,
            'volatility': volatility
        }
        
        print(f"Opened LONG position: {size:.4f} units at ${price:.2f}")
    
    def _calculate_volatility(self):
        """Calculate volatility for position sizing"""
        if len(self.data_close) < self.p.volatility_window:
            return 0.02  # Default volatility
        
        recent_prices = self.data_close.get(size=self.p.volatility_window)
        returns = pd.Series(recent_prices).pct_change().dropna()
        volatility = returns.std()
        
        return max(volatility, 0.001)  # Prevent division by zero
    
    def notify_trade(self, trade):
        """Track trade execution"""
        if trade.isclosed:
            trade_info = {
                'date': self.datas[0].datetime.date(0),
                'type': 'LONG' if trade.history[0].event.size > 0 else 'SHORT',
                'entry_price': trade.history[0].event.price,
                'exit_price': trade.history[-1].event.price,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / trade.value * 100,
                'size': trade.history[0].event.size,
                'duration': (trade.history[-1].event.dt - trade.history[0].event.dt).days
            }
            self.trade_history.append(trade_info)
            
            print(f"Trade closed: PnL=${trade.pnl:.2f} ({trade.pnlcomm/trade.value*100:.2f}%)")
    
    def notify_cashvalue(self, cash, value):
        """Track portfolio value"""
        self.portfolio_values.append({
            'date': self.datas[0].datetime.date(0),
            'cash': cash,
            'value': value
        })
```

---

## 📊 Step 3.3: Backtesting Framework

### Backtesting Engine
```python
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
            'sharpe_ratio': sharpe_analyzer['sharperatio'],
            'max_drawdown': drawdown_analyzer['max']['drawdown'],
            'max_drawdown_pct': drawdown_analyzer['max']['drawdown'] * 100,
            'total_trades': trades_analyzer.get('total', {}).get('closed', 0),
            'won_trades': trades_analyzer.get('won', {}).get('total', 0),
            'lost_trades': trades_analyzer.get('lost', {}).get('total', 0),
            'win_rate': trades_analyzer.get('won', {}).get('total', 0) / max(trades_analyzer.get('total', {}).get('closed', 1), 1),
        }
        
        return metrics
    
    def plot_results(self, title="Backtest Results"):
        """Plot backtest results"""
        self.cerebro.plot(style='candlestick', title=title, figsize=(15, 8))
```

---

## 🛡️ Step 3.4: Risk Management Integration

### Risk Management System
```python
class RiskManager:
    def __init__(self):
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_position_size = 0.2  # 20% max per position
        self.max_drawdown = 0.15  # 15% max drawdown
        self.consecutive_losses = 3  # Stop after 3 consecutive losses
        
        self.daily_pnl = 0
        self.consecutive_loss_count = 0
        self.peak_portfolio_value = 0
        
    def check_risk_limits(self, current_value, position_size, last_trade_pnl):
        """Check if trade violates risk limits"""
        # Update tracking
        self.daily_pnl += last_trade_pnl
        
        if last_trade_pnl < 0:
            self.consecutive_loss_count += 1
        else:
            self.consecutive_loss_count = 0
        
        # Update peak value
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        # Check risk limits
        current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        risk_checks = {
            'daily_loss_exceeded': abs(self.daily_pnl) > self.max_daily_loss * self.peak_portfolio_value,
            'position_too_large': position_size > self.max_position_size,
            'max_drawdown_exceeded': current_drawdown > self.max_drawdown,
            'too_many_consecutive_losses': self.consecutive_loss_count >= self.consecutive_losses
        }
        
        # Return True if all checks pass
        return not any(risk_checks.values()), risk_checks
    
    def get_position_size_limit(self, volatility, base_size=0.02):
        """Calculate position size based on volatility"""
        # Volatility-adjusted position sizing
        if volatility > 0:
            vol_adjusted_size = base_size / volatility
        else:
            vol_adjusted_size = base_size
            
        return min(vol_adjusted_size, self.max_position_size)
    
    def reset_daily_tracking(self):
        """Reset daily tracking for new trading day"""
        self.daily_pnl = 0
```

---

## ✅ Phase 3 Deliverables

### Required Files
- [ ] `src/strategies/hybrid_ai_strategy.py` - Main trading strategy
- [ ] `src/strategies/regime_detector.py` - Market regime detection
- [ ] `src/backtesting/backtest_engine.py` - Backtesting framework
- [ ] `src/risk/risk_manager.py` - Risk management system
- [ ] `notebooks/03_strategy_development.ipynb` - Strategy development notebook

### Success Criteria
- [ ] Backtesting shows positive alpha
- [ ] Risk controls prevent >20% drawdown
- [ ] Strategy outperforms buy-and-hold
- [ ] Regime detection working correctly
- [ ] Volatility-based sizing functional

### Validation Commands
```bash
# Test strategy implementation
python -c "from src.strategies.hybrid_ai_strategy import HybridAIStrategy; print('Strategy loaded')"

# Test backtesting engine
python -c "from src.backtesting.backtest_engine import BacktestEngine; engine = BacktestEngine(); print('Backtest engine ready')"

# Test risk management
python -c "from src.risk.risk_manager import RiskManager; rm = RiskManager(); print('Risk manager loaded')"

# Run strategy backtest
python tests/test_trading_strategy.py
```

---

## 🚨 Common Issues

### Strategy Implementation Problems
- **Feature Mismatch**: Ensure backtest features match training features
- **Lookahead Bias**: Use only historical data for decisions
- **Transaction Costs**: Include realistic trading costs

### Backtesting Issues
- **Data Quality**: Ensure clean, continuous price data
- **Survivorship Bias**: Use realistic universe of assets
- **Market Impact**: Consider large trade effects

### Risk Management Issues
- **Over-conservative**: Don't restrict trading too much
- **Lagging Indicators**: Risk signals may be too slow
- **Parameter Sensitivity**: Test different risk parameters

---

## 📚 Next Steps

After completing Phase 3:
1. Validate strategy performance across different market conditions
2. Optimize risk parameters
3. Proceed to [Phase 4: Production](phase-4-production.md)

---

*Phase 3 typically takes 2 weeks to complete with thorough backtesting.*
