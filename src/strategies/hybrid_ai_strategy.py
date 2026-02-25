import backtrader as bt
from datetime import datetime
import numpy as np
import pandas as pd
from src.strategies.regime_detector import MarketRegimeDetector

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
        self.current_position_data = None
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
            recent_prices = [self.data_close[-i] for i in range(self.p.regime_window + 1)]
            recent_prices.reverse()
            recent_returns = pd.Series(recent_prices).pct_change().dropna()
            self.current_regime = self.regime_detector.calculate_regime(recent_returns).iloc[-1]
        
        # Regime filter (paper's approach)
        if self.p.enable_regime_filter and not self.regime_detector.get_regime_signal(self.current_regime):
            if self.position:
                self.close()
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
                # Paper looks at sentiment scores before 9:30 AM for the day
                # In backtest we assume daily sentiment is available
                sentiment_score = self._get_sentiment_score(current_date)
                sentiment_allowed = sentiment_score >= self.p.sentiment_threshold
            
            # Trading logic (paper's long-only approach)
            if not self.position and prediction == 1 and sentiment_allowed:
                # Open long position
                self._open_long_position(current_price, probability)
                
            elif self.position and prediction == 0:
                # Close position based on signal
                self.close()
                
        except Exception as e:
            # Silently handle for now
            pass
    
    def _prepare_features(self):
        """Prepare features for ML prediction"""
        if self.feature_engineer is None or self.model is None:
            return None
            
        try:
            # This is a placeholder for actual feature engineering logic
            # In a real scenario, you'd pull enough history to calculate features
            if len(self.data_close) < 100:  # Arbitrary threshold for features
                return None
                
            # Create a mock dataframe with the right columns if needed
            # For phase 3 we use a simplified approach
            features = pd.DataFrame(np.random.randn(1, len(self.feature_engineer.feature_columns)))
            return features
            
        except Exception as e:
            return None
    
    def _get_sentiment_score(self, date):
        """Get sentiment score for current date"""
        if self.sentiment_analyzer is None:
            return 0.0
        # Placeholder for integration
        return 0.0
    
    def _open_long_position(self, price, probability):
        """Open long position with volatility-based sizing"""
        # Calculate position size (paper's volatility-based sizing)
        volatility = self._calculate_volatility()
        
        # Volatility-based cash sizing (paper formula: weight = target_vol / current_vol)
        if volatility > 0:
            risk_adjusted_size = self.p.position_size_pct / volatility
            target_pct = min(risk_adjusted_size, self.p.max_position_pct)
        else:
            target_pct = self.p.position_size_pct
        
        # Execute trade
        self.order_target_percent(target=target_pct)
        
        # Track position
        self.current_position_data = {
            'entry_price': price,
            'entry_date': self.datas[0].datetime.date(0),
            'probability': probability,
            'volatility': volatility,
            'target_pct': target_pct
        }
    
    def _calculate_volatility(self):
        """Calculate volatility for position sizing"""
        if len(self.data_close) < self.p.volatility_window:
            return 0.02  # Default volatility
        
        recent_prices = [self.data_close[-i] for i in range(self.p.volatility_window + 1)]
        recent_prices.reverse()
        returns = pd.Series(recent_prices).pct_change().dropna()
        volatility = returns.std()
        
        return max(volatility, 0.001)  # Prevent division by zero
    
    def notify_trade(self, trade):
        """Track trade execution"""
        if trade.isclosed:
            trade_info = {
                'date': self.datas[0].datetime.date(0),
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / trade.value * 100 if trade.value != 0 else 0,
                'size': trade.size,
                'duration': trade.barlen
            }
            self.trade_history.append(trade_info)
    
    def notify_cashvalue(self, cash, value):
        """Track portfolio value"""
        self.portfolio_values.append({
            'date': self.datas[0].datetime.date(0),
            'cash': cash,
            'value': value
        })
