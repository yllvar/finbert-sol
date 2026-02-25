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
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        else:
            current_drawdown = 0
            
        risk_checks = {
            'daily_loss_exceeded': abs(self.daily_pnl) > self.max_daily_loss * self.peak_portfolio_value if self.peak_portfolio_value > 0 else False,
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
