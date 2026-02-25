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
        if isinstance(returns, list):
            returns = pd.Series(returns)
            
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
