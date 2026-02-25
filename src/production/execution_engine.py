import traceback
from datetime import datetime
import json

class TradingExecutionEngine:
    def __init__(self, paper_trader, strategy, risk_manager, config=None):
        self.paper_trader = paper_trader
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config or {}
        
        self.active_positions = {}
        self.daily_pnl = 0
        self.execution_log = []
        
    def execute_daily_strategy(self):
        """Execute daily trading strategy"""
        try:
            print(f"[{datetime.now()}] Starting daily strategy execution...")
            
            # Get current market data
            market_data = self._get_current_market_data()
            
            # Generate trading signals
            # Note: Strategy.generate_signals() should be implemented to wrap the prediction logic
            signals = self.strategy.generate_signals(market_data)
            
            # Get current positions
            current_positions = self.paper_trader.get_open_positions()
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                self._execute_signal(symbol, signal, current_positions)
            
            # Log daily performance
            self._log_daily_performance()
            
        except Exception as e:
            print(f"Error in daily execution: {e}")
            self._log_error(e)
    
    def _execute_signal(self, symbol, signal, current_positions):
        """Execute trading signal for a symbol"""
        current_position = current_positions.get(symbol, {'quantity': 0})
        current_quantity = current_position['quantity']
        
        # Risk check
        portfolio_value = self._get_portfolio_value()
        risk_ok, risk_info = self.risk_manager.check_risk_limits(
            portfolio_value,
            abs(signal.get('quantity', 0)),
            self.daily_pnl
        )
        
        if not risk_ok:
            print(f"Risk check failed for {symbol}: {risk_info}")
            return
        
        # Execute trades
        if signal['action'] == 'BUY' and current_quantity == 0:
            # Open long position
            order_result = self.paper_trader.place_order(
                symbol=symbol,
                side='buy',
                order_type='market',
                quantity=signal['quantity']
            )
            self._log_order(symbol, 'BUY', signal['quantity'], order_result)
            
        elif signal['action'] == 'SELL' and current_quantity > 0:
            # Close position
            order_result = self.paper_trader.place_order(
                symbol=symbol,
                side='sell',
                order_type='market',
                quantity=current_quantity
            )
            self._log_order(symbol, 'SELL', current_quantity, order_result)
    
    def _get_current_market_data(self):
        """Get current market data for all symbols"""
        # Placeholder for real data ingestion
        symbol = self.config.get('strategy', {}).get('symbol', 'SOL-USD-PERP')
        return {
            symbol: {
                'price': 100.0,
                'volume': 1000000,
                'timestamp': datetime.now()
            }
        }
    
    def _get_portfolio_value(self):
        """Get current portfolio value"""
        balance_info = self.paper_trader.get_account_balance()
        return balance_info.get('total_value', 0.0)
    
    def _log_order(self, symbol, action, quantity, result):
        """Log order execution"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_id': result.get('order_id'),
            'status': result.get('status', 'unknown')
        }
        
        self.execution_log.append(log_entry)
        print(f"Order executed: {action} {quantity} {symbol} - ID: {result.get('order_id')}")
    
    def _log_daily_performance(self):
        """Log daily performance metrics"""
        portfolio_value = self._get_portfolio_value()
        
        performance_log = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'daily_pnl': self.daily_pnl,
            'open_positions_count': len(self.paper_trader.get_open_positions()),
            'orders_executed': len([log for log in self.execution_log if 'action' in log])
        }
        
        # In production this should append to a file
        print(f"Daily performance: ${portfolio_value:.2f} (PnL: ${self.daily_pnl:.2f})")
    
    def _log_error(self, error):
        """Log execution errors"""
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'message': str(error),
            'traceback': traceback.format_exc()
        }
        self.execution_log.append(error_log)
