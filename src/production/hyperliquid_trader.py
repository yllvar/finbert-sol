import requests
import json
from datetime import datetime
import time

class HyperliquidPaperTrader:
    def __init__(self, base_url="https://api.hyperliquid.xyz", paper_mode=True):
        self.base_url = base_url
        self.paper_mode = paper_mode
        self.session = requests.Session()
        self.account_id = None
        
    def setup_paper_account(self):
        """Setup paper trading account"""
        # Note: This is a placeholder for actual Hyperliquid API calls
        # In a real scenario, you'd use their SDK or specific signing logic
        print(f"Setting up paper account at {self.base_url}...")
        self.account_id = "paper_account_12345"
        return {"account_id": self.account_id, "status": "active"}
    
    def get_account_balance(self):
        """Get paper trading account balance"""
        # Mocking balance for phase 4 demonstration
        return {
            "total_value": 100000.0,
            "cash": 95000.0,
            "withdrawable": 95000.0
        }
    
    def place_order(self, symbol, side, order_type, quantity, price=None):
        """Place paper trading order"""
        order_result = {
            "order_id": f"ord_{int(time.time())}",
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price or 100.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        return order_result
    
    def get_open_positions(self):
        """Get current open positions"""
        # Mocking positions
        return {}
    
    def cancel_order(self, order_id):
        """Cancel open order"""
        return {"status": "cancelled", "order_id": order_id}
