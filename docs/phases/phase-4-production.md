# Phase 4: Production (Week 7-8)

## 🎯 Objectives
Deploy paper trading system with monitoring, automation, and operational procedures.

## 📋 Prerequisites
- Complete Phase 3: Validated trading strategy
- Positive backtesting results
- Risk management controls working

---

## 🚀 Step 4.1: Paper Trading Setup

### Hyperliquid Paper Trading Integration
```python
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
        endpoint = f"{self.base_url}/info/paper/setup"
        
        response = self.session.post(endpoint)
        response.raise_for_status()
        
        account_info = response.json()
        self.account_id = account_info['account_id']
        
        print(f"Paper trading account setup: {self.account_id}")
        return account_info
    
    def get_account_balance(self):
        """Get paper trading account balance"""
        endpoint = f"{self.base_url}/info/paper/balance"
        params = {"account_id": self.account_id}
        
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def place_order(self, symbol, side, order_type, quantity, price=None):
        """Place paper trading order"""
        endpoint = f"{self.base_url}/paper/order"
        
        order_data = {
            "account_id": self.account_id,
            "symbol": symbol,
            "side": side,  # "buy" or "sell"
            "type": order_type,  # "market" or "limit"
            "quantity": quantity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if price and order_type == "limit":
            order_data["price"] = price
        
        response = self.session.post(endpoint, json=order_data)
        response.raise_for_status()
        
        return response.json()
    
    def get_open_positions(self):
        """Get current open positions"""
        endpoint = f"{self.base_url}/info/paper/positions"
        params = {"account_id": self.account_id}
        
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def cancel_order(self, order_id):
        """Cancel open order"""
        endpoint = f"{self.base_url}/paper/order/cancel"
        
        cancel_data = {
            "account_id": self.account_id,
            "order_id": order_id
        }
        
        response = self.session.post(endpoint, json=cancel_data)
        response.raise_for_status()
        
        return response.json()
```

### Trading Execution Engine
```python
class TradingExecutionEngine:
    def __init__(self, paper_trader, strategy, risk_manager):
        self.paper_trader = paper_trader
        self.strategy = strategy
        self.risk_manager = risk_manager
        
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
        risk_ok, risk_info = self.risk_manager.check_risk_limits(
            self._get_portfolio_value(),
            abs(signal['quantity']),
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
        # This would integrate with your data pipeline
        # For now, return dummy data
        return {
            'SOL-USD-PERP': {
                'price': 100.0,
                'volume': 1000000,
                'timestamp': datetime.now()
            }
        }
    
    def _get_portfolio_value(self):
        """Get current portfolio value"""
        balance_info = self.paper_trader.get_account_balance()
        return balance_info['total_value']
    
    def _log_order(self, symbol, action, quantity, result):
        """Log order execution"""
        log_entry = {
            'timestamp': datetime.now(),
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
            'date': datetime.now().date(),
            'portfolio_value': portfolio_value,
            'daily_pnl': self.daily_pnl,
            'positions': self.paper_trader.get_open_positions(),
            'orders_executed': len([log for log in self.execution_log if log['timestamp'].date() == datetime.now().date()])
        }
        
        self.execution_log.append(performance_log)
        print(f"Daily performance: ${portfolio_value:.2f} (PnL: ${self.daily_pnl:.2f})")
    
    def _log_error(self, error):
        """Log execution errors"""
        error_log = {
            'timestamp': datetime.now(),
            'type': 'error',
            'message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.execution_log.append(error_log)
```

---

## 📊 Step 4.2: Model Retention Pipeline

### Daily Model Retraining
```python
import joblib
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from datetime import datetime, timedelta

class ModelRetentionPipeline:
    def __init__(self, model_path="models/", retrain_frequency="daily"):
        self.model_path = model_path
        self.retrain_frequency = retrain_frequency
        self.last_retrain_date = None
        
    def should_retrain(self):
        """Check if model should be retrained"""
        if self.last_retrain_date is None:
            return True
        
        if self.retrain_frequency == "daily":
            return datetime.now().date() > self.last_retrain_date
        elif self.retrain_frequency == "weekly":
            return (datetime.now().date() - self.last_retrain_date).days >= 7
        
        return False
    
    def retrain_model(self, data_loader, feature_engineer, model_class):
        """Retrain model with latest data"""
        try:
            print(f"[{datetime.now()}] Starting model retraining...")
            
            # Get latest training data
            latest_data = data_loader.get_latest_data(days_back=90)
            
            # Prepare features
            features = feature_engineer.prepare_features(latest_data)
            labels = feature_engineer.create_labels(latest_data)
            
            # Split data
            split_idx = int(len(features) * 0.7)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]
            
            # Train new model
            new_model = model_class()
            new_model.train_model(X_train, y_train, X_test, y_test)
            
            # Evaluate new model
            eval_results = new_model.evaluate_model(X_test, y_test)
            
            # Compare with current model
            current_performance = self._load_current_model_performance()
            
            if self._should_deploy_new_model(eval_results, current_performance):
                self._deploy_model(new_model, eval_results)
                print("Model deployed successfully")
            else:
                print("New model performance not sufficient, keeping current model")
            
            self.last_retrain_date = datetime.now().date()
            
        except Exception as e:
            print(f"Error in model retraining: {e}")
            self._log_retraining_error(e)
    
    def _load_current_model_performance(self):
        """Load current model performance metrics"""
        try:
            with open(f"{self.model_path}model_performance.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'accuracy': 0.5}  # Default baseline
    
    def _should_deploy_new_model(self, new_results, current_results):
        """Decide whether to deploy new model"""
        # Deploy if accuracy improves by at least 1%
        accuracy_improvement = new_results['accuracy'] - current_results.get('accuracy', 0.5)
        
        # Also check other metrics
        min_accuracy_threshold = 0.60  # Paper's baseline
        
        return (accuracy_improvement >= 0.01 and 
                new_results['accuracy'] >= min_accuracy_threshold)
    
    def _deploy_model(self, model, performance):
        """Deploy new model to production"""
        # Save model
        model_file = f"{self.model_path}production_model_{datetime.now().strftime('%Y%m%d')}.joblib"
        joblib.dump(model, model_file)
        
        # Update production symlink
        production_link = f"{self.model_path}production_model.joblib"
        if os.path.exists(production_link):
            os.remove(production_link)
        os.symlink(model_file, production_link)
        
        # Save performance metrics
        with open(f"{self.model_path}model_performance.json", 'w') as f:
            json.dump(performance, f, indent=2)
        
        print(f"Model deployed: {model_file}")
    
    def _log_retraining_error(self, error):
        """Log retraining errors"""
        error_log = {
            'timestamp': datetime.now(),
            'type': 'retraining_error',
            'message': str(error)
        }
        
        with open(f"{self.model_path}retraining_errors.jsonl", 'a') as f:
            f.write(json.dumps(error_log) + '\n')
```

---

## 📈 Step 4.3: Monitoring & Alerting

### Performance Monitoring Dashboard
```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class PerformanceMonitor:
    def __init__(self, log_file="logs/trading_performance.jsonl"):
        self.log_file = log_file
        self.alert_thresholds = {
            'daily_loss_limit': -0.05,  # 5% daily loss
            'max_drawdown': -0.15,       # 15% max drawdown
            'min_win_rate': 0.45,        # 45% minimum win rate
            'consecutive_losses': 3       # Max consecutive losses
        }
        
    def load_performance_data(self, days_back=30):
        """Load recent performance data"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        performance_data = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    log_entry = json.loads(line.strip())
                    log_date = datetime.fromisoformat(log_entry['timestamp'])
                    
                    if log_date >= cutoff_date:
                        performance_data.append(log_entry)
        except FileNotFoundError:
            pass
        
        return pd.DataFrame(performance_data)
    
    def calculate_daily_metrics(self, data):
        """Calculate daily performance metrics"""
        if data.empty:
            return {}
        
        # Group by date
        daily_data = data.groupby(data['timestamp'].dt.date).agg({
            'portfolio_value': 'last',
            'daily_pnl': 'sum',
            'orders_executed': 'sum'
        }).reset_index()
        
        # Calculate returns
        daily_data['daily_return'] = daily_data['portfolio_value'].pct_change()
        
        # Calculate running metrics
        daily_data['cumulative_return'] = (1 + daily_data['daily_return']).cumprod() - 1
        daily_data['running_max'] = daily_data['portfolio_value'].cummax()
        daily_data['drawdown'] = (daily_data['portfolio_value'] - daily_data['running_max']) / daily_data['running_max']
        
        return daily_data
    
    def check_alert_conditions(self, metrics):
        """Check for alert conditions"""
        alerts = []
        
        if not metrics.empty:
            latest = metrics.iloc[-1]
            
            # Daily loss alert
            if latest['daily_return'] < self.alert_thresholds['daily_loss_limit']:
                alerts.append({
                    'type': 'daily_loss',
                    'severity': 'high',
                    'message': f"Daily loss {latest['daily_return']:.2%} exceeds limit",
                    'timestamp': datetime.now()
                })
            
            # Maximum drawdown alert
            if latest['drawdown'] < self.alert_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'max_drawdown',
                    'severity': 'critical',
                    'message': f"Drawdown {latest['drawdown']:.2%} exceeds limit",
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def generate_performance_report(self, days_back=30):
        """Generate comprehensive performance report"""
        data = self.load_performance_data(days_back)
        metrics = self.calculate_daily_metrics(data)
        
        if metrics.empty:
            return "No performance data available"
        
        # Calculate summary statistics
        total_return = metrics['cumulative_return'].iloc[-1]
        volatility = metrics['daily_return'].std()
        sharpe_ratio = metrics['daily_return'].mean() / volatility * np.sqrt(252) if volatility > 0 else 0
        max_drawdown = metrics['drawdown'].min()
        
        report = f"""
Performance Report (Last {days_back} days)
========================================
Total Return: {total_return:.2%}
Volatility: {volatility:.2%}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.2%}
Trading Days: {len(metrics)}
        """
        
        return report
    
    def plot_performance_chart(self, days_back=30):
        """Plot performance chart"""
        data = self.load_performance_data(days_back)
        metrics = self.calculate_daily_metrics(data)
        
        if metrics.empty:
            print("No data available for plotting")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Portfolio value
        ax1.plot(metrics['timestamp'], metrics['portfolio_value'], label='Portfolio Value')
        ax1.set_title('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Daily returns
        ax2.bar(metrics['timestamp'], metrics['daily_return'], label='Daily Returns')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Daily Returns')
        ax2.legend()
        ax2.grid(True)
        
        # Drawdown
        ax3.fill_between(metrics['timestamp'], metrics['drawdown'], 0, 
                        where=(metrics['drawdown'] < 0), alpha=0.3, color='red', label='Drawdown')
        ax3.set_title('Drawdown')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
```

### Alert System
```python
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class AlertSystem:
    def __init__(self, smtp_config=None):
        self.smtp_config = smtp_config
        self.alert_log = []
        
    def send_email_alert(self, alert):
        """Send email alert"""
        if not self.smtp_config:
            print(f"ALERT: {alert['message']}")
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = self.smtp_config['to_email']
            msg['Subject'] = f"Trading Alert: {alert['type'].upper()}"
            
            body = f"""
Trading System Alert
===================

Type: {alert['type']}
Severity: {alert['severity']}
Message: {alert['message']}
Timestamp: {alert['timestamp']}

Please check the trading system immediately.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Email alert sent: {alert['type']}")
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def log_alert(self, alert):
        """Log alert for tracking"""
        self.alert_log.append(alert)
        
        # Save to file
        with open("logs/alerts.jsonl", "a") as f:
            f.write(json.dumps(alert) + "\n")
```

---

## ⚙️ Step 4.4: Automation & Deployment

### Daily Execution Automation
```python
import schedule
import time
import threading
from datetime import time as dt_time

class TradingBot:
    def __init__(self, execution_engine, monitor, alert_system):
        self.execution_engine = execution_engine
        self.monitor = monitor
        self.alert_system = alert_system
        self.running = False
        
    def start(self):
        """Start trading bot"""
        self.running = True
        
        # Schedule daily execution at 9:00 AM EST
        schedule.every().day.at("09:00").do(self._daily_execution)
        
        # Schedule monitoring checks every hour
        schedule.every().hour.do(self._monitoring_check)
        
        # Schedule model retraining weekly
        schedule.every().sunday.at("02:00").do(self._weekly_retraining)
        
        print("Trading bot started")
        
        # Run scheduler in separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    
    def stop(self):
        """Stop trading bot"""
        self.running = False
        schedule.clear()
        print("Trading bot stopped")
    
    def _run_scheduler(self):
        """Run scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _daily_execution(self):
        """Daily strategy execution"""
        try:
            print("Starting daily strategy execution...")
            self.execution_engine.execute_daily_strategy()
            
            # Check for alerts after execution
            self._monitoring_check()
            
        except Exception as e:
            print(f"Error in daily execution: {e}")
            alert = {
                'type': 'execution_error',
                'severity': 'high',
                'message': f"Daily execution failed: {str(e)}",
                'timestamp': datetime.now()
            }
            self.alert_system.send_email_alert(alert)
    
    def _monitoring_check(self):
        """Hourly monitoring check"""
        try:
            alerts = self.monitor.check_alert_conditions(
                self.monitor.calculate_daily_metrics(
                    self.monitor.load_performance_data()
                )
            )
            
            for alert in alerts:
                self.alert_system.send_email_alert(alert)
                
        except Exception as e:
            print(f"Error in monitoring check: {e}")
    
    def _weekly_retraining(self):
        """Weekly model retraining"""
        try:
            print("Starting weekly model retraining...")
            # This would integrate with your model retention pipeline
            print("Model retraining completed")
            
        except Exception as e:
            print(f"Error in model retraining: {e}")
```

---

## ✅ Phase 4 Deliverables

### Required Files
- [ ] `src/production/hyperliquid_trader.py` - Paper trading integration
- [ ] `src/production/execution_engine.py` - Trading execution
- [ ] `src/production/model_retention.py` - Model retraining pipeline
- [ ] `src/production/monitoring.py` - Performance monitoring
- [ ] `src/production/trading_bot.py` - Automation system
- [ ] `config/production_config.json` - Production configuration

### Success Criteria
- [ ] Paper trading executes without errors
- [ ] Daily performance monitoring active
- [ ] Alert system working correctly
- [ ] Model retraining pipeline functional
- [ ] System uptime > 95%

### Validation Commands
```bash
# Test paper trading integration
python -c "from src.production.hyperliquid_trader import HyperliquidPaperTrader; print('Paper trader ready')"

# Test execution engine
python -c "from src.production.execution_engine import TradingExecutionEngine; print('Execution engine ready')"

# Test monitoring system
python -c "from src.production.monitoring import PerformanceMonitor; print('Monitor ready')"

# Start trading bot
python src/production/trading_bot.py
```

---

## 🚨 Common Issues

### Paper Trading Problems
- **API Limits**: Respect rate limits and implement backoff
- **Order Execution**: Handle partial fills and rejections
- **Data Synchronization**: Ensure real-time data accuracy

### Model Retention Issues
- **Data Quality**: Ensure training data quality before retraining
- **Performance Degradation**: Monitor model performance over time
- **Deployment Failures**: Test model deployment thoroughly

### Monitoring Issues
- **False Alerts**: Tune alert thresholds to reduce noise
- **Missing Data**: Handle gaps in performance logs
- **System Overload**: Monitor resource usage

---

## 📚 Next Steps

After completing Phase 4:
1. Monitor paper trading performance for 2-4 weeks
2. Optimize parameters based on live results
3. Consider live trading deployment (with additional safeguards)

---

## 🎯 Production Readiness Checklist

- [ ] Paper trading account setup and funded
- [ ] All components tested individually
- [ ] End-to-end workflow tested
- [ ] Error handling and logging complete
- [ ] Alert system configured and tested
- [ ] Documentation updated with operational procedures
- [ ] Backup and recovery procedures in place
- [ ] Security measures implemented (API keys, access controls)

---

*Phase 4 typically takes 2 weeks to complete with thorough testing and monitoring setup.*
