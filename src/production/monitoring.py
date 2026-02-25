import pandas as pd
import json
import os
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, log_folder="logs/trading/"):
        self.log_folder = log_folder
        self.performance_file = os.path.join(log_folder, "performance.jsonl")
        self.alert_file = os.path.join(log_folder, "alerts.jsonl")
        
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            
        self.alert_thresholds = {
            'daily_loss_limit': -0.05,
            'max_drawdown': -0.15
        }
        
    def log_performance(self, metrics):
        with open(self.performance_file, "a") as f:
            metrics['timestamp'] = datetime.now().isoformat()
            f.write(json.dumps(metrics) + "\n")
            
    def check_alert_conditions(self, latest_metrics):
        """Check for alert conditions"""
        alerts = []
        
        daily_return = latest_metrics.get('daily_return', 0)
        if daily_return < self.alert_thresholds['daily_loss_limit']:
            alerts.append({
                'type': 'daily_loss',
                'severity': 'high',
                'message': f"Daily loss {daily_return:.2%} exceeds limit",
                'timestamp': datetime.now().isoformat()
            })
            
        # Potentially check cumulative drawdown here
        
        for alert in alerts:
            self.log_alert(alert)
        return alerts

    def log_alert(self, alert):
        with open(self.alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
            
    def generate_report(self):
        # Placeholder for simple stat summary
        return "Performance Monitor active. Logs at " + self.performance_file
