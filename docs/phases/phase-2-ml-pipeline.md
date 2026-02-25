# Phase 2: Machine Learning Pipeline (Week 3-4)

## 🎯 Objectives
Implement the paper's ML methodology with your SOL data, including XGBoost baseline and FinBERT sentiment analysis.

## 📋 Prerequisites
- Complete Phase 1: Data pipeline working
- Validated SOL historical data
- Hyperliquid API integration

---

## 🤖 Step 2.1: Feature Engineering Pipeline

### Feature Preprocessing
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Prepare features for ML training"""
        # Select relevant features
        feature_cols = [
            # Order flow features
            'best_bid', 'best_ask', 'mid_price', 'spread', 'spread_bps',
            'depth_imbalance_5', 'depth_imbalance_10', 'ofi_ratio',
            
            # Technical indicators
            'roc_1h', 'roc_4h', 'sma_diff_1h', 'sma_diff_4h',
            'golden_cross', 'bb_width_1h', 'bb_position_1h',
            
            # Market microstructure
            'toxicity_bid_proxy', 'toxicity_ask_proxy', 'toxicity_imbalance',
            'bid_vwap_5', 'ask_vwap_5', 'vwap_mid_5',
            
            # Cross-asset signals
            'btc_close', 'sol_ret_1h', 'btc_ret_1h', 'rel_strength_1h',
            
            # Time features
            'hour', 'day_of_week', 'is_us_open'
        ]
        
        # Ensure all features exist
        available_features = [f for f in feature_cols if f in df.columns]
        
        # Handle missing values
        features = df[available_features].fillna(method='ffill').fillna(0)
        
        # Create lagged features for prediction
        features['returns_lag_1h'] = df['sol_ret_1h'].shift(1)
        features['returns_lag_4h'] = df['sol_ret_1h'].shift(4)
        
        # Create rolling features
        features['volatility_24h'] = df['sol_ret_1h'].rolling(24).std()
        features['volume_ratio'] = df['bid_volume_5'] / (df['ask_volume_5'] + 1e-8)
        
        self.feature_columns = features.columns.tolist()
        return features.dropna()
    
    def create_labels(self, df, horizon='1h'):
        """Create binary labels for next return direction"""
        if horizon == '1h':
            target = df['sol_ret_1h'].shift(-1)  # Next hour return
        elif horizon == '4h':
            target = df['sol_ret_1h'].shift(-4)  # Next 4-hour return
        
        # Binary classification: 1 if positive return, 0 otherwise
        labels = (target > 0).astype(int)
        return labels
    
    def scale_features(self, features, fit=True):
        """Scale features using StandardScaler"""
        if fit:
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        return pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
```

### Train/Test Split
```python
def create_train_test_split(features, labels, test_size=0.3, random_state=42):
    """Create time-aware train/test split"""
    # Use chronological split to avoid lookahead bias
    split_idx = int(len(features) * (1 - test_size))
    
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = labels.iloc[:split_idx]
    y_test = labels.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test
```

---

## 🎯 Step 2.2: XGBoost Baseline Implementation

### XGBoost Model (Paper Configuration)
```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class XGBoostTrader:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model with paper's hyperparameters"""
        # Paper's configuration: 200 estimators, depth 6, learning rate 0.05
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        # Convert to binary predictions
        predictions = (probs > 0.5).astype(int)
        
        return predictions, probs
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, probs = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print(f"Paper Baseline: 0.630")
        print(f"Improvement: {accuracy - 0.630:+.3f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probs
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot top feature importance"""
        if self.feature_importance is None:
            return
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
```

---

## 🧠 Step 2.3: FinBERT Sentiment Analysis

### FinBERT Integration
```python
from transformers import pipeline
import torch
from datetime import datetime, time

class FinBERTSentiment:
    def __init__(self):
        # Load FinBERT model for financial sentiment analysis
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a single text"""
        try:
            result = self.sentiment_pipeline(text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'positive': result['score'] if result['label'] == 'positive' else 0,
                'negative': result['score'] if result['label'] == 'negative' else 0,
                'neutral': result['score'] if result['label'] == 'neutral' else 0
            }
        except Exception as e:
            return {
                'label': 'neutral',
                'score': 0.5,
                'positive': 0,
                'negative': 0,
                'neutral': 1.0
            }
    
    def aggregate_daily_sentiment(self, news_data):
        """Aggregate sentiment scores by day and ticker"""
        daily_sentiment = {}
        
        for date, articles in news_data.items():
            if not articles:
                daily_sentiment[date] = 0.0
                continue
            
            # Analyze each article
            sentiments = []
            for article in articles:
                # Only consider news before 9:30 AM EST
                if article['time'] < time(9, 30):
                    sentiment = self.analyze_text_sentiment(article['text'])
                    # Convert to -1 to 1 scale
                    score = (sentiment['positive'] - sentiment['negative'])
                    sentiments.append(score)
            
            # Aggregate daily sentiment (paper's formula)
            if sentiments:
                daily_sentiment[date] = np.mean(sentiments)
            else:
                daily_sentiment[date] = 0.0
        
        return daily_sentiment
    
    def create_sentiment_filter(self, sentiment_scores, threshold=-0.70):
        """Create sentiment filter as per paper methodology"""
        # Trades suspended if sentiment < -0.70
        return {date: (score >= threshold) for date, score in sentiment_scores.items()}
```

### Mock News Data (For Testing)
```python
def create_mock_news_data():
    """Create mock news data for testing"""
    mock_news = {
        '2024-01-15': [
            {'text': 'Solana network shows strong performance with new DeFi integrations', 'time': time(8, 30)},
            {'text': 'SOL price surges as institutional interest grows', 'time': time(9, 00)}
        ],
        '2024-01-16': [
            {'text': 'Technical issues reported on Solana blockchain', 'time': time(8, 45)},
            {'text': 'SOL experiences volatility amid market uncertainty', 'time': time(9, 15)}
        ],
        '2024-01-17': [
            {'text': 'Solana ecosystem expands with new partnerships', 'time': time(8, 00)}
        ]
    }
    return mock_news
```

---

## 📊 Step 2.4: Model Evaluation Framework

### Comprehensive Evaluation
```python
class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_baseline_models(self, X_train, X_test, y_train, y_test):
        """Compare with baseline models"""
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        models = {
            'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost (Paper)': XGBoostTrader()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'XGBoost (Paper)':
                model.train_model(X_train, y_train, X_test, y_test)
                eval_result = model.evaluate_model(X_test, y_test)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                eval_result = {
                    'accuracy': accuracy,
                    'predictions': predictions
                }
            
            results[name] = eval_result
            print(f"{name} Accuracy: {eval_result['accuracy']:.3f}")
        
        self.results = results
        return results
    
    def backtest_strategy(self, predictions, returns, sentiment_filter=None):
        """Simple backtesting of ML predictions"""
        strategy_returns = []
        
        for i, (pred, actual_return) in enumerate(zip(predictions, returns)):
            # Apply sentiment filter if available
            if sentiment_filter and i in sentiment_filter:
                if not sentiment_filter[i]:
                    strategy_returns.append(0)  # No trade
                    continue
            
            # Long only strategy (paper's approach)
            if pred == 1:  # Predict positive return
                strategy_returns.append(actual_return)
            else:
                strategy_returns.append(0)  # No position
        
        return np.array(strategy_returns)
    
    def calculate_performance_metrics(self, strategy_returns, benchmark_returns):
        """Calculate trading performance metrics"""
        cumulative_strategy = np.cumprod(1 + strategy_returns)
        cumulative_benchmark = np.cumprod(1 + benchmark_returns)
        
        # Calculate metrics
        total_return = cumulative_strategy[-1] - 1
        benchmark_return = cumulative_benchmark[-1] - 1
        
        # Annualized metrics (assuming hourly data)
        hours_per_year = 365 * 24
        annual_return = (1 + total_return) ** (hours_per_year / len(strategy_returns)) - 1
        
        # Sharpe ratio
        excess_returns = strategy_returns - benchmark_returns
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(hours_per_year)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(cumulative_strategy)
        drawdown = (cumulative_strategy - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': total_return - benchmark_return
        }
```

---

## ✅ Phase 2 Deliverables

### Required Files
- [ ] `src/models/xgboost_trader.py` - XGBoost implementation
- [ ] `src/models/finbert_sentiment.py` - Sentiment analysis
- [ ] `src/features/feature_engineer.py` - Feature pipeline
- [ ] `src/evaluation/model_evaluator.py` - Evaluation framework
- [ ] `notebooks/02_ml_pipeline.ipynb` - ML development notebook

### Success Criteria
- [ ] XGBoost accuracy ≥ 60% (paper baseline)
- [ ] Sentiment analysis pipeline functional
- [ ] Feature engineering pipeline complete
- [ ] Model evaluation framework working
- [ ] Backtesting shows positive alpha

### Validation Commands
```bash
# Test feature engineering
python -c "from src.features.feature_engineer import FeatureEngineer; fe = FeatureEngineer(); print('Feature engineer loaded')"

# Test XGBoost training
python -c "from src.models.xgboost_trader import XGBoostTrader; model = XGBoostTrader(); print('XGBoost model ready')"

# Test sentiment analysis
python -c "from src.models.finbert_sentiment import FinBERTSentiment; sentiment = FinBERTSentiment(); print('FinBERT loaded')"

# Run model evaluation
python tests/test_ml_pipeline.py
```

---

## 🚨 Common Issues

### Model Training Problems
- **Overfitting**: Use cross-validation and early stopping
- **Data Leakage**: Ensure proper train/test split
- **Imbalanced Classes**: Use class weights or sampling

### Feature Engineering Issues
- **Missing Values**: Implement proper imputation strategies
- **Scale Mismatch**: Use same scaler for train/test
- **Lookahead Bias**: Ensure only historical data used

### Sentiment Analysis Issues
- **Model Loading**: Check GPU memory for FinBERT
- **API Limits**: Cache sentiment results
- **News Data**: Use mock data for development

---

## 📚 Next Steps

After completing Phase 2:
1. Validate model performance against paper baseline
2. Test sentiment integration
3. Proceed to [Phase 3: Trading Strategy](phase-3-trading-strategy.md)

---

*Phase 2 typically takes 2 weeks to complete with proper validation.*
