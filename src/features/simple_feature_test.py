"""
Simplified Feature Engineering Pipeline for testing
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_sol_data


def simple_feature_test():
    """Test feature engineering with smaller dataset"""
    print("🚀 Testing Simplified Feature Engineering...")
    
    # Load smaller sample for testing
    print("📊 Loading sample data...")
    df = load_sol_data(2024, 1)
    
    # Use only first 10000 rows for testing
    df_sample = df.head(10000).copy()
    print(f"✅ Sample data loaded: {df_sample.shape}")
    
    # Select core features
    core_features = [
        'best_bid', 'best_ask', 'mid_price', 'spread', 'spread_bps',
        'depth_imbalance_5', 'bid_volume_5', 'ask_volume_5',
        'ofi_ratio', 'roc_1h', 'roc_4h', 'golden_cross', 'bb_width_1h',
        'btc_close', 'sol_ret_1h', 'btc_ret_1h', 'hour', 'day_of_week'
    ]
    
    # Filter to available features
    available_features = [f for f in core_features if f in df_sample.columns]
    features = df_sample[available_features].copy()
    
    print(f"✅ Selected {len(available_features)} core features")
    
    # Handle missing values
    features = features.ffill().fillna(0)
    
    # Create simple lagged features
    if 'sol_ret_1h' in features.columns:
        features['returns_lag_1h'] = features['sol_ret_1h'].shift(1)
        features['returns_lag_4h'] = features['sol_ret_1h'].shift(4)
    
    # Create simple rolling features
    if 'sol_ret_1h' in features.columns:
        features['volatility_6h'] = features['sol_ret_1h'].rolling(6).std()
        features['volatility_24h'] = features['sol_ret_1h'].rolling(24).std()
    
    # Create labels
    if 'sol_ret_1h' in df_sample.columns:
        labels = (df_sample['sol_ret_1h'].shift(-1) > 0).astype(int)
        # Remove NaN from labels
        valid_indices = labels.notna()
        features_clean = features[valid_indices]
        labels_clean = labels[valid_indices]
        
        print(f"✅ Features and labels aligned: {len(features_clean)} samples")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean.fillna(0))
        
        print(f"✅ Features scaled: {features_scaled.shape}")
        print(f"   Positive labels: {labels_clean.sum()}/{len(labels_clean)} ({labels_clean.mean():.2%})")
        
        return features_scaled, labels_clean, available_features
    
    return None, None, None


if __name__ == "__main__":
    features, labels, feature_names = simple_feature_test()
    
    if features is not None:
        print("\n🎉 Simplified feature engineering test complete!")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Feature names: {len(feature_names)}")
    else:
        print("❌ Feature engineering test failed!")
