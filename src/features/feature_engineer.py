"""
Feature Engineering Pipeline for SOL Trading System
Implements the paper's feature engineering approach with your existing data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_sol_data
from utils.technical_indicators import add_technical_features


class FeatureEngineer:
    """
    Feature engineering pipeline for SOL trading ML models
    Based on "Generating Alpha" paper methodology
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML training
        
        Args:
            df: Raw DataFrame with SOL data
        
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Preparing features...")
        
        # Start with a copy to avoid modifying original
        features = df.copy()
        
        # Ensure timestamp is datetime and set as index
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            if not isinstance(features.index, pd.DatetimeIndex):
                features.set_index('timestamp', inplace=True)
        
        # Select core features from your existing data
        core_features = [
            # Order flow features
            'best_bid', 'best_ask', 'mid_price', 'spread', 'spread_bps',
            'depth_imbalance_5', 'depth_imbalance_10', 'depth_imbalance_20',
            'bid_volume_5', 'ask_volume_5', 'bid_volume_10', 'ask_volume_10',
            'ofi_ratio', 'order_count_imbalance_5', 'order_count_imbalance_10',
            
            # Market microstructure
            'avg_bid_order_size', 'avg_ask_order_size', 'large_order_imbalance',
            'bid_slope', 'ask_slope', 'slope_imbalance',
            'bid_concentration', 'ask_concentration', 'concentration_imbalance',
            'depth_spread_5_pct', 'depth_spread_10_pct',
            'bid_vwap_5', 'ask_vwap_5', 'vwap_mid_5',
            'bid_vwap_10', 'ask_vwap_10', 'vwap_mid_10',
            
            # Order flow toxicity
            'toxicity_bid_proxy', 'toxicity_ask_proxy', 'toxicity_imbalance',
            'toxicity_trend',
            
            # Technical indicators (already in your data)
            'roc_1h', 'roc_4h', 'sma_diff_1h', 'sma_diff_4h',
            'golden_cross', 'bb_width_1h', 'bb_position_1h',
            
            # Cross-asset signals
            'btc_close', 'btc_high', 'btc_low', 'btc_volume',
            'sol_ret_1h', 'btc_ret_1h', 'rel_strength_1h',
            'btc_return_lag_1h', 'btc_return_lag_2h', 'btc_return_lag_4h',
            'btc_lead_signal_1h', 'btc_lead_signal_2h', 'btc_lead_signal_4h',
            'btc_lead_combined',
            
            # Time features
            'hour', 'day_of_week', 'is_us_open'
        ]
        
        # Filter to available features
        available_features = [f for f in core_features if f in features.columns]
        missing_features = [f for f in core_features if f not in features.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        features = features[available_features].copy()
        
        # Handle missing values
        print("🔧 Handling missing values...")
        features = self._handle_missing_values(features)
        
        # Create lagged features for prediction
        print("🔧 Creating lagged features...")
        features = self._create_lagged_features(features)
        
        # Create rolling features
        print("🔧 Creating rolling features...")
        features = self._create_rolling_features(features)
        
        # Create interaction features
        print("🔧 Creating interaction features...")
        features = self._create_interaction_features(features)
        
        # Create regime features
        print("🔧 Creating regime features...")
        features = self._create_regime_features(features)
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        print(f"✅ Feature engineering complete: {features.shape}")
        print(f"   Total features: {len(self.feature_columns)}")
        
        return features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        # Forward fill for time series data
        df = df.ffill()
        
        # Backward fill for any remaining NaNs
        df = df.bfill()
        
        # Fill any remaining NaNs with 0 (for numerical stability)
        df = df.fillna(0)
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for prediction"""
        # Lag returns (paper's approach)
        if 'sol_ret_1h' in df.columns:
            df['returns_lag_1h'] = df['sol_ret_1h'].shift(1)
            df['returns_lag_4h'] = df['sol_ret_1h'].shift(4)
            df['returns_lag_24h'] = df['sol_ret_1h'].shift(24)
        
        # Lag BTC returns
        if 'btc_ret_1h' in df.columns:
            df['btc_returns_lag_1h'] = df['btc_ret_1h'].shift(1)
            df['btc_returns_lag_4h'] = df['btc_ret_1h'].shift(4)
        
        # Lag spread
        if 'spread' in df.columns:
            df['spread_lag_1h'] = df['spread'].shift(1)
            df['spread_lag_4h'] = df['spread'].shift(4)
        
        # Lag volume imbalance
        if 'depth_imbalance_5' in df.columns:
            df['imbalance_lag_1h'] = df['depth_imbalance_5'].shift(1)
            df['imbalance_lag_4h'] = df['depth_imbalance_5'].shift(4)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        windows = [6, 12, 24]  # 6h, 12h, 24h windows
        
        for window in windows:
            # Rolling volatility
            if 'sol_ret_1h' in df.columns:
                df[f'volatility_{window}h'] = df['sol_ret_1h'].rolling(window).std()
            
            # Rolling mean of spread
            if 'spread' in df.columns:
                df[f'spread_mean_{window}h'] = df['spread'].rolling(window).mean()
            
            # Rolling mean of imbalance
            if 'depth_imbalance_5' in df.columns:
                df[f'imbalance_mean_{window}h'] = df['depth_imbalance_5'].rolling(window).mean()
            
            # Rolling volume ratio
            if 'bid_volume_5' in df.columns and 'ask_volume_5' in df.columns:
                volume_ratio = df['bid_volume_5'] / (df['ask_volume_5'] + 1e-8)
                df[f'volume_ratio_mean_{window}h'] = volume_ratio.rolling(window).mean()
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        # Price * spread interaction
        if 'mid_price' in df.columns and 'spread' in df.columns:
            df['price_spread_interaction'] = df['mid_price'] * df['spread']
        
        # Volume * imbalance interaction
        if 'bid_volume_5' in df.columns and 'depth_imbalance_5' in df.columns:
            df['volume_imbalance_interaction'] = df['bid_volume_5'] * df['depth_imbalance_5']
        
        # BTC correlation features
        if 'sol_ret_1h' in df.columns and 'btc_ret_1h' in df.columns:
            df['btc_sol_correlation_12h'] = df['sol_ret_1h'].rolling(12).corr(df['btc_ret_1h'])
            df['btc_sol_spread'] = df['sol_ret_1h'] - df['btc_ret_1h']
        
        # Toxicity * volume interaction
        if 'toxicity_imbalance' in df.columns and 'bid_volume_5' in df.columns:
            df['toxicity_volume_interaction'] = df['toxicity_imbalance'] * df['bid_volume_5']
        
        # Time-based interactions
        if 'hour' in df.columns and 'is_us_open' in df.columns:
            df['us_session_hour'] = df['hour'] * df['is_us_open']
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime detection features"""
        # Rolling return regime (paper's approach)
        if 'sol_ret_1h' in df.columns:
            # 20-hour rolling mean for regime detection
            df['regime_rolling_mean'] = df['sol_ret_1h'].rolling(20).mean()
            df['regime_bullish'] = (df['regime_rolling_mean'] > 0).astype(int)
        
        # Volatility regime
        if 'sol_ret_1h' in df.columns:
            df['volatility_regime'] = df['sol_ret_1h'].rolling(24).std()
            df['high_volatility'] = (df['volatility_regime'] > df['volatility_regime'].quantile(0.75)).astype(int)
        
        # Volume regime
        if 'bid_volume_5' in df.columns:
            df['volume_regime'] = df['bid_volume_5'].rolling(24).mean()
            df['high_volume'] = (df['volume_regime'] > df['volume_regime'].quantile(0.75)).astype(int)
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: str = '1h') -> pd.Series:
        """
        Create binary labels for next return direction
        
        Args:
            df: DataFrame with returns
            horizon: Prediction horizon ('1h', '4h', '24h')
        
        Returns:
            Series with binary labels (1=up, 0=down)
        """
        print(f"🎯 Creating {horizon} labels...")
        
        if 'sol_ret_1h' not in df.columns:
            raise ValueError("sol_ret_1h column not found in DataFrame")
        
        if horizon == '1h':
            target = df['sol_ret_1h'].shift(-1)  # Next hour return
        elif horizon == '4h':
            target = df['sol_ret_1h'].shift(-4)  # Next 4-hour return
        elif horizon == '24h':
            target = df['sol_ret_1h'].shift(-24)  # Next 24-hour return
        else:
            raise ValueError(f"Unsupported horizon: {horizon}")
        
        # Binary classification: 1 if positive return, 0 otherwise
        labels = (target > 0).astype(int)
        
        # Keep the same index as the original DataFrame (shifted labels will have NaN at the end)
        # This ensures alignment with features
        
        print(f"✅ Labels created: {labels.sum()}/{len(labels)} positive ({labels.mean():.2%})")
        
        return labels
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        
        Args:
            features: DataFrame with features
            fit: Whether to fit scaler (True for training, False for inference)
        
        Returns:
            Scaled DataFrame
        """
        if fit:
            scaled_features = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_features = self.scaler.transform(features)
        
        return pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
    def create_train_test_split(self, features: pd.DataFrame, labels: pd.Series, 
                              test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create time-aware train/test split
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            test_size: Proportion of data for testing
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("📊 Creating train/test split...")
        
        # Use chronological split to avoid lookahead bias (paper's approach)
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = labels.iloc[:split_idx]
        y_test = labels.iloc[split_idx:]
        
        print(f"✅ Split created:")
        print(f"   Train: {len(X_train)} samples ({y_train.mean():.2%} positive)")
        print(f"   Test:  {len(X_test)} samples ({y_test.mean():.2%} positive)")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        if not self.feature_columns:
            return {}
        
        groups = {
            'price_features': [f for f in self.feature_columns if any(x in f for x in ['price', 'bid', 'ask', 'mid'])],
            'spread_features': [f for f in self.feature_columns if 'spread' in f],
            'volume_features': [f for f in self.feature_columns if any(x in f for x in ['volume', 'size'])],
            'imbalance_features': [f for f in self.feature_columns if 'imbalance' in f],
            'toxicity_features': [f for f in self.feature_columns if 'toxicity' in f],
            'technical_features': [f for f in self.feature_columns if any(x in f for x in ['roc', 'sma', 'bb', 'golden'])],
            'btc_features': [f for f in self.feature_columns if 'btc' in f],
            'lagged_features': [f for f in self.feature_columns if 'lag' in f],
            'rolling_features': [f for f in self.feature_columns if any(x in f for x in ['rolling', 'volatility'])],
            'interaction_features': [f for f in self.feature_columns if 'interaction' in f],
            'regime_features': [f for f in self.feature_columns if 'regime' in f],
            'time_features': [f for f in self.feature_columns if any(x in f for x in ['hour', 'day', 'session'])]
        }
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups


def main():
    """Test feature engineering pipeline"""
    print("🚀 Testing Feature Engineering Pipeline...")
    
    # Load sample data
    print("📊 Loading sample data...")
    df = load_sol_data(2024, 1)
    print(f"✅ Loaded data: {df.shape}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Prepare features
    features = engineer.prepare_features(df)
    print(f"✅ Features prepared: {features.shape}")
    
    # Create labels
    labels = engineer.create_labels(df, horizon='1h')
    print(f"✅ Labels created: {len(labels)}")
    
    # Align features and labels (remove NaN from labels)
    # Labels will have NaN at the end due to shifting, so we need to align
    print(f"🔍 Debug: Features shape: {features.shape}, index type: {type(features.index)}")
    print(f"🔍 Debug: Labels shape: {labels.shape}, index type: {type(labels.index)}")
    
    # Reset indices to ensure alignment
    features_reset = features.reset_index(drop=True)
    labels_reset = labels.reset_index(drop=True)
    
    valid_indices = labels_reset.notna()
    features_aligned = features_reset[valid_indices]
    labels_aligned = labels_reset[valid_indices]
    
    print(f"✅ Aligned features and labels: {len(features_aligned)} samples")
    
    if len(features_aligned) == 0:
        print("❌ No aligned samples found!")
        return engineer, None, None, None, None
    
    # Scale features
    print("🔧 Scaling features...")
    
    # Check for infinite or very large values
    features_clean = features_aligned.replace([np.inf, -np.inf], np.nan)
    features_clean = features_clean.fillna(0)
    
    # Additional check for very large values
    for col in features_clean.columns:
        if features_clean[col].abs().max() > 1e6:
            print(f"⚠️  Capping large values in column: {col}")
            features_clean[col] = features_clean[col].clip(-1e6, 1e6)
    
    features_scaled = engineer.scale_features(features_clean, fit=True)
    print(f"✅ Features scaled: {features_scaled.shape}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = engineer.create_train_test_split(
        features_scaled, labels_aligned
    )
    
    # Show feature groups
    feature_groups = engineer.get_feature_importance_groups()
    print(f"✅ Feature groups created:")
    for group, feats in feature_groups.items():
        print(f"   {group}: {len(feats)} features")
    
    print("\n🎉 Feature engineering pipeline test complete!")
    return engineer, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
