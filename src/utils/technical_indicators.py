"""
Technical indicators for SOL trading system
"""
import pandas as pd
import numpy as np
from typing import Union


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        prices: Price series
        span: EMA span
    
    Returns:
        EMA series
    """
    return prices.ewm(span=span).mean()


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'MACD': macd,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    })


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        prices: Price series
        window: RSI window
    
    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Price series
        window: Moving average window
        num_std: Number of standard deviations
    
    Returns:
        DataFrame with upper, middle, lower bands and width
    """
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    width = (upper_band - lower_band) / sma
    
    result = pd.DataFrame({
        'BB_Upper': upper_band,
        'BB_Middle': sma,
        'BB_Lower': lower_band,
        'BB_Width': width
    })
    
    # Add position within bands
    result['BB_Position'] = (prices - lower_band) / (upper_band - lower_band)
    
    return result


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        df: DataFrame with high, low, close columns
        window: ATR window
    
    Returns:
        ATR series
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_roc(prices: pd.Series, period: int = 1) -> pd.Series:
    """
    Calculate Rate of Change
    
    Args:
        prices: Price series
        period: ROC period
    
    Returns:
        ROC series
    """
    return prices.pct_change(period)


def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator
    
    Args:
        df: DataFrame with high, low, close columns
        k_window: %K window
        d_window: %D window
    
    Returns:
        DataFrame with %K and %D
    """
    lowest_low = df['low'].rolling(window=k_window).min()
    highest_high = df['high'].rolling(window=k_window).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return pd.DataFrame({
        'Stoch_K': k_percent,
        'Stoch_D': d_percent
    })


def calculate_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Williams %R
    
    Args:
        df: DataFrame with high, low, close columns
        window: Lookback window
    
    Returns:
        Williams %R series
    """
    highest_high = df['high'].rolling(window=window).max()
    lowest_low = df['low'].rolling(window=window).min()
    
    williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    
    return williams_r


def calculate_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index
    
    Args:
        df: DataFrame with high, low, close columns
        window: CCI window
    
    Returns:
        CCI series
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    
    return cci


def add_technical_features(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Name of price column
    
    Returns:
        DataFrame with added technical features
    """
    result = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    prices = df[price_col]
    
    # EMAs
    result['EMA_50'] = calculate_ema(prices, 50)
    result['EMA_200'] = calculate_ema(prices, 200)
    result['EMA_Ratio'] = result['EMA_50'] / result['EMA_200']
    
    # MACD
    macd_df = calculate_macd(prices)
    result['MACD'] = macd_df['MACD']
    result['MACD_Signal'] = macd_df['MACD_Signal']
    result['MACD_Histogram'] = macd_df['MACD_Histogram']
    
    # RSI
    result['RSI_14'] = calculate_rsi(prices, 14)
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(prices)
    result = pd.concat([result, bb_df], axis=1)
    
    # ATR (requires high/low)
    if all(col in df.columns for col in ['high', 'low']):
        result['ATR'] = calculate_atr(df)
        result['Volatility'] = prices.rolling(20).std()
    
    # Rate of Change
    result['ROC_1h'] = calculate_roc(prices, 1)
    result['ROC_4h'] = calculate_roc(prices, 4)
    
    # SMA differences
    result['SMA_20'] = prices.rolling(20).mean()
    result['SMA_50'] = prices.rolling(50).mean()
    result['SMA_Diff_1h'] = prices.rolling(1).mean() - prices.rolling(1).mean().shift(1)
    result['SMA_Diff_4h'] = prices.rolling(4).mean() - prices.rolling(4).mean().shift(4)
    
    # Golden Cross
    result['Golden_Cross'] = (result['SMA_20'] > result['SMA_50']).astype(int)
    
    # Stochastic (requires high/low)
    if all(col in df.columns for col in ['high', 'low']):
        stoch_df = calculate_stochastic(df)
        result = pd.concat([result, stoch_df], axis=1)
    
    return result


def main():
    """Test technical indicators"""
    print("Testing technical indicators...")
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    # Simulate price movement
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * (1 + np.cumsum(returns))
    
    # Create OHLCV
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.01, 100)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Created sample data: {df.shape}")
    
    # Add technical indicators
    df_with_features = add_technical_features(df)
    print(f"✅ Added technical features: {df_with_features.shape}")
    
    # Show sample features
    feature_cols = ['EMA_50', 'EMA_200', 'MACD', 'RSI_14', 'BB_Width', 'ATR', 'Golden_Cross']
    print(f"✅ Sample features:")
    print(df_with_features[feature_cols].tail())
    
    print("\n✅ Technical indicators test complete!")


if __name__ == "__main__":
    main()
