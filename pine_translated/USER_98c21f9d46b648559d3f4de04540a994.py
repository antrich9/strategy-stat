import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    Entry logic only for the "optimised Reduced sharp turn strategy tap MTF FVG" strategy.
    Generates long and short entries based on FVG (Fair Value Gap) detection,
    RSI confirmation, and multi-timeframe trend alignment.
    """
    
    # Wilder RSI (ta.rsi)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    volfilt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    # ATR filter
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(20).mean() / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    
    # Trend filter
    sma54 = df['close'].rolling(54).mean()
    loc2 = sma54 > sma54.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG detection
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # 240tf data approximation using current timeframe close
    tf_close = df['close'].copy()
    tf_sma50 = tf_close.rolling(50).mean()
    
    # State tracking
    last