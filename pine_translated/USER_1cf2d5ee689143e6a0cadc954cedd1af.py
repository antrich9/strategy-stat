import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    
    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    # Convert time to datetime
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Define London time windows
    # London morning: 06:45 to 09:45
    # London afternoon: 13:45 to 15:45
    
    # Resample to 4H for 4H indicators
    # Group by 4H periods and aggregate OHLCV data
    df_4h = df.copy()
    df_4h.set_index('datetime', inplace=True)
    
    # Resample to 4H (240 minutes) using explicit offset
    ohlc_4h = df_4h.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Drop NaN rows from resampling
    ohlc_4h.dropna(inplace=True)
    
    # Calculate Volume Filter
    vol_sma_4h = ohlc_4h['volume'].rolling(window=9).mean()
    volfilt = ohlc_4h['volume'] > vol_sma_4h * 1.5
    
    # Calculate ATR Filter
    atr_length = 20
    high_low = ohlc_4h['high'] - ohlc_4h['low']
    high_close = (ohlc_4h['high'] - ohlc_4h['close'].shift()).abs()
    low_close = (ohlc_4h['low'] - ohlc_4h['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length).mean()
    atr_filter = atr / 1.5
    
    # Calculate Trend Filter
    sma_54 = ohlc_4h['close'].rolling(window=54).mean()
    trend_up = sma_54 > sma_54.shift(1)
    
    # Identify FVGs
    bullish_fvg = (ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & volfilt & (ohlc_4h['low'] - ohlc_4h['high'].shift(2) > atr_filter)
    bearish_fvg = (ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & volfilt & (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > atr_filter)
    
    # Detect new 4H candle
    is_new_4h = ohlc_4h.index != ohlc_4h.index.shift(1)
    
    # Combine conditions for entries
    entries = []
    last_fvg = None
    
    for i in range(1, len(ohlc_4h)):
        if is_new_4h.iloc[i]:
            # Check for bullish FVG and bearish trend
            if bullish_fvg.iloc[i] and not trend_up.iloc[i]:
                entries.append({
                    'timestamp': ohlc_4h.index[i],
                    'type': 'long',
                    'price': ohlc_4h['close'].iloc[i]
                })
                last_fvg = 'bullish'
            
            # Check for bearish FVG and bullish trend
            elif bearish_fvg.iloc[i] and trend_up.iloc[i]:
                entries.append({
                    'timestamp': ohlc_4h.index[i],
                    'type': 'short',
                    'price': ohlc_4h['close'].iloc[i]
                })
                last_fvg = 'bearish'
    
    return entries