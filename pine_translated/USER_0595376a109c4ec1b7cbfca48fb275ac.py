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
    df = df.copy()
    
    # Convert time to datetime for resampling
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('datetime')
    
    # Resample to 4H
    ohlc_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    ohlc_4h = ohlc_4h.dropna()
    
    # Calculate 4H indicators
    # ATR using Wilder's method
    high = ohlc_4h['high']
    low = ohlc_4h['low']
    close = ohlc_4h['close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_length = 20
    atr = tr.ewm(alpha=1.0/atr_length, adjust=False).mean()
    atr_4h = atr / 1.5
    
    # Volume filter
    sma_vol = close.rolling(9).mean()
    vol_filt = df['volume_4h'] > sma_vol * 1.5
    
    # Trend filter using 54 SMA
    sma54 = close.rolling(54).mean()
    trend_up = sma54 > sma54.shift(1)
    
    # Fair Value Gap detection
    bull_fvg = (low > high.shift(2)) & vol_filt & (low - high.shift(2) > atr_4h) & trend_up
    bear_fvg = (high < low.shift(2)) & vol_filt & (low.shift(2) - high > atr_4h) & ~trend_up
    
    # Sharp turn detection
    sharp_turn_long = bull_fvg & bear_fvg.shift(1)
    sharp_turn_short = bear_fvg & bull_fvg.shift(1)
    
    return sharp_turn_long, sharp_turn_short