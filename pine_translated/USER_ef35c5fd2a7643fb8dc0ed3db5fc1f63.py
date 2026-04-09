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
    
    # Parse timestamps and prepare data
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 4H and daily candles
    tf_4h = df.set_index('datetime').resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    tf_daily = df.set_index('datetime').resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    
    # Shift 4H and daily data to align with current bars
    high_4h = tf_4h['high'].shift(1)
    low_4h = tf_4h['low'].shift(1)
    high_4h_1 = tf_4h['high'].shift(2)
    low_4h_1 = tf_4h['low'].shift(2)
    high_4h_2 = tf_4h['high'].shift(3)
    low_4h_2 = tf_4h['low'].shift(3)
    
    daily_high = tf_daily['high'].shift(1)
    daily_low = tf_daily['low'].shift(1)
    daily_close = tf_daily['close'].shift(1)
    daily_open = tf_daily['open'].shift(1)
    prev_day_high = tf_daily['high'].shift(2)
    prev_day_low = tf_daily['low'].shift(2)
    prev_day_high_2 = tf_daily['high'].shift(3)
    prev_day_low_2 = tf_daily['low'].shift(3)
    
    # Detect FVG conditions
    bullish_fvg = low_4h > prev_day_high_2
    bearish_fvg = high_4h < prev_day_low_2
    
    # Determine entry signals
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bullish_fvg[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price': df['close'].iloc[i]
            })
            trade_num += 1
        elif bearish_fvg[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[i],
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries