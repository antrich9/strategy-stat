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
    
    ema_length = 50
    fib_level = 0.5
    
    ema = df['close'].ewm(span=ema_length, adjust=False).mean()
    
    swing_high_mask = df['high'] > df['high'].rolling(window=5).max().shift(1)
    swing_low_mask = df['low'] < df['low'].rolling(window=5).min().shift(1)
    
    last_swing_high = np.nan
    last_swing_low = np.nan
    
    pullback_long = pd.Series(np.nan, index=df.index)
    pullback_short = pd.Series(np.nan, index=df.index)
    
    for i in range(len(df)):
        if swing_high_mask.iloc[i]:
            last_swing_high = df['high'].iloc[i]
        if swing_low_mask.iloc[i]:
            last_swing_low = df['low'].iloc[i]
        
        if not np.isnan(last_swing_high) and not np.isnan(last_swing_low):
            pullback_long.iloc[i] = last_swing_low + fib_level * (last_swing_high - last_swing_low)
            pullback_short.iloc[i] = last_swing_high - fib_level * (last_swing_high - last_swing_low)
    
    long_condition = df['close'] > ema
    short_condition = df['close'] < ema
    
    long_entry = long_condition & (df['close'] > pullback_long)
    short_entry = short_condition & (df['close'] < pullback_short)
    
    entries = []
    trade_num = 1
    position_open = False
    
    for i in range(1, len(df)):
        if position_open:
            continue
        
        if pd.isna(pullback_long.iloc[i]) or pd.isna(pullback_short.iloc[i]):
            continue
        
        if long_entry.iloc[i] and not long_entry.iloc[i-1]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            position_open = True
        elif short_entry.iloc[i] and not short_entry.iloc[i-1]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            position_open = True
    
    return entries