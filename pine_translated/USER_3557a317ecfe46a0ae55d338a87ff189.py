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
    df = df.sort_values('time').reset_index(drop=True)
    
    is_swing_high = (df['high'].shift(1) < df['high'].shift(2)) & (df['high'].shift(3) < df['high'].shift(2)) & (df['high'].shift(4) < df['high'].shift(2))
    is_swing_low = (df['low'].shift(1) > df['low'].shift(2)) & (df['low'].shift(3) > df['low'].shift(2)) & (df['low'].shift(4) > df['low'].shift(2))
    
    bfvg = df['low'] > df['high'].shift(2)
    sfvg = df['high'] < df['low'].shift(2)
    
    entries = []
    trade_num = 1
    last_swing_type = "none"
    
    for i in range(4, len(df)):
        if is_swing_high.iloc[i]:
            last_swing_type = "high"
        if is_swing_low.iloc[i]:
            last_swing_type = "low"
        
        if bfvg.iloc[i] and last_swing_type == "low":
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        if sfvg.iloc[i] and last_swing_type == "high":
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries