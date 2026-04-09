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
    
    # Ensure we have enough data
    if len(df) < 5:
        return []
    
    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Time validation (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12)
    valid_time = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dt = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
        hour = dt.hour
        if (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12):
            valid_time[i] = True
    
    # Calculate OB and FVG conditions
    ob_up = np.zeros(len(df), dtype=bool)
    ob_down = np.zeros(len(df), dtype=bool)
    fvg_up = np.zeros(len(df), dtype=bool)
    fvg_down = np.zeros(len(df), dtype=bool)
    
    for i in range(1, len(df) - 2):
        try:
            ob_up[i] = is_ob_up(i)
            ob_down[i] = is_ob_down(i)
            fvg_up[i] = is_fvg_up(i)
            fvg_down[i] = is_fvg_down(i)
        except:
            pass
    
    # Entry conditions
    long_entry = ob_up & fvg_up & valid_time
    short_entry = ob_down & fvg_down & valid_time
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i] if hasattr(long_entry, 'iloc') else long_entry[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        elif short_entry.iloc[i] if hasattr(short_entry, 'iloc') else short_entry[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(entry_ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries