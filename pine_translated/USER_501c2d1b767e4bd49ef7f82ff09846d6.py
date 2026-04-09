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
    
    # Calculate EMAs (Wilder's method using adjust=False)
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Shift for previous bar values
    ema8_prev = ema8.shift(1)
    ema20_prev = ema20.shift(1)
    
    # Crossover: ema8 crosses above ema20
    # At bar i: ema8[i] > ema20[i] AND ema8[i-1] <= ema20[i-1]
    crossover = (ema8 > ema20) & (ema8_prev <= ema20_prev)
    
    # Crossunder: ema8 crosses below ema20
    # At bar i: ema8[i] < ema20[i] AND ema8[i-1] >= ema20[i-1]
    crossunder = (ema8 < ema20) & (ema8_prev >= ema20_prev)
    
    # Long condition: ema8 crosses above ema20 AND ema20 > ema50
    long_condition = crossover & (ema20 > ema50)
    
    # Short condition: ema8 crosses below ema20 AND ema20 < ema50
    short_condition = crossunder & (ema20 < ema50)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip bars where required indicators are NaN
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue
            
        if long_condition.iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[i],
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries