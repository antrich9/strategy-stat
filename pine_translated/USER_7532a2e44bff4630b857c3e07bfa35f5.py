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
    high = df['high']
    low = df['low']
    timestamps = df['time']
    
    prev_day_high = high.shift(1)
    prev_day_low = low.shift(1)
    
    high_taken = (high > prev_day_high) & (high.shift(1) <= prev_day_high.shift(1))
    low_taken = (low < prev_day_low) & (low.shift(1) >= prev_day_low.shift(1))
    
    ny_hour = (pd.to_datetime(timestamps, unit='s').dt.hour - 5) % 24
    in_session = ((ny_hour >= 3) & (ny_hour < 7)) | ((ny_hour >= 8) & (ny_hour < 12))
    
    choch_high = prev_day_high.copy()
    choch_low = prev_day_low.copy()
    
    long_choch = (high > choch_high) & (high.shift(1) <= choch_high.shift(1))
    short_choch = (low < choch_low) & (low.shift(1) >= choch_low.shift(1))
    
    long_fvg = low.shift(2) > high
    short_fvg = high.shift(2) < low
    
    long_condition = long_choch & long_fvg & in_session
    short_condition = short_choch & short_fvg & in_session
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(timestamps.iloc[i])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': float(low.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(low.iloc[i]),
                'raw_price_b': float(low.iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(timestamps.iloc[i])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': float(high.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(high.iloc[i]),
                'raw_price_b': float(high.iloc[i])
            })
            trade_num += 1
    
    return entries