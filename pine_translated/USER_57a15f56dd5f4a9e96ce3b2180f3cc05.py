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
    close = df['close']
    open_price = df['open']
    low = df['low']
    high = df['high']
    
    close_1 = close.shift(1).fillna(0)
    open_1 = open_price.shift(1).fillna(0)
    close_2 = close.shift(2).fillna(0)
    open_2 = open_price.shift(2).fillna(0)
    high_2 = high.shift(2).fillna(0)
    
    bull_fvg = (low > high_2) & (close_1 > high_2) & (close_2 > open_2) & (close_1 > open_1) & (close > open_price)
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(df)):
        if bull_fvg.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
    
    return entries