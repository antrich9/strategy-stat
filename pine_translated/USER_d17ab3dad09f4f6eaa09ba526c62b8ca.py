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
    high = df['high']
    low = df['low']
    
    atr = df['high'].rolling(200).max() - df['low'].rolling(200).min()
    atr = close.ewm(span=200, adjust=False).mean() * 0.02
    
    filter_width = 0.0
    
    low_shifted_1 = low.shift(1)
    low_shifted_3 = low.shift(3)
    high_shifted_1 = high.shift(1)
    high_shifted_3 = high.shift(3)
    close_shifted_2 = close.shift(2)
    
    bull = (low_shifted_3 > high_shifted_1) & (close_shifted_2 < low_shifted_3) & (close > low_shifted_3) & ((low_shifted_3 - high_shifted_1) > atr * filter_width)
    bear = (low_shifted_1 > high_shifted_3) & (close_shifted_2 > high_shifted_3) & (close < high_shifted_3) & ((low_shifted_1 - high_shifted_3) > atr * filter_width)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bull.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif bear.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries