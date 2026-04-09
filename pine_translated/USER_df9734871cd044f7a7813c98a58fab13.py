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
    
    new_day = pd.to_datetime(df['time'], unit='s').dt.date.diff() != pd.Timedelta(days=0)
    new_day.iloc[0] = True
    
    prev_high = np.nan
    prev_low = np.nan
    breakout_counter = 0
    breakout_direction = None
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if new_day.iloc[i]:
            if i > 0:
                prev_high = high.iloc[i - 1]
                prev_low = low.iloc[i - 1]
            breakout_counter = 0
            breakout_direction = None
        
        if close.iloc[i] > prev_high:
            if breakout_direction != "up":
                breakout_counter = 1
                breakout_direction = "up"
            else:
                breakout_counter += 1
        elif close.iloc[i] < prev_low:
            if breakout_direction != "down":
                breakout_counter = 1
                breakout_direction = "down"
            else:
                breakout_counter += 1
        else:
            breakout_counter = 0
            breakout_direction = None
        
        confirmation_candles = 1
        if breakout_direction == "down" and breakout_counter == confirmation_candles:
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
            trade_num += 1
    
    return entries