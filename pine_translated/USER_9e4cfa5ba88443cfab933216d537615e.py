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
    entries = []
    trade_num = 1
    
    if len(df) < 2:
        return entries
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values
    
    bar_time = np.datetime64('1970-01-01') + (time * 10**6).astype('timedelta64[ns]')
    day_num = (bar_time.astype('datetime64[D]') - np.datetime64('1970-01-01')).astype(int)
    new_day = np.zeros(len(df), dtype=bool)
    new_day[1:] = np.diff(day_num) != 0
    
    prev_day_high = np.zeros(len(df))
    prev_day_low = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if new_day[i]:
            prev_day_high[i] = high[i-1]
            prev_day_low[i] = low[i-1]
        else:
            prev_day_high[i] = prev_day_high[i-1]
            prev_day_low[i] = prev_day_low[i-1]
    
    pbh = prev_day_high[1:]
    pbl = prev_day_low[1:]
    
    if np.all(pbh == 0) and np.all(pbl == 0):
        return entries
    
    swept_low = low[1:] < pbl
    swept_high = high[1:] > pbh
    broke_high = close[1:] > pbh
    broke_low = close[1:] < pbl
    
    bias = np.zeros(len(df))
    bias_diff = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if new_day[i]:
            bias[i] = 0
        if swept_low[i] and broke_high[i]:
            bias[i] = 1
        elif swept_high[i] and broke_low[i]:
            bias[i] = -1
        elif low[i] < pbl[i]:
            bias[i] = -1
        elif high[i] > pbh[i]:
            bias[i] = 1
        bias_diff[i] = bias[i] - bias[i-1]
    
    long_cond = bias_diff > 0
    short_cond = bias_diff < 0
    
    for i in range(1, len(df)):
        if new_day[i]:
            continue
        if pbh[i] == 0 and pbl[i] == 0:
            continue
            
        if long_cond[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time[i]),
                'entry_time': datetime.fromtimestamp(time[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1
        
        if short_cond[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time[i]),
                'entry_time': datetime.fromtimestamp(time[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1
    
    return entries