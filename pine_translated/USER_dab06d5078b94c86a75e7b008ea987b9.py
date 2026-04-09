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
    times = df['time'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    days = timestamps.date
    
    is_new_day = np.zeros(n, dtype=bool)
    if n > 1:
        is_new_day[1:] = days[1:] != days[:-1]
    
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    
    current_day_high = -np.inf
    current_day_low = np.inf
    
    for i in range(n):
        if is_new_day[i]:
            prev_day_high[i] = current_day_high
            prev_day_low[i] = current_day_low
            current_day_high = highs[i]
            current_day_low = lows[i]
        else:
            if highs[i] > current_day_high:
                current_day_high = highs[i]
            if lows[i] < current_day_low:
                current_day_low = lows[i]
    
    for i in range(1, n):
        if np.isnan(prev_day_high[i]):
            prev_day_high[i] = prev_day_high[i-1]
        if np.isnan(prev_day_low[i]):
            prev_day_low[i] = prev_day_low[i-1]
    
    flagpdl = closes < prev_day_low
    flagpdh = closes > prev_day_high
    
    ob_up = np.full(n, False)
    ob_down = np.full(n, False)
    fvg_up = np.full(n, False)
    fvg_down = np.full(n, False)
    
    if n > 2:
        valid_idx = np.arange(2, n)
        ob_up[valid_idx] = (closes[2:] > opens[2:]) & (closes[1:-1] < opens[1:-1]) & (closes[2:] > highs[1:-1])
        ob_down[valid_idx] = (closes[2:] < opens[2:]) & (closes[1:-1] > opens[1:-1]) & (closes[2:] < lows[1:-1])
        fvg_up[valid_idx] = lows[2:] > highs[:-2]
        fvg_down[valid_idx] = highs[2:] < lows[:-2]
    
    hours = timestamps.hour
    is_bullish_time = (hours >= 7) & (hours <= 9)
    is_bearish_time = (hours >= 12) & (hours <= 14)
    
    bullish_entry = ob_up & fvg_up & flagpdl & is_bullish_time
    bearish_entry = ob_down & fvg_down & flagpdh & is_bearish_time
    
    entries = []
    trade_num = 1
    
    for i in range(n):
        ts = int(times[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if bullish_entry[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
        
        if bearish_entry[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
    
    return entries