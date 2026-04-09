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
    results = []
    trade_num = 0
    
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    time_arr = df['time'].values
    
    n = len(df)
    
    # Pre-compute helper arrays
    is_up = np.zeros(n, dtype=bool)
    is_down = np.zeros(n, dtype=bool)
    
    for i in range(n):
        is_up[i] = close_arr[i] > open_arr[i]
        is_down[i] = close_arr[i] < open_arr[i]
    
    # OB and FVG conditions
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        idx = i
        idx_prev = i - 1
        idx_prev2 = i - 2
        
        ob_up[i] = is_down[idx_prev] and is_up[idx] and close_arr[idx] > high_arr[idx_prev]
        ob_down[i] = is_up[idx_prev] and is_down[idx] and close_arr[idx] < low_arr[idx_prev]
        
        fvg_up[i] = low_arr[idx] > high_arr[idx_prev2]
        fvg_down[i] = high_arr[idx] < low_arr[idx_prev2]
    
    # Previous day high/low (using daily high/low of previous row)
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_day_high[i] = high_arr[i-1]
        prev_day_low[i] = low_arr[i-1]
    
    # Swing detection (simplified for available data)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)
    
    for i in range(5, n):
        if (high_arr[i-3] < high_arr[i-2] and 
            high_arr[i-1] <= high_arr[i-2] and 
            high_arr[i-2] >= high_arr[i-4] and 
            high_arr[i-2] >= high_arr[i-5]):
            is_swing_high[i] = True
        
        if (low_arr[i-3] > low_arr[i-2] and 
            low_arr[i-1] >= low_arr[i-2] and 
            low_arr[i-2] <= low_arr[i-4] and 
            low_arr[i-2] <= low_arr[i-5]):
            is_swing_low[i] = True
    
    # Detect entries
    for i in range(n):
        # Skip first few bars
        if i < 5 or np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]):
            continue
        
        ts_ms = time_arr[i]
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        
        # London time windows: 07:45-09:45 and 14:45-16:45
        hour = dt.hour
        minute = dt.minute
        
        is_morning = (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute <= 44)
        is_afternoon = (hour == 14 and minute >= 45) or (hour == 15) or (hour == 16 and minute <= 44)
        is_within_time_window = is_morning or is_afternoon
        
        if not is_within_time_window:
            continue
        
        current_price = close_arr[i]
        
        bullish_cond = ob_up[i] and fvg_up[i]
        bearish_cond = ob_down[i] and fvg_down[i]
        
        if bullish_cond:
            trade_num += 1
            entry_time = dt.isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts_ms),
                'entry_time': entry_time,
                'entry_price_guess': float(current_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(current_price),
                'raw_price_b': float(current_price)
            })
        
        if bearish_cond:
            trade_num += 1
            entry_time = dt.isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts_ms),
                'entry_time': entry_time,
                'entry_price_guess': float(current_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(current_price),
                'raw_price_b': float(current_price)
            })
    
    return results