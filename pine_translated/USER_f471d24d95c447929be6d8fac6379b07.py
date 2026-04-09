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
    trade_num = 1
    
    if len(df) < 10:
        return results
    
    high = df['high']
    low = df['low']
    close = df['close']
    time_col = df['time']
    
    # Zigzag parameters
    prd = 2
    max_array_size = 50
    
    # Initialize zigzag arrays
    zigzag = []
    direction = 0
    
    # Calculate confirmed swing highs and lows (non-repainting)
    confirmedSwingHigh = np.full(len(df), np.nan)
    confirmedSwingHighBar = np.full(len(df), np.nan)
    confirmedSwingLow = np.full(len(df), np.nan)
    confirmedSwingLowBar = np.full(len(df), np.nan)
    
    for i in range(prd, len(df)):
        pivot_high = True
        for j in range(1, prd + 1):
            if high.iloc[i - j] > high.iloc[i] or (j == prd and high.iloc[i - j] == high.iloc[i]):
                pivot_high = False
                break
        if pivot_high:
            confirmedSwingHigh[i] = high.iloc[i]
            confirmedSwingHighBar[i] = i - prd
        
        pivot_low = True
        for j in range(1, prd + 1):
            if low.iloc[i - j] < low.iloc[i] or (j == prd and low.iloc[i - j] == low.iloc[i]):
                pivot_low = False
                break
        if pivot_low:
            confirmedSwingLow[i] = low.iloc[i]
            confirmedSwingLowBar[i] = i - prd
    
    # Calculate Fibonacci 0.5 level using zigzag-like approach
    fib_50 = np.full(len(df), np.nan)
    
    swing_highs = []
    swing_lows = []
    swing_high_times = []
    swing_low_times = []
    
    for i in range(prd, len(df)):
        if not np.isnan(confirmedSwingHigh[i]):
            swing_highs.append(float(confirmedSwingHigh[i]))
            swing_high_times.append(int(time_col.iloc[i]))
        if not np.isnan(confirmedSwingLow[i]):
            swing_lows.append(float(confirmedSwingLow[i]))
            swing_low_times.append(int(time_col.iloc[i]))
    
    # Calculate fib_50 based on recent swings
    for i in range(len(df)):
        valid_highs = [h for h in swing_highs if h is not None]
        valid_lows = [l for l in swing_lows if l is not None]
        
        if len(valid_highs) >= 2 and len(valid_lows) >= 2:
            recent_highs = valid_highs[-2:]
            recent_lows = valid_lows[-2:]
            
            if recent_lows[-1] < recent_highs[-1] and len(valid_lows) >= 2:
                fib_0 = min(recent_lows)
                fib_1 = max(recent_highs)
                fib_50[i] = fib_0 + (fib_1 - fib_0) * 0.5
    
    # Calculate BOS and CHoCH
    bullish_bos = np.full(len(df), False)
    bearish_bos = np.full(len(df), False)
    bullish_choch = np.full(len(df), False)
    bearish_choch = np.full(len(df), False)
    
    dir_arr = np.zeros(len(df))
    current_dir = 0
    
    for i in range(len(df)):
        if not np.isnan(confirmedSwingHigh[i]):
            current_dir = 1
        elif not np.isnan(confirmedSwingLow[i]):
            current_dir = -1
        dir_arr[i] = current_dir
        
        canCheckBOS = not np.isnan(confirmedSwingHighBar[i]) and not np.isnan(confirmedSwingLowBar[i])
        
        if canCheckBOS and i > 5:
            if current_dir == 1 and close.iloc[i] > confirmedSwingHigh[i] and not np.isnan(confirmedSwingHigh[i]):
                bullish_bos[i] = True
            if current_dir == -1 and close.iloc[i] < confirmedSwingLow[i] and not np.isnan(confirmedSwingLow[i]):
                bearish_bos[i] = True
            if current_dir == -1 and close.iloc[i] > confirmedSwingHigh[i] and not np.isnan(confirmedSwingHigh[i]):
                bullish_choch[i] = True
            if current_dir == 1 and close.iloc[i] < confirmedSwingLow[i] and not np.isnan(confirmedSwingLow[i]):
                bearish_choch[i] = True
    
    # Calculate fractals for orderblock detection
    isFractalHigh = np.full(len(df), False)
    isFractalLow = np.full(len(df), False)
    
    for i in range(4, len(df)):
        if high.iloc[i] < high.iloc[i-1] and (high.iloc[i-2] < high.iloc[i-1] or (high.iloc[i-2] == high.iloc[i-1] and high.iloc[i-3] < high.iloc[i-2])):
            isFractalHigh[i] = True
        if low.iloc[i] > low.iloc[i-1] and (low.iloc[i-2] > low.iloc[i-1] or (low.iloc[i-2] == low.iloc[i-1] and low.iloc[i-3] > low.iloc[i-2])):
            isFractalLow[i] = True
    
    # Orderblock detection - bullish and bearish
    bullish_ob = np.full(len(df), False)
    bearish_ob = np.full(len(df), False)
    
    for i in range(6, len(df)):
        # Bullish orderblock: last 3 bars form a bearish candle, current bar is bullish, 
        # and price hasn't broken above recent high fractal
        if close.iloc[i] > open.iloc[i] and close.iloc[i-1] < open.iloc[i-1] and close.iloc[i-2] < open.iloc[i-2]:
            recent_high_fractal_idx = None
            for j in range(i-1, max(0, i-20), -1):
                if isFractalHigh[j]:
                    recent_high_fractal_idx = j
                    break
            
            if recent_high_fractal_idx is not None and close.iloc[i] < high.iloc[recent_high_fractal_idx]:
                if not np.isnan(fib_50[i]) and low.iloc[i] > fib_50[i]:
                    bullish_ob[i] = True
    
    for i in range(6, len(df)):
        # Bearish orderblock: last 3 bars form a bullish candle, current bar is bearish,
        # and price hasn't broken below recent low fractal
        if close.iloc[i] < open.iloc[i] and close.iloc[i-1] > open.iloc[i-1] and close.iloc[i-2] > open.iloc[i-2]:
            recent_low_fractal_idx = None
            for j in range(i-1, max(0, i-20), -1):
                if isFractalLow[j]:
                    recent_low_fractal_idx = j
                    break
            
            if recent_low_fractal_idx is not None and close.iloc[i] > low.iloc[recent_low_fractal_idx]:
                if not np.isnan(fib_50[i]) and high.iloc[i] < fib_50[i]:
                    bearish_ob[i] = True
    
    # Build entry conditions
    long_entry_cond = bullish_bos | bullish_choch
    short_entry_cond = bearish_bos | bearish_choch
    
    # Apply Fibonacci filter - price should be near fib_50
    for i in range(len(df)):
        if not np.isnan(fib_50[i]):
            dist_to_fib = abs(close.iloc[i] - fib_50[i]) / fib_50[i]
            if dist_to_fib > 0.02:
                long_entry_cond = long_entry_cond & (close.iloc[i] < fib_50[i])
                short_entry_cond = short_entry_cond & (close.iloc[i] > fib_50[i])
    
    # Generate entries
    open_col = df['open']
    
    for i in range(10, len(df)):
        if np.isnan(fib_50[i]):
            continue
        
        if long_entry_cond.iloc[i] and bullish_ob.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if short_entry_cond.iloc[i] and bearish_ob.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return results