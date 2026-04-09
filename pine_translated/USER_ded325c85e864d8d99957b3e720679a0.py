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
    
    close = df['close']
    time = df['time']
    
    # Wilder ATR implementation for threshold
    high = df['high']
    low = df['low']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/10, adjust=False).mean()
    threshold_multiplier = 3.0
    depth = 10
    
    # Calculate deviation threshold dynamically
    deviation = (atr / close * 100) * threshold_multiplier
    
    # ZigZag-like pivot detection using local extrema
    # This is a simplified approximation of the ZigZag library functionality
    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)
    
    for i in range(depth, len(df) - depth):
        # Check for pivot high
        is_pivot_high = True
        for j in range(1, depth + 1):
            if close.iloc[i] <= close.iloc[i - j] * (1 + deviation.iloc[i] / 100):
                is_pivot_high = False
                break
        if is_pivot_high:
            for j in range(1, depth + 1):
                if close.iloc[i] <= close.iloc[i + j] * (1 + deviation.iloc[i] / 100):
                    is_pivot_high = False
                    break
        if is_pivot_high:
            pivot_high.iloc[i] = True
        
        # Check for pivot low
        is_pivot_low = True
        for j in range(1, depth + 1):
            if close.iloc[i] >= close.iloc[i - j] * (1 - deviation.iloc[i] / 100):
                is_pivot_low = False
                break
        if is_pivot_low:
            for j in range(1, depth + 1):
                if close.iloc[i] >= close.iloc[i + j] * (1 - deviation.iloc[i] / 100):
                    is_pivot_low = False
                    break
        if is_pivot_low:
            pivot_low.iloc[i] = True
    
    # Find last pivot points
    last_pivot_high_idx = pivot_high[pivot_high].index.max() if pivot_high.any() else None
    last_pivot_low_idx = pivot_low[pivot_low].index.max() if pivot_low.any() else None
    
    if last_pivot_high_idx is None or last_pivot_low_idx is None:
        return entries
    
    # Determine swing high and swing low
    if last_pivot_high_idx > last_pivot_low_idx:
        swing_high = close.iloc[last_pivot_high_idx]
        swing_low = close.iloc[last_pivot_low_idx]
        start_price = swing_low
        end_price = swing_high
        height = abs(swing_high - swing_low)
    else:
        swing_high = close.iloc[last_pivot_high_idx]
        swing_low = close.iloc[last_pivot_low_idx]
        start_price = swing_high
        end_price = swing_low
        height = abs(swing_high - swing_low)
    
    # Calculate Fibonacci levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 3.618, 4.236]
    enabled_levels = [True, True, True, True, True, True, True, True, True, True, True]
    
    # Crossing detection function
    def crossing_level(close_series, level_price, i):
        if i < 1:
            return False
        r = level_price
        return (r > close_series.iloc[i] and r < close_series.iloc[i-1]) or (r < close_series.iloc[i] and r > close_series.iloc[i-1])
    
    # Entry logic: Long when price crosses above 0.382 Fibonacci level
    # Based on the alert condition for 0.382 level
    for i in range(depth + 1, len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(close.iloc[i]):
            continue
        
        # Calculate current Fibonacci levels
        for fib_val, enabled in zip(fib_levels, enabled_levels):
            if not enabled:
                continue
            fib_price = start_price + height * fib_val
            
            # Check for crossover at 0.382 level (bullish entry signal)
            if fib_val == 0.382 and crossing_level(close, fib_price, i):
                # Bullish entry: price crossed above 0.382 level
                ts = int(time.iloc[i])
                entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                break  # Only one entry per bar
    
    return entries