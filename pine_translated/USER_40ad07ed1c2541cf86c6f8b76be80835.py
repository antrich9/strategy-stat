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
    
    if len(df) < 5:
        return results
    
    # Calculate EMA(close, 9) and EMA(close, 21)
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    ema21 = df['close'].ewm(span=21, adjust=False).mean()
    
    # Calculate SMA(close, 50) for trend detection
    sma50 = df['close'].rolling(50).mean()
    
    # Detect new days using time differences
    times = df['time'].values
    is_new_day = np.zeros(len(df), dtype=bool)
    is_new_day[1:] = (times[1:] - times[:-1]) >= 86400  # 1 day in seconds
    
    # Get previous day high/low using 1-day shift
    prev_day_high = df['high'].shift(1).rolling(window=1440).max()
    prev_day_low = df['low'].shift(1).rolling(window=1440).min()
    
    # Flags for previous day high/low sweep
    flag_pdh = False
    flag_pdl = False
    waiting_for_entry = False
    
    # OB/FVG conditions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        if idx - 1 < 0 or idx >= len(df):
            return False
        return is_down(idx - 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx - 1]
    
    def is_ob_down(idx):
        if idx - 1 < 0 or idx >= len(df):
            return False
        return is_up(idx - 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx - 1]
    
    def is_fvg_up(idx):
        if idx - 2 < 0 or idx >= len(df):
            return False
        return df['low'].iloc[idx] > df['high'].iloc[idx - 2]
    
    def is_fvg_down(idx):
        if idx - 2 < 0 or idx >= len(df):
            return False
        return df['high'].iloc[idx] < df['low'].iloc[idx - 2]
    
    # Detect sweeps
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    pdh_swept = close_arr > prev_day_high.values
    pdl_swept = close_arr < prev_day_low.values
    
    # Fibonacci levels calculation
    # Calculate swing high/low for fib
    swing_high = df['high'].rolling(window=20).max().shift(1)
    swing_low = df['low'].rolling(window=20).min().shift(1)
    fib_range = swing_high - swing_low
    fib_618 = swing_low + fib_range * 0.618
    fib_786 = swing_low + fib_range * 0.786
    
    # Entry conditions
    ema9_arr = ema9.values
    ema21_arr = ema21.values
    sma50_arr = sma50.values
    fib618_arr = fib_618.values
    fib786_arr = fib_786.values
    prevdh_arr = prev_day_high.values
    prevdl_arr = prev_day_low.values
    
    long_conditions = np.zeros(len(df), dtype=bool)
    short_conditions = np.zeros(len(df), dtype=bool)
    
    for i in range(3, len(df)):
        if pd.isna(sma50_arr[i]) or pd.isna(fib618_arr[i]) or pd.isna(prevdh_arr[i]):
            continue
            
        # Long entry: Previous day low swept + OB up + FVG up + price above EMA9 + EMA9 > EMA21
        if pdl_swept[i] and is_ob_up(i) and is_fvg_up(i):
            if ema9_arr[i] > ema21_arr[i] and close_arr[i] > ema9_arr[i]:
                long_conditions[i] = True
        
        # Short entry: Previous day high swept + OB down + FVG down + price below EMA9 + EMA9 < EMA21
        if pdh_swept[i] and is_ob_down(i) and is_fvg_down(i):
            if ema9_arr[i] < ema21_arr[i] and close_arr[i] < ema9_arr[i]:
                short_conditions[i] = True
    
    # Generate entries
    for i in range(len(df)):
        if long_conditions[i] or short_conditions[i]:
            direction = 'long' if long_conditions[i] else 'short'
            ts = int(df['time'].iloc[i])
            
            entry = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            results.append(entry)
            trade_num += 1
    
    return results