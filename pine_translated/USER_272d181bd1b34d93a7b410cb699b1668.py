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
    
    if len(df) < 2:
        return results
    
    # Time column is unix timestamp in milliseconds (Pine Script uses ms)
    # If times are very small (< 1e12), assume seconds and multiply by 1000
    sample_ts = df['time'].iloc[0]
    if sample_ts < 1e12:
        df = df.copy()
        df['time'] = df['time'] * 1000
    
    # Detect new days (barrier)
    day_change = (df['time'] // 86400000).diff().fillna(0) != 0
    is_new_day = day_change.astype(bool)
    
    # Trading windows: 07:45-09:45 and 15:45-16:45 London time
    # Extract hour and minute from timestamp
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    
    # Morning window: 07:45 to 09:45
    in_morning_window = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
                        ((df['hour'] >= 8) & (df['hour'] < 9)) | \
                        ((df['hour'] == 9) & (df['minute'] <= 45))
    
    # Afternoon window: 15:45 to 16:45
    in_afternoon_window = ((df['hour'] == 15) & (df['minute'] >= 45)) | \
                          ((df['hour'] >= 16) & (df['hour'] < 16)) | \
                          ((df['hour'] == 16) & (df['minute'] <= 45))
    
    in_trading_window = in_morning_window | in_afternoon_window
    
    # Get previous day high/low using daily resample (approximation)
    # This requires aggregation - for simplicity we track rolling prev day levels
    # Since we can't use request.security, we use daily high/low from previous day bars
    
    # Group by day and get high/low
    df['day'] = df['time'] // 86400000
    daily_info = df.groupby('day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_info['prev_day_high'] = daily_info['high'].shift(1)
    daily_info['prev_day_low'] = daily_info['low'].shift(1)
    
    # Merge back
    df = df.merge(daily_info[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    
    # Fill forward within day
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Check for sweeps
    prev_day_high_swept = df['high'] > df['prev_day_high']
    prev_day_low_swept = df['low'] < df['prev_day_low']
    
    # Flags for liquidity sweep detection (similar to var bool flagpdh/flagpdl)
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    # Current day high/low tracking for the condition
    df['current_day_high'] = df.groupby('day')['high'].cummax()
    df['current_day_low'] = df.groupby('day')['low'].cummin()
    
    # Shift for previous bar comparison
    prev_high_swept = prev_day_high_swept.shift(1).fillna(False)
    prev_low_swept = prev_day_low_swept.shift(1).fillna(False)
    prev_curr_high = df['current_day_high'].shift(1)
    prev_curr_low = df['current_day_low'].shift(1)
    
    # Compute flagpdh: prev day high swept AND current day low > prev day low
    cond_flagpdh = prev_high_swept & (df['current_day_low'] > df['prev_day_low'].shift(1))
    cond_flagpdl = prev_low_swept & (df['current_day_high'] < df['prev_day_high'].shift(1))
    
    # Reset on new day
    for i in range(1, len(df)):
        if is_new_day.iloc[i]:
            flagpdh.iloc[i] = False
            flagpdl.iloc[i] = False
        else:
            if cond_flagpdh.iloc[i]:
                flagpdh.iloc[i] = True
            elif cond_flagpdl.iloc[i]:
                flagpdl.iloc[i] = True
            else:
                flagpdh.iloc[i] = False
                flagpdl.iloc[i] = False
    
    # Entry signals: 
    # Long: prev day low swept during trading window (after flagpdh)
    # Short: prev day high swept during trading window (after flagpdl)
    # But from the script, the actual strategy.entry() calls are not visible
    # The structure exists but no explicit entry execution is shown
    
    # Since no explicit strategy.entry() calls exist in the provided Pine Script,
    # we return an empty list as per rules (only replicate actual entry calls)
    
    return results