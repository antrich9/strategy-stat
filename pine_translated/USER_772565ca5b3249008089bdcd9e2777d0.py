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
    
    # Ensure we have enough data
    if len(df) < 5:
        return results
    
    # Create a copy to avoid modifying original
    data = df.copy().reset_index(drop=True)
    
    # Helper functions
    def is_up(idx):
        return data['close'].iloc[idx] > data['open'].iloc[idx]
    
    def is_down(idx):
        return data['close'].iloc[idx] < data['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and data['close'].iloc[idx] > data['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and data['close'].iloc[idx] < data['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return data['low'].iloc[idx] > data['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return data['high'].iloc[idx] < data['low'].iloc[idx + 2]
    
    def get_session(ts):
        """Extract hour from unix timestamp"""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.hour
    
    # Initialize flags and tracking variables
    flagpdl = False
    flagpdh = False
    waiting_for_entry = False
    waiting_for_short_entry = False
    
    prev_day_high = np.nan
    prev_day_low = np.nan
    
    # Iterate through bars
    for i in range(1, len(data)):
        current_time = data['time'].iloc[i]
        current_close = data['close'].iloc[i]
        current_high = data['high'].iloc[i]
        current_low = data['low'].iloc[i]
        prev_close = data['close'].iloc[i - 1]
        
        hour = get_session(current_time)
        
        # Check for new day (midnight UTC or start of new calendar day)
        is_new_day = False
        if i > 0:
            prev_dt = datetime.fromtimestamp(data['time'].iloc[i-1], tz=timezone.utc)
            curr_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
            if prev_dt.date() < curr_dt.date():
                is_new_day = True
        
        if is_new_day:
            flagpdl = False
            flagpdh = False
            waiting_for_entry = False
            waiting_for_short_entry = False
            prev_day_high = np.nan
            prev_day_low = np.nan
        
        # Update previous day high/low at the start of each day
        if i >= 1:
            prev_dt = datetime.fromtimestamp(data['time'].iloc[i-1], tz=timezone.utc)
            curr_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
            if prev_dt.date() < curr_dt.date():
                # Previous day is i-1
                prev_day_high = data['high'].iloc[i-1]
                prev_day_low = data['low'].iloc[i-1]
        
        # Check for price sweeping previous day high/low
        if not np.isnan(prev_day_high) and current_close > prev_day_high:
            flagpdh = True
        
        if not np.isnan(prev_day_low) and current_close < prev_day_low:
            flagpdl = True
        
        # Calculate OB and FVG for current bar
        if i >= 2:
            ob_up = is_ob_up(i - 1)
            ob_down = is_ob_down(i - 1)
        else:
            ob_up = False
            ob_down = False
        
        if i >= 2:
            fvg_up = is_fvg_up(i - 2)
            fvg_down = is_fvg_down(i - 2)
        else:
            fvg_up = False
            fvg_down = False
        
        # Long entry conditions
        # Time window 1: 07:00-09:59 UTC
        in_long_time_window = 7 <= hour < 10
        
        # Entry when previous day low was swept + bullish OB+FVG
        if in_long_time_window:
            if flagpdl and ob_up and fvg_up:
                entry_ts = int(current_time)
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(current_close)
                
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                flagpdl = False  # Reset after entry
        
        # Short entry conditions
        # Time window 2: 12:00-14:59 UTC
        in_short_time_window = 12 <= hour < 15
        
        # Entry when previous day high was swept + bearish OB+FVG
        if in_short_time_window:
            if flagpdh and ob_down and fvg_down:
                entry_ts = int(current_time)
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(current_close)
                
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                flagpdh = False  # Reset after entry
    
    return results