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
    trade_num = 0
    
    # State variables (like Pine Script var)
    current_position = "flat"
    short_signal = False
    long_signal = False
    
    # For tracking daily high/low
    prev_day_high = np.nan
    prev_day_low = np.nan
    
    first_bar = True
    current_day_start_ts = None
    
    # Track current day high/low
    current_day_high = np.nan
    current_day_low = np.nan
    
    n = len(df)
    
    for i in range(n):
        current_ts = df['time'].iloc[i]
        
        # Check for new day using date change
        is_new_day = False
        if first_bar:
            first_bar = False
        else:
            prev_ts = df['time'].iloc[i-1]
            prev_date = datetime.fromtimestamp(prev_ts, tz=timezone.utc).date()
            curr_date = datetime.fromtimestamp(current_ts, tz=timezone.utc).date()
            is_new_day = (curr_date != prev_date)
        
        if is_new_day:
            # Update prev day high/low for next bar's signal checking
            prev_day_high = current_day_high
            prev_day_low = current_day_low
            # Reset for new day
            current_day_high = df['high'].iloc[i]
            current_day_low = df['low'].iloc[i]
        else:
            if pd.isna(current_day_high):
                current_day_high = df['high'].iloc[i]
                current_day_low = df['low'].iloc[i]
            else:
                current_day_high = max(current_day_high, df['high'].iloc[i])
                current_day_low = min(current_day_low, df['low'].iloc[i])
        
        # Skip if we don't have previous day values yet
        if pd.isna(prev_day_high) or pd.isna(prev_day_low):
            continue
        
        close_price = df['close'].iloc[i]
        
        # Short Condition: close > d_info.ph and current_position == "flat"
        if close_price > prev_day_high and current_position == "flat":
            short_signal = True
            long_signal = False
        
        # Long Condition: close < d_info.pl and current_position == "flat"
        if close_price < prev_day_low and current_position == "flat":
            long_signal = True
            short_signal = False
        
        # Only long entries are active (short is commented out in original)
        if long_signal and current_position == "flat":
            trade_num += 1
            entry_price = df['close'].iloc[i]
            entry_time_str = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            current_position = "long"
            long_signal = False
    
    return entries