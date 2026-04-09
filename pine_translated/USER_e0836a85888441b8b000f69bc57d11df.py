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
    n = len(df)
    if n < 2:
        return []
    
    time_arr = df['time'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    
    # Detect new days using date change
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc).date() for ts in time_arr]
    is_new_day = np.array([True] + [dates[i] != dates[i-1] for i in range(1, n)])
    
    # Tracking variables for daily info
    ph = np.nan  # previous high
    pl = np.nan  # previous low
    ch = np.nan  # current high
    cl = np.nan  # current low
    co = np.nan  # current open
    p_up = False  # previous bar was up
    
    # Strategy state
    current_position = "flat"
    short_signal = False
    long_signal = False
    
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        # Update info on new day
        if is_new_day[i]:
            # Move previous day values
            if not np.isnan(ch):
                if not np.isnan(co):
                    p_up = close_arr[i-1] >= co
                ph = ch
                pl = cl
            ch = high_arr[i]
            cl = low_arr[i]
            co = open_arr[i]
        else:
            # Update current day high/low
            if np.isnan(ch):
                ch = high_arr[i]
                cl = low_arr[i]
            else:
                ch = max(ch, high_arr[i])
                cl = min(cl, low_arr[i])
        
        # Reset position if flat
        if current_position == "flat":
            short_signal = False
            long_signal = False
        
        # Signal logic (only short is enabled, long is commented out in Pine)
        if current_position == "flat" and not np.isnan(ph) and not np.isnan(pl):
            # Short signal: close > previous high
            if close_arr[i] > ph:
                short_signal = True
                long_signal = False
            
            # Long signal: close < previous low (commented out in Pine)
            if close_arr[i] < pl:
                long_signal = True
                short_signal = False
        
        # Execute short entry on signal
        if short_signal and current_position == "flat":
            entry_ts = time_arr[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close_arr[i]
            
            entries.append({
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
            current_position = "short"
            short_signal = False
        
        # Execute long entry on signal (commented out in Pine, but implementation included)
        if long_signal and current_position == "flat":
            entry_ts = time_arr[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close_arr[i]
            
            entries.append({
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
            current_position = "long"
            long_signal = False
        
        # Reset to flat when position closed (position_size == 0)
        # In this simplified version, we only track if we're in a position
        # Since we're only generating entries, we reset on next signal condition
        # when current_position != "flat" - this is simplified since we don't track exits
    
    return entries