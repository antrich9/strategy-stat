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
    
    prd = 10
    follow = True
    bull = True
    bear = True
    levels = [0.5, 0.618]
    
    n = len(df)
    entries = []
    trade_num = 0
    
    up = df['high'].iloc[0]
    dn = df['low'].iloc[0]
    iup = 0
    idn = 0
    swing_low = np.nan
    swing_high = np.nan
    iswing_low = -1
    iswing_high = -1
    pos = 0
    
    golden_high = np.nan
    golden_low = np.nan
    entered_zone = False
    
    up_prev = up
    dn_prev = dn
    
    for i in range(prd, n):
        high_val = df['high'].iloc[i]
        low_val = df['low'].iloc[i]
        close_val = df['close'].iloc[i]
        time_val = df['time'].iloc[i]
        
        up_prev = up
        dn_prev = dn
        
        pvt_hi = df['high'].iloc[i-prd]
        pvt_lo = df['low'].iloc[i-prd]
        
        if bull:
            max_in_window = df['high'].iloc[i-prd+1:i+1].max() if i-prd+1 <= i else high_val
            if pvt_hi == max_in_window:
                up = pvt_hi
                iup = i - prd
        
        if bear:
            min_in_window = df['low'].iloc[i-prd+1:i+1].min() if i-prd+1 <= i else low_val
            if pvt_lo == min_in_window:
                dn = pvt_lo
                idn = i - prd
        
        if bull and up > up_prev and pos <= 0:
            swing_low = dn
            iswing_low = idn
            swing_high = up
            iswing_high = iup
            
            golden_high = swing_low + (swing_high - swing_low) * levels[0]
            golden_low = swing_low + (swing_high - swing_low) * levels[1]
            
            entered_zone = False
            pos = 1
        
        if bear and dn < dn_prev and pos >= 0:
            swing_low = dn
            iswing_low = idn
            swing_high = up
            iswing_high = iup
            
            golden_high = swing_low + (swing_high - swing_low) * levels[1]
            golden_low = swing_low + (swing_high - swing_low) * levels[0]
            
            entered_zone = False
            pos = -1
        
        if not np.isnan(golden_low) and not np.isnan(golden_high):
            if pos == 1 and not entered_zone:
                if golden_low <= close_val <= golden_high:
                    trade_num += 1
                    entry_time = datetime.fromtimestamp(time_val, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': time_val,
                        'entry_time': entry_time,
                        'entry_price_guess': close_val,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': close_val,
                        'raw_price_b': close_val
                    })
                    entered_zone = True
            
            elif pos == -1 and not entered_zone:
                if golden_low <= close_val <= golden_high:
                    trade_num += 1
                    entry_time = datetime.fromtimestamp(time_val, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': time_val,
                        'entry_time': entry_time,
                        'entry_price_guess': close_val,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': close_val,
                        'raw_price_b': close_val
                    })
                    entered_zone = True
    
    return entries