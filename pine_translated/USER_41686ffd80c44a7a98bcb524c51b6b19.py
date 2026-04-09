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
    
    # State variables
    ph = np.nan
    pl = np.nan
    ch = np.nan
    cl = np.nan
    co = np.nan
    p_up = False
    
    current_position = "flat"
    entry_price = np.nan
    long_signal = False
    short_signal = False
    prev_date = None
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        current_date = pd.to_datetime(ts, unit='s').date()
        
        d_info_update = (prev_date is not None and current_date != prev_date)
        
        if d_info_update:
            if not np.isnan(ch):
                if df['close'].iloc[i-1] >= co:
                    p_up = True
                else:
                    p_up = False
                ph = ch
                pl = cl
                ch = df['high'].iloc[i]
                cl = df['low'].iloc[i]
                co = df['open'].iloc[i]
            else:
                ch = df['high'].iloc[i]
                cl = df['low'].iloc[i]
        else:
            if np.isnan(ch):
                ch = df['high'].iloc[i]
                cl = df['low'].iloc[i]
            else:
                ch = max(df['high'].iloc[i], ch)
                cl = min(df['low'].iloc[i], cl)
        
        if d_info_update and df['close'].iloc[i] > ph and current_position == "flat":
            short_signal = True
            long_signal = False
        
        if d_info_update and df['close'].iloc[i] < pl and current_position == "flat":
            long_signal = True
            short_signal = False
        
        if long_signal and current_position == "flat":
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            current_position = "long"
            entry_price = df['close'].iloc[i]
            long_signal = False
        
        if short_signal and current_position == "flat":
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            current_position = "short"
            entry_price = df['close'].iloc[i]
            short_signal = False
        
        if i > 0 and ((current_position == "long" and df['close'].iloc[i] != df['close'].iloc[i]) or 
                      (current_position == "short" and df['close'].iloc[i] != df['close'].iloc[i])):
            pass
        
        prev_date = current_date
    
    return entries