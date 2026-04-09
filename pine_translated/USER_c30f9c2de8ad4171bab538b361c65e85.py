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
    
    fvg_size_percent = 0.1
    results = []
    trade_num = 0
    
    fvg_high = np.nan
    fvg_low = np.nan
    
    for i in range(len(df)):
        current_ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)
        
        in_time_window = (dt.hour >= 15) and (dt.hour < 16)
        
        # FVG detection (only if within time window and no FVG is already stored)
        if in_time_window and pd.isna(fvg_high):
            if i >= 2:
                low_prev = df['low'].iloc[i-1]
                high_prev = df['high'].iloc[i-1]
                high_2_prev = df['high'].iloc[i-2]
                
                if pd.notna(low_prev) and pd.notna(high_prev) and pd.notna(high_2_prev):
                    fvg_size = df['close'].iloc[i] * fvg_size_percent / 100
                    
                    if (low_prev > high_2_prev) and ((high_prev - low_prev) >= fvg_size):
                        fvg_high = high_prev
                        fvg_low = low_prev
        
        # Reset FVG outside time window
        if not in_time_window:
            fvg_high = np.nan
            fvg_low = np.nan
        
        # Entry on retracement into FVG
        if not pd.isna(fvg_high):
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            
            if pd.notna(current_low) and pd.notna(current_close):
                if (current_low <= fvg_high) and (current_close >= fvg_low):
                    trade_num += 1
                    entry_price = current_close
                    
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': current_ts,
                        'entry_time': dt.isoformat(),
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    
                    # Reset FVG after entry
                    fvg_high = np.nan
                    fvg_low = np.nan
    
    return results