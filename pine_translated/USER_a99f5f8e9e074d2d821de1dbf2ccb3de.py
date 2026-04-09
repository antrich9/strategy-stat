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
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    time_col = df['time']
    
    n = len(df)
    
    # Calculate previous day high and low
    prev_day_high = pd.Series(np.nan, index=df.index)
    prev_day_low = pd.Series(np.nan, index=df.index)
    
    day_highs = []
    day_lows = []
    day_starts = []
    
    current_day_start_idx = 0
    current_day_high = high.iloc[0]
    current_day_low = low.iloc[0]
    
    for i in range(n):
        current_ts = time_col.iloc[i]
        current_day = current_ts // 86400
        
        if i > 0:
            prev_ts = time_col.iloc[i-1]
            prev_day = prev_ts // 86400
            
            if current_day != prev_day:
                day_highs.append(current_day_high)
                day_lows.append(current_day_low)
                day_starts.append(i)
                prev_day_high.iloc[i] = current_day_high
                prev_day_low.iloc[i] = current_day_low
                current_day_high = high.iloc[i]
                current_day_low = low.iloc[i]
            else:
                if i >= 1 and i - 1 < len(day_highs):
                    prev_day_high.iloc[i] = day_highs[i - 1]
                    prev_day_low.iloc[i] = day_lows[i - 1]
                current_day_high = max(current_day_high, high.iloc[i])
                current_day_low = min(current_day_low, low.iloc[i])
        else:
            current_day_high = high.iloc[i]
            current_day_low = low.iloc[i]
    
    flagpdl = pd.Series(False, index=df.index)
    flagpdh = pd.Series(False, index=df.index)
    
    for i in range(1, n):
        if close.iloc[i] > prev_day_high.iloc[i]:
            flagpdh.iloc[i] = True
        if close.iloc[i] < prev_day_low.iloc[i]:
            flagpdl.iloc[i] = True
    
    # Helper functions for OB and FVG
    def is_up(idx):
        return close.iloc[idx] > open_.iloc[idx]
    
    def is_down(idx):
        return close.iloc[idx] < open_.iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and close.iloc[idx] > high.iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and close.iloc[idx] < low.iloc[idx + 1]
    
    def is_fvg_up(idx):
        return low.iloc[idx] > high.iloc[idx + 2]
    
    def is_fvg_down(idx):
        return high.iloc[idx] < low.iloc[idx + 2]
    
    for i in range(2, n):
        if pd.isna(prev_day_high.iloc[i]) or pd.isna(prev_day_low.iloc[i]):
            continue
        
        ob_up_val = is_ob_up(i)
        fvg_up_val = is_fvg_up(i)
        ob_down_val = is_ob_down(i)
        fvg_down_val = is_fvg_down(i)
        
        if ob_up_val and fvg_up_val and flagpdl.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if ob_down_val and fvg_down_val and flagpdh.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time_col.iloc[i]),
                'entry_time': datetime.fromtimestamp(time_col.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return results