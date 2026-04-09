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
    
    if len(df) < 3:
        return entries
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_vals = df['open']
    time = df['time']
    
    # FVG conditions from Pine Script
    # Top FVG (bearish): low[2] <= open[1] and high[0] >= close[1] and close[0] < low[1]
    top_fvg_bway = (low.shift(2) <= open_vals.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    
    # Top FVG weak: low[2] <= open[1] and high[0] >= close[1] and close[0] > low[1]
    top_fvg_weak = (low.shift(2) <= open_vals.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))
    
    # Bottom FVG (bullish): high[2] >= open[1] and low[0] <= close[1] and close[0] > high[1]
    bottom_fvg_bway = (high.shift(2) >= open_vals.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    
    # Bottom FVG weak: high[2] >= open[1] and low[0] <= close[1] and close[0] < high[1]
    bottom_fvg_weak = (high.shift(2) >= open_vals.shift(1)) & (low <= close.shift(1)) & (close < high.shift(1))
    
    # FVG size conditions
    top_fvg_size = low.shift(2) - high
    bottom_fvg_size = low - high.shift(2)
    
    # Valid FVG conditions (size > 0)
    valid_top_fvg = top_fvg_bway & (top_fvg_size > 0) | top_fvg_weak & (top_fvg_size > 0)
    valid_bottom_fvg = bottom_fvg_bway & (bottom_fvg_size > 0) | bottom_fvg_weak & (bottom_fvg_size > 0)
    
    # OB detection helpers
    is_up = close > open_vals
    is_down = close < open_vals
    
    # Check for stacked OB + FVG conditions (from end of script)
    # isObUp(index) => isDown(index + 1) and isUp(index) and close[index]...
    # This appears to be checking forbullish order blocks stacked with FVGs
    
    for i in range(3, len(df)):
        # Skip if any required values are NaN
        if pd.isna(high.iloc[i]) or pd.isna(low.iloc[i]) or pd.isna(close.iloc[i]):
            continue
        if pd.isna(low.iloc[i-2]) or pd.isna(open_vals.iloc[i-1]) or pd.isna(close.iloc[i-1]) or pd.isna(low.iloc[i-1]):
            continue
        if pd.isna(high.iloc[i-2]) or pd.isna(high.iloc[i-1]):
            continue
        
        entry_ts = int(time.iloc[i])
        entry_price = float(close.iloc[i])
        
        # Check for bullish FVG setup (potential long entry)
        # Bull FVG: bottom imbalance detected, FVG not yet tested
        if valid_bottom_fvg.iloc[i]:
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        # Check for bearish FVG setup (potential short entry)
        # Bear FVG: top imbalance detected, FVG not yet tested
        if valid_top_fvg.iloc[i]:
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries