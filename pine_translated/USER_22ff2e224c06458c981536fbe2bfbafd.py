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
    
    close = df['close']
    open_vals = df['open']
    high = df['high']
    low = df['low']
    
    # EMA lengths
    ema9_length = 9
    ema18_length = 18
    
    # Calculate EMAs using Wilder's method (equivalent to Pine's ta.ema)
    ema91 = close.ewm(span=ema9_length, adjust=False).mean()
    ema181 = close.ewm(span=ema18_length, adjust=False).mean()
    
    # Helper functions for OB and FVG identification
    def is_up(idx):
        return close.iloc[idx] > open_vals.iloc[idx]
    
    def is_down(idx):
        return close.iloc[idx] < open_vals.iloc[idx]
    
    def is_ob_up(idx):
        if idx + 1 >= len(df) or idx >= len(df):
            return False
        return (is_down(idx + 1) and is_up(idx) and 
                close.iloc[idx] > high.iloc[idx + 1])
    
    def is_ob_down(idx):
        if idx + 1 >= len(df) or idx >= len(df):
            return False
        return (is_up(idx + 1) and is_down(idx) and 
                close.iloc[idx] < low.iloc[idx + 1])
    
    def is_fvg_up(idx):
        if idx + 2 >= len(df):
            return False
        return low.iloc[idx] > high.iloc[idx + 2]
    
    def is_fvg_down(idx):
        if idx + 2 >= len(df):
            return False
        return high.iloc[idx] < low.iloc[idx + 2]
    
    # Build boolean series for conditions
    # Need at least 3 bars for FVG check
    valid_bars = len(df) > 3
    
    if not valid_bars:
        return entries
    
    # Calculate OB and FVG conditions
    # For index i, obUp uses isObUp(1) which checks conditions at index i-1
    # We need to shift the conditions appropriately
    
    ob_up_conditions = []
    ob_down_conditions = []
    fvg_up_conditions = []
    fvg_down_conditions = []
    
    for i in range(len(df)):
        if i >= 1 and i < len(df) - 1:
            ob_up_conditions.append(is_ob_up(i - 1))
            ob_down_conditions.append(is_ob_down(i - 1))
        else:
            ob_up_conditions.append(False)
            ob_down_conditions.append(False)
        
        if i < len(df) - 2:
            fvg_up_conditions.append(is_fvg_up(i))
            fvg_down_conditions.append(is_fvg_down(i))
        else:
            fvg_up_conditions.append(False)
            fvg_down_conditions.append(False)
    
    ob_up_series = pd.Series(ob_up_conditions, index=df.index)
    ob_down_series = pd.Series(ob_down_conditions, index=df.index)
    fvg_up_series = pd.Series(fvg_up_conditions, index=df.index)
    fvg_down_series = pd.Series(fvg_down_conditions, index=df.index)
    
    # Bullish entry condition: obUp1 and fvgUp1 and close > ema91 and close > ema181
    bull_entry = (ob_up_series & fvg_up_series & 
                  (close > ema91) & (close > ema181))
    
    # Bearish entry condition: obDown1 and fvgDown1 and close < ema91 and close < ema181
    bear_entry = (ob_down_series & fvg_down_series & 
                  (close < ema91) & (close < ema181))
    
    # Iterate through bars and generate entries
    for i in range(len(df)):
        # Skip if EMA values are NaN (need enough bars for EMA calculation)
        if pd.isna(ema91.iloc[i]) or pd.isna(ema181.iloc[i]):
            continue
        
        # Check for bullish entry
        if bull_entry.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
        
        # Check for bearish entry
        if bear_entry.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
    
    return entries