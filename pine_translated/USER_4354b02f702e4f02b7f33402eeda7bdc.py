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
    bb = 20
    input_retSince = 2
    input_retValid = 2
    
    n = len(df)
    entries = []
    trade_num = 1
    
    sBreak = np.nan
    rBreak = np.nan
    sRetSince = 0
    rRetSince = 0
    sRetValid = False
    rRetValid = False
    
    pl_vals = np.full(n, np.nan)
    ph_vals = np.full(n, np.nan)
    sTop_vals = np.full(n, np.nan)
    sBot_vals = np.full(n, np.nan)
    rTop_vals = np.full(n, np.nan)
    rBot_vals = np.full(n, np.nan)
    
    for i in range(bb, n):
        pl_vals[i] = df['low'].iloc[i-bb] if all(df['low'].iloc[i-bb:i+bb].index) else np.nan
        if i >= 2*bb:
            window = df['low'].iloc[i-bb:i].values
            min_idx = np.argmin(window)
            if min_idx == bb - 1:
                pl_vals[i] = window[min_idx]
    
    for i in range(bb, n):
        ph_vals[i] = df['high'].iloc[i-bb] if all(df['high'].iloc[i-bb:i+bb].index) else np.nan
        if i >= 2*bb:
            window = df['high'].iloc[i-bb:i].values
            max_idx = np.argmax(window)
            if max_idx == bb - 1:
                ph_vals[i] = window[max_idx]
    
    for i in range(bb, n):
        if not pd.isna(pl_vals[i]):
            s_yLoc = df['low'].iloc[i-bb-1] if df['low'].iloc[i-bb-1] > df['low'].iloc[i-bb+1] else df['low'].iloc[i-bb+1]
            sBot_vals[i] = min(s_yLoc, pl_vals[i])
            sTop_vals[i] = pl_vals[i]
        if not pd.isna(ph_vals[i]):
            r_yLoc = df['high'].iloc[i-bb-1] if df['high'].iloc[i-bb-1] > df['high'].iloc[i-bb+1] else df['high'].iloc[i-bb+1]
            rTop_vals[i] = max(r_yLoc, ph_vals[i])
            rBot_vals[i] = ph_vals[i]
    
    for i in range(bb, n):
        if i > 0:
            if not pd.isna(pl_vals[i]) and (pd.isna(pl_vals[i-1]) or pl_vals[i] != pl_vals[i-1]):
                sBreak = np.nan
            if not pd.isna(ph_vals[i]) and (pd.isna(ph_vals[i-1]) or ph_vals[i] != ph_vals[i-1]):
                rBreak = np.nan
        
        sBot = sBot_vals[i] if not pd.isna(sBot_vals[i]) else -np.inf
        sTop = sTop_vals[i] if not pd.isna(sTop_vals[i]) else np.inf
        rTop = rTop_vals[i] if not pd.isna(rTop_vals[i]) else np.inf
        rBot = rBot_vals[i] if not pd.isna(rBot_vals[i]) else -np.inf
        
        close_prev = df['close'].iloc[i-1]
        close_i = df['close'].iloc[i]
        high_i = df['high'].iloc[i]
        low_i = df['low'].iloc[i]
        
        if pd.isna(sBreak) and close_prev < sBot and close_i >= sBot:
            sBreak = True
        
        if pd.isna(rBreak) and close_prev > rTop and close_i <= rTop:
            rBreak = True
        
        s1 = not pd.isna(sBreak) and (i - np.where(~pd.isna(pl_vals[:i+1]))[0][-1]) > input_retSince and high_i >= sTop and close_i <= sBot
        s2 = not pd.isna(sBreak) and (i - np.where(~pd.isna(pl_vals[:i+1]))[0][-1]) > input_retSince and high_i >= sTop and close_i >= sBot and close_i <= sTop
        s3 = not pd.isna(sBreak) and (i - np.where(~pd.isna(pl_vals[:i+1]))[0][-1]) > input_retSince and high_i >= sBot and high_i <= sTop
        s4 = not pd.isna(sBreak) and (i - np.where(~pd.isna(pl_vals[:i+1]))[0][-1]) > input_retSince and high_i >= sBot and high_i <= sTop and close_i < sBot
        
        r1 = not pd.isna(rBreak) and (i - np.where(~pd.isna(ph_vals[:i+1]))[0][-1]) > input_retSince and low_i <= rBot and close_i >= rTop
        r2 = not pd.isna(rBreak) and (i - np.where(~pd.isna(ph_vals[:i+1]))[0][-1]) > input_retSince and low_i <= rBot and close_i <= rTop and close_i >= rBot
        r3 = not pd.isna(rBreak) and (i - np.where(~pd.isna(ph_vals[:i+1]))[0][-1]) > input_retSince and low_i <= rTop and low_i >= rBot
        r4 = not pd.isna(rBreak) and (i - np.where(~pd.isna(ph_vals[:i+1]))[0][-1]) > input_retSince and low_i <= rTop and low_i >= rBot and close_i > rTop
        
        sRetActive = s1 or s2 or s3 or s4
        rRetActive = r1 or r2 or r3 or r4
        
        sRetEvent = sRetActive
        rRetEvent = rRetActive
        
        if sRetEvent:
            sRetSince = 1
        elif sRetSince > 0:
            sRetSince += 1
        if rRetEvent:
            rRetSince = 1
        elif rRetSince > 0:
            rRetSince += 1
        
        if sRetSince > 0 and sRetSince <= input_retValid:
            sRetValid = (close_i <= sTop and high_i >= sBot)
        else:
            sRetValid = False
        
        if rRetSince > 0 and rRetSince <= input_retValid:
            rRetValid = (close_i >= rBot and low_i <= rTop)
        else:
            rRetValid = False
        
        if sRetValid:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            sBreak = np.nan
            sRetSince = 0
            sRetValid = False
        
        if rRetValid:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            rBreak = np.nan
            rRetSince = 0
            rRetValid = False
    
    return entries