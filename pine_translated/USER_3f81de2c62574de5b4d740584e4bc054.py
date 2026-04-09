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
    
    # Strategy parameters (default values from Pine Script)
    atrLength = 14
    ciLength = 14
    lengthMD = 10
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    tradeDirection = 'Both'
    
    n = len(df)
    results = []
    trade_num = 1
    
    # Calculate ATR using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=range(n), dtype=float)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, n):
        atr.iloc[i] = (atr.iloc[i-1] * (atrLength - 1) + tr.iloc[i]) / atrLength
    
    # Calculate pivot points
    bb = input_lookback
    
    # pivotlow and pivothigh
    pl = pd.Series(np.nan, index=range(n))
    ph = pd.Series(np.nan, index=range(n))
    
    for i in range(bb, n - bb):
        # pivotlow: lowest low in window of size bb before and after
        left_low_idx = low.iloc[i-bb:i].idxmin()
        right_low_idx = low.iloc[i+1:i+bb+1].idxmin()
        if left_low_idx == i - bb and right_low_idx == i + 1:
            pl.iloc[i] = low.iloc[i]
        
        # pivothigh: highest high in window
        left_high_idx = high.iloc[i-bb:i].idxmax()
        right_high_idx = high.iloc[i+1:i+bb+1].idxmax()
        if left_high_idx == i - bb and right_high_idx == i + 1:
            ph.iloc[i] = high.iloc[i]
    
    # Forward fill NaN for fixnan
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Support and Resistance boxes
    # s_yLoc: location for support box
    s_yLoc = pd.Series(np.nan, index=range(n))
    for i in range(bb + 1, n):
        if not pd.isna(low.iloc[i - bb + 1]) and not pd.isna(low.iloc[i - bb - 1]):
            s_yLoc.iloc[i] = low.iloc[i - bb - 1] if low.iloc[i - bb + 1] > low.iloc[i - bb - 1] else low.iloc[i - bb + 1]
    
    # r_yLoc: location for resistance box
    r_yLoc = pd.Series(np.nan, index=range(n))
    for i in range(bb + 1, n):
        if not pd.isna(high.iloc[i - bb + 1]) and not pd.isna(high.iloc[i - bb - 1]):
            r_yLoc.iloc[i] = high.iloc[i - bb + 1] if high.iloc[i - bb + 1] > high.iloc[i - bb - 1] else high.iloc[i - bb - 1]
    
    # Box boundaries
    sTop = pd.Series(np.nan, index=range(n))
    sBot = pd.Series(np.nan, index=range(n))
    rTop = pd.Series(np.nan, index=range(n))
    rBot = pd.Series(np.nan, index=range(n))
    
    for i in range(n):
        if not pd.isna(pl.iloc[i]):
            box_start_idx = i - bb
            if box_start_idx >= 0 and box_start_idx < n:
                sTop.iloc[i] = pl.iloc[i]
                if not pd.isna(s_yLoc.iloc[box_start_idx]):
                    sBot.iloc[i] = s_yLoc.iloc[box_start_idx]
        
        if not pd.isna(ph.iloc[i]):
            box_start_idx = i - bb
            if box_start_idx >= 0 and box_start_idx < n:
                rTop.iloc[i] = ph.iloc[i]
                if not pd.isna(r_yLoc.iloc[box_start_idx]):
                    rBot.iloc[i] = r_yLoc.iloc[box_start_idx]
    
    # Breakout flags
    sBreak = pd.Series(False, index=range(n))
    rBreak = pd.Series(False, index=range(n))
    
    cu = pd.Series(False, index=range(n))
    co = pd.Series(False, index=range(n))
    
    for i in range(1, n):
        if pd.notna(sBot.iloc[i]):
            cu.iloc[i] = close.iloc[i] < sBot.iloc[i] and close.iloc[i-1] >= sBot.iloc[i]
        if pd.notna(rTop.iloc[i]):
            co.iloc[i] = close.iloc[i] > rTop.iloc[i] and close.iloc[i-1] <= rTop.iloc[i]
    
    # Update breakout flags
    for i in range(n):
        if cu.iloc[i] and not sBreak.iloc[i]:
            sBreak.iloc[i] = True
        if co.iloc[i] and not rBreak.iloc[i]:
            rBreak.iloc[i] = True
        
        if i > 0 and i - bb >= 0:
            if pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i-1]) and pl.iloc[i] != pl.iloc[i-1]:
                if pd.isna(sBreak.iloc[i]):
                    sBreak.iloc[i] = False
            if pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i-1]) and ph.iloc[i] != ph.iloc[i-1]:
                if pd.isna(rBreak.iloc[i]):
                    rBreak.iloc[i] = False
    
    # Retest conditions for support (s1, s2, s3, s4)
    s1 = pd.Series(False, index=range(n))
    s2 = pd.Series(False, index=range(n))
    s3 = pd.Series(False, index=range(n))
    s4 = pd.Series(False, index=range(n))
    
    # Retest conditions for resistance (r1, r2, r3, r4)
    r1 = pd.Series(False, index=range(n))
    r2 = pd.Series(False, index=range(n))
    r3 = pd.Series(False, index=range(n))
    r4 = pd.Series(False, index=range(n))
    
    # Calculate barssince for sBreak and rBreak
    barssince_sBreak = pd.Series(n, index=range(n))
    barssince_rBreak = pd.Series(n, index=range(n))
    last_sBreak_idx = -1000
    last_rBreak_idx = -1000
    
    for i in range(n):
        if sBreak.iloc[i]:
            last_sBreak_idx = i
        if last_sBreak_idx >= 0:
            barssince_sBreak.iloc[i] = i - last_sBreak_idx if last_sBreak_idx < i else n
        
        if rBreak.iloc[i]:
            last_rBreak_idx = i
        if last_rBreak_idx >= 0:
            barssince_rBreak.iloc[i] = i - last_rBreak_idx if last_rBreak_idx < i else n
    
    for i in range(n):
        if barssince_sBreak.iloc[i] > input_retSince:
            if pd.notna(sTop.iloc[i]) and pd.notna(sBot.iloc[i]):
                if high.iloc[i] >= sTop.iloc[i] and close.iloc[i] <= sBot.iloc[i]:
                    s1.iloc[i] = True
                if high.iloc[i] >= sTop.iloc[i] and close.iloc[i] >= sBot.iloc[i] and close.iloc[i] <= sTop.iloc[i]:
                    s2.iloc[i] = True
                if high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i]:
                    s3.iloc[i] = True
                if high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i] and close.iloc[i] < sBot.iloc[i]:
                    s4.iloc[i] = True
        
        if barssince_rBreak.iloc[i] > input_retSince:
            if pd.notna(rTop.iloc[i]) and pd.notna(rBot.iloc[i]):
                if low.iloc[i] <= rBot.iloc[i] and close.iloc[i] >= rTop.iloc[i]:
                    r1.iloc[i] = True
                if low.iloc[i] <= rBot.iloc[i] and close.iloc[i] <= rTop.iloc[i] and close.iloc[i] >= rBot.iloc[i]:
                    r2.iloc[i] = True
                if low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i]:
                    r3.iloc[i] = True
                if low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i] and close.iloc[i] > rTop.iloc[i]:
                    r4.iloc[i] = True
    
    # Retest validation
    sRetValid = pd.Series(False, index=range(n))
    rRetValid = pd.Series(False, index=range(n))
    
    sRetEvent = pd.Series(False, index=range(n))
    rRetEvent = pd.Series(False, index=range(n))
    
    sRetValue = pd.Series(np.nan, index=range(n))
    rRetValue = pd.Series(np.nan, index=range(n))
    
    # For support retest
    s_retOccurred = False
    s_retEvent_detected = False
    s_retEvent_idx = -1
    s_retEvent_val = np.nan
    s_potential_retest_idx = -1
    
    for i in range(n):
        s_active = s1.iloc[i] or s2.iloc[i] or s3.iloc[i] or s4.iloc[i]
        s_current_event = s_active and (i == 0 or not (s1.iloc[i-1] or s2.iloc[i-1] or s3.iloc[i-1] or s4.iloc[i-1]))
        
        if s_current_event:
            s_retEvent_detected = True
            s_retEvent_idx = i
            s_retEvent_val = low.iloc[i]
            s_retOccurred = False
            s_potential_retest_idx = i
        
        if s_retEvent_detected:
            bars_since = i - s_retEvent_idx
            
            if bars_since > 0 and bars_since <= input_retValid:
                ret_conditions = close.iloc[i] <= s_retEvent_val
                if ret_conditions and not s_retOccurred:
                    sRetValid.iloc[i] = True
                    s_retOccurred = True
            
            if bars_since > input_retValid:
                s_retEvent_detected = False
    
    # For resistance retest
    r_retOccurred = False
    r_retEvent_detected = False
    r_retEvent_idx = -1
    r_retEvent_val = np.nan
    r_potential_retest_idx = -1
    
    for i in range(n):
        r_active = r1.iloc[i] or r2.iloc[i] or r3.iloc[i] or r4.iloc[i]
        r_current_event = r_active and (i == 0 or not (r1.iloc[i-1] or r2.iloc[i-1] or r3.iloc[i-1] or r4.iloc[i-1]))
        
        if r_current_event:
            r_retEvent_detected = True
            r_retEvent_idx = i
            r_retEvent_val = high.iloc[i]
            r_retOccurred = False
            r_potential_retest_idx = i
        
        if r_retEvent_detected:
            bars_since = i - r_retEvent_idx
            
            if bars_since > 0 and bars_since <= input_retValid:
                ret_conditions = close.iloc[i] >= r_retEvent_val
                if ret_conditions and not r_retOccurred:
                    rRetValid.iloc[i] = True
                    r_retOccurred = True
            
            if bars_since > input_retValid:
                r_retEvent_detected = False
    
    # Generate entries based on tradeDirection
    for i in range(n):
        if pd.isna(close.iloc[i]):
            continue
        
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(close.iloc[i])
        
        if tradeDirection in ['Long', 'Both'] and sRetValid.iloc[i]:
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
        
        if tradeDirection in ['Short', 'Both'] and rRetValid.iloc[i]:
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
    
    return results