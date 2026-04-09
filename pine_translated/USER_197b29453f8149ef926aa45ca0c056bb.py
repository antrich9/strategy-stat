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
    close = df['close']
    high = df['high']
    low = df['low']
    time = df['time']
    
    bb = 20
    input_retSince = 2
    input_retValid = 2
    tradeDirection = 'Both'
    
    # Helper functions
    def calc_tma(src, length):
        ma = src.rolling(length).mean()
        return ma.rolling(length).mean()
    
    # Calculate TMA (not directly used in entries but part of the system)
    tma = calc_tma(close, bb)
    
    # Pivot calculations
    pl = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)
    
    for i in range(bb, len(df)):
        window_low = low.iloc[i-bb:i+1]
        window_high = high.iloc[i-bb:i+1]
        min_idx = window_low.idxmin()
        max_idx = window_high.idxmax()
        if min_idx == i - bb:
            pl.iloc[i] = low.iloc[i - bb]
        if max_idx == i - bb:
            ph.iloc[i] = high.iloc[i - bb]
    
    pl = pl.fillna(method='ffill')
    ph = ph.fillna(method='ffill')
    
    # Box calculations
    s_yLoc = pd.Series(np.nan, index=df.index)
    r_yLoc = pd.Series(np.nan, index=df.index)
    
    for i in range(bb + 1, len(df)):
        s_yLoc.iloc[i] = low.iloc[bb - 1] if low.iloc[bb + 1] > low.iloc[bb - 1] else low.iloc[bb + 1]
        r_yLoc.iloc[i] = high.iloc[bb + 1] if high.iloc[bb + 1] > high.iloc[bb - 1] else high.iloc[bb - 1]
    
    s_yLoc = s_yLoc.fillna(method='ffill')
    r_yLoc = r_yLoc.fillna(method='ffill')
    
    # Box boundaries
    sTop = pd.Series(np.nan, index=df.index)
    sBot = pd.Series(np.nan, index=df.index)
    rTop = pd.Series(np.nan, index=df.index)
    rBot = pd.Series(np.nan, index=df.index)
    
    for i in range(bb, len(df)):
        sTop.iloc[i] = s_yLoc.iloc[i]
        sBot.iloc[i] = pl.iloc[i]
        rTop.iloc[i] = ph.iloc[i]
        rBot.iloc[i] = r_yLoc.iloc[i]
    
    # Breakout detection
    cu = pd.Series(False, index=df.index)
    co = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if i >= bb and not pd.isna(sBot.iloc[i]) and not pd.isna(sBot.iloc[i-1]):
            cu.iloc[i] = close.iloc[i] < sBot.iloc[i] and close.iloc[i-1] >= sBot.iloc[i-1]
        if i >= bb and not pd.isna(rTop.iloc[i]) and not pd.isna(rTop.iloc[i-1]):
            co.iloc[i] = close.iloc[i] > rTop.iloc[i] and close.iloc[i-1] <= rTop.iloc[i-1]
    
    # Retest state
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    for i in range(bb, len(df)):
        if pd.isna(sTop.iloc[i]) or pd.isna(sBot.iloc[i]):
            continue
        
        if cu.iloc[i] and not sBreak.iloc[i-1]:
            sBreak.iloc[i] = True
        elif i > 0:
            sBreak.iloc[i] = sBreak.iloc[i-1]
        
        if co.iloc[i] and not rBreak.iloc[i-1]:
            rBreak.iloc[i] = True
        elif i > 0:
            rBreak.iloc[i] = rBreak.iloc[i-1]
        
        # Check retest validity
        sBarsSince = 0
        rBarsSince = 0
        for j in range(1, i+1):
            if sBreak.iloc[i-j]:
                sBarsSince = j
                break
        for j in range(1, i+1):
            if rBreak.iloc[i-j]:
                rBarsSince = j
                break
        
        if sBarsSince > input_retSince:
            cond1 = high.iloc[i] >= sTop.iloc[i] and close.iloc[i] <= sBot.iloc[i]
            cond2 = high.iloc[i] >= sTop.iloc[i] and close.iloc[i] >= sBot.iloc[i] and close.iloc[i] <= sTop.iloc[i]
            cond3 = high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i]
            cond4 = high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i] and close.iloc[i] < sBot.iloc[i]
            if cond1 or cond2 or cond3 or cond4:
                sRetValid.iloc[i] = True
        
        if rBarsSince > input_retSince:
            cond1 = low.iloc[i] <= rBot.iloc[i] and close.iloc[i] >= rTop.iloc[i]
            cond2 = low.iloc[i] <= rBot.iloc[i] and close.iloc[i] <= rTop.iloc[i] and close.iloc[i] >= rBot.iloc[i]
            cond3 = low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i]
            cond4 = low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i] and close.iloc[i] > rTop.iloc[i]
            if cond1 or cond2 or cond3 or cond4:
                rRetValid.iloc[i] = True
        
        # Reset on pivot change
        if i > 0 and i < len(df) - 1:
            if not pd.isna(pl.iloc[i]) and (pd.isna(pl.iloc[i-1]) or pl.iloc[i] != pl.iloc[i-1]):
                if not sBreak.iloc[i]:
                    sBreak.iloc[i] = False
            if not pd.isna(ph.iloc[i]) and (pd.isna(ph.iloc[i-1]) or ph.iloc[i] != ph.iloc[i-1]):
                if not rBreak.iloc[i]:
                    rBreak.iloc[i] = False
    
    # Generate entries
    entries = []
    trade_num = 1
    
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    
    prev_sRetValid = False
    prev_rRetValid = False
    
    for i in range(bb, len(df)):
        if sRetValid.iloc[i] and not prev_sRetValid:
            long_entry.iloc[i] = True
        if rRetValid.iloc[i] and not prev_rRetValid:
            short_entry.iloc[i] = True
        
        prev_sRetValid = sRetValid.iloc[i]
        prev_rRetValid = rRetValid.iloc[i]
    
    if tradeDirection == 'Long':
        short_entry = pd.Series(False, index=df.index)
    elif tradeDirection == 'Short':
        long_entry = pd.Series(False, index=df.index)
    
    for i in range(bb, len(df)):
        if long_entry.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries