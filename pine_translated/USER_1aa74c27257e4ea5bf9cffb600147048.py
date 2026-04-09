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
    
    # Default input values from Pine Script
    bb = 20  # lookback range
    input_retSince = 2  # bars since breakout
    input_retValid = 2  # retest detection limiter
    length = 20  # Volatility Stop Length
    factor = 2.0  # Volatility Stop Multiplier
    atrLength = 14  # ATR Length
    
    n = len(df)
    entries = []
    trade_num = 1
    
    # Pivot levels
    pl = pd.Series(index=df.index, dtype=float)
    ph = pd.Series(index=df.index, dtype=float)
    
    for i in range(bb, n - bb):
        pivot_low_val = df['low'].iloc[i - bb:i + bb + 1].min()
        if df['low'].iloc[i] == pivot_low_val:
            pl.iloc[i] = pivot_low_val
        else:
            pl.iloc[i] = np.nan
            
        pivot_high_val = df['high'].iloc[i - bb:i + bb + 1].max()
        if df['high'].iloc[i] == pivot_high_val:
            ph.iloc[i] = pivot_high_val
        else:
            ph.iloc[i] = np.nan
    
    # Fill forward NaN with last valid
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Box boundaries - use fixed window based on bb
    sTop = pd.Series(index=df.index, dtype=float)
    sBot = pd.Series(index=df.index, dtype=float)
    rTop = pd.Series(index=df.index, dtype=float)
    rBot = pd.Series(index=df.index, dtype=float)
    
    for i in range(bb + 1, n):
        s_yLoc = df['low'].iloc[i - 1] if df['low'].iloc[i - 1] > df['low'].iloc[i + 1] else df['low'].iloc[i + 1]
        sTop.iloc[i] = pl.iloc[i] if not pd.isna(pl.iloc[i]) else sTop.iloc[i - 1] if i > 0 else np.nan
        sBot.iloc[i] = s_yLoc if not pd.isna(pl.iloc[i]) else sBot.iloc[i - 1] if i > 0 else np.nan
        
        r_yLoc = df['high'].iloc[i - 1] if df['high'].iloc[i - 1] > df['high'].iloc[i + 1] else df['high'].iloc[i + 1]
        rTop.iloc[i] = r_yLoc if not pd.isna(ph.iloc[i]) else rTop.iloc[i - 1] if i > 0 else np.nan
        rBot.iloc[i] = ph.iloc[i] if not pd.isna(ph.iloc[i]) else rBot.iloc[i - 1] if i > 0 else np.nan
    
    # Breakout detection
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    for i in range(1, n):
        if pd.isna(sTop.iloc[i]) or pd.isna(sBot.iloc[i]):
            continue
        # Support breakout: close crosses below sBot
        if df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i - 1] >= sBot.iloc[i - 1]:
            sBreak.iloc[i] = True
        # Resistance breakout: close crosses above rTop
        if pd.isna(rTop.iloc[i]) or pd.isna(rBot.iloc[i]):
            continue
        if df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i - 1] <= rTop.iloc[i - 1]:
            rBreak.iloc[i] = True
    
    # Retest conditions
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    for i in range(bb + 1, n):
        if pd.isna(sTop.iloc[i]) or pd.isna(sBot.iloc[i]):
            continue
        bars_since_break = -1
        for j in range(i - 1, -1, -1):
            if sBreak.iloc[j]:
                bars_since_break = i - j
                break
        if bars_since_break > input_retSince:
            # s1: high >= sTop and close <= sBot
            if df['high'].iloc[i] >= sTop.iloc[i] and df['close'].iloc[i] <= sBot.iloc[i]:
                if bars_since_break <= input_retValid:
                    sRetValid.iloc[i] = True
            # s2: high >= sTop and close >= sBot and close <= sTop
            if df['high'].iloc[i] >= sTop.iloc[i] and df['close'].iloc[i] >= sBot.iloc[i] and df['close'].iloc[i] <= sTop.iloc[i]:
                if bars_since_break <= input_retValid:
                    sRetValid.iloc[i] = True
            # s3: high >= sBot and high <= sTop
            if df['high'].iloc[i] >= sBot.iloc[i] and df['high'].iloc[i] <= sTop.iloc[i]:
                if bars_since_break <= input_retValid:
                    sRetValid.iloc[i] = True
            # s4: high >= sBot and high <= sTop and close < sBot
            if df['high'].iloc[i] >= sBot.iloc[i] and df['high'].iloc[i] <= sTop.iloc[i] and df['close'].iloc[i] < sBot.iloc[i]:
                if bars_since_break <= input_retValid:
                    sRetValid.iloc[i] = True
        
        if pd.isna(rTop.iloc[i]) or pd.isna(rBot.iloc[i]):
            continue
        bars_since_break = -1
        for j in range(i - 1, -1, -1):
            if rBreak.iloc[j]:
                bars_since_break = i - j
                break
        if bars_since_break > input_retSince:
            # r1: low <= rBot and close >= rTop
            if df['low'].iloc[i] <= rBot.iloc[i] and df['close'].iloc[i] >= rTop.iloc[i]:
                if bars_since_break <= input_retValid:
                    rRetValid.iloc[i] = True
            # r2: low <= rBot and close <= rTop and close >= rBot
            if df['low'].iloc[i] <= rBot.iloc[i] and df['close'].iloc[i] <= rTop.iloc[i] and df['close'].iloc[i] >= rBot.iloc[i]:
                if bars_since_break <= input_retValid:
                    rRetValid.iloc[i] = True
            # r3: low <= rTop and low >= rBot
            if df['low'].iloc[i] <= rTop.iloc[i] and df['low'].iloc[i] >= rBot.iloc[i]:
                if bars_since_break <= input_retValid:
                    rRetValid.iloc[i] = True
            # r4: low <= rTop and low >= rBot and close > rTop
            if df['low'].iloc[i] <= rTop.iloc[i] and df['low'].iloc[i] >= rBot.iloc[i] and df['close'].iloc[i] > rTop.iloc[i]:
                if bars_since_break <= input_retValid:
                    rRetValid.iloc[i] = True
    
    # Volatility Stop calculation
    vStop = pd.Series(index=df.index, dtype=float)
    uptrend = pd.Series(False, index=df.index)
    
    # Calculate ATR using Wilder method
    tr = pd.Series(index=df.index, dtype=float)
    for i in range(1, n):
        high_low = df['high'].iloc[i] - df['low'].iloc[i]
        high_close = abs(df['high'].iloc[i] - df['close'].iloc[i - 1])
        low_close = abs(df['low'].iloc[i] - df['close'].iloc[i - 1])
        tr.iloc[i] = max(high_low, high_close, low_close)
    
    # First ATR value is simple average of first 'length' values
    atr = pd.Series(index=df.index, dtype=float)
    if n > atrLength:
        first_atr = tr.iloc[1:atrLength + 1].mean()
        atr.iloc[atrLength] = first_atr
    
    for i in range(atrLength + 1, n):
        atr.iloc[i] = (atr.iloc[i - 1] * (atrLength - 1) + tr.iloc[i]) / atrLength
    
    max_val = df['close'].iloc[0]
    min_val = df['close'].iloc[0]
    stop_val = np.nan
    
    for i in range(length, n):
        src = df['close'].iloc[i]
        atrM = atr.iloc[i] * factor if not pd.isna(atr.iloc[i]) else tr.iloc[i]
        
        max_val = max(max_val, src)
        min_val = min(min_val, src)
        
        if pd.isna(stop_val):
            stop_val = src
        else:
            if uptrend.iloc[i - 1]:
                stop_val = max(stop_val, max_val - atrM)
            else:
                stop_val = min(stop_val, min_val + atrM)
        
        cur_uptrend = src - stop_val >= 0.0
        
        if cur_uptrend != uptrend.iloc[i - 1]:
            max_val = src
            min_val = src
            stop_val = max_val - atrM if cur_uptrend else min_val + atrM
        
        uptrend.iloc[i] = cur_uptrend
        vStop.iloc[i] = stop_val
    
    # Generate entries
    for i in range(bb + 1, n):
        if pd.isna(vStop.iloc[i]) or pd.isna(uptrend.iloc[i]):
            continue
        
        entry_signal = uptrend.iloc[i] and (not uptrend.iloc[i - 1] if i > 0 else False)
        retest_confirm = sRetValid.iloc[i] or rRetValid.iloc[i]
        
        if entry_signal and retest_confirm:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry)
            trade_num += 1
    
    return entries