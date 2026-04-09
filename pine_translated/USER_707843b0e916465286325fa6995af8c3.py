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
    
    # Parameters (from Pine Script inputs)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = "Both"  # Can be "Long", "Short", or "Both"
    
    # Heiken Ashi parameters
    maMethod = "EMA"
    maLength1 = 6
    maLength2 = 4
    
    # Support/Resistance parameters
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'
    input_breakout = True
    input_retest = True
    
    # Calculate Heiken Ashi
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    haClose = (o + h + l + c) / 4
    
    haOpen = np.zeros(len(df))
    haOpen[0] = (o[0] + c[0]) / 2
    for i in range(1, len(df)):
        haOpen[i] = (haOpen[i-1] + haClose[i-1]) / 2
    
    haHigh = np.maximum(h, np.maximum(haOpen, haClose))
    haLow = np.minimum(l, np.minimum(haOpen, haClose))
    
    # Apply Moving Averages for Smoothness
    if maMethod == "EMA":
        smoothHAOpen = pd.Series(haOpen).ewm(span=maLength1, adjust=False).mean().values
        smoothHAClose = pd.Series(haClose).ewm(span=maLength2, adjust=False).mean().values
    else:
        smoothHAOpen = pd.Series(haOpen).rolling(maLength1).mean().values
        smoothHAClose = pd.Series(haClose).rolling(maLength2).mean().mean().values
    
    # Calculate ATR (Wilder ATR)
    tr = np.zeros(len(df))
    tr[0] = h[0] - l[0]
    for i in range(1, len(df)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    
    atr = np.zeros(len(df))
    atr[atrLength-1] = np.mean(tr[:atrLength])
    for i in range(atrLength, len(df)):
        atr[i] = (atr[i-1] * (atrLength - 1) + tr[i]) / atrLength
    
    # Calculate pivot points
    pl = np.zeros(len(df))
    ph = np.zeros(len(df))
    
    for i in range(input_lookback, len(df)):
        window_low = l[i-input_lookback:i+input_lookback+1]
        window_high = h[i-input_lookback:i+input_lookback+1]
        pl[i] = np.min(window_low)
        ph[i] = np.max(window_high)
    
    # Initialize variables
    sBreak = np.zeros(len(df), dtype=bool)
    rBreak = np.zeros(len(df), dtype=bool)
    sTop = np.zeros(len(df))
    sBot = np.zeros(len(df))
    rTop = np.zeros(len(df))
    rBot = np.zeros(len(df))
    
    entries = []
    trade_num = 1
    
    # Iterate through bars
    for i in range(len(df)):
        if i < input_lookback:
            continue
            
        # Update boxes
        if i > 0 and np.isnan(pl[i-1]) and not np.isnan(pl[i]):
            # Support box changed
            sTop[i] = sBot[i-1]
            sBot[i] = pl[i]
        else:
            sTop[i] = sTop[i-1]
            sBot[i] = sBot[i-1]
            
        if i > 0 and np.isnan(ph[i-1]) and not np.isnan(ph[i]):
            # Resistance box changed
            rTop[i] = rBot[i-1]
            rBot[i] = ph[i]
        else:
            rTop[i] = rTop[i-1]
            rBot[i] = rBot[i-1]
        
        # Breakout conditions
        if not np.isnan(sBot[i]) and c[i] < sBot[i] and c[i-1] >= sBot[i]:
            sBreak[i] = True
            if input_breakout:
                sBreak[i] = True
        
        if not np.isnan(rTop[i]) and c[i] > rTop[i] and c[i-1] <= rTop[i]:
            rBreak[i] = True
            if input_breakout:
                rBreak[i] = True
        
        # Retest conditions
        sRetValid = False
        rRetValid = False
        
        if sBreak[i]:
            # Long retest: price comes back to support
            if h[i] >= sTop[i] and c[i] <= sBot[i]:
                sRetValid = True
            elif h[i] >= sTop[i] and c[i] >= sBot[i] and c[i] <= sTop[i]:
                sRetValid = True
            elif h[i] >= sBot[i] and h[i] <= sTop[i]:
                sRetValid = True
            elif h[i] >= sBot[i] and h[i] <= sTop[i] and c[i] < sBot[i]:
                sRetValid = True
        
        if rBreak[i]:
            # Short retest: price comes back to resistance
            if l[i] <= rBot[i] and c[i] >= rTop[i]:
                rRetValid = True
            elif l[i] <= rBot[i] and c[i] <= rTop[i] and c[i] >= rBot[i]:
                rRetValid = True
            elif l[i] <= rTop[i] and l[i] >= rBot[i]:
                rRetValid = True
            elif l[i] <= rTop[i] and l[i] >= rBot[i] and c[i] > rTop[i]:
                rRetValid = True
        
        # Generate entries
        if input_retest:
            if sRetValid and (tradeDirection == "Long" or tradeDirection == "Both"):
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': df['time'].iloc[i],
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': c[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': c[i],
                    'raw_price_b': c[i]
                })
                trade_num += 1
            
            if rRetValid and (tradeDirection == "Short" or tradeDirection == "Both"):
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': df['time'].iloc[i],
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': c[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': c[i],
                    'raw_price_b': c[i]
                })
                trade_num += 1
    
    return entries