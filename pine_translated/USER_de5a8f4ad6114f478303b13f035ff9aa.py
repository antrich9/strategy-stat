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
    
    # Strategy parameters
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = "Both"
    hmaLength = 21
    
    # Input parameters
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'
    
    rTon = input_repType == 'On'
    rTcc = input_repType == 'Off: Candle Confirmation'
    rThv = input_repType == 'Off: High & Low'
    
    # Calculate HMA
    price = df['close']
    hmaSource = price
    half_length = int(hmaLength / 2)
    sqrt_length = int(np.sqrt(hmaLength))
    
    wma1 = hmaSource.ewm(span=half_length, adjust=False).mean()
    wma2_val = hmaSource.ewm(span=hmaLength, adjust=False).mean()
    wma_diff = 2 * wma1 - wma2_val
    hma = wma_diff.ewm(span=sqrt_length, adjust=False).mean()
    
    # HMA conditions
    hmaLongCondition = df['close'] > hma
    hmaShortCondition = df['close'] < hma
    
    # Trade direction conditions
    longCondition = tradeDirection == "Long" or tradeDirection == "Both"
    shortCondition = tradeDirection == "Short" or tradeDirection == "Both"
    
    # Calculate ATR (Wilder ATR)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[atrLength - 1] = tr.iloc[:atrLength].mean()
    for i in range(atrLength, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (atrLength - 1) + tr.iloc[i]) / atrLength
    
    # Pivot calculations
    bb = input_lookback
    
    # pivotlow: lowest low over bb bars ending at bb bars back
    pl = pd.Series(index=df.index, dtype=float)
    for i in range(bb, len(df)):
        window = low.iloc[i - bb:i - bb + bb + 1]
        pl.iloc[i] = window.idxmin()
        pl.iloc[i] = low.iloc[int(pl.iloc[i])]
    
    # Alternative approach using rolling
    pl = pd.Series(index=df.index, dtype=float)
    for i in range(bb, len(df)):
        pl.iloc[i] = low.iloc[i - bb]
        for j in range(1, bb + 1):
            if low.iloc[i - bb + j] <= low.iloc[i - bb]:
                pl.iloc[i] = low.iloc[i - bb + j]
    
    # Actually, let's use a proper pivotlow implementation
    pl = pd.Series(np.nan, index=df.index)
    for i in range(bb, len(df) - bb):
        window = low.iloc[i - bb:i + bb + 1]
        min_idx = window.idxmin()
        if min_idx == i:
            pl.iloc[i] = low.iloc[i]
    
    # pivothigh: highest high over bb bars ending at bb bars back
    ph = pd.Series(np.nan, index=df.index)
    for i in range(bb, len(df) - bb):
        window = high.iloc[i - bb:i + bb + 1]
        max_idx = window.idxmax()
        if max_idx == i:
            ph.iloc[i] = high.iloc[i]
    
    # Box heights
    s_yLoc = pd.Series(index=df.index, dtype=float)
    for i in range(bb + 1, len(df)):
        if low.iloc[i - bb + 1] > low.iloc[i - bb - 1]:
            s_yLoc.iloc[i] = low.iloc[i - bb - 1]
        else:
            s_yLoc.iloc[i] = low.iloc[i - bb + 1]
    
    r_yLoc = pd.Series(index=df.index, dtype=float)
    for i in range(bb + 1, len(df)):
        if high.iloc[i - bb + 1] > high.iloc[i - bb - 1]:
            r_yLoc.iloc[i] = high.iloc[i - bb + 1]
        else:
            r_yLoc.iloc[i] = high.iloc[i - bb - 1]
    
    # sBox and rBox values
    sBot = pl.copy()
    sTop = pl.copy()
    rBot = ph.copy()
    rTop = ph.copy()
    
    # Update box values where applicable
    for i in range(bb, len(df)):
        if not np.isnan(pl.iloc[i]):
            sBot.iloc[i] = pl.iloc[i]
            sTop.iloc[i] = pl.iloc[i]
        if not np.isnan(ph.iloc[i]):
            rBot.iloc[i] = ph.iloc[i]
            rTop.iloc[i] = ph.iloc[i]
    
    # Fill forward the box boundaries
    sBot = sBot.ffill()
    sTop = sTop.ffill()
    rBot = rBot.ffill()
    rTop = rTop.ffill()
    
    # Breakout detection - need to track state across bars
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    # Calculate crossunder and crossover for boxes
    cu = pd.Series(False, index=df.index)
    co = pd.Series(False, index=df.index)
    
    # For repainting logic
    for i in range(1, len(df)):
        # Support break (cross under sBot)
        if rTon:
            cu.iloc[i] = close.iloc[i] < sBot.iloc[i] and close.iloc[i-1] >= sBot.iloc[i-1]
        elif rThv:
            cu.iloc[i] = low.iloc[i] < sBot.iloc[i] and low.iloc[i-1] >= sBot.iloc[i-1]
        else:  # rTcc
            cu.iloc[i] = close.iloc[i] < sBot.iloc[i] and close.iloc[i-1] >= sBot.iloc[i-1]  # Simplified for candle confirmation
        
        # Resistance break (crossover rTop)
        if rTon:
            co.iloc[i] = close.iloc[i] > rTop.iloc[i] and close.iloc[i-1] <= rTop.iloc[i-1]
        elif rThv:
            co.iloc[i] = high.iloc[i] > rTop.iloc[i] and high.iloc[i-1] <= rTop.iloc[i-1]
        else:
            co.iloc[i] = close.iloc[i] > rTop.iloc[i] and close.iloc[i-1] <= rTop.iloc[i-1]
    
    # Track breakout state
    for i in range(1, len(df)):
        if cu.iloc[i] and not sBreak.iloc[i-1]:
            sBreak.iloc[i] = True
        elif pd.notna(pl.iloc[i]) and np.isnan(pl.iloc[i-1]) and not sBreak.iloc[i-1]:
            sBreak.iloc[i] = False
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1]
        
        if co.iloc[i] and not rBreak.iloc[i-1]:
            rBreak.iloc[i] = True
        elif pd.notna(ph.iloc[i]) and np.isnan(ph.iloc[i-1]) and not rBreak.iloc[i-1]:
            rBreak.iloc[i] = False
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1]
    
    # Retest condition function
    def retestCondition(breakout_series, c1, c2, c3, c4):
        """Returns True where retest conditions are met"""
        retActive = c1 | c2 | c3 | c4
        retEvent = retActive & ~retActive.shift(1).fillna(False)
        
        # barssince retEvent
        bars_since = pd.Series(np.nan, index=df.index)
        count = 0
        for i in range(len(df)):
            if retEvent.iloc[i]:
                count = 0
            if not np.isnan(breakout_series.iloc[i]) and breakout_series.iloc[i]:
                count += 1
            else:
                count = 0
            bars_since.iloc[i] = count if count > 0 else np.nan
        
        # For the condition: barssince(na(breakout)) > input_retSince
        bars_since_breakout = pd.Series(np.nan, index=df.index)
        count = 0
        for i in range(len(df)):
            if pd.isna(breakout_series.iloc[i]) or not breakout_series.iloc[i]:
                count = 0
            else:
                count += 1
            bars_since_breakout.iloc[i] = count
        
        cond1 = bars_since_breakout > input_retSince
        
        return retActive & cond1
    
    # Support retest conditions (s1-s4)
    s1 = retestCondition(sBreak, high >= sTop, high >= sTop & (close >= sBot) & (close <= sTop), high >= sBot & (high <= sTop), high >= sBot & (high <= sTop) & (close < sBot))
    
    s1 = (high >= sTop) & (close <= sBot)
    s2 = (high >= sTop) & (close >= sBot) & (close <= sTop)
    s3 = (high >= sBot) & (high <= sTop)
    s4 = (high >= sBot) & (high <= sTop) & (close < sBot)
    
    sRetActive = s1 | s2 | s3 | s4
    
    # Resistance retest conditions (r1-r4)
    r1 = (low <= rBot) & (close >= rTop)
    r2 = (low <= rBot) & (close <= rTop) & (close >= rBot)
    r3 = (low <= rTop) & (low >= rBot)
    r4 = (low <= rTop) & (low >= rBot) & (close > rTop)
    
    rRetActive = r1 | r2 | r3 | r4
    
    # Calculate barssince for breakout
    sBarsSinceBreak = pd.Series(np.nan, index=df.index)
    count = 0
    for i in range(len(df)):
        if pd.isna(sBreak.iloc[i]) or not sBreak.iloc[i]:
            count = 0
        else:
            count += 1
        sBarsSinceBreak.iloc[i] = count if count > 0 else np.nan
    
    rBarsSinceBreak = pd.Series(np.nan, index=df.index)
    count = 0
    for i in range(len(df)):
        if pd.isna(rBreak.iloc[i]) or not rBreak.iloc[i]:
            count = 0
        else:
            count += 1
        rBarsSinceBreak.iloc[i] = count if count > 0 else np.nan
    
    # Apply bars since breakout condition
    sRetActive = sRetActive & (sBarsSinceBreak > input_retSince)
    rRetActive = rRetActive & (rBarsSinceBreak > input_retSince)
    
    # Retest event (rising edge)
    sRetEvent = sRetActive & ~sRetActive.shift(1).fillna(False)
    rRetEvent = rRetActive & ~rRetActive.shift(1).fillna(False)
    
    # Calculate barssince for retEvent
    sBarsSinceRetEvent = pd.Series(np.nan, index=df.index)
    count = 0
    for i in range(len(df)):
        if sRetEvent.iloc[i]:
            count = 0
        else:
            count += 1
        sBarsSinceRetEvent.iloc[i] = count
    
    rBarsSinceRetEvent = pd.Series(np.nan, index=df.index)
    count = 0
    for i in range(len(df)):
        if rRetEvent.iloc[i]:
            count = 0
        else:
            count += 1
        rBarsSinceRetEvent.iloc[i] = count
    
    # Ret valid check
    # For support (ph type): retConditions = close >= retValue
    # For resistance (pl type): retConditions = close <= retValue
    
    # Get retValue (y1 at retEvent bars)
    sRetValue = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if sRetEvent.iloc[i]:
            idx = i - sBarsSinceRetEvent.iloc[i]
            if idx >= 0 and idx < len(high):
                sRetValue.iloc[i] = high.iloc[idx]
    
    rRetValue = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if rRetEvent.iloc[i]:
            idx = i - rBarsSinceRetEvent.iloc[i]
            if idx >= 0 and idx < len(low):
                rRetValue.iloc[i] = low.iloc[idx]
    
    # Forward fill retValue
    sRetValue = sRetValue.ffill()
    rRetValue = rRetValue.ffill()
    
    # Ret conditions
    sRetConditions = close >= sRetValue
    rRetConditions = close <= rRetValue
    
    # Ret valid: barssince(retEvent) > 0 and barssince(retEvent) <= input_retValid and retConditions
    sRetValid = (sBarsSinceRetEvent > 0) & (sBarsSinceRetEvent <= input_retValid) & sRetConditions & sRetEvent
    rRetValid = (rBarsSinceRetEvent > 0) & (rBarsSinceRetEvent <= input_retValid) & rRetConditions & rRetEvent
    
    # For entries, we need actual retValid signals
    # A retValid occurs when we have a valid retest
    sRetValidSignal = pd.Series(False, index=df.index)
    rRetValidSignal = pd.Series(False, index=df.index)
    
    # Track retOccurred state
    sRetOccurred = False
    rRetOccurred = False
    
    for i in range(1, len(df)):
        # Support retest validation
        if sRetEvent.iloc[i]:
            sRetOccurred = False
        
        bars_since = sBarsSinceRetEvent.iloc[i]
        if not np.isnan(bars_since) and bars_since > 0 and bars_since <= input_retValid:
            if sRetConditions.iloc[i] and not sRetOccurred:
                sRetValidSignal.iloc[i] = True
                sRetOccurred = True
        
        if (not np.isnan(bars_since) and bars_since > input_retValid) or sRetEvent.iloc[i]:
            if sRetOccurred:
                sRetOccurred = False
        
        # Resistance retest validation
        if rRetEvent.iloc[i]:
            rRetOccurred = False
        
        bars_since = rBarsSinceRetEvent.iloc[i]
        if not np.isnan(bars_since) and bars_since > 0 and bars_since <= input_retValid:
            if rRetConditions.iloc[i] and not rRetOccurred:
                rRetValidSignal.iloc[i] = True
                rRetOccurred = True
        
        if (not np.isnan(bars_since) and bars_since > input_retValid) or rRetEvent.iloc[i]:
            if rRetOccurred:
                rRetOccurred = False
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Long entries: longCondition AND hmaLongCondition AND sRetValidSignal
    if longCondition:
        for i in range(len(df)):
            if hmaLongCondition.iloc[i] and sRetValidSignal.iloc[i]:
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                }
                entries.append(entry)
                trade_num += 1
    
    # Short entries: shortCondition AND hmaShortCondition AND rRetValidSignal
    if shortCondition:
        for i in range(len(df)):
            if hmaShortCondition.iloc[i] and rRetValidSignal.iloc[i]:
                entry = {
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                }
                entries.append(entry)
                trade_num += 1
    
    return entries