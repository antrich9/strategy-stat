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
    
    # Strategy parameters (matching Pine Script defaults)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    lengthKPO = 14
    smoothKPO = 3
    
    # Input parameters
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    tradeDirection = "Both"
    
    bb = input_lookback
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Kase Peak Oscillator
    avgPrice = close.rolling(lengthKPO).mean()
    kpo = (close - avgPrice).ewm(span=smoothKPO, adjust=False).mean()
    
    # Wilder ATR calculation
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, min_periods=atrLength).mean()
    
    # Helper functions for pivot point detection
    def calc_pivot_low(src, length):
        result = pd.Series(np.nan, index=src.index)
        for i in range(length, len(src) - length):
            left = src.iloc[i - length:i]
            right = src.iloc[i + 1:i + length + 1]
            if src.iloc[i] == left.min() and src.iloc[i] == right.min():
                result.iloc[i] = src.iloc[i]
        return result
    
    def calc_pivot_high(src, length):
        result = pd.Series(np.nan, index=src.index)
        for i in range(length, len(src) - length):
            left = src.iloc[i - length:i]
            right = src.iloc[i + 1:i + length + 1]
            if src.iloc[i] == left.max() and src.iloc[i] == right.max():
                result.iloc[i] = src.iloc[i]
        return result
    
    pl = calc_pivot_low(low, bb)
    ph = calc_pivot_high(high, bb)
    
    # fixnan - forward fill NaN values
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Box height calculation
    s_yLoc = pd.Series(np.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1)), index=low.index)
    r_yLoc = pd.Series(np.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1)), index=high.index)
    
    # Breakout conditions (crossover/crossunder with repainting)
    rTon = True  # Repainting On
    rTcc = False
    rThv = False
    
    def repaint_crossover(series1, series2):
        if rTon:
            return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
        elif rThv:
            return (high > series2) & (high.shift(1) <= series2.shift(1))
        else:
            return (series1 > series2) & (series1.shift(1) <= series2.shift(1)) & (df['time'] != df['time'].shift(1))
    
    def repaint_crossunder(series1, series2):
        if rTon:
            return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
        elif rThv:
            return (low < series2) & (low.shift(1) >= series2.shift(1))
        else:
            return (series1 < series2) & (series1.shift(1) >= series2.shift(1)) & (df['time'] != df['time'].shift(1))
    
    # Initialize state variables
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    sBox_top = pd.Series(np.nan, index=df.index)
    sBox_bottom = pd.Series(np.nan, index=df.index)
    rBox_top = pd.Series(np.nan, index=df.index)
    rBox_bottom = pd.Series(np.nan, index=df.index)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    sRetEvent = pd.Series(False, index=df.index)
    rRetEvent = pd.Series(False, index=df.index)
    
    # Track retest event values
    sRetValue = pd.Series(np.nan, index=df.index)
    rRetValue = pd.Series(np.nan, index=df.index)
    
    retOccurred_s = False
    retOccurred_r = False
    
    # Main loop to detect breakouts and retests
    for i in range(bb + 1, len(df)):
        # Update support box when pivot low changes
        if pd.notna(pl.iloc[i]) and (i == 0 or pl.iloc[i] != pl.iloc[i-1]):
            sBox_bottom.iloc[i] = pl.iloc[i]
            sBox_top.iloc[i] = s_yLoc.iloc[i]
        elif i > 0:
            sBox_bottom.iloc[i] = sBox_bottom.iloc[i-1]
            sBox_top.iloc[i] = sBox_top.iloc[i-1]
        
        # Update resistance box when pivot high changes
        if pd.notna(ph.iloc[i]) and (i == 0 or ph.iloc[i] != ph.iloc[i-1]):
            rBox_top.iloc[i] = ph.iloc[i]
            rBox_bottom.iloc[i] = r_yLoc.iloc[i]
        elif i > 0:
            rBox_top.iloc[i] = rBox_top.iloc[i-1]
            rBox_bottom.iloc[i] = rBox_bottom.iloc[i-1]
        
        # Handle box deletion on pivot change
        if i > 0:
            if pd.notna(pl.iloc[i]) and pl.iloc[i] != pl.iloc[i-1]:
                if not sBreak.iloc[i-1] if i > 0 else True:
                    sBox_bottom.iloc[i] = np.nan
                    sBox_top.iloc[i] = np.nan
                sBreak.iloc[i] = False
            
            if pd.notna(ph.iloc[i]) and ph.iloc[i] != ph.iloc[i-1]:
                if not rBreak.iloc[i-1] if i > 0 else True:
                    rBox_top.iloc[i] = np.nan
                    rBox_bottom.iloc[i] = np.nan
                rBreak.iloc[i] = False
        
        # Detect breakouts
        sBot = sBox_bottom.iloc[i] if pd.notna(sBox_bottom.iloc[i]) else low.iloc[i]
        sTop = sBox_top.iloc[i] if pd.notna(sBox_top.iloc[i]) else low.iloc[i]
        rTop = rBox_top.iloc[i] if pd.notna(rBox_top.iloc[i]) else high.iloc[i]
        rBot = rBox_bottom.iloc[i] if pd.notna(rBox_bottom.iloc[i]) else high.iloc[i]
        
        if pd.notna(sBox_bottom.iloc[i]):
            cu = repaint_crossunder(close, pd.Series([sBot] * len(df), index=df.index))
            if cu.iloc[i] and not sBreak.iloc[i-1] if i > 0 else True:
                sBreak.iloc[i] = True
        
        if pd.notna(rBox_top.iloc[i]):
            co = repaint_crossover(close, pd.Series([rTop] * len(df), index=df.index))
            if co.iloc[i] and not rBreak.iloc[i-1] if i > 0 else True:
                rBreak.iloc[i] = True
        
        # Calculate bars since breakout
        bars_since_sBreak = np.where(sBreak.iloc[:i+1].iloc[::-1].cumsum().iloc[::-1] > 0,
                                     (sBreak.iloc[:i+1] == False).cumsum().iloc[::-1].cumsum().iloc[::-1].iloc[:i+1].iloc[i], 999)
        bars_since_rBreak = np.where(rBreak.iloc[:i+1].iloc[::-1].cumsum().iloc[::-1] > 0,
                                     (rBreak.iloc[:i+1] == False).cumsum().iloc[::-1].cumsum().iloc[::-1].iloc[:i+1].iloc[i], 999)
        
        # Retest conditions for support
        if pd.notna(sBox_bottom.iloc[i]) and pd.notna(sBox_top.iloc[i]):
            s1 = bars_since_sBreak > input_retSince and high.iloc[i] >= sTop and close.iloc[i] <= sBot
            s2 = bars_since_sBreak > input_retSince and high.iloc[i] >= sTop and close.iloc[i] >= sBot and close.iloc[i] <= sTop
            s3 = bars_since_sBreak > input_retSince and high.iloc[i] >= sBot and high.iloc[i] <= sTop
            s4 = bars_since_sBreak > input_retSince and high.iloc[i] >= sBot and high.iloc[i] <= sTop and close.iloc[i] < sBot
            sActive = s1 or s2 or s3 or s4
        else:
            sActive = False
        
        # Retest conditions for resistance
        if pd.notna(rBox_bottom.iloc[i]) and pd.notna(rBox_top.iloc[i]):
            r1 = bars_since_rBreak > input_retSince and low.iloc[i] <= rBot and close.iloc[i] >= rTop
            r2 = bars_since_rBreak > input_retSince and low.iloc[i] <= rBot and close.iloc[i] <= rTop and close.iloc[i] >= rBot
            r3 = bars_since_rBreak > input_retSince and low.iloc[i] <= rTop and low.iloc[i] >= rBot
            r4 = bars_since_rBreak > input_retSince and low.iloc[i] <= rTop and low.iloc[i] >= rBot and close.iloc[i] > rTop
            rActive = r1 or r2 or r3 or r4
        else:
            rActive = False
        
        # Retest event detection
        if i > 0:
            if sActive and not sActive:
                sRetEvent.iloc[i] = True
                sRetValue.iloc[i] = high.iloc[i]
                retOccurred_s = False
            elif sActive:
                sRetEvent.iloc[i] = sRetEvent.iloc[i-1]
                sRetValue.iloc[i] = sRetValue.iloc[i-1]
            
            if rActive and not rActive:
                rRetEvent.iloc[i] = True
                rRetValue.iloc[i] = low.iloc[i]
                retOccurred_r = False
            elif rActive:
                rRetEvent.iloc[i] = rRetEvent.iloc[i-1]
                rRetValue.iloc[i] = rRetValue.iloc[i-1]
        
        # Calculate retest validity
        bars_since_sRetEvent = 0
        bars_since_rRetEvent = 0
        
        if sRetEvent.iloc[i]:
            bars_since_sRetEvent = min(i, 1000)
            for j in range(i-1, -1, -1):
                if not sRetEvent.iloc[j]:
                    bars_since_sRetEvent = i - j
                    break
        
        if rRetEvent.iloc[i]:
            bars_since_rRetEvent = min(i, 1000)
            for j in range(i-1, -1, -1):
                if not rRetEvent.iloc[j]:
                    bars_since_rRetEvent = i - j
                    break
        
        # Check retValid conditions
        if bars_since_sRetEvent > 0 and bars_since_sRetEvent <= input_retValid and pd.notna(sRetValue.iloc[i]):
            retConditions = close.iloc[i] <= sRetValue.iloc[i] if rTon else (high.iloc[i] >= sRetValue.iloc[i] if rThv else close.iloc[i] <= sRetValue.iloc[i])
            if retConditions and not retOccurred_s:
                sRetValid.iloc[i] = True
                retOccurred_s = True
        
        if bars_since_rRetEvent > 0 and bars_since_rRetEvent <= input_retValid and pd.notna(rRetValue.iloc[i]):
            retConditions = close.iloc[i] >= rRetValue.iloc[i] if rTon else (low.iloc[i] <= rRetValue.iloc[i] if rThv else close.iloc[i] >= rRetValue.iloc[i])
            if retConditions and not retOccurred_r:
                rRetValid.iloc[i] = True
                retOccurred_r = True
    
    # Generate entries based on valid retests
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        entry = None
        
        if tradeDirection in ["Long", "Both"] and sRetValid.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            }
            trade_num += 1
        
        elif tradeDirection in ["Short", "Both"] and rRetValid.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            }
            trade_num += 1
        
        if entry:
            entries.append(entry)
    
    return entries