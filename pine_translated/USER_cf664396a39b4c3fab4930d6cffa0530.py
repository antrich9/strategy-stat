import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    atrLength = 14
    atrMultiplier = 1.5
    baselinePeriod = 50
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    
    # Calculate baseline SMA
    baseline = df['close'].rolling(baselinePeriod).mean()
    
    # Calculate pivots
    bb = input_lookback
    
    # For pivotlow: find the lowest low over bb bars ending at current bar
    # pivotlow(source, leftbars, rightbars) returns the value of the pivot low point
    # We need to implement this
    
    # Calculate ATR using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Stop loss and take profit levels
    stopLossLong = close - (atr * atrMultiplier)
    stopLossShort = close + (atr * atrMultiplier)
    takeProfitLong = close + ((atr * atrMultiplier) * 1.5)
    takeProfitShort = close - ((atr * atrMultiplier) * 1.5)
    
    # Pivot calculations
    # pivotlow(low, bb, bb) means looking back bb bars and forward bb bars
    # For a pivot low at index i, we need i-bb to i+bb to be considered
    # Since we can't look forward, we'll use a simpler approach
    
    # For pivot low: 
    pl = pd.Series(index=df.index, dtype=float)
    for i in range(bb, len(df) - bb):
        window = low.iloc[i-bb:i+bb+1]
        if low.iloc[i] == window.min():
            pl.iloc[i] = low.iloc[i]
    
    # Similar for pivot high
    ph = pd.Series(index=df.index, dtype=float)
    for i in range(bb, len(df) - bb):
        window = high.iloc[i-bb:i+bb+1]
        if high.iloc[i] == window.max():
            ph.iloc[i] = high.iloc[i]
    
    # Forward fill
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Calculate support and resistance boxes
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    sTop = pd.Series(np.nan, index=df.index)
    sBot = pd.Series(np.nan, index=df.index)
    rTop = pd.Series(np.nan, index=df.index)
    rBot = pd.Series(np.nan, index=df.index)
    
    for i in range(len(df)):
        if not pd.isna(pl.iloc[i]):
            # Support box
            s_yLoc = low.iloc[bb + 1] if low.iloc[bb + 1] > low.iloc[bb - 1] else low.iloc[bb - 1]
            sTop.iloc[i] = pl.iloc[i]
            sBot.iloc[i] = s_yLoc
        
        if not pd.isna(ph.iloc[i]):
            # Resistance box
            r_yLoc = high.iloc[bb + 1] if high.iloc[bb + 1] > high.iloc[bb - 1] else high.iloc[bb - 1]
            rTop.iloc[i] = r_yLoc
            rBot.iloc[i] = ph.iloc[i]
    
    # Detect breakouts
    # cu: support breakout (price crosses below support box bottom)
    # co: resistance breakout (price crosses above resistance box top)
    cu = pd.Series(False, index=df.index)
    co = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if sBot.iloc[i-1] > 0 and sBot.iloc[i] > 0:
            if close.iloc[i-1] >= sBot.iloc[i-1] and close.iloc[i] < sBot.iloc[i]:
                cu.iloc[i] = True
        
        if rTop.iloc[i-1] > 0 and rTop.iloc[i] > 0:
            if close.iloc[i-1] <= rTop.iloc[i-1] and close.iloc[i] > rTop.iloc[i]:
                co.iloc[i] = True
    
    # Update break flags
    for i in range(len(df)):
        if cu.iloc[i] and not sBreak.iloc[i-1]:
            sBreak.iloc[i] = True
        if co.iloc[i] and not rBreak.iloc[i-1]:
            rBreak.iloc[i] = True
    
    # Retest conditions
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    # Calculate retests
    for i in range(len(df)):
        if sBreak.iloc[i]:
            # Support retest
            for j in range(i+1, min(i+input_retValid+1, len(df))):
                if low.iloc[j] <= sBot.iloc[i] and close.iloc[j] >= sTop.iloc[i]:
                    sRetValid.iloc[j] = True
                    break
        
        if rBreak.iloc[i]:
            # Resistance retest
            for j in range(i+1, min(i+input_retValid+1, len(df))):
                if high.iloc[j] >= rTop.iloc[i] and close.iloc[j] <= rBot.iloc[i]:
                    rRetValid.iloc[j] = True
                    break
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if sRetValid.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['timestamp'].iloc[i],
                'entry_price': close.iloc[i],
                'stop_loss': stopLossLong.iloc[i],
                'take_profit': takeProfitLong.iloc[i],
                'exit_ts': 0,
                'exit_price': 0.0
            })
            trade_num += 1
        
        if rRetValid.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['timestamp'].iloc[i],
                'entry_price': close.iloc[i],
                'stop_loss': stopLossShort.iloc[i],
                'take_profit': takeProfitShort.iloc[i],
                'exit_ts': 0,
                'exit_price': 0.0
            })
            trade_num += 1
    
    return entries