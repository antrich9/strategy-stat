import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    Returns list of dicts with entry signals.
    """
    entries = []
    trade_num = 0
    
    n = len(df)
    if n < 10:
        return entries
    
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    
    is_0935_bar = pd.Series(False, index=df.index)
    for idx in range(1, n):
        ts = df['time'].iloc[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.hour == 9 and dt.minute == 35:
            is_0935_bar.iloc[idx] = True
    
    high0930 = pd.Series(np.nan, index=df.index)
    low0930 = pd.Series(np.nan, index=df.index)
    for idx in range(1, n):
        if is_0935_bar.iloc[idx]:
            high0930.iloc[idx] = high.iloc[idx - 1]
            low0930.iloc[idx] = low.iloc[idx - 1]
    
    sweptHigh = False
    sweptLow = False
    foundFVG = False
    tradeToday = False
    firstSweepFVGTaken = False
    
    bullFVG = pd.Series(False, index=df.index)
    bearFVG = pd.Series(False, index=df.index)
    for idx in range(2, n):
        bullFVG.iloc[idx] = (low.iloc[idx] > high.iloc[idx - 2]) and (low.iloc[idx - 1] > high.iloc[idx - 2])
        bearFVG.iloc[idx] = (high.iloc[idx] < low.iloc[idx - 2]) and (high.iloc[idx - 1] < low.iloc[idx - 2])
    
    for idx in range(1, n):
        ts = df['time'].iloc[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.hour == 9 and dt.minute == 35:
            sweptHigh = False
            sweptLow = False
            foundFVG = False
            tradeToday = False
            firstSweepFVGTaken = False
        
        if not pd.isna(high0930.iloc[idx]) and not sweptHigh and high.iloc[idx] > high0930.iloc[idx]:
            sweptHigh = True
        
        if not pd.isna(low0930.iloc[idx]) and not sweptLow and low.iloc[idx] < low0930.iloc[idx]:
            sweptLow = True
        
        if idx >= 2:
            showBullFVG = (sweptHigh or sweptLow) and bullFVG.iloc[idx] and not foundFVG and not firstSweepFVGTaken
            showBearFVG = (sweptHigh or sweptLow) and bearFVG.iloc[idx] and not foundFVG and not firstSweepFVGTaken
            
            if showBullFVG or showBearFVG:
                trade_num += 1
                direction = 'long' if showBullFVG else 'short'
                entry_ts = int(df['time'].iloc[idx])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(close.iloc[idx])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                
                foundFVG = True
                firstSweepFVGTaken = True
                tradeToday = True
    
    return entries