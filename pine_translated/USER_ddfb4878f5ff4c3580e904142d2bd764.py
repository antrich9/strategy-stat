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
    length = 5
    useCloseCandle = False
    tradetype = "Long and Short"
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    h = high.rolling(length * 2 + 1).max()
    l = low.rolling(length * 2 + 1).min()
    
    dirUp = False
    lastLow = high.iloc[0] * 100
    lastHigh = 0.0
    
    entries = []
    trade_num = 1
    
    for i in range(length * 2 + 1, len(df)):
        length_val = length
        len_shifted = length_val
        
        isMin = l.iloc[i] == low.iloc[i - len_shifted] if i - len_shifted >= 0 else False
        isMax = h.iloc[i] == high.iloc[i - len_shifted] if i - len_shifted >= 0 else False
        
        if dirUp:
            if isMin and low.iloc[i - len_shifted] < lastLow:
                lastLow = low.iloc[i - len_shifted]
            if isMax and high.iloc[i - len_shifted] > lastLow:
                lastHigh = high.iloc[i - len_shifted]
                dirUp = False
        else:
            if isMax and high.iloc[i - len_shifted] > lastHigh:
                lastHigh = high.iloc[i - len_shifted]
            if isMin and low.iloc[i - len_shifted] < lastHigh:
                lastLow = low.iloc[i - len_shifted]
                dirUp = True
                if isMax and high.iloc[i - len_shifted] > lastLow:
                    lastHigh = high.iloc[i - len_shifted]
                    dirUp = False
        
        recentTouch = False
        for j in range(1, 11):
            if i + j < len(df) and i + j + 1 < len(df):
                if (low.iloc[i + j] <= lastLow and low.iloc[i + j + 1] > lastLow) or (high.iloc[i + j] >= lastHigh and high.iloc[i + j + 1] < lastHigh):
                    recentTouch = True
                    break
        
        price_for_long = close.iloc[i] if useCloseCandle else high.iloc[i]
        price_for_short = close.iloc[i] if useCloseCandle else low.iloc[i]
        
        high_1 = high.iloc[i - 1] if i - 1 >= 0 else np.nan
        low_1 = low.iloc[i - 1] if i - 1 >= 0 else np.nan
        lastHigh_1 = lastHigh
        lastLow_1 = lastLow
        
        longCondition = price_for_long >= lastHigh and high_1 < lastHigh_1 and not recentTouch and (tradetype == "Long and Short" or tradetype == "Long")
        shortCondition = price_for_short <= lastLow and low_1 > lastLow_1 and not recentTouch and (tradetype == "Long and Short" or tradetype == "Short")
        
        if longCondition and not shortCondition:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = close.iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
        
        if shortCondition and not longCondition:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price_guess = close.iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
    
    return entries