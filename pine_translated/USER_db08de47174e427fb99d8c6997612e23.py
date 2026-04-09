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
    leftBars = 15
    rightBars = 5
    
    entries = []
    trade_num = 0
    
    n = len(df)
    if n <= leftBars + rightBars + 1:
        return entries
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    pivotHigh = np.full(n, np.nan)
    pivotLow = np.full(n, np.nan)
    
    for i in range(leftBars + rightBars, n):
        isHigh = True
        for j in range(1, leftBars + 1):
            if high[i - j] >= high[i]:
                isHigh = False
                break
        if isHigh:
            for j in range(1, rightBars + 1):
                if high[i + j] > high[i]:
                    isHigh = False
                    break
        if isHigh:
            pivotHigh[i - rightBars] = high[i]
        
        isLow = True
        for j in range(1, leftBars + 1):
            if low[i - j] <= low[i]:
                isLow = False
                break
        if isLow:
            for j in range(1, rightBars + 1):
                if low[i + j] < low[i]:
                    isLow = False
                    break
        if isLow:
            pivotLow[i - rightBars] = low[i]
    
    for i in range(leftBars + rightBars + 1, n):
        ts = int(df['time'].iloc[i])
        current_close = close[i]
        
        bullish_break = not np.isnan(pivotHigh[i - 1]) and current_close > pivotHigh[i - 1]
        bearish_break = not np.isnan(pivotLow[i - 1]) and current_close < pivotLow[i - 1]
        
        if bullish_break:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': current_close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': current_close,
                'raw_price_b': current_close
            })
        
        if bearish_break:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': current_close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': current_close,
                'raw_price_b': current_close
            })
    
    return entries