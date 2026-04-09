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
    open_col = df['open']
    
    n = len(df)
    if n == 0:
        return []
    
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    jmaJMA = np.zeros(n)
    e0JMA = np.zeros(n)
    e1JMA = np.zeros(n)
    e2JMA = np.zeros(n)
    
    for i in range(1, n):
        srcjmaJMA = close.iloc[i]
        e0JMA[i] = (1 - alphajmaJMA) * srcjmaJMA + alphajmaJMA * e0JMA[i-1]
        e1JMA[i] = (srcjmaJMA - e0JMA[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA[i-1]
        e2JMA[i] = (e0JMA[i] + phasejmaJMARatiojmaJMA * e1JMA[i] - jmaJMA[i-1]) * ((1 - alphajmaJMA) ** 2) + (alphajmaJMA ** 2) * e2JMA[i-1]
        jmaJMA[i] = e2JMA[i] + jmaJMA[i-1]
    
    jmaJMA_series = pd.Series(jmaJMA, index=df.index)
    
    jma_prev = jmaJMA_series.shift(1)
    
    if usejmaJMA:
        if usecolorjmaJMA:
            signalmaJMALong = (jmaJMA_series > jma_prev) & (close > jmaJMA_series)
            signalmaJMAShort = (jmaJMA_series < jma_prev) & (close < jmaJMA_series)
        else:
            signalmaJMALong = close > jmaJMA_series
            signalmaJMAShort = close < jmaJMA_series
    else:
        signalmaJMALong = pd.Series(True, index=df.index)
        signalmaJMAShort = pd.Series(True, index=df.index)
    
    if inverseJMA:
        finalLongSignalJMA = signalmaJMAShort
        finalShortSignalJMA = signalmaJMALong
    else:
        finalLongSignalJMA = signalmaJMALong
        finalShortSignalJMA = signalmaJMAShort
    
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        ts = int(df['time'].iloc[i])
        price = close.iloc[i]
        
        if finalLongSignalJMA.iloc[i]:
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(price),
                'raw_price_b': float(price)
            })
            trade_num += 1
        
        if finalShortSignalJMA.iloc[i]:
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(price),
                'raw_price_b': float(price)
            })
            trade_num += 1
    
    return entries