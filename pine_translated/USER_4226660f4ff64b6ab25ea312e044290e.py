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
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    n = len(df)
    prd = 50
    
    ph = np.zeros(n)
    pl = np.zeros(n)
    phL = np.zeros(n)
    plL = np.zeros(n)
    dir_arr = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            continue
        
        hb = pd.Series(high).rolling(window=prd, min_periods=prd).apply(lambda x: x.argmax(), raw=True).iloc[i]
        lb = pd.Series(low).rolling(window=prd, min_periods=prd).apply(lambda x: x.argmin(), raw=True).iloc[i]
        
        if i - hb == 0:
            ph[i] = high.iloc[i]
            phL[i] = i
        else:
            ph[i] = ph[i-1]
            phL[i] = phL[i-1]
        
        if i - lb == 0:
            pl[i] = low.iloc[i]
            plL[i] = i
        else:
            pl[i] = pl[i-1]
            plL[i] = plL[i-1]
        
        dir_arr[i] = 1 if phL[i] > plL[i] else -1
    
    return entries