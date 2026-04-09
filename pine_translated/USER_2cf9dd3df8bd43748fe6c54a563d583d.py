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
    entries = []
    trade_num = 0
    
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    n = len(df)
    if n < 3:
        return entries
    
    # Parameters
    prd = 10  # Structure Period
    bull = True
    bear = True
    follow = True
    levels = [0.50, 0.618]
    
    # Variables
    pos = 0  # 0=flat, 1+=long, -1-=short
    
    Up = np.zeros(n)
    Dn = np.zeros(n)
    iUp = np.zeros(n, dtype=int)
    iDn = np.zeros(n, dtype=int)
    
    # Initialize with first values
    Up[0] = high.iloc[0]
    Dn[0] = low.iloc[0]
    
    # Calculate pivots
    for i in range(prd, n):
        # Pivot High: high[i] is highest from i-prd to i
        window_high = high.iloc[i-prd:i+1]
        if high.iloc[i] == window_high.max():
            Up[i] = high.iloc[i]
            iUp[i] = i
    
        # Pivot Low: low[i] is lowest from i-prd to i
        window_low = low.iloc[i-prd:i+1]
        if low.iloc[i] == window_low.min():
            Dn[i] = low.iloc[i]
            iDn[i] = i
    
    # Forward fill Up and Dn
    for i in range(1, n):
        Up[i] = high.iloc[i] if Up[i] != 0 else Up[i-1]
        if Up[i] == 0:
            Up[i] = Up[i-1]
        if iUp[i] == 0:
            iUp[i] = iUp[i-1]
    
    for i in range(1, n):
        if Dn[i] == 0:
            Dn[i] = Dn[i-1]
        else:
            Dn[i] = low.iloc[i]
        if iDn[i] == 0:
            iDn[i] = iDn[i-1]
    
    # Track entries
    for i in range(2, n):
        # Entry conditions from Pine Script:
        # if Up > Up[1] and pos <= 0: LONG entry
        # if Dn < Dn[1] and pos >= 0: SHORT entry
        
        up_curr = Up[i]
        up_prev = Up[i-1]
        dn_curr = Dn[i]
        dn_prev = Dn[i-1]
        
        direction = None
        
        if up_curr > up_prev and pos <= 0:
            direction = 'long'
        elif dn_curr < dn_prev and pos >= 0:
            direction = 'short'
        
        if direction:
            trade_num += 1
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
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
            
            # Update position state
            if direction == 'long':
                pos = 1
            else:
                pos = -1
    
    return entries