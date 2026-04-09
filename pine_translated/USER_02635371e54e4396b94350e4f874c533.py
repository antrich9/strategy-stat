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
    
    prd = 2
    max_array_size = 50
    
    zigzag = []
    direction = 0
    
    fib_50 = pd.Series(np.nan, index=df.index)
    
    for i in range(prd, len(df)):
        newbar = True
        
        len_val = i - prd + 1
        
        ph = df['high'].iloc[i] if df['high'].iloc[i] == df['high'].iloc[i-len_val+1:i+1].max() else np.nan
        pl = df['low'].iloc[i] if df['low'].iloc[i] == df['low'].iloc[i-len_val+1:i+1].min() else np.nan
        
        if pd.notna(pl) and pd.isna(ph):
            new_dir = -1
        elif pd.notna(ph) and pd.isna(pl):
            new_dir = 1
        else:
            new_dir = direction
        
        dirchanged = new_dir != direction
        direction = new_dir
        
        if pd.notna(ph) or pd.notna(pl):
            if dirchanged:
                zigzag.insert(0, ph if direction == 1 else pl)
                zigzag.insert(0, i)
            else:
                zigzag[0] = ph if direction == 1 else pl
        
        if len(zigzag) >= 6:
            fib_0 = zigzag[2]
            fib_1 = zigzag[4]
            fib_50.iloc[i] = fib_0 + (fib_1 - fib_0) * 0.5
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(df)):
        if pd.isna(fib_50.iloc[i]) or pd.isna(fib_50.iloc[i-1]):
            continue
        
        cond_long = fib_50.iloc[i-1] >= df['close'].iloc[i-1] and fib_50.iloc[i] < df['close'].iloc[i]
        cond_short = fib_50.iloc[i-1] <= df['close'].iloc[i-1] and fib_50.iloc[i] > df['close'].iloc[i]
        
        if cond_long:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        elif cond_short:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries