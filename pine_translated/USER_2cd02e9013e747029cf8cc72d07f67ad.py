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
    trade_num = 1
    
    length = 5
    p = int(length / 2)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    dh = pd.Series(np.sign(high - high.shift(1)).rolling(p).sum())
    dl = pd.Series(np.sign(low - low.shift(1)).rolling(p).sum())
    
    bullf = (dh == -p) & (dh.shift(p) == p) & (high.shift(p) == high.rolling(length).max())
    bearf = (dl == p) & (dl.shift(p) == -p) & (low.shift(p) == low.rolling(length).min())
    
    bullf_count = bullf.cumsum()
    bearf_count = bearf.cumsum()
    
    upper_value = pd.Series(np.nan, index=df.index)
    upper_loc = pd.Series(0, index=df.index)
    upper_iscrossed = pd.Series(False, index=df.index)
    
    lower_value = pd.Series(np.nan, index=df.index)
    lower_loc = pd.Series(0, index=df.index)
    lower_iscrossed = pd.Series(False, index=df.index)
    
    os = 0
    
    for i in range(1, len(df)):
        n = i
        
        if bullf.iloc[i]:
            upper_value.iloc[i] = high.iloc[i-p]
            upper_loc.iloc[i] = n - p
            upper_iscrossed.iloc[i] = False
        
        if bearf.iloc[i]:
            lower_value.iloc[i] = low.iloc[i-p]
            lower_loc.iloc[i] = n - p
            lower_iscrossed.iloc[i] = False
        
        bull_cross = close.iloc[i] > upper_value.iloc[i-p] if pd.notna(upper_value.iloc[i-p]) else False
        bear_cross = close.iloc[i] < lower_value.iloc[i-p] if pd.notna(lower_value.iloc[i-p]) else False
        
        bull_cond = bull_cross and not upper_iscrossed.iloc[i-p] if pd.notna(upper_iscrossed.iloc[i-p]) else False
        bear_cond = bear_cross and not lower_iscrossed.iloc[i-p] if pd.notna(lower_iscrossed.iloc[i-p]) else False
        
        if bull_cond:
            upper_iscrossed.iloc[i] = True
            os = 1
        elif bear_cond:
            lower_iscrossed.iloc[i] = True
            os = -1
    
    for i in range(1, len(df)):
        n = i
        
        bull_cross = close.iloc[i] > upper_value.iloc[i-p] if pd.notna(upper_value.iloc[i-p]) else False
        bear_cross = close.iloc[i] < lower_value.iloc[i-p] if pd.notna(lower_value.iloc[i-p]) else False
        
        bull_cond = bull_cross and not upper_iscrossed.iloc[i-p] if pd.notna(upper_iscrossed.iloc[i-p]) else False
        bear_cond = bear_cross and not lower_iscrossed.iloc[i-p] if pd.notna(lower_iscrossed.iloc[i-p]) else False
        
        if bull_cond:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            os = 1
        elif bear_cond:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            os = -1
    
    return entries