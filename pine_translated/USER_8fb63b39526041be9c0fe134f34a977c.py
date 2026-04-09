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
    
    # Calculate EMAs
    shortEma = close.ewm(span=8, adjust=False).mean()
    mediumEma = close.ewm(span=20, adjust=False).mean()
    longEma = close.ewm(span=50, adjust=False).mean()
    
    # Market phase detection
    uptrend = (shortEma > mediumEma) & (mediumEma > longEma)
    downtrend = (shortEma < mediumEma) & (mediumEma < longEma)
    
    # Detect crossovers and crossunders
    short_cross_above_medium = (shortEma > mediumEma) & (shortEma.shift(1) <= mediumEma.shift(1))
    short_cross_below_medium = (shortEma < mediumEma) & (shortEma.shift(1) >= mediumEma.shift(1))
    
    # Entry conditions
    long_entry = short_cross_above_medium & uptrend
    short_entry = short_cross_below_medium & downtrend
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip bars where EMAs are NaN
        if pd.isna(shortEma.iloc[i]) or pd.isna(mediumEma.iloc[i]) or pd.isna(longEma.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        entry_price = float(close.iloc[i])
        
        if long_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries