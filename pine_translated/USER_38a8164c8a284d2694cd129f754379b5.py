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
    time = df['time']
    
    lookback = 3  # default from Pine Script
    pivot_type = "Close"  # default from Pine Script
    
    # Compute pivot high and low based on pivot_type
    if pivot_type == "Close":
        src = close
    else:
        src = high  # for High/Low type, use high for pivot high and low for pivot low
    
    # Compute pivot high: max in window of 2*lookback+1, shifted by lookback to align with pivot point
    pivot_high = src.rolling(2*lookback+1).max().shift(lookback)
    pivot_low = src.rolling(2*lookback+1).min().shift(lookback)
    
    # Get close at the bar where pivot is detected (same shift as pivot)
    close_at_pivot = close.shift(lookback)
    
    highLevel = close_at_pivot.where(pivot_high.notna())
    lowLevel = close_at_pivot.where(pivot_low.notna())
    
    # Compute uptrend and downtrend signals
    uptrendSignal = high > highLevel
    downtrendSignal = low < lowLevel
    
    # Compute inUptrend state (similar to Pine Script's f_signal)
    inUptrend = pd.Series(False, index=close.index)
    inDowntrend = pd.Series(False, index=close.index)
    
    for i in range(1, len(close)):
        if pd.isna(uptrendSignal.iloc[i-1]) or pd.isna(downtrendSignal.iloc[i-1]):
            inUptrend.iloc[i] = inUptrend.iloc[i-1]
            inDowntrend.iloc[i] = inDowntrend.iloc[i-1]
            continue
        if uptrendSignal.iloc[i-1]:
            inUptrend.iloc[i] = True
        elif downtrendSignal.iloc[i-1]:
            inUptrend.iloc[i] = False
        else:
            inUptrend.iloc[i] = inUptrend.iloc[i-1]
        inDowntrend.iloc[i] = not inUptrend.iloc[i]
    
    # Generate entries on trend change
    entries = []
    trade_num = 1
    
    # Long entries: when inUptrend becomes true (previous was false)
    long_entry = inUptrend & ~inUptrend.shift(1).fillna(False)
    # Short entries: when inDowntrend becomes true (previous was false)
    short_entry = inDowntrend & ~inDowntrend.shift(1).fillna(False)
    
    for i in range(len(close)):
        if long_entry.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries