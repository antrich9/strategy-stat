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
    # Time window filters (in UTC hours)
    # betweenTime = '0700-0959' -> hours 7-9
    # betweenTime1 = '1200-1459' -> hours 12-14
    df['hour'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    
    time_filter_long = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 12) & (df['hour'] <= 14))
    
    # Previous day high and low using 1-day shift
    df['prev_day_high'] = df['high'].shift(1).rolling(window=2).max().shift(1)
    df['prev_day_low'] = df['low'].shift(1).rolling(window=2).min().shift(1)
    
    # Detect new day (start of day)
    df['is_new_day'] = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).date()) != df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).date()).shift(1)
    df['is_new_day'] = df['is_new_day'].fillna(False)
    
    # Flags for PDH and PDL sweep
    flagpdh = False
    flagpdl = False
    
    # WaitForEntry flags
    waitingForEntry = False
    waitingForShortEntry1 = False
    
    # OB and FVG detection functions
    def isUp(close, open_prices, idx):
        return close.iloc[idx] > open_prices.iloc[idx]
    
    def isDown(close, open_prices, idx):
        return close.iloc[idx] < open_prices.iloc[idx]
    
    def isObUp(close, open_prices, high_prices, idx):
        if idx < 1:
            return False
        return isDown(close, open_prices, idx + 1) and isUp(close, open_prices, idx) and close.iloc[idx] > high_prices.iloc[idx + 1]
    
    def isObDown(close, open_prices, low_prices, idx):
        if idx < 1:
            return False
        return isUp(close, open_prices, idx + 1) and isDown(close, open_prices, idx) and close.iloc[idx] < low_prices.iloc[idx + 1]
    
    def isFvgUp(low_prices, high_prices, idx):
        if idx < 2:
            return False
        return low_prices.iloc[idx] > high_prices.iloc[idx + 2]
    
    def isFvgDown(high_prices, low_prices, idx):
        if idx < 2:
            return False
        return high_prices.iloc[idx] < low_prices.iloc[idx + 2]
    
    # Calculate OB and FVG
    obUp = isObUp(df['close'], df['open'], df['high'], 1)
    obDown = isObDown(df['close'], df['open'], df['low'], 1)
    fvgUp = isFvgUp(df['low'], df['high'], 0)
    fvgDown = isFvgDown(df['high'], df['low'], 0)
    
    df['ob_up'] = False
    df['ob_down'] = False
    df['fvg_up'] = False
    df['fvg_down'] = False
    
    for i in range(2, len(df)):
        df.loc[df.index[i], 'ob_up'] = isObUp(df['close'], df['open'], df['high'], i)
        df.loc[df.index[i], 'ob_down'] = isObDown(df['close'], df['open'], df['low'], i)
        df.loc[df.index[i], 'fvg_up'] = isFvgUp(df['low'], df['high'], i)
        df.loc[df.index[i], 'fvg_down'] = isFvgDown(df['high'], df['low'], i)
    
    # Detect PDH and PDL sweep
    df['pdh_sweep'] = (df['close'] > df['prev_day_high']) & df['prev_day_high'].notna()
    df['pdl_sweep'] = (df['close'] < df['prev_day_low']) & df['prev_day_low'].notna()
    
    # Entry conditions
    long_entry_cond = (
        (df['ob_up'] == True) & 
        (df['fvg_up'] == True) & 
        (df['pdh_sweep'] == True) &
        time_filter_long
    )
    
    short_entry_cond = (
        (df['ob_down'] == True) & 
        (df['fvg_down'] == True) & 
        (df['pdl_sweep'] == True) &
        time_filter_long
    )
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        row = df.iloc[i]
        
        # Detect new day and reset flags
        if row['is_new_day']:
            flagpdh = False
            flagpdl = False
            waitingForEntry = False
            waitingForShortEntry1 = False
        
        # Update flags on sweep
        if row['pdh_sweep']:
            flagpdh = True
        if row['pdl_sweep']:
            flagpdl = True
        
        entry_price = row['close']
        ts = int(row['time'])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        # Long entry
        if long_entry_cond.iloc[i] and flagpdh:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        # Short entry
        if short_entry_cond.iloc[i] and flagpdl:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries