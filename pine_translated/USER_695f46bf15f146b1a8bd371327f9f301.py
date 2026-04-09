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
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Time window: London morning 7:45-9:45 and afternoon 14:45-16:45
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['dayofweek'] = df['time'].dt.dayofweek
    
    morning_start = (df['hour'] == 7) & (df['minute'] >= 45)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 45)
    morning_window = morning_start | ((df['hour'] == 8) | ((df['hour'] == 9) & morning_end))
    morning_window = ((df['hour'] == 7) & (df['minute'] >= 45)) | ((df['hour'] == 8)) | ((df['hour'] == 9) & (df['minute'] <= 45))
    
    afternoon_start = (df['hour'] == 14) & (df['minute'] >= 45)
    afternoon_end = (df['hour'] == 16) & (df['minute'] <= 45)
    afternoon_window = ((df['hour'] == 14) & (df['minute'] >= 45)) | ((df['hour'] == 15)) | ((df['hour'] == 16) & (df['minute'] <= 45))
    
    isWithinTimeWindow = morning_window | afternoon_window
    
    # OB and FVG conditions
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    def isUp(idx):
        return close.iloc[idx] > open_.iloc[idx]
    
    def isDown(idx):
        return close.iloc[idx] < open_.iloc[idx]
    
    # Calculate OB and FVG series
    obUp = (isDown(1)) & (isUp(0)) & (close.iloc[0] > high.iloc[1])
    obDown = (isUp(1)) & (isDown(0)) & (close.iloc[0] < low.iloc[1])
    fvgUp = low.iloc[0] > high.iloc[2]
    fvgDown = high.iloc[0] < low.iloc[2]
    
    # Stack OB + FVG conditions
    long_condition = obUp & fvgUp & isWithinTimeWindow
    short_condition = obDown & fvgDown & isWithinTimeWindow
    
    # Optional filters (enabled by default in code)
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    atr = (high - low).rolling(20).mean() / 1.5
    atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 3:
            continue
            
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i].timestamp()),
                'entry_time': df['time'].iloc[i].isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            
        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i].timestamp()),
                'entry_time': df['time'].iloc[i].isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries