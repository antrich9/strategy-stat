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
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    
    dt = df['datetime'].dt
    london_hour = dt.tz_convert('Europe/London').hour
    london_minute = dt.tz_convert('Europe/London').minute
    
    in_window = (
        ((london_hour == 10) & (london_minute >= 45)) |
        ((london_hour == 11) & (london_minute <= 45))
    )
    
    resampled = df.set_index('datetime').resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = resampled['high']
    low_4h = resampled['low']
    close_4h = resampled['close']
    volume_4h = resampled['volume']
    
    volfilt1 = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    
    tr = pd.concat([high_4h - low_4h, (high_4h - close_4h.shift(1)).abs(), (low_4h - close_4h.shift(1)).abs()], axis=1).max(axis=1)
    atr_4h = tr.rolling(20).mean() / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h)
    
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    entries = []
    trade_num = 1
    lastFVG = 0
    
    for i in range(2, len(resampled)):
        dt4h = resampled.index[i]
        
        if not in_window[resampled.index.get_loc(dt4h)]:
            continue
        
        if bfvg1.iloc[i] and lastFVG == -1:
            direction = 'long'
            start_ts = int(dt4h.strftime('%Y%m%d%H%M%S'))
            idx = df['time'].searchsorted(start_ts)
            
            if idx < len(df):
                entry_ts = int(df['time'].iloc[idx])
                entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[idx])
                
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
                
                trade_num += 1
                lastFVG = 1
                
        elif sfvg1.iloc[i] and lastFVG == 1:
            direction = 'short'
            start_ts = int(dt4h.strftime('%Y%m%d%H%M%S'))
            idx = df['time'].searchsorted(start_ts)
            
            if idx < len(df):
                entry_ts = int(df['time'].iloc[idx])
                entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[idx])
                
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
                
                trade_num += 1
                lastFVG = -1
                
        elif bfvg1.iloc[i]:
            lastFVG = 1
        elif sfvg1.iloc[i]:
            lastFVG = -1
    
    return entries