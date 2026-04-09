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
    
    results = []
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    time = df['time']
    
    # Volume Filter
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    # ATR Filter (Wilder)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = tr.ewm(span=20, adjust=False).mean() / 1.5
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    # Trend Filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # OB conditions
    obUp = (close.shift(1) > open_price.shift(1)) & (close > open_price) & (close > high.shift(1))
    obDown = (close.shift(1) < open_price.shift(1)) & (close < open_price) & (close < low.shift(1))
    
    # FVG conditions for OB alignment
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)
    
    # Combined entry conditions (OB + FVG stacked)
    long_condition = obUp & fvgUp & bfvg
    short_condition = obDown & fvgDown & sfvg
    
    # Time window check
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    london_offset = pd.tseries.offsets.Hour(n=0)
    
    morning_start = ts.dt.normalize() + pd.Timedelta(hours=7, minutes=45)
    morning_end = ts.dt.normalize() + pd.Timedelta(hours=9, minutes=45)
    afternoon_start = ts.dt.normalize() + pd.Timedelta(hours=14, minutes=45)
    afternoon_end = ts.dt.normalize() + pd.Timedelta(hours=16, minutes=45)
    
    is_within_time_window = ((ts >= morning_start) & (ts < morning_end)) | ((ts >= afternoon_start) & (ts < afternoon_end))
    
    # Apply time window
    long_condition = long_condition & is_within_time_window
    short_condition = short_condition & is_within_time_window
    
    # Skip bars where indicators are NaN
    valid_bars = ~(bfvg.isna() | sfvg.isna() | obUp.isna() | obDown.isna() | is_within_time_window.isna())
    long_condition = long_condition & valid_bars
    short_condition = short_condition & valid_bars
    
    # Find entries
    for i in range(len(df)):
        if i < 2:
            continue
            
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            results.append({
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
            
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            results.append({
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
    
    return results