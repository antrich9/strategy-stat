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
    open_price = df['open']
    volume = df['volume']
    
    # Volume Filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_sma = volume.rolling(9).mean()
    volfilt = close > 0  # placeholder, will be computed properly below
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR Filter: ta.atr(20) / 1.5
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_filter_val = atr / 1.5
    atrfilt = ((low - high.shift(2)) > atr_filter_val) | ((low.shift(2) - high) > atr_filter_val)
    
    # Trend Filter: ta.sma(close, 54) > ta.sma(close, 54)[1]
    loc = close.ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2] (FVG detection)
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2]
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Helper functions for OB and FVG conditions
    def isUp(idx):
        return close.shift(idx) > open_price.shift(idx)
    
    def isDown(idx):
        return close.shift(idx) < open_price.shift(idx)
    
    def isObUp(idx):
        return isDown(idx + 1) & isUp(idx) & (close.shift(idx) > high.shift(idx + 1))
    
    def isObDown(idx):
        return isUp(idx + 1) & isDown(idx) & (close.shift(idx) < low.shift(idx + 1))
    
    def isFvgUp(idx):
        return low.shift(idx) > high.shift(idx + 2)
    
    def isFvgDown(idx):
        return high.shift(idx) < low.shift(idx + 2)
    
    # Calculate OB and FVG conditions at specific offsets
    obUp = isObUp(1)
    obDown = isObDown(1)
    fvgUp = isFvgUp(0)
    fvgDown = isFvgDown(0)
    
    # Entries list
    entries = []
    trade_num = 1
    
    # Iterate through bars
    for i in range(len(df)):
        # Skip if any required indicator is NaN
        if i < 3:
            continue
        
        # Bullish entry conditions
        if bfvg.iloc[i] and obUp.iloc[i] and fvgUp.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        
        # Bearish entry conditions
        if sfvg.iloc[i] and obDown.iloc[i] and fvgDown.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries