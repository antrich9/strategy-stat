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
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_prices = df['open']
    volume = df['volume']
    time = df['time']
    
    # Helper functions
    def isUp(idx):
        return close.iloc[idx] > open_prices.iloc[idx]
    
    def isDown(idx):
        return close.iloc[idx] < open_prices.iloc[idx]
    
    def isObUp(idx):
        return isDown(idx + 1) and isUp(idx) and close.iloc[idx] > high.iloc[idx + 1]
    
    def isObDown(idx):
        return isUp(idx + 1) and isDown(idx) and close.iloc[idx] < low.iloc[idx + 1]
    
    def isFvgUp(idx):
        return (low.iloc[idx] > high.iloc[idx + 2])
    
    def isFvgDown(idx):
        return (high.iloc[idx] < low.iloc[idx + 2])
    
    # Calculate indicators
    obUp = pd.Series(False, index=df.index)
    obDown = pd.Series(False, index=df.index)
    fvgUp = pd.Series(False, index=df.index)
    fvgDown = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        try:
            obUp.iloc[i] = isObUp(i - 1)
            obDown.iloc[i] = isObDown(i - 1)
            fvgUp.iloc[i] = isFvgUp(i)
            fvgDown.iloc[i] = isFvgDown(i)
        except:
            pass
    
    # Volume filter
    vol_sma = volume.rolling(9).mean()
    volfilt = (volume.shift(1) > vol_sma * 1.5) | (volume.shift(1).isna())
    
    # ATR filter (Wilder)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = ((low - high.shift(2) > atr / 1.5) | (low.shift(2) - high > atr / 1.5)) | (atr.isna())
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 | loc.isna()
    locfilts = (~loc2) | loc.isna()
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts
    
    # Time window (Europe/London times)
    # Morning: 07:45 - 09:45, Afternoon: 14:45 - 16:45
    timestamps = pd.to_datetime(df['time'], unit='ms', utc=True)
    hours = timestamps.dt.hour
    minutes = timestamps.dt.minute
    
    isWithinMorningWindow = ((hours == 7) & (minutes >= 45)) | ((hours == 8) | (hours == 9) & (minutes < 45))
    isWithinAfternoonWindow = ((hours == 14) & (minutes >= 45)) | ((hours == 15) | (hours == 16) & (minutes < 45))
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Entry conditions
    long_condition = (bfvg | obUp | fvgUp) & isWithinTimeWindow
    short_condition = (sfvg | obDown | fvgDown) & isWithinTimeWindow
    
    # Generate entries
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries