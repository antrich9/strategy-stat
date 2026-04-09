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
    
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    time = df['time']
    
    dt = pd.to_datetime(time, unit='s', utc=True)
    
    london_morning_start = (dt.dt.hour == 7) & (dt.dt.minute >= 45)
    london_morning_end = (dt.dt.hour == 9) & (dt.dt.minute <= 44)
    isWithinMorningWindow = london_morning_start | ((dt.dt.hour == 8) | ((dt.dt.hour == 9) & (dt.dt.minute <= 44)))
    isWithinMorningWindow = ((dt.dt.hour == 7) & (dt.dt.minute >= 45)) | (dt.dt.hour == 8) | ((dt.dt.hour == 9) & (dt.dt.minute <= 44))
    
    london_afternoon_start = (dt.dt.hour == 14) & (dt.dt.minute >= 45)
    london_afternoon_end = (dt.dt.hour == 16) & (dt.dt.minute <= 44)
    isWithinAfternoonWindow = ((dt.dt.hour == 14) & (dt.dt.minute >= 45)) | (dt.dt.hour == 15) | ((dt.dt.hour == 16) & (dt.dt.minute <= 44))
    
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    sma9 = close.rolling(9).mean()
    
    volfilt = volume.shift(1) > sma9 * 1.5
    
    tr = np.maximum(
        np.maximum(high - low, np.abs(high - close.shift(1))),
        np.abs(low - close.shift(1))
    )
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[0] = tr.iloc[0]
    alpha = 1.0 / 20.0
    for i in range(1, len(df)):
        atr.iloc[i] = alpha * tr.iloc[i] + (1 - alpha) * atr.iloc[i-1]
    atr = atr / 1.5
    
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    isUp = close > open_price
    isDown = close < open_price
    
    isObUp = isDown.shift(1) & isUp & (close > high.shift(1))
    isObDown = isUp.shift(1) & isDown & (close < low.shift(1))
    
    isFvgUp = low > high.shift(2)
    isFvgDown = high < low.shift(2)
    
    bfvg = isFvgUp & volfilt & atrfilt & loc2
    sfvg = isFvgDown & volfilt & atrfilt & (~loc2)
    
    long_condition = bfvg & isObUp & isWithinTimeWindow
    short_condition = sfvg & isObDown & isWithinTimeWindow
    
    for i in range(len(df)):
        if i < 2:
            continue
        if pd.isna(long_condition.iloc[i]) or pd.isna(short_condition.iloc[i]):
            continue
        if long_condition.iloc[i]:
            ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
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
            ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return results