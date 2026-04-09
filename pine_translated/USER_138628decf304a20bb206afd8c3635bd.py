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
    fvgTH = 0.5
    
    localTime = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    hour = localTime.dt.hour
    minute = localTime.dt.minute
    totalMinutes = hour * 60 + minute
    
    isWithinMorningWindow = (totalMinutes >= 405) & (totalMinutes < 585)
    isWithinAfternoonWindow = (totalMinutes >= 885) & (totalMinutes < 1005)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/144, adjust=False).mean()
    atr_filtered = atr * fvgTH
    
    bullG = (low > high.shift(1))
    bearG = (high < low.shift(1))
    
    bull = (low - high.shift(2) > atr_filtered) & \
           (low > high.shift(2)) & \
           (close.shift(1) > high.shift(2)) & \
           ~(bullG | bullG.shift(1))
    
    bear = (low.shift(2) - high > atr_filtered) & \
           (high < low.shift(2)) & \
           (close.shift(1) < low.shift(2)) & \
           ~(bearG | bearG.shift(1))
    
    long_entry = bull & isWithinTimeWindow
    short_entry = bear & isWithinTimeWindow
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries