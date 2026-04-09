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
    
    # Trading windows
    startHour, endHour = 7, 8
    startMinute, endMinute = 0, 59
    startHour2, endHour2 = 10, 11
    startMinute2, endMinute2 = 0, 59
    startHour3, endHour3 = 4, 4
    startMinute3, endMinute3 = 0, 59
    startHour4, endHour4 = 2, 2
    startMinute4, endMinute4 = 0, 59
    
    # Filters (default enabled as in original inp1/inp2/inp3 = false so filters disabled)
    inp1, inp2, inp3 = False, False, False
    
    # Pre-calculate volume SMA(9)
    vol_sma9 = df['volume'].rolling(9).mean()
    
    # Pre-calculate ATR(20) using Wilder's method
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean()
    
    # Pre-calculate SMA(54) for location filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # FVG conditions
    bfvg = df['low'] > df['high'].shift(2)
    sfvg = df['high'] < df['low'].shift(2)
    
    # Imbalance conditions
    top_imb_bway = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1)) & (df['close'] < df['low'].shift(1))
    top_imb_xbway = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1)) & (df['close'] > df['low'].shift(1))
    bot_imb_bway = (df['high'].shift(2) >= df['open'].shift(1)) & (df['low'] <= df['close'].shift(1)) & (df['close'] > df['high'].shift(1))
    bot_imb_xbway = (df['high'].shift(2) >= df['open'].shift(1)) & (df['low'] <= df['close'].shift(1)) & (df['close'] < df['high'].shift(1))
    
    # Imbalance size > 0
    top_imb_size = (df['low'].shift(2) - df['high']) > 0
    bot_imb_size = (df['low'] - df['high'].shift(2)) > 0
    
    # Apply filters
    vol_filt = vol_sma9.shift(1) * 1.5 < df['volume'].shift(1)
    vol_filt = inp1 & vol_filt
    
    low_atr_diff = df['low'] - df['high'].shift(2)
    high_atr_diff = df['low'].shift(2) - df['high']
    atr_filt = (low_atr_diff > atr) | (high_atr_diff > atr)
    atr_filt = inp2 & atr_filt
    
    locfiltb = loc2 & inp3
    locfilts = (~loc2) & inp3
    
    # Combined conditions
    long_cond = bfvg & vol_filt & atr_filt & locfiltb & (top_imb_bway | top_imb_xbway) & top_imb_size
    short_cond = sfvg & vol_filt & atr_filt & locfilts & (bot_imb_bway | bot_imb_xbway) & bot_imb_size
    
    for i in range(2, len(df)):
        ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        hour = (ts % 86400) // 3600
        minute = (ts % 3600) // 60
        
        is_within_window = (
            (hour > startHour or (hour == startHour and minute >= startMinute)) and (hour < endHour or (hour == endHour and minute <= endMinute))
        ) or (
            (hour > startHour2 or (hour == startHour2 and minute >= startMinute2)) and (hour < endHour2 or (hour == endHour2 and minute <= endMinute2))
        ) or (
            (hour > startHour3 or (hour == startHour3 and minute >= startMinute3)) and (hour < endHour3 or (hour == endHour3 and minute <= endMinute3))
        ) or (
            (hour > startHour4 or (hour == startHour4 and minute >= startMinute4)) and (hour < endHour4 or (hour == endHour4 and minute <= endMinute4))
        )
        
        if not is_within_window:
            continue
        
        entry_price = df['close'].iloc[i]
        
        if long_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        if short_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
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