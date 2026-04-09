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
    
    # Resample to daily timeframe
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    daily = df.groupby(pd.Grouper(key='time_dt', freq='D')).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'last'
    }).dropna()
    
    daily = daily.reset_index(drop=True)
    
    # Get daily high, low, open, close shifted appropriately
    dailyHigh = daily['high']
    dailyLow = daily['low']
    dailyOpen = daily['open']
    dailyClose = daily['close']
    
    # Previous day's data
    prevDayHigh = dailyHigh.shift(1)
    prevDayLow = dailyLow.shift(1)
    
    # Get daily data with historical reference for FVG detection
    dailyHigh1 = dailyHigh.shift(1)
    dailyLow1 = dailyLow.shift(1)
    dailyHigh2 = dailyHigh.shift(2)
    dailyLow2 = dailyLow.shift(2)
    
    # Volume filter
    volfilt = df.groupby(pd.Grouper(key='time_dt', freq='D'))['volume'].sum().shift(1) > df.groupby(pd.Grouper(key='time_dt', freq='D'))['volume'].sum().rolling(9).mean() * 1.5
    
    # ATR filter (Wilder ATR)
    high_low = daily['high'] - daily['low']
    high_close = np.abs(daily['high'] - daily['close'].shift(1))
    low_close = np.abs(daily['low'] - daily['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder ATR
    atr = true_range.ewm(alpha=1/20, adjust=False).mean()
    atr2 = atr / 1.5
    
    atrfilt_long = (dailyLow - dailyHigh2.shift(1) > atr2) | (dailyLow.shift(2) - dailyHigh.shift(1) > atr2)
    atrfilt_short = (dailyHigh - dailyLow2.shift(1) < atr2) | (dailyHigh.shift(2) - dailyLow.shift(1) < atr2)
    
    # Trend filter (SMA of close)
    close_daily = daily['close']
    loc = close_daily.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions
    # Bullish FVG: dailyLow > dailyHigh2 (gap up from 2 days ago)
    # Bearish FVG: dailyHigh < dailyLow2 (gap down from 2 days ago)
    
    bfvg = (dailyLow > dailyHigh2.shift(1)) & locfiltb & atrfilt_long & volfilt.reindex(daily.index, fill_value=True)
    sfvg = (dailyHigh < dailyLow2.shift(1)) & locfilts & atrfilt_short & volfilt.reindex(daily.index, fill_value=True)
    
    # Swing detection
    is_swing_high = (dailyHigh1 < dailyHigh2) & (dailyHigh.shift(3) < dailyHigh2) & (dailyHigh.shift(4) < dailyHigh2)
    is_swing_low = (dailyLow1 > dailyLow2) & (dailyLow.shift(3) > dailyLow2) & (dailyLow.shift(4) > dailyLow2)
    
    # Track last swing type
    lastSwingType = pd.Series(index=daily.index, dtype=str).fillna('none')
    lastSwingType_prev = 'none'
    
    entries = []
    trade_num = 1
    
    # We need to map daily conditions back to intraday bars
    df_indexed = df.set_index('time_dt')
    
    for i in range(len(daily)):
        ts = daily['time'].iloc[i]
        
        # Get the daily bar's start and end times
        day_start = daily['time_dt'].iloc[i]
        if i < len(daily) - 1:
            day_end = daily['time_dt'].iloc[i + 1]
        else:
            day_end = day_start + pd.Timedelta(days=1)
        
        # Find corresponding intraday bars
        mask = (df['time_dt'] >= day_start) & (df['time_dt'] < day_end)
        day_bars = df[mask]
        
        if len(day_bars) == 0:
            continue
        
        # Update swing type tracking
        if is_swing_high.iloc[i]:
            lastSwingType_prev = 'dailyHigh'
        elif is_swing_low.iloc[i]:
            lastSwingType_prev = 'dailyLow'
        
        # Long entry: bfvg with dailyLow swing
        if bfvg.iloc[i] and lastSwingType_prev == 'dailyLow':
            # Entry on first bar of the day
            entry_bar = day_bars.iloc[0]
            entry_price = entry_bar['close']
            entry_ts = int(entry_bar['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
        
        # Short entry: sfvg with dailyHigh swing
        if sfvg.iloc[i] and lastSwingType_prev == 'dailyHigh':
            # Entry on first bar of the day
            entry_bar = day_bars.iloc[0]
            entry_price = entry_bar['close']
            entry_ts = int(entry_bar['time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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