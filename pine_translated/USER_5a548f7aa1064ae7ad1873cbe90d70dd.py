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
    # Convert time to datetime for time window filtering
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # London time windows: 7:45-9:45 and 14:45-16:45
    london_morning_start = 7 * 60 + 45  # 465 minutes
    london_morning_end = 9 * 60 + 45    # 585 minutes
    london_afternoon_start = 14 * 60 + 45  # 885 minutes
    london_afternoon_end = 16 * 60 + 45    # 1005 minutes
    
    isWithinMorningWindow = (df['time_minutes'] >= london_morning_start) & (df['time_minutes'] < london_morning_end)
    isWithinAfternoonWindow = (df['time_minutes'] >= london_afternoon_start) & (df['time_minutes'] < london_afternoon_end)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Previous day high/low from 4H data - using 240min resampling approach
    # For 15m data, PDH/PDL from 4H chart means we need to identify 4H boundaries
    # Simplified: Use rolling high/low with 16 bars (15m * 16 = 240m = 4H)
    prevDayHigh = df['high'].rolling(window=16, min_periods=16).max().shift(1)
    prevDayLow = df['low'].rolling(window=16, min_periods=16).min().shift(1)
    
    # PDH and PDL sweeps
    previousDayHighTaken = df['high'] > prevDayHigh
    previousDayLowTaken = df['low'] < prevDayLow
    
    # 4H data reconstruction for swing detection
    high_4h = df['high'].rolling(window=16, min_periods=16).max()
    low_4h = df['low'].rolling(window=16, min_periods=16).min()
    
    # Swing detection using 4H data
    is_swing_high_4h = (
        (high_4h.shift(3) < high_4h.shift(2)) & 
        (high_4h.shift(1) <= high_4h.shift(2)) & 
        (high_4h.shift(2) >= high_4h.shift(4)) & 
        (high_4h.shift(2) >= high_4h.shift(5))
    )
    is_swing_low_4h = (
        (low_4h.shift(3) > low_4h.shift(2)) & 
        (low_4h.shift(1) >= low_4h.shift(2)) & 
        (low_4h.shift(2) <= low_4h.shift(4)) & 
        (low_4h.shift(2) <= low_4h.shift(5))
    )
    
    # Flags for tracking sweeps
    flagpdh = previousDayHighTaken.cumsum() > 0
    flagpdl = previousDayLowTaken.cumsum() > 0
    
    # ATR (Wilder RSI style smoothing)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Filter valid bars
    valid_bars = isWithinTimeWindow & ~isWithinTimeWindow.isna() & ~atr.isna() & ~prevDayHigh.isna() & ~prevDayLow.isna()
    
    # Build boolean series for conditions
    long_condition = isWithinTimeWindow & previousDayHighTaken
    short_condition = isWithinTimeWindow & previousDayLowTaken
    
    # Filter conditions for valid bars
    long_condition = long_condition & valid_bars
    short_condition = short_condition & valid_bars
    
    # Iterate and generate entries
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        if not valid_bars.iloc[i]:
            continue
        
        if long_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        
        if short_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries