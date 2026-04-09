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
    # Initialize ATR(144) * 0.5 using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    
    atr = tr.ewm(alpha=1.0/144, adjust=False).mean()
    fvgTH = 0.5
    atr_threshold = atr * fvgTH
    
    # Time windows (Europe/London)
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    
    morning_start = ts.dt.tz_localize(None).dt.hour * 3600 + ts.dt.tz_localize(None).dt.minute * 60 + ts.dt.tz_localize(None).dt.second
    is_morning = (morning_start >= 6*3600 + 45*60) & (morning_start < 9*3600 + 45*60)
    
    afternoon_start = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    is_afternoon = (afternoon_start >= 14*3600 + 45*60) & (afternoon_start < 16*3600 + 45*60)
    
    is_within_time_window = is_morning | is_afternoon
    
    # Previous day high/low (simplified: using 240tf high/low approximation)
    pdh = high.rolling(96).max().shift(1)
    pdl = low.rolling(96).min().shift(1)
    
    # FVG conditions
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    bull = ((low - high.shift(2)) > atr_threshold) & \
           (low > high.shift(2)) & \
           (close.shift(1) > high.shift(2)) & \
           ~bullG & ~bullG.shift(1)
    
    bear = ((low.shift(2) - high) > atr_threshold) & \
           (high < low.shift(2)) & \
           (close.shift(1) < low.shift(2)) & \
           ~bearG & ~bearG.shift(1)
    
    # Sweep conditions
    prevDayHighTaken = high > pdh
    prevDayLowTaken = low < pdl
    
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if high.iloc[i] > pdh.iloc[i] if not pd.isna(pdh.iloc[i]) else False:
            flagpdh.iloc[i] = True
            flagpdl.iloc[i] = False
        elif low.iloc[i] < pdl.iloc[i] if not pd.isna(pdl.iloc[i]) else False:
            flagpdl.iloc[i] = True
            flagpdh.iloc[i] = False
        else:
            flagpdh.iloc[i] = flagpdh.iloc[i-1]
            flagpdl.iloc[i] = flagpdl.iloc[i-1]
    
    # Entry signals: long on bull FVG with time window, short on bear FVG with time window
    long_entry = bull & is_within_time_window & flagpdh & (~bullG) & (~bullG.shift(1))
    short_entry = bear & is_within_time_window & flagpdl & (~bearG) & (~bearG.shift(1))
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
        if np.isnan(atr_threshold.iloc[i]):
            continue
        
        if long_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries