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
    
    # Helper functions for indicators
    def isUp(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def isDown(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def isObUp(idx):
        if idx + 1 >= len(df) or idx >= len(df):
            return False
        return isDown(idx + 1) and isUp(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def isObDown(idx):
        if idx + 1 >= len(df) or idx >= len(df):
            return False
        return isUp(idx + 1) and isDown(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def isFvgUp(idx):
        if idx + 2 >= len(df):
            return False
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def isFvgDown(idx):
        if idx + 2 >= len(df):
            return False
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Get timezone info from timestamps
    def get_timezone():
        return timezone.utc
    
    # Time filter functions
    def is_in_session(timestamp, start_hour, start_min, end_hour, end_min):
        dt = datetime.fromtimestamp(timestamp, tz=get_timezone())
        start = dt.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        end = dt.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
        return start <= dt <= end
    
    def is_in_session_0700_0959(ts):
        return is_in_session(ts, 7, 0, 9, 59)
    
    def is_in_session_1200_1459(ts):
        return is_in_session(ts, 12, 0, 14, 59)
    
    # Get daily high/low - need to implement rolling day calculation
    # Since we don't have security() function, we'll calculate previous day high/low
    df['day_start'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.normalize()
    df['prev_day_high'] = df['high'].rolling(window='1D', min_periods=1).max().shift(1)
    df['prev_day_low'] = df['low'].rolling(window='1D', min_periods=1).min().shift(1)
    
    # Calculate OB and FVG conditions
    obUp = pd.Series(False, index=df.index)
    obDown = pd.Series(False, index=df.index)
    fvgUp = pd.Series(False, index=df.index)
    fvgDown = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        obUp.iloc[i] = isObUp(i)
        obDown.iloc[i] = isObDown(i)
        fvgUp.iloc[i] = isFvgUp(i)
        fvgDown.iloc[i] = isFvgDown(i)
    
    # Calculate flag conditions (PDH/PDL sweep)
    flagpdh = df['close'] > df['prev_day_high']
    flagpdl = df['close'] < df['prev_day_low']
    
    # Long conditions: flagpdh AND (obUp AND fvgUp) AND time filter
    long_cond_1 = flagpdh & obUp & fvgUp & df['time'].apply(is_in_session_0700_0959)
    long_cond_2 = flagpdh & obUp & fvgUp & df['time'].apply(is_in_session_1200_1459)
    long_cond = long_cond_1 | long_cond_2
    
    # Short conditions: flagpdl AND (obDown AND fvgDown) AND time filter
    short_cond_1 = flagpdl & obDown & fvgDown & df['time'].apply(is_in_session_0700_0959)
    short_cond_2 = flagpdl & obDown & fvgDown & df['time'].apply(is_in_session_1200_1459)
    short_cond = short_cond_1 | short_cond_2
    
    # Generate entries
    for i in range(len(df)):
        if pd.isna(df['prev_day_high'].iloc[i]) or pd.isna(df['prev_day_low'].iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=get_timezone()).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=get_timezone()).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries