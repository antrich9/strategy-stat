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
    
    # Constants from Pine Script
    fastLength = 50
    slowLength = 200
    start_hour = 7
    end_hour = 10
    end_minute = 59
    
    # Calculate EMAs
    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()
    
    # Detect new day (change in day)
    df['day'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.date
    isNewDay = df['day'] != df['day'].shift(1)
    isNewDay.iloc[0] = True  # First bar is considered new day
    
    # Calculate previous day high and low
    # Using 1-day offset to get previous day's values
    prevDayHigh = df['high'].shift(1).where(isNewDay.shift(1).fillna(False), np.nan)
    prevDayLow = df['low'].shift(1).where(isNewDay.shift(1).fillna(False), np.nan)
    
    # Forward fill previous day high/low until new day
    prevDayHigh = prevDayHigh.ffill()
    prevDayLow = prevDayLow.ffill()
    
    # Adjusted time for UTC+1 with DST
    df['dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['month'] = df['dt'].dt.month
    df['dayofweek'] = df['dt'].dt.dayofweek  # Monday=0, Sunday=6
    df['dayofmonth'] = df['dt'].dt.day
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    
    # DST check: March (month>=3) with Sunday>=25 or after March, October before Sunday<25 or before October
    is_dst = ((df['month'] >= 3) & ((df['month'] == 3) & (df['dayofweek'] == 6) & (df['dayofmonth'] >= 25) | (df['month'] > 3))) & \
             ((df['month'] <= 10) & ((df['month'] == 10) & (df['dayofweek'] == 6) & (df['dayofmonth'] < 25) | (df['month'] < 10)))
    
    # Adjust hour for UTC+1 with DST
    adjusted_hour = df['hour'] + 1 + is_dst.astype(int)
    adjusted_hour = adjusted_hour % 24
    adjusted_minute = df['minute']
    
    # Trading window check
    in_trading_window = (adjusted_hour >= start_hour) & (adjusted_hour <= end_hour) & \
                        ~((adjusted_hour == end_hour) & (adjusted_minute > end_minute))
    
    # Flags for previous day high/low sweep
    flagpdh = df['close'] > prevDayHigh
    flagpdl = df['close'] < prevDayLow
    
    # Reset flags at start of new day
    flagpdh = flagpdh.where(~isNewDay, False)
    flagpdl = flagpdl.where(~isNewDay, False)
    
    # Forward fill flags within each day
    flagpdh = flagpdh.groupby(df['day']).transform('max')
    flagpdl = flagpdl.groupby(df['day']).transform('max')
    
    # Trend conditions
    uptrend = fastEMA > slowEMA
    downtrend = fastEMA < slowEMA
    
    # Entry conditions
    long_condition = in_trading_window & flagpdl & uptrend
    short_condition = in_trading_window & flagpdh & downtrend
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip if indicators might be NaN (need enough bars for EMA)
        if i < slowLength:
            continue
        
        entry_price = df['close'].iloc[i]
        ts = df['time'].iloc[i]
        
        if long_condition.iloc[i]:
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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