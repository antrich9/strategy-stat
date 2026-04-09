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
    
    # Convert time to datetime for timezone-aware operations
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Check if it's Friday (dayofweek: Monday=1, Sunday=7, Friday=5)
    df['is_friday'] = df['datetime'].dt.dayofweek == 4
    
    # Trading window logic: London time (7:45-9:45 and 15:45-16:45)
    # Convert to London local time (UTC+0/UTC+1 for DST - simplified to UTC for this implementation)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Morning window: 7:45 to 9:45
    morning_start = (df['hour'] == 7) & (df['minute'] >= 45)
    morning_mid = (df['hour'] == 8)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 45)
    is_within_morning_window = morning_start | morning_mid | morning_end
    
    # Afternoon window: 15:45 to 16:45
    afternoon_start = (df['hour'] == 15) & (df['minute'] >= 45)
    afternoon_mid = (df['hour'] == 16) & (df['minute'] <= 45)
    is_within_afternoon_window = afternoon_start | afternoon_mid
    
    # Combine windows and exclude Fridays
    df['in_trading_window'] = (is_within_morning_window | is_within_afternoon_window) & (~df['is_friday'])
    
    # Get previous day high and low
    # For daily high/low, we need to look back at previous day's data
    df['day'] = df['datetime'].dt.date
    daily_agg = df.groupby('day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    daily_agg = daily_agg[['day', 'prev_day_high', 'prev_day_low']]
    
    # Merge previous day levels back to main dataframe
    df = df.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    
    # Forward fill the previous day high/low within each day
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Shift to get the previous day's high/low (not current day's)
    df['prev_day_high'] = df['prev_day_high'].shift(1)
    df['prev_day_low'] = df['prev_day_low'].shift(1)
    
    # Get current day high and low (rolling 240-bar or daily aggregation)
    # Since we don't have 240-min data, we'll use daily rolling high/low within the current day
    df['current_day_high'] = df.groupby('day')['high'].cummax()
    df['current_day_low'] = df.groupby('day')['low'].cummin()
    
    # Detect sweeps of previous day high/low
    df['previous_day_high_taken'] = df['high'] > df['prev_day_high']
    df['previous_day_low_taken'] = df['low'] < df['prev_day_low']
    
    # Initialize flags
    df['flagpdh'] = False
    df['flagpdl'] = False
    
    # Iterate to set flags properly (using cumulative logic)
    # flagpdh: previous day high swept AND current day low > prev day low
    # flagpdl: previous day low swept AND current day high < prev day high
    
    for i in range(1, len(df)):
        if df['previous_day_high_taken'].iloc[i] and df['current_day_low'].iloc[i] > df['prev_day_low'].iloc[i]:
            df.loc[df.index[i], 'flagpdh'] = True
            df.loc[df.index[i], 'flagpdl'] = False
        elif df['previous_day_low_taken'].iloc[i] and df['current_day_high'].iloc[i] < df['prev_day_high'].iloc[i]:
            df.loc[df.index[i], 'flagpdl'] = True
            df.loc[df.index[i], 'flagpdh'] = False
        else:
            df.loc[df.index[i], 'flagpdh'] = False
            df.loc[df.index[i], 'flagpdl'] = False
    
    # Entry conditions based on the logic:
    # Long entry: flagpdh is true (prev day high swept, prev day low not swept) and in trading window
    # Short entry: flagpdl is true (prev day low swept, prev day high not swept) and in trading window
    
    df['long_entry'] = df['flagpdh'] & df['in_trading_window']
    df['short_entry'] = df['flagpdl'] & df['in_trading_window']
    
    # Generate entry signals
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['long_entry'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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
            
        elif df['short_entry'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
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