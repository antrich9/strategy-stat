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
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Extract hour and minute for trading window logic
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Check if the bar is during daylight saving time (DST)
    # UK DST: Starts last Sunday in March, Ends last Sunday in October
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # DST condition: month == 3 and Sunday and day >= 25, or month > 3
    # AND month == 10 and Sunday and day < 25, or month < 10
    is_dst_start = (df['month'] == 3) & (df['dayofweek'] == 6) & (df['day'] >= 25)
    is_dst_other = df['month'] > 3
    is_dst_end = (df['month'] == 10) & (df['dayofweek'] == 6) & (df['day'] < 25)
    is_dst_other_end = df['month'] < 10
    
    is_dst = ((is_dst_start) | (is_dst_other)) & ((is_dst_end) | (is_dst_other_end))
    
    # Adjust hour for DST (add 1 hour during DST)
    adjusted_hour = df['hour'] + is_dst.astype(int)
    
    # Trading window 1: 07:00 - 10:59
    start_hour_1 = 7
    end_hour_1 = 10
    end_minute_1 = 59
    
    # Trading window 2: 15:00 - 16:59
    start_hour_2 = 15
    end_hour_2 = 16
    end_minute_2 = 59
    
    # Check if within trading windows
    in_window_1 = (adjusted_hour >= start_hour_1) & (adjusted_hour <= end_hour_1)
    in_window_1_exclusive = (adjusted_hour == end_hour_1) & (df['minute'] > end_minute_1)
    in_window_1 = in_window_1 & ~in_window_1_exclusive
    
    in_window_2 = (adjusted_hour >= start_hour_2) & (adjusted_hour <= end_hour_2)
    in_window_2_exclusive = (adjusted_hour == end_hour_2) & (df['minute'] > end_minute_2)
    in_window_2 = in_window_2 & ~in_window_2_exclusive
    
    in_trading_window = in_window_1 | in_window_2
    
    # Previous day high and low using daily resampling
    # For each bar, get the previous day's high and low
    daily_df = df.resample('D', on='datetime').agg({'high': 'max', 'low': 'min'}).dropna()
    
    # Create a mapping from date to previous day's high/low
    daily_df = daily_df.reset_index()
    daily_df['date'] = daily_df['datetime'].dt.date
    
    # Get previous day values by shifting
    daily_df['prev_day_high'] = daily_df['high'].shift(1)
    daily_df['prev_day_low'] = daily_df['low'].shift(1)
    
    # Map previous day values back to main dataframe
    df['date'] = df['datetime'].dt.date
    df = df.merge(daily_df[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    # Current day high/low from 240-minute timeframe (simulated by checking within current day)
    df['current_day_high'] = df['high'].cummax()
    df['current_day_low'] = df['low'].cummin()
    
    # Check for new day
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    
    # Reset tracking variables on new day
    df['flagpdh'] = False
    df['flagpdl'] = False
    
    # Previous day high/low taken conditions
    prev_day_high = df['prev_day_high']
    prev_day_low = df['prev_day_low']
    
    prev_day_high_taken = df['high'] > prev_day_high
    prev_day_low_taken = df['low'] < prev_day_low
    
    # Track flags with state
    flagpdh = False
    flagpdl = False
    
    for i in range(len(df)):
        if df['is_new_day'].iloc[i]:
            flagpdh = False
            flagpdl = False
        
        if prev_day_high_taken.iloc[i] and df['current_day_low'].iloc[i] > prev_day_low.iloc[i]:
            flagpdh = True
        elif prev_day_low_taken.iloc[i] and df['current_day_high'].iloc[i] < prev_day_high.iloc[i]:
            flagpdl = True
        else:
            flagpdh = False
            flagpdl = False
        
        df.iloc[i, df.columns.get_loc('flagpdh')] = flagpdh
        df.iloc[i, df.columns.get_loc('flagpdl')] = flagpdl
    
    # Entry conditions
    df['long_entry'] = df['flagpdh'] & in_trading_window
    df['short_entry'] = df['flagpdl'] & in_trading_window
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:  # Skip first two bars for indicator stability
            continue
        
        if df['long_entry'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
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
        
        if df['short_entry'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
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