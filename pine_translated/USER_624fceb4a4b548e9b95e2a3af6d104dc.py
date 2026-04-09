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
    
    # Extract OHLC data
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # Previous day high/low - using 1-day lookahead in Pine, approximated here
    # We'll use daily data shifted by 1 day
    daily_high = df['high'].resample('D', kind='timestamp').max()
    daily_low = df['low'].resample('D', kind='timestamp').min()
    
    # Shift to get previous day values
    prev_day_high = daily_high.shift(1)
    prev_day_low = daily_low.shift(1)
    
    # Reindex to match original dataframe
    prev_day_high = prev_day_high.reindex(df.index, method='ffill')
    prev_day_low = prev_day_low.reindex(df.index, method='ffill')
    
    # Current day high/low (240min = 4h bars)
    current_day_high = df['high'].rolling(window=6, min_periods=1).max()  # Approx 240min
    current_day_low = df['low'].rolling(window=6, min_periods=1).min()
    
    # Calculate previous day high/low taken
    previous_day_high_taken = high > prev_day_high
    previous_day_low_taken = low < prev_day_low
    
    # Initialize flags
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    # Calculate flagpdh and flagpdl
    for i in range(1, len(df)):
        if previous_day_high_taken.iloc[i] and current_day_low.iloc[i] > prev_day_low.iloc[i]:
            flagpdh.iloc[i] = True
        elif previous_day_low_taken.iloc[i] and current_day_high.iloc[i] < prev_day_high.iloc[i]:
            flagpdl.iloc[i] = True
    
    # Swing detection (2-bar lookback for swing high/low)
    is_swing_high = pd.Series(False, index=df.index)
    is_swing_low = pd.Series(False, index=df.index)
    
    for i in range(4, len(df)):
        main_bar_high = high.iloc[i-2]
        main_bar_low = low.iloc[i-2]
        
        # Swing High: high[i-1] < main_bar_high and high[i-3] < main_bar_high and high[i-4] < main_bar_high
        if high.iloc[i-1] < main_bar_high and high.iloc[i-3] < main_bar_high and high.iloc[i-4] < main_bar_high:
            is_swing_high.iloc[i] = True
        
        # Swing Low: low[i-1] > main_bar_low and low[i-3] > main_bar_low and low[i-4] > main_bar_low
        if low.iloc[i-1] > main_bar_low and low.iloc[i-3] > main_bar_low and low.iloc[i-4] > main_bar_low:
            is_swing_low.iloc[i] = True
    
    # Trading window (London time: 07:45-09:45 and 15:45-16:45)
    is_within_morning_window = pd.Series(False, index=df.index)
    is_within_afternoon_window = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        ts = time.iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Morning window: 07:45 to 09:45
        if (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute <= 45):
            is_within_morning_window.iloc[i] = True
        
        # Afternoon window: 15:45 to 16:45
        if (hour == 15 and minute >= 45) or (hour == 16 and minute <= 45):
            is_within_afternoon_window.iloc[i] = True
    
    is_within_time_window = is_within_morning_window | is_within_afternoon_window
    
    # Entry conditions (long and short)
    # Long: flagpdh is true and within time window
    # Short: flagpdl is true and within time window
    
    long_condition = flagpdh & is_within_time_window
    short_condition = flagpdl & is_within_time_window
    
    # Generate entries
    for i in range(1, len(df)):
        ts = int(time.iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
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