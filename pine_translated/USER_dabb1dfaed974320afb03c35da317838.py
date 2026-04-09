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
    
    # Convert timestamp to datetime for London time checks
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['date'] = df['datetime'].dt.date
    
    # Calculate previous day high and low
    df['prev_day_high'] = df['high'].shift(1).where(df['date'] != df['date'].shift(1))
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['low'].shift(1).where(df['date'] != df['date'].shift(1))
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Current day high and low (so far)
    df['current_day_high'] = df.groupby('date')['high'].cummax()
    df['current_day_low'] = df.groupby('date')['low'].cummin()
    
    # Check if previous day high/low has been swept
    df['prev_day_high_swept'] = df['high'] > df['prev_day_high']
    df['prev_day_low_swept'] = df['low'] < df['prev_day_low']
    
    # Track flags for sweep conditions
    flagpdh = False
    flagpdl = False
    
    # London trading windows: 07:45-09:45 and 15:45-16:45
    df['in_london_window'] = (
        ((df['hour'] == 7) & (df['minute'] >= 45)) |
        ((df['hour'] == 8)) |
        ((df['hour'] == 9) & (df['minute'] <= 45)) |
        ((df['hour'] == 15) & (df['minute'] >= 45)) |
        ((df['hour'] == 16) & (df['minute'] <= 45))
    )
    
    prev_date = None
    
    for i in range(len(df)):
        current_date = df['date'].iloc[i]
        
        # Reset flags at start of new day
        if prev_date != current_date:
            flagpdh = False
            flagpdl = False
            prev_date = current_date
        
        # Get values for current bar
        prev_day_high = df['prev_day_high'].iloc[i]
        prev_day_low = df['prev_day_low'].iloc[i]
        current_day_high = df['current_day_high'].iloc[i]
        current_day_low = df['current_day_low'].iloc[i]
        prev_day_high_swept = df['prev_day_high_swept'].iloc[i]
        prev_day_low_swept = df['prev_day_low_swept'].iloc[i]
        in_window = df['in_london_window'].iloc[i]
        
        # Skip if any required value is NaN
        if pd.isna(prev_day_high) or pd.isna(prev_day_low):
            continue
        
        # Update flags based on sweep conditions
        if prev_day_high_swept and current_day_low > prev_day_low:
            flagpdh = True
        elif prev_day_low_swept and current_day_high < prev_day_high:
            flagpdl = True
        else:
            flagpdh = False
            flagpdl = False
        
        # Entry conditions based on flags and trading window
        if in_window:
            if flagpdh:
                # Long entry: previous day high swept, current day low above prev day low
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                
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
                flagpdh = False  # Reset after entry
            
            if flagpdl:
                # Short entry: previous day low swept, current day high below prev day high
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                
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
                flagpdl = False  # Reset after entry
    
    return entries