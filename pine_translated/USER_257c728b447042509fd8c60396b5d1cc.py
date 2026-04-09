import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytz

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
    ny_tz = pytz.timezone('America/New_York')
    
    # Convert timestamps to NY time for window checking
    df['time_ny'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(ny_tz)
    df['hour'] = df['time_ny'].dt.hour
    df['minute'] = df['time_ny'].dt.minute
    
    # London morning window: 2:45 - 5:45 NY
    morning_start = (df['hour'] == 2) & (df['minute'] >= 45)
    morning_mid = (df['hour'] >= 3) & (df['hour'] < 5)
    morning_end = (df['hour'] == 5) & (df['minute'] < 45)
    is_within_morning_window = morning_start | morning_mid | morning_end
    
    # London afternoon window: 8:45 - 10:45 NY
    afternoon_start = (df['hour'] == 8) & (df['minute'] >= 45)
    afternoon_mid = (df['hour'] >= 9) & (df['hour'] < 10)
    afternoon_end = (df['hour'] == 10) & (df['minute'] < 45)
    is_within_afternoon_window = afternoon_start | afternoon_mid | afternoon_end
    
    df['is_within_time_window'] = is_within_morning_window | is_within_afternoon_window
    
    # Calculate previous day high/low
    df['day_change'] = df['time_ny'].dt.date != df['time_ny'].dt.date.shift(1)
    df['new_day'] = df['day_change'].fillna(True)
    
    df['pdh'] = df['high'].rolling(window=2).max().shift(1)
    df['pdl'] = df['low'].rolling(window=2).min().shift(1)
    
    # Bias conditions
    df['swept_low'] = df['low'] < df['pdl']
    df['swept_high'] = df['high'] > df['pdh']
    df['broke_high'] = df['close'] > df['pdh']
    df['broke_low'] = df['close'] < df['pdl']
    
    entries = []
    trade_num = 1
    
    bias = 0
    prev_date = None
    
    for i in range(1, len(df)):
        ts = df['time'].iloc[i]
        current_date = df['time_ny'].iloc[i].date()
        
        # Reset bias on new day
        if current_date != prev_date:
            bias = 0
            prev_date = current_date
        
        # Update bias
        if df['swept_low'].iloc[i] and df['broke_high'].iloc[i]:
            bias = 1
        elif df['swept_high'].iloc[i] and df['broke_low'].iloc[i]:
            bias = -1
        elif df['low'].iloc[i] < df['pdl'].iloc[i]:
            bias = -1
        elif df['high'].iloc[i] > df['pdh'].iloc[i]:
            bias = 1
        
        # Check for NaN in required indicators
        if pd.isna(df['pdh'].iloc[i]) or pd.isna(df['pdl'].iloc[i]):
            continue
        
        # Generate entries based on bias and time window
        if df['is_within_time_window'].iloc[i]:
            if bias == 1:
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
            elif bias == -1:
                entry_price = df['close'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
    
    return entries