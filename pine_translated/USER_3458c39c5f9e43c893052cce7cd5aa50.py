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
    results = []
    trade_num = 0
    
    # Calculate ATR (Wilder ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr = pd.Series(np.zeros(len(df)), index=df.index)
    if len(true_range) > 0:
        atr.iloc[0] = true_range.iloc[0]
        for i in range(1, len(df)):
            atr.iloc[i] = (atr.iloc[i-1] * 13 + true_range.iloc[i]) / 14
    
    # Get previous day high/low using D timeframe
    prev_day_high = pd.Series(np.nan, index=df.index)
    prev_day_low = pd.Series(np.nan, index=df.index)
    
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['time'], unit='s', utc=True).dt.date
    df_copy['prev_date'] = df_copy['date'].shift(1)
    
    for i in range(1, len(df)):
        if df_copy['date'].iloc[i] != df_copy['date'].iloc[i-1]:
            prev_day_high.iloc[i] = df['high'].iloc[i-1]
            prev_day_low.iloc[i] = df['low'].iloc[i-1]
        else:
            if i > 0 and i < len(prev_day_high):
                prev_day_high.iloc[i] = prev_day_high.iloc[i-1]
                prev_day_low.iloc[i] = prev_day_low.iloc[i-1]
    
    for i in range(1, len(df)):
        if pd.notna(prev_day_high.iloc[i]):
            break
        prev_day_high.iloc[i] = df['high'].iloc[i]
        prev_day_low.iloc[i] = df['low'].iloc[i]
    
    # Get 240 timeframe high/low (current day)
    current_day_high = df['high'].rolling(96).max().shift(1)
    current_day_low = df['low'].rolling(96).min().shift(1)
    current_day_high = df['high'].cummax().shift(1)
    current_day_low = df['low'].cummin().shift(1)
    
    # Check if new day
    is_new_day = df_copy['date'] != df_copy['date'].shift(1)
    
    # Define London time trading windows
    dt_utc = pd.to_datetime(df['time'], unit='s', utc=True)
    
    is_within_morning_window = ((dt_utc.hour == 7) & (dt_utc.minute >= 45)) | \
                               ((dt_utc.hour == 8) | (dt_utc.hour == 9)) | \
                               ((dt_utc.hour == 9) & (dt_utc.minute <= 44))
    
    is_within_afternoon_window = ((dt_utc.hour == 15) & (dt_utc.minute >= 45)) | \
                                  (dt_utc.hour == 16) & (dt_utc.minute <= 44)
    
    is_within_time_window = is_within_morning_window | is_within_afternoon_window
    
    # Detect previous day high/low sweeps
    prev_day_high_swept = df['high'] > prev_day_high
    prev_day_low_swept = df['low'] < prev_day_low
    
    # Track flags - simplified version
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if is_new_day.iloc[i]:
            flagpdh.iloc[i] = False
            flagpdl.iloc[i] = False
        else:
            flagpdh.iloc[i] = flagpdh.iloc[i-1]
            flagpdl.iloc[i] = flagpdl.iloc[i-1]
            
            if prev_day_high_swept.iloc[i] and current_day_low.iloc[i] > prev_day_low.iloc[i]:
                flagpdh.iloc[i] = True
            elif prev_day_low_swept.iloc[i] and current_day_high.iloc[i] < prev_day_high.iloc[i]:
                flagpdl.iloc[i] = True
    
    # Entry conditions - long when prev day high swept, short when prev day low swept
    # Only within trading windows
    long_entry_condition = flagpdh & is_within_time_window & (df['high'] > prev_day_high)
    short_entry_condition = flagpdl & is_within_time_window & (df['low'] < prev_day_low)
    
    long_entry_condition = (prev_day_high_swept & 
                           (df['low'] < prev_day_high) & 
                           is_within_time_window)
    
    short_entry_condition = (prev_day_low_swept & 
                            (df['high'] > prev_day_low) & 
                            is_within_time_window)
    
    # Generate entries
    for i in range(1, len(df)):
        if pd.isna(prev_day_high.iloc[i]) or pd.isna(prev_day_low.iloc[i]):
            continue
        
        if is_within_time_window.iloc[i]:
            # Long entry: price sweeps previous day high then pulls back
            if prev_day_high_swept.iloc[i] and df['close'].iloc[i] < prev_day_high.iloc[i]:
                trade_num += 1
                entry_price = df['close'].iloc[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
            
            # Short entry: price sweeps previous day low then pulls back
            if prev_day_low_swept.iloc[i] and df['close'].iloc[i] > prev_day_low.iloc[i]:
                trade_num += 1
                entry_price = df['close'].iloc[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
    
    return results