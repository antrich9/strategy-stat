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
    
    # Input parameters (default values from Pine Script)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Time window parameters (London time)
    morning_start_hour, morning_start_min = 7, 45
    morning_end_hour, morning_end_min = 9, 45
    afternoon_start_hour, afternoon_start_min = 14, 45
    afternoon_end_hour, afternoon_end_min = 16, 45
    
    # Helper functions for time window
    def is_within_time_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Morning window: 07:45 - 09:45
        in_morning = (hour > morning_start_hour or (hour == morning_start_hour and minute >= morning_start_min)) and \
                     (hour < morning_end_hour or (hour == morning_end_hour and minute <= morning_end_min))
        
        # Afternoon window: 14:45 - 16:45
        in_afternoon = (hour > afternoon_start_hour or (hour == afternoon_start_hour and minute >= afternoon_start_min)) and \
                       (hour < afternoon_end_hour or (hour == afternoon_end_hour and minute <= afternoon_end_min))
        
        return in_morning or in_afternoon
    
    # Calculate indicators
    # Volume filter
    if inp1:
        vol_sma = df['volume'].rolling(9).mean()
        volfilt = df['volume'].shift(1) > vol_sma * 1.5
    else:
        volfilt = pd.Series(True, index=df.index)
    
    # ATR filter (Wilder ATR)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    if inp2:
        atrfilt_long = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
        atrfilt_short = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt_long
        atrfilt_long = (df['low'] > df['high'].shift(2)) & volfilt
        atrfilt = atrfilt_long | atrfilt_short
    else:
        atrfilt = pd.Series(True, index=df.index)
    
    # Location filter (trend)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    if inp3:
        locfilt_long = loc2
        locfilt_short = ~loc2
    else:
        locfilt_long = pd.Series(True, index=df.index)
        locfilt_short = pd.Series(True, index=df.index)
    
    # FVG detection
    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfilt_long
    
    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilt_short
    
    # Time window check
    time_window = df['time'].apply(is_within_time_window)
    
    # Combined entry conditions
    long_condition = bfvg & time_window
    short_condition = sfvg & time_window
    
    # Previous day high/low (using 1-day shift for previous day values)
    # Since we don't have higher timeframe data, we'll skip this or approximate
    # For a proper implementation, we'd need to resample to daily timeframe
    
    # Iterate through bars to generate entries
    for i in range(len(df)):
        # Skip if indicators are NaN
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            results.append({
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
        
        if short_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            results.append({
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
    
    return results