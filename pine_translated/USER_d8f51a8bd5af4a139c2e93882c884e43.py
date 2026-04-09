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
    
    # Convert timestamp to datetime for time window checks
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_of_day'] = df['hour'] * 60 + df['minute']
    
    # Define time windows (London sessions)
    # Morning: 07:45 - 09:45 UTC
    # Afternoon: 14:45 - 16:45 UTC
    morning_start = 7 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    
    # Check if within time window
    is_within_time_window = (
        ((df['time_of_day'] >= morning_start) & (df['time_of_day'] < morning_end)) |
        ((df['time_of_day'] >= afternoon_start) & (df['time_of_day'] < afternoon_end))
    )
    
    # For swing detection, we need 4H data but will approximate using local extrema on 15m data
    # Swing high: high[3] < high[2] and high[1] <= high[2] and high[2] >= high[4] and high[2] >= high[5]
    # In pandas: shift(3) < shift(2) and shift(1) <= shift(2) and shift(2) >= shift(4) and shift(2) >= shift(5)
    
    is_swing_high = (
        (df['high'].shift(3) < df['high'].shift(2)) &
        (df['high'].shift(1) <= df['high'].shift(2)) &
        (df['high'].shift(2) >= df['high'].shift(4)) &
        (df['high'].shift(2) >= df['high'].shift(5))
    )
    
    # Swing low: low[3] > low[2] and low[1] >= low[2] and low[2] <= low[4] and low[2] <= low[5]
    is_swing_low = (
        (df['low'].shift(3) > df['low'].shift(2)) &
        (df['low'].shift(1) >= df['low'].shift(2)) &
        (df['low'].shift(2) <= df['low'].shift(4)) &
        (df['low'].shift(2) <= df['low'].shift(5))
    )
    
    # Track if we've seen a swing high/low recently (within last 5 bars)
    # to determine trend direction
    bullish_count = 0
    bearish_count = 0
    trend_direction = "Neutral"
    
    # Detect trend direction based on consecutive swing highs/lows
    for i in range(6, len(df)):
        if pd.isna(df['high'].iloc[i-2]):
            continue
            
        current_swing_high = (
            (df['high'].iloc[i-3] < df['high'].iloc[i-2]) &
            (df['high'].iloc[i-1] <= df['high'].iloc[i-2]) &
            (df['high'].iloc[i-2] >= df['high'].iloc[i-4]) &
            (df['high'].iloc[i-2] >= df['high'].iloc[i-5])
        )
        
        current_swing_low = (
            (df['low'].iloc[i-3] > df['low'].iloc[i-2]) &
            (df['low'].iloc[i-1] >= df['low'].iloc[i-2]) &
            (df['low'].iloc[i-2] <= df['low'].iloc[i-4]) &
            (df['low'].iloc[i-2] <= df['low'].iloc[i-5])
        )
        
        if current_swing_high:
            bullish_count += 1
            bearish_count = 0
        elif current_swing_low:
            bearish_count += 1
            bullish_count = 0
        
        if bullish_count > 1:
            trend_direction = "Bullish"
        elif bearish_count > 1:
            trend_direction = "Bearish"
        else:
            trend_direction = "Neutral"
    
    # Reset counters for actual entry generation
    bullish_count = 0
    bearish_count = 0
    
    # Generate entries based on swing detection within time window
    for i in range(6, len(df)):
        if pd.isna(df['high'].iloc[i-2]) or not is_within_time_window.iloc[i]:
            continue
        
        current_swing_high = (
            (df['high'].iloc[i-3] < df['high'].iloc[i-2]) &
            (df['high'].iloc[i-1] <= df['high'].iloc[i-2]) &
            (df['high'].iloc[i-2] >= df['high'].iloc[i-4]) &
            (df['high'].iloc[i-2] >= df['high'].iloc[i-5])
        )
        
        current_swing_low = (
            (df['low'].iloc[i-3] > df['low'].iloc[i-2]) &
            (df['low'].iloc[i-1] >= df['low'].iloc[i-2]) &
            (df['low'].iloc[i-2] <= df['low'].iloc[i-4]) &
            (df['low'].iloc[i-2] <= df['low'].iloc[i-5])
        )
        
        if current_swing_high:
            bullish_count += 1
            bearish_count = 0
        elif current_swing_low:
            bearish_count += 1
            bullish_count = 0
        
        # Determine current trend
        if bullish_count > 1:
            current_trend = "Bullish"
        elif bearish_count > 1:
            current_trend = "Bearish"
        else:
            current_trend = "Neutral"
        
        # Entry logic: trade with trend when trend is established
        if current_trend == "Bullish" and current_swing_low:
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
            
        elif current_trend == "Bearish" and current_swing_high:
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