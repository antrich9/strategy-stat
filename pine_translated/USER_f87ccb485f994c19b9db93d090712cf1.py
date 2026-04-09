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
    
    # Define London time windows
    london_morning_start = 7 * 3600 + 45 * 60  # 7:45 UTC
    london_morning_end = 9 * 3600 + 45 * 60    # 9:45 UTC
    london_afternoon_start = 14 * 3600 + 45 * 60  # 14:45 UTC
    london_afternoon_end = 16 * 3600 + 45 * 60   # 16:45 UTC
    
    # Extract hour from timestamp (assuming timestamp is in seconds)
    hour = (df['time'] % 86400) // 3600
    minute = (df['time'] % 3600) // 60
    total_minutes = hour * 60 + minute
    
    # Create boolean Series for trading windows
    is_within_morning_window = (total_minutes >= (7 * 60 + 45)) & (total_minutes < (9 * 60 + 45))
    is_within_afternoon_window = (total_minutes >= (14 * 60 + 45)) & (total_minutes < (16 * 60 + 45))
    in_trading_window = is_within_morning_window | is_within_afternoon_window
    
    # Initialize variables
    last_swing_high = np.nan
    last_swing_low = np.nan
    bullish_count = 0
    bearish_count = 0
    
    entries = []
    trade_num = 1
    
    # Iterate through bars starting from index 5 for swing detection
    for i in range(5, len(df)):
        is_swing_high = False
        is_swing_low = False
        
        # Swing detection using 4H data (using current close as proxy)
        # Swing High: high[i-2] is peak
        if (df['close'].iloc[i-3] < df['close'].iloc[i-2] and 
            df['close'].iloc[i-1] <= df['close'].iloc[i-2] and 
            df['close'].iloc[i-2] >= df['close'].iloc[i-4] and 
            df['close'].iloc[i-2] >= df['close'].iloc[i-5]):
            is_swing_high = True
        
        # Swing Low: low[i-2] is trough
        if (df['close'].iloc[i-3] > df['close'].iloc[i-2] and 
            df['close'].iloc[i-1] >= df['close'].iloc[i-2] and 
            df['close'].iloc[i-2] <= df['close'].iloc[i-4] and 
            df['close'].iloc[i-2] <= df['close'].iloc[i-5]):
            is_swing_low = True
        
        # Update swing points and counters
        if is_swing_high:
            last_swing_high = df['close'].iloc[i-2]
            bullish_count += 1
            bearish_count = 0
        
        if is_swing_low:
            last_swing_low = df['close'].iloc[i-2]
            bearish_count += 1
            bullish_count = 0
        
        # Determine trend direction
        trend = "Neutral"
        if bullish_count > 1:
            trend = "Bullish"
        elif bearish_count > 1:
            trend = "Bearish"
        
        # Entry logic based on trend and time window
        if trend == "Bullish" and in_trading_window.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif trend == "Bearish" and in_trading_window.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries