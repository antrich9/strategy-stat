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
    
    n = len(df)
    if n < 3:
        return entries
    
    # Convert unix timestamp to datetime for time filtering
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # Helper function to check if time is within session
    def in_session(time_minutes, session_start_str, session_end_str):
        start_h, start_m = map(int, session_start_str.split(':'))
        end_h, end_m = map(int, session_end_str.split(':'))
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        return start <= time_minutes <= end
    
    # Calculate previous day high/low flags
    # In Pine Script, high[1] and low[1] refer to previous bar values
    # prevDayHigh and prevDayLow are fetched from daily timeframe
    # For simplicity, we use rolling window to approximate daily high/low
    # As we don't have daily data, we use a 24-period approximation or rolling max/min
    
    # Calculate rolling 24-bar high and low as proxy for previous day high/low
    df['prev_day_high_proxy'] = df['high'].shift(1).rolling(window=24, min_periods=24).max().shift(1)
    df['prev_day_low_proxy'] = df['low'].shift(1).rolling(window=24, min_periods=24).min().shift(1)
    
    # Detect when previous day high is swept (close crosses above)
    df['high_swept'] = (df['close'] > df['prev_day_high_proxy']) & (df['close'].shift(1) <= df['prev_day_high_proxy'])
    
    # Detect when previous day low is swept (close crosses below)
    df['low_swept'] = (df['close'] < df['prev_day_low_proxy']) & (df['close'].shift(1) >= df['prev_day_low_proxy'])
    
    # Carry forward the swept flags until new day (reset handled via daily change detection)
    df['is_new_day'] = df['datetime'].dt.date != df['datetime'].dt.date.shift(1)
    
    # Calculate FVG conditions
    # Bullish FVG: low > high of 2 bars ago
    df['fvg_bull'] = df['low'] > df['high'].shift(2)
    
    # Bearish FVG: high < low of 2 bars ago
    df['fvg_bear'] = df['high'] < df['low'].shift(2)
    
    # Calculate Order Block conditions
    # Bullish OB: down bar followed by up bar, with close above previous high
    df['ob_bull'] = ((df['close'].shift(1) < df['open'].shift(1)) & 
                    (df['close'] > df['open']) & 
                    (df['close'] > df['high'].shift(1)))
    
    # Bearish OB: up bar followed by down bar, with close below previous low
    df['ob_bear'] = ((df['close'].shift(1) > df['open'].shift(1)) & 
                    (df['close'] < df['open']) & 
                    (df['close'] < df['low'].shift(1)))
    
    # Combined stacked OB+FVG signals
    df['bullish_stacked'] = df['ob_bull'] & df['fvg_bull']
    df['bearish_stacked'] = df['ob_bear'] & df['fvg_bear']
    
    # Process entries
    for i in range(2, n):
        # Skip if any required values are NaN
        if (pd.isna(df['prev_day_high_proxy'].iloc[i]) or 
            pd.isna(df['prev_day_low_proxy'].iloc[i]) or
            pd.isna(df['high'].iloc[i]) or
            pd.isna(df['low'].iloc[i]) or
            pd.isna(df['high'].iloc[i-2]) or
            pd.isna(df['low'].iloc[i-2])):
            continue
        
        current_time_min = df['time_minutes'].iloc[i]
        
        # LONG ENTRY CONDITIONS:
        # 1. Previous day high swept (close > prevDayHigh)
        # 2. Bullish stacked OB+FVG present
        # 3. Within long session (0700-0959)
        
        if in_session(current_time_min, '0700', '0959'):
            high_swept = df['close'].iloc[i] > df['prev_day_high_proxy'].iloc[i]
            
            if high_swept and df['bullish_stacked'].iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
        
        # SHORT ENTRY CONDITIONS:
        # 1. Previous day low swept (close < prevDayLow)
        # 2. Bearish stacked OB+FVG present
        # 3. Within short session (1200-1459)
        
        if in_session(current_time_min, '1200', '1459'):
            low_swept = df['close'].iloc[i] < df['prev_day_low_proxy'].iloc[i]
            
            if low_swept and df['bearish_stacked'].iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
    
    return entries