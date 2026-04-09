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
    
    # Convert time to datetime for grouping by day
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['datetime'].dt.date
    
    # Calculate previous day high and low
    df['prev_day_high'] = df['high'].shift(1).where(df['day'] != df['day'].shift(1))
    df['prev_day_high'] = df['prev_day_high'].ffill()
    
    df['prev_day_low'] = df['low'].shift(1).where(df['day'] != df['day'].shift(1))
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Flag conditions for PDH and PDL breaks
    df['flagpdh'] = df['high'] > df['prev_day_high']
    df['flagpdl'] = df['low'] < df['prev_day_low']
    
    # OB (Order Block) detection
    # isUp: close > open (bullish candle)
    # isDown: close < open (bearish candle)
    # isObUp: isDown(1) and isUp(0) and close(0) > high(1)
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    
    df['ob_up'] = (df['is_down'].shift(1)) & (df['is_up']) & (df['close'] > df['high'].shift(1))
    df['ob_down'] = (df['is_up'].shift(1)) & (df['is_down']) & (df['close'] < df['low'].shift(1))
    
    # FVG (Fair Value Gap) detection
    # isFvgUp: low(0) > high(2)
    # isFvgDown: high(0) < low(2)
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    # Stacked OB + FVG conditions
    df['bullish_entry'] = df['ob_up'] & df['fvg_up']
    df['bearish_entry'] = df['ob_down'] & df['fvg_down']
    
    # Alternative entries based on PDH/PDL breaks with FVG confirmation
    df['bullish_pdh_break'] = df['flagpdh'] & df['fvg_up'].shift(1)
    df['bearish_pdl_break'] = df['flagpdl'] & df['fvg_down'].shift(1)
    
    # Combine all entry conditions
    df['final_long_entry'] = df['bullish_entry'] | df['bullish_pdh_break']
    df['final_short_entry'] = df['bearish_entry'] | df['bearish_pdl_break']
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip if indicators are NaN or invalid
        if pd.isna(df['close'].iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        
        # Check long entry
        if df['final_long_entry'].iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        # Check short entry
        if df['final_short_entry'].iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries