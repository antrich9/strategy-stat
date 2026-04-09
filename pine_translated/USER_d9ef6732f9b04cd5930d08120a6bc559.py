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
    trade_num = 1
    
    # Convert time to datetime for timezone operations
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Detect new days
    df['day'] = df['dt'].dt.date
    
    # Previous day high/low - shift by 1 to get previous day values
    df['prev_day_high'] = df['high'].shift(1)
    df['prev_day_low'] = df['low'].shift(1)
    
    # Detect when prev day high/low is swept
    df['prevdh_swept'] = df['close'] > df['prev_day_high']
    df['prevdl_swept'] = df['close'] < df['prev_day_low']
    
    # Forward-fill swept flags (persist until reset)
    df['flagpdh'] = df['prevdh_swept'].cummax()
    df['flagpdl'] = df['prevdl_swept'].cummax()
    
    # Reset flags on new day
    df['new_day'] = df['day'] != df['day'].shift(1)
    df.loc[df['new_day'], 'flagpdh'] = False
    df.loc[df['new_day'], 'flagpdl'] = False
    
    # Reapply current bar sweep
    df['flagpdh'] = df['prevdh_swept'] | (df['flagpdh'].shift(1).fillna(False) & ~df['new_day'])
    df['flagpdl'] = df['prevdl_swept'] | (df['flagpdl'].shift(1).fillna(False) & ~df['new_day'])
    
    # Order Block detection
    # isObUp: isDown(i+1) and isUp(i) and close[i] > high[i+1]
    # isDown: close < open
    # isUp: close > open
    isDown_i1 = df['close'].shift(1) < df['open'].shift(1)
    isUp_i = df['close'] > df['open']
    close_gt_high_i1 = df['close'] > df['high'].shift(1)
    df['obUp'] = isDown_i1.shift(1) & isUp_i & close_gt_high_i1.shift(1)
    
    isUp_i1 = df['close'].shift(1) > df['open'].shift(1)
    isDown_i = df['close'] < df['open']
    close_lt_low_i1 = df['close'] < df['low'].shift(1)
    df['obDown'] = isUp_i1.shift(1) & isDown_i & close_lt_low_i1.shift(1)
    
    # FVG detection
    # isFvgUp: low[i] > high[i+2]
    df['fvgUp'] = df['low'] > df['high'].shift(2)
    # isFvgDown: high[i] < low[i+2]
    df['fvgDown'] = df['high'] < df['low'].shift(2)
    
    # Time filter - bars between 07:00-09:59 and 12:00-14:59 UTC
    df['hour'] = df['dt'].dt.hour
    time_filter_long = ((df['hour'] >= 7) & (df['hour'] < 10))
    time_filter_short = ((df['hour'] >= 12) & (df['hour'] < 15))
    
    # Combined entry conditions
    df['long_condition'] = (df['fvgUp']) & (df['obUp']) & (df['flagpdl']) & (time_filter_long)
    df['short_condition'] = (df['fvgDown']) & (df['obDown']) & (df['flagpdh']) & (time_filter_short)
    
    # Reset flags on new day to prevent entries on first bar
    df.loc[df['new_day'], 'long_condition'] = False
    df.loc[df['new_day'], 'short_condition'] = False
    
    # Need minimum 3 bars for FVG/OB calculation
    min_bars = max(3, df['obUp'].notna().idxmax() if df['obUp'].notna().any() else 3)
    
    # Iterate and generate entries
    for i in range(min_bars, len(df)):
        if pd.isna(df['obUp'].iloc[i]) or pd.isna(df['fvgUp'].iloc[i]):
            continue
            
        if df['long_condition'].iloc[i]:
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
            trade_num += 1
            
        elif df['short_condition'].iloc[i]:
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
            trade_num += 1
    
    return results