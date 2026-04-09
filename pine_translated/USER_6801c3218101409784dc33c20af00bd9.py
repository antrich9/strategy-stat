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
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['time_dt'].dt.date
    
    # Get previous day high and low
    daily_agg = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    daily_agg = daily_agg[['date', 'prev_day_high', 'prev_day_low']]
    df = df.merge(daily_agg, on='date', how='left')
    
    # Time filters - 0700-0959 for long, 1200-1459 for short
    df['hour'] = df['time_dt'].dt.hour
    df['minute'] = df['time_dt'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['is_long_window'] = (df['time_minutes'] >= 420) & (df['time_minutes'] <= 599)  # 07:00-09:59
    df['is_short_window'] = (df['time_minutes'] >= 720) & (df['time_minutes'] <= 899)  # 12:00-14:59
    
    # Fair Value Gap detection
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    # Order Block detection
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    df['ob_up'] = df['is_down'].shift(1) & df['is_up'] & (df['close'] > df['high'].shift(1))
    df['ob_down'] = df['is_up'].shift(1) & df['is_down'] & (df['close'] < df['low'].shift(1))
    
    # Stacked OB + FVG
    df['stacked_bullish'] = df['ob_up'] & df['fvg_up']
    df['stacked_bearish'] = df['ob_down'] & df['fvg_down']
    
    # Detect PDH/PDL sweeps using crossover/crossunder
    df['close_above_pdh'] = df['close'] > df['prev_day_high']
    df['close_below_pdl'] = df['close'] < df['prev_day_low']
    df['pdh_swept'] = df['close_above_pdh'] & (~df['close_above_pdh'].shift(1).fillna(False))
    df['pdl_swept'] = df['close_below_pdl'] & (~df['close_below_pdl'].shift(1).fillna(False))
    
    # Track flags across bars
    flagpdh = False
    flagpdl = False
    prev_date = None
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        current_date = df['date'].iloc[i]
        
        # Reset flags on new day
        if prev_date is not None and current_date != prev_date:
            flagpdh = False
            flagpdl = False
        
        # Update sweep flags
        if df['pdh_swept'].iloc[i]:
            flagpdh = True
        if df['pdl_swept'].iloc[i]:
            flagpdl = True
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        # Long entry: stacked bullish, PDL swept, in long time window
        if df['stacked_bullish'].iloc[i] and flagpdl and df['is_long_window'].iloc[i]:
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
            flagpdl = False  # Reset after entry
        
        # Short entry: stacked bearish, PDH swept, in short time window
        if df['stacked_bearish'].iloc[i] and flagpdh and df['is_short_window'].iloc[i]:
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
            flagpdh = False  # Reset after entry
        
        prev_date = current_date
    
    return entries