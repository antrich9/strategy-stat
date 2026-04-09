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
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Get previous day high and low using daily resample
    df['day'] = df['time'].dt.date
    daily_agg = df.groupby('day').agg({
        'high': 'max',
        'low': 'min',
        'time': 'first'
    }).reset_index(drop=True)
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    df = df.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    
    # Detect new day
    df['is_new_day'] = df['day'] != df['day'].shift(1)
    
    # Previous day high/low sweep detection
    df['flagpdh'] = df['close'] > df['prev_day_high']
    df['flagpdl'] = df['close'] < df['prev_day_low']
    
    # Order Block conditions
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    df['ob_up'] = (df['is_down'].shift(1)) & (df['is_up']) & (df['close'] > df['high'].shift(1))
    df['ob_down'] = (df['is_up'].shift(1)) & (df['is_down']) & (df['close'] < df['low'].shift(1))
    
    # Fair Value Gap conditions
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    # Stacked OB + FVG
    df['bullish_stack'] = df['ob_up'] & df['fvg_up']
    df['bearish_stack'] = df['ob_down'] & df['fvg_down']
    
    # Reset flags at start of new day
    df.loc[df['is_new_day'], 'flagpdh'] = False
    df.loc[df['is_new_day'], 'flagpdl'] = False
    
    # Forward fill flags within day
    df['flagpdh'] = df.groupby('day')['flagpdh'].cummax()
    df['flagpdl'] = df.groupby('day')['flagpdl'].cummax()
    
    # Create entry conditions (after sweep)
    df['long_cond'] = df['bullish_stack'] & df['flagpdh']
    df['short_cond'] = df['bearish_stack'] & df['flagpdl']
    
    # Entry signals - when condition becomes true
    df['long_entry'] = df['long_cond'] & (~df['long_cond'].shift(1).fillna(False))
    df['short_entry'] = df['short_cond'] & (~df['short_cond'].shift(1).fillna(False))
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['long_entry'].iloc[i]:
            ts = int(df['time'].iloc[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif df['short_entry'].iloc[i]:
            ts = int(df['time'].iloc[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries