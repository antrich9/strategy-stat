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
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Convert time to datetime for date operations
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # Detect new day
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    
    # Calculate previous day's high and low using shift
    # For each day, we need the high/low of the previous calendar day
    df['prev_day_high'] = df['high'].shift(1)
    df['prev_day_low'] = df['low'].shift(1)
    
    # When is_new_day is True, the prev_day_high/low should be from 2 bars ago (previous calendar day)
    # On first bar of a new day, shift(1) gives the same day's previous bar, not previous day
    # Fix: recalculate using groupby
    for i in range(1, len(df)):
        if df['is_new_day'].iloc[i]:
            df.loc[df.index[i], 'prev_day_high'] = df['high'].iloc[i - 1]
            df.loc[df.index[i], 'prev_day_low'] = df['low'].iloc[i - 1]
    
    # Forward fill prev day H/L within each day (they should be constant for the day)
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Detect sweeps of previous day's high and low
    # Sweep high: close crosses above prev_day_high
    df['sweep_high'] = (df['close'] > df['prev_day_high']) & (df['close'].shift(1) <= df['prev_day_high'])
    # Sweep low: close crosses below prev_day_low
    df['sweep_low'] = (df['close'] < df['prev_day_low']) & (df['close'].shift(1) >= df['prev_day_low'])
    
    # Track sweep flags - reset on new day
    flagpdh = False
    flagpdl = False
    
    sweep_high_series = pd.Series(False, index=df.index)
    sweep_low_series = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        if df['is_new_day'].iloc[i]:
            flagpdh = False
            flagpdl = False
        
        if df['sweep_high'].iloc[i]:
            flagpdh = True
        if df['sweep_low'].iloc[i]:
            flagpdl = True
            
        sweep_high_series.iloc[i] = flagpdh
        sweep_low_series.iloc[i] = flagpdl
    
    df['high_swept'] = sweep_high_series
    df['low_swept'] = sweep_low_series
    
    # Calculate EMAs for trend confirmation
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate ATR (Wilder's method using EWM)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/200, adjust=False).mean()
    
    # Detect bullish and bearish FVG (Fair Value Gap)
    # FVG: 3-bar pattern where middle bar has a gap between high and low of adjacent bars
    # Bullish FVG: low of bar 3 > high of bar 1
    # Bearish FVG: high of bar 3 < low of bar 1
    df['bull_fvg'] = (df['low'].shift(-2) > df['high'].shift(0)) & \
                     ((df['high'].shift(0) - df['low'].shift(-2)) > 0)
    df['bear_fvg'] = (df['high'].shift(-2) < df['low'].shift(0)) & \
                     ((df['low'].shift(0) - df['high'].shift(-2)) > 0)
    
    # Entry conditions based on ICT concepts:
    # Long: Previous day low swept (liquidity grab) + bullish FVG + price above EMA50
    # Short: Previous day high swept (liquidity grab) + bearish FVG + price below EMA50
    df['long_condition'] = df['low_swept'] & df['bull_fvg'] & (df['close'] > df['ema50'])
    df['short_condition'] = df['high_swept'] & df['bear_fvg'] & (df['close'] < df['ema50'])
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Track if we're already in a position to avoid double entries
    in_position = False
    
    for i in range(len(df)):
        # Skip if ATR is NaN (not enough data)
        if pd.isna(df['atr'].iloc[i]):
            continue
        # Skip if EMAs are NaN
        if pd.isna(df['ema50'].iloc[i]) or pd.isna(df['ema200'].iloc[i]):
            continue
            
        direction = None
        
        if df['long_condition'].iloc[i] and not in_position:
            direction = 'long'
            in_position = True
        elif df['short_condition'].iloc[i] and not in_position:
            direction = 'short'
            in_position = True
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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
        
        # Reset position flag on new day
        if df['is_new_day'].iloc[i]:
            in_position = False
    
    return entries