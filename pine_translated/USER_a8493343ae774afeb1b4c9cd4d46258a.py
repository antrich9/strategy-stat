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
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Detect new days
    df['day'] = df['time'].dt.date
    df['is_new_day'] = df['day'] != df['day'].shift(1)
    
    # Get previous day high and low using shift
    df['prev_day_high'] = df['high'].shift(1).where(df['is_new_day'].shift(1))
    df['prev_day_low'] = df['low'].shift(1).where(df['is_new_day'].shift(1))
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Sweep flags
    df['pdh_swept'] = df['close'] > df['prev_day_high']
    df['pdl_swept'] = df['close'] < df['prev_day_low']
    
    # Forward fill sweep flags within day
    df['flagpdh'] = df.groupby('day')['pdh_swept'].cummax()
    df['flagpdl'] = df.groupby('day')['pdl_swept'].cummax()
    
    # OB and FVG detection
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    
    df['ob_up'] = (df['is_down'].shift(1)) & (df['is_up']) & (df['close'] > df['high'].shift(1))
    df['ob_down'] = (df['is_up'].shift(1)) & (df['is_down']) & (df['close'] < df['low'].shift(1))
    
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    df['stacked_bullish'] = df['ob_up'].shift(1) & df['fvg_up']
    df['stacked_bearish'] = df['ob_down'].shift(1) & df['fvg_down']
    
    # Fibonacci levels based on previous swing (using recent high/low as proxy)
    df['swing_high'] = df['high'].rolling(20).max().shift(1)
    df['swing_low'] = df['low'].rolling(20).min().shift(1)
    df['fib618'] = df['swing_low'] + (df['swing_high'] - df['swing_low']) * 0.618
    df['fib786'] = df['swing_low'] + (df['swing_high'] - df['swing_low']) * 0.786
    
    # Entry conditions
    # Long: PDL swept + stacked bullish OB/FVG + price near fib618
    # Short: PDH swept + stacked bearish OB/FVG + price near fib786
    
    df['bullish_fib_zone'] = (df['close'] >= df['fib618'] * 0.99) & (df['close'] <= df['fib618'] * 1.01)
    df['bearish_fib_zone'] = (df['close'] >= df['fib786'] * 0.99) & (df['close'] <= df['fib786'] * 1.01)
    
    df['long_entry'] = df['flagpdl'] & df['stacked_bullish'] & df['bullish_fib_zone']
    df['short_entry'] = df['flagpdh'] & df['stacked_bearish'] & df['bearish_fib_zone']
    
    # Track in position state to avoid duplicate entries
    in_long = False
    in_short = False
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Skip if indicators are NaN
        if pd.isna(row['fib618']) or pd.isna(row['fib786']):
            continue
        if pd.isna(row['stacked_bullish']) or pd.isna(row['stacked_bearish']):
            continue
            
        entry_price = row['close']
        ts = df['time'].iloc[i].timestamp()
        
        # Long entry
        if row['long_entry'] and not in_long:
            in_long = True
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_short = False  # Reset short on long entry
            
        # Short entry
        elif row['short_entry'] and not in_short:
            in_short = True
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            in_long = False  # Reset long on short entry
    
    return entries