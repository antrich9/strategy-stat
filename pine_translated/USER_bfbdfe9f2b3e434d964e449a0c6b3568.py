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
    
    # Convert time to datetime for grouping by day
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['datetime'].dt.date
    
    # Calculate previous day's high and low
    daily_high = df.groupby('day')['high'].max().shift(1)
    daily_low = df.groupby('day')['low'].min().shift(1)
    
    # Map previous day high/low back to each bar
    df['prev_day_high'] = df['day'].map(daily_high)
    df['prev_day_low'] = df['day'].map(daily_low)
    
    # Detect new day
    df['is_new_day'] = df['day'].diff().notna() & (df['day'].diff() != pd.Timedelta(0))
    df['is_new_day'].iloc[0] = True
    
    # Flag variables for sweep conditions
    flagpdl = False
    flagpdh = False
    
    # OB and FVG conditions (boolean Series)
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    
    is_ob_up = is_down.shift(-1) & is_up & (df['close'].shift(-1) > df['high'].shift(-1))
    is_ob_down = is_up.shift(-1) & is_down & (df['close'].shift(-1) < df['low'].shift(-1))
    
    is_fvg_up = df['low'] > df['high'].shift(-2)
    is_fvg_down = df['high'] < df['low'].shift(-2)
    
    plot_ob_fvg = True  # Since it's an input that defaults to true
    
    # Long and short entry variables
    long_waiting_for_entry = False
    long_fib_level = np.nan
    short_waiting_for_entry = False
    short_fib_level = np.nan
    
    # Iterate through each bar
    for i in range(len(df)):
        # Reset flags at new day
        if df['is_new_day'].iloc[i]:
            flagpdl = False
            flagpdh = False
        
        # Check for sweep
        if df['close'].iloc[i] > df['prev_day_high'].iloc[i]:
            flagpdh = True
        if df['close'].iloc[i] < df['prev_day_low'].iloc[i]:
            flagpdl = True
        
        # Get current bar conditions
        ob_up = is_ob_up.iloc[i]
        ob_down = is_ob_down.iloc[i]
        fvg_up = is_fvg_up.iloc[i]
        fvg_down = is_fvg_down.iloc[i]
        
        prev_day_high_val = df['prev_day_high'].iloc[i]
        prev_day_low_val = df['prev_day_low'].iloc[i]
        
        # Skip if NaN
        if pd.isna(prev_day_high_val) or pd.isna(prev_day_low_val):
            continue
        if pd.isna(ob_up) or pd.isna(ob_down) or pd.isna(fvg_up) or pd.isna(fvg_down):
            continue
        
        # Long Entry Logic - Stage 1: Set waiting
        if not long_waiting_for_entry and ob_up and fvg_up:
            long_waiting_for_entry = True
            long_fib_level = prev_day_low_val + (prev_day_high_val - prev_day_low_val) * 0.618
        
        # Long Entry Logic - Stage 2: Entry trigger
        if long_waiting_for_entry:
            # Check crossunder(low, longFibLevel)
            if i > 0:
                prev_low = df['low'].iloc[i-1]
                curr_low = df['low'].iloc[i]
                
                if not np.isnan(long_fib_level):
                    if prev_low >= long_fib_level and curr_low < long_fib_level:
                        entry_price = df['close'].iloc[i]
                        
                        results.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(df['time'].iloc[i]),
                            'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                            'entry_price_guess': entry_price,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': entry_price,
                            'raw_price_b': entry_price
                        })
                        
                        trade_num += 1
                        long_waiting_for_entry = False
                        long_fib_level = np.nan
        
        # Short Entry Logic - Stage 1: Set waiting
        if not short_waiting_for_entry and ob_down and fvg_down:
            short_waiting_for_entry = True
            short_fib_level = prev_day_high_val - (prev_day_high_val - prev_day_low_val) * 0.618
        
        # Short Entry Logic - Stage 2: Entry trigger
        if short_waiting_for_entry:
            # Check crossover(high, shortFibLevel)
            if i > 0:
                prev_high = df['high'].iloc[i-1]
                curr_high = df['high'].iloc[i]
                
                if not np.isnan(short_fib_level):
                    if prev_high <= short_fib_level and curr_high > short_fib_level:
                        entry_price = df['close'].iloc[i]
                        
                        results.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': int(df['time'].iloc[i]),
                            'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                            'entry_price_guess': entry_price,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': entry_price,
                            'raw_price_b': entry_price
                        })
                        
                        trade_num += 1
                        short_waiting_for_entry = False
                        short_fib_level = np.nan
    
    return results