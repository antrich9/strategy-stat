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

    # Time-based day detection using timezone-aware timestamps
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Detect new trading days (9:30 AM ET)
    # Convert to ET (UTC-5 or UTC-4 depending on DST - using simple approach)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day'] = df['datetime'].dt.day
    
    # New trading day at 9:30
    new_trading_day = (df['hour'] == 9) & (df['minute'] == 30)
    
    # Calculate PDH and PDL
    df['pdh'] = np.nan
    df['pdl'] = np.nan
    
    # Track daily high/low
    daily_high = df['high'].copy()
    daily_low = df['low'].copy()
    
    # For each row, get previous day's high and low
    for i in range(len(df)):
        if i > 0:
            prev_day_end = i - 1
            while prev_day_end >= 0 and df['day'].iloc[prev_day_end] == df['day'].iloc[i]:
                prev_day_end -= 1
            if prev_day_end >= 0:
                df.loc[df.index[i], 'pdh'] = df['high'].iloc[prev_day_end]
                df.loc[df.index[i], 'pdl'] = df['low'].iloc[prev_day_end]
    
    # State variables
    swept_high = False
    swept_low = False
    found_fvg = False
    trade_today = False
    first_sweep_fvg_taken = False
    
    # For entries, we need to check conditions at each bar
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Reset daily flags at market open (09:30)
        if row['hour'] == 9 and row['minute'] == 30:
            swept_high = False
            swept_low = False
            found_fvg = False
            trade_today = False
            first_sweep_fvg_taken = False
        
        pdh = row['pdh']
        pdl = row['pdl']
        
        # Check PDH/PDL sweep
        if not np.isnan(pdh) and not swept_high and row['high'] > pdh:
            swept_high = True
        
        if not np.isnan(pdl) and not swept_low and row['low'] < pdl:
            swept_low = True
        
        # FVG Detection (simplified)
        if i >= 2:
            bull_fvg = row['low'] > df['high'].iloc[i-2] and df['low'].iloc[i-1] > df['high'].iloc[i-2]
            bear_fvg = row['high'] < df['low'].iloc[i-2] and df['high'].iloc[i-1] < df['low'].iloc[i-2]
            
            # Entry conditions
            show_bull_fvg = (swept_high or swept_low) and bull_fvg and not found_fvg and not first_sweep_fvg_taken
            show_bear_fvg = (swept_high or swept_low) and bear_fvg and not found_fvg and not first_sweep_fvg_taken
            
            if show_bull_fvg:
                found_fvg = True
                entry_price = row['close']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
            
            if show_bear_fvg:
                found_fvg = True
                entry_price = row['close']
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(row['time']),
                    'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
    
    return entries