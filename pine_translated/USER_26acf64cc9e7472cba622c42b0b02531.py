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
    
    # Resample to 4H candles
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('datetime').resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h = df_4h.reset_index()
    
    if len(df_4h) < 3:
        return []
    
    # Calculate 4H FVG conditions
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    # Bullish FVG: low > high[2]
    bfvg = low_4h > high_4h.shift(2)
    # Bearish FVG: high < low[2]
    sfvg = high_4h < low_4h.shift(2)
    
    # Detect new 4H candles
    is_new_4h = df_4h['datetime'].dt.hour.shift(1) != df_4h['datetime'].dt.hour
    is_new_4h.iloc[0] = True
    
    entries = []
    trade_num = 1
    last_fvg = 0
    
    # Iterate through 4H candles (skip first 2 for FVG calculation)
    for i in range(2, len(df_4h)):
        current_fvg = 0
        if bfvg.iloc[i]:
            current_fvg = 1
        elif sfvg.iloc[i]:
            current_fvg = -1
        
        # Entry logic: Sharp Turn (FVG flip)
        entry_direction = None
        if bfvg.iloc[i] and last_fvg == -1:
            entry_direction = 'long'
        elif sfvg.iloc[i] and last_fvg == 1:
            entry_direction = 'short'
        
        if entry_direction:
            ts = int(df_4h['datetime'].iloc[i].timestamp())
            entry = {
                'trade_num': trade_num,
                'direction': entry_direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df_4h['close'].iloc[i]),
                'raw_price_b': float(df_4h['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
        
        # Update FVG state if current bar has FVG
        if current_fvg != 0:
            last_fvg = current_fvg
    
    return entries