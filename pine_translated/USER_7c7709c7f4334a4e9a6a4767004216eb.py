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
    trade_num = 0
    
    if len(df) < 3:
        return entries
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    # FVG Detection parameters (default values from inputs)
    bullish_fvg_cond = pd.Series(False, index=df.index)
    bearish_fvg_cond = pd.Series(False, index=df.index)
    
    # Bullish FVG: candle 2 body between candle 1 and candle 3 wicks (up direction)
    # body_method: close.iloc[i-1] > low.iloc[i-2] and close.iloc[i-1] < high.iloc[i] and open.iloc[i-1] > low.iloc[i-2] and open.iloc[i-1] < high.iloc[i]
    # wicks_method: low.iloc[i] < low.iloc[i-2] and high.iloc[i-1] > high.iloc[i]
    
    # Detect FVGs starting from bar 2
    for i in range(2, len(df)):
        # Bullish FVG detection (body method)
        bull_body = (min(open_price.iloc[i-1], close.iloc[i-1]) > min(open_price.iloc[i-2], close.iloc[i-2])) and \
                    (max(open_price.iloc[i-1], close.iloc[i-1]) < max(open_price.iloc[i], close.iloc[i]))
        
        # Bullish FVG detection (wicks method)
        bull_wicks = (low.iloc[i] < low.iloc[i-2]) and (high.iloc[i-1] > high.iloc[i])
        
        # Bearish FVG detection (body method)
        bear_body = (min(open_price.iloc[i-1], close.iloc[i-1]) < max(open_price.iloc[i-2], close.iloc[i-2])) and \
                    (max(open_price.iloc[i-1], close.iloc[i-1]) > min(open_price.iloc[i], close.iloc[i]))
        
        # Bearish FVG detection (wicks method)
        bear_wicks = (high.iloc[i] > high.iloc[i-2]) and (low.iloc[i-1] < low.iloc[i])
        
        bullish_fvg_cond.iloc[i] = bull_body or bull_wicks
        bearish_fvg_cond.iloc[i] = bear_body or bear_wicks
    
    # Generate entries based on detected FVGs
    for i in range(len(df)):
        if bullish_fvg_cond.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
        
        if bearish_fvg_cond.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
    
    return entries