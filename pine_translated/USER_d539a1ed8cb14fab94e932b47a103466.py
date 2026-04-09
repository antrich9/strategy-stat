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
    
    bull_fvg = df['low'] > df['high'].shift(2)
    bear_fvg = df['high'] < df['low'].shift(2)
    
    bull_zone_top = df['high'].shift(2)
    bear_zone_bottom = df['low'].shift(2)
    
    bull_zones = {}
    bear_zones = {}
    
    entries = []
    trade_num = 1
    
    for i in range(5, len(df)):
        current_ts = df['time'].iloc[i]
        entry_price = df['close'].iloc[i]
        entry_time = datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat()
        
        if bull_fvg.iloc[i-2]:
            bull_zones[i-2] = bull_zone_top.iloc[i-2]
        
        if bear_fvg.iloc[i-2]:
            bear_zones[i-2] = bear_zone_bottom.iloc[i-2]
        
        for det_idx in list(bull_zones.keys()):
            if df['low'].iloc[i] < bull_zones[det_idx]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': current_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                del bull_zones[det_idx]
        
        for det_idx in list(bear_zones.keys()):
            if df['high'].iloc[i] > bear_zones[det_idx]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': current_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1