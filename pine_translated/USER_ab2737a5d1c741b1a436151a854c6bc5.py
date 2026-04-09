import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Prepare time-based filters
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    time_filter = (df['hour'] >= 10) & (df['hour'] < 12)
    
    # Compute bullish breakaway FVG condition
    bfvg = (df['low'] > df['high'].shift(2))
    
    # Compute bearish breakaway FVG condition
    sfvg = (df['high'] < df['low'].shift(2))
    
    # Apply filters and time condition
    long_condition = bfvg & time_filter
    short_condition = sfvg & time_filter
    
    # Prepare entry list
    entries = []
    trade_num = 1
    
    # Process long entries
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
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
    
    # Process short entries
    for i in range(len(df)):
        if short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
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
    
    return entries