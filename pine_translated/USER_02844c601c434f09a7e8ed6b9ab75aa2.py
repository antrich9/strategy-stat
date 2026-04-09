import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['datetime'].dt.date
    
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    daily['prev_day_close'] = daily['close'].shift(1)
    
    daily['bias'] = 0
    daily.loc[daily['prev_day_close'] > daily['prev_day_high'], 'bias'] = 1
    daily.loc[daily['prev_day_close'] < daily['prev_day_low'], 'bias'] = -1
    
    df = df.merge(daily[['date', 'bias', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    prev_day_high = df['prev_day_high']
    prev_day_low = df['prev_day_low']
    bias = df['bias']
    
    short_raid = (bias == -1) & (prev_day_high.notna()) & \
                 (df['high'] > prev_day_high) & \
                 (prev_day_high.shift(1).notna()) & \
                 (df['high'].shift(1) <= prev_day_high.shift(1))
    
    long_raid = (bias == 1) & (prev_day_low.notna()) & \
                (df['low'] < prev_day_low) & \
                (prev_day_low.shift(1).notna()) & \
                (df['low'].shift(1) >= prev_day_low.shift(1))
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if pd.isna(prev_day_high.iloc[i]) or pd.isna(prev_day_low.iloc[i]):
            continue
        
        direction = None
        if short_raid.iloc[i]:
            direction = 'short'
        elif long_raid.iloc[i]:
            direction = 'long'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries