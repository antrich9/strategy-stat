import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['day'] = df['datetime'].dt.date
    
    daily_agg = df.groupby('day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg.columns = ['day', 'day_high', 'day_low']
    daily_agg['prev_day_high'] = daily_agg['day_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['day_low'].shift(1)
    df = df.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    df['new_day'] = df['day'] != df['day'].shift(1)
    df['swept_low'] = df['low'] < df['prev_day_low']
    df['swept_high'] = df['high'] > df['prev_day_high']
    df['broke_high'] = df['close'] > df['prev_day_high']
    df['broke_low'] = df['close'] < df['prev_day_low']
    
    bias = 0
    df['bias'] = 0.0
    
    for i in range(1, len(df)):
        if df['new_day'].iloc[i]:
            bias = 0
        
        if df['swept_low'].iloc[i] and df['broke_high'].iloc[i]:
            bias = 1
        elif df['swept_high'].iloc[i] and df['broke_low'].iloc[i]:
            bias = -1
        elif df['low'].iloc[i] < df['prev_day_low'].iloc[i]:
            bias = -1
        elif df['high'].iloc[i] > df['prev_day_high'].iloc[i]:
            bias = 1
        
        df.at[i, 'bias'] = bias
    
    for i in range(1, len(df)):
        if df.at[i, 'bias'] == 1 and df.at[i, 'broke_high']:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif df.at[i, 'bias'] == -1 and df.at[i, 'broke_low']:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries