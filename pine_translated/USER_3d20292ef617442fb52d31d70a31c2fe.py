import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Previous day high/low calculation using daily aggregation
    df['date'] = df['datetime'].dt.date
    daily_agg = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    ).reset_index()
    daily_agg['prev_day_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['daily_low'].shift(1)
    df = df.merge(daily_agg[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    df.drop('date', axis=1, inplace=True)
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Detect liquidity sweeps (flagpdh, flagpdl)
    df['flagpdh'] = df['close'] > df['prev_day_high']
    df['flagpdl'] = df['close'] < df['prev_day_low']
    
    # Detect Order Blocks and FVG
    df['is_up'] = df['close'] > df['open']
    df['is_down'] = df['close'] < df['open']
    
    df['ob_up'] = (
        df['is_down'].shift(1) &
        df['is_up'] &
        (df['close'] > df['high'].shift(1))
    )
    
    df['ob_down'] = (
        df['is_up'].shift(1) &
        df['is_down'] &
        (df['close'] < df['low'].shift(1))
    )
    
    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)
    
    # Time filter for sessions (0700-0959 and 1200-1459 in GMT+1)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    
    # Convert to GMT+1 by adding 1 hour to UTC times (assuming input is UTC)
    df['minute_of_day'] = (df['minute_of_day'] + 60) % (24 * 60)
    df['hour'] = df['minute_of_day'] // 60
    
    df['in_long_session'] = (df['hour'] >= 7) & (df['hour'] < 10)
    df['in_short_session'] = (df['hour'] >= 12) & (df['hour'] < 15)
    
    # Entry conditions
    df['long_entry'] = (
        df['flagpdh'] &
        df['ob_up'] &
        df['fvg_up'] &
        df['in_long_session']
    )
    
    df['short_entry'] = (
        df['flagpdl'] &
        df['ob_down'] &
        df['fvg_down'] &
        df['in_short_session']
    )
    
    # Build entries list
    entries_list = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
            
        if df['long_entry'].iloc[i]:
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            
        elif df['short_entry'].iloc[i]:
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries_list