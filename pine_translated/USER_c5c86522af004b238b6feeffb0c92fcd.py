import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['day'] = df['datetime'].dt.date
    
    daily_agg = df.groupby('day').agg({
        'high': 'max',
        'low': 'min'
    }).reset_index()
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    
    df = df.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    
    df['previousDayHighTaken'] = df['high'] > df['prev_day_high']
    df['previousDayLowTaken'] = df['low'] < df['prev_day_low']
    
    # Current day 240min high/low for flag logic
    df['4h_high'] = df['high'].rolling(window=24, min_periods=1).max()  # Approximate 240min bars
    df['4h_low'] = df['low'].rolling(window=24, min_periods=1).min()
    
    # Time windows: 07:45-09:45 and 15:45-16:45 London
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    morning_start = 7 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 15 * 60 + 45
    afternoon_end = 16 * 60 + 45
    
    isWithinMorningWindow = (df['time_minutes'] >= morning_start) & (df['time_minutes'] < morning_end)
    isWithinAfternoonWindow = (df['time_minutes'] >= afternoon_start) & (df['time_minutes'] < afternoon_end)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Flag logic
    flagpdh = False
    flagpdl = False
    entries_list = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['prev_day_high'].iloc[i]) or pd.isna(df['prev_day_low'].iloc[i]):
            continue
            
        prev_day_high = df['prev_day_high'].iloc[i]
        prev_day_low = df['prev_day_low'].iloc[i]
        high_sweep = df['previousDayHighTaken'].iloc[i]
        low_sweep = df['previousDayLowTaken'].iloc[i]
        current_day_high = df['4h_high'].iloc[i]
        current_day_low = df['4h_low'].iloc[i]
        in_window = in_trading_window.iloc[i]
        
        if high_sweep and current_day_low > prev_day_low:
            flagpdh = True
        elif low_sweep and current_day_high < prev_day_high:
            flagpdl = True
        else:
            flagpdh = False
            flagpdl = False
        
        if flagpdl and in_window:
            ts = int(df['time'].iloc[i])
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif flagpdh and in_window:
            ts = int(df['time'].iloc[i])
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries_list