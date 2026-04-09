import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    # Calculate previous day high/low using daily aggregation
    df['day'] = pd.to_datetime(df['time'], unit='ms').dt.date
    daily_agg = df.groupby('day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg.columns = ['day', 'prev_day_high', 'prev_day_low']
    # Shift to align previous day values with current day
    daily_agg['prev_day_high'] = daily_agg['prev_day_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['prev_day_low'].shift(1)
    
    # Calculate 240-minute (4-hour) high/low for the current period
    df['minute_dt'] = pd.to_datetime(df['time'], unit='ms')
    df['interval_4h'] = df['minute_dt'].dt.floor('4h')
    interval_agg = df.groupby('interval_4h').agg({'high': 'max', 'low': 'min'}).reset_index()
    interval_agg.columns = ['interval_4h', 'current_4h_high', 'current_4h_low']
    
    # Merge aggregated values back
    df = df.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    df = df.merge(interval_agg, on='interval_4h', how='left')
    
    # Trading windows (7:45-9:45 and 15:45-16:45)
    df['hour'] = df['minute_dt'].dt.hour
    df['minute'] = df['minute_dt'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['morning_window'] = (df['time_minutes'] >= 465) & (df['time_minutes'] <= 585)
    df['afternoon_window'] = (df['time_minutes'] >= 945) & (df['time_minutes'] <= 1005)
    df['in_trading_window'] = (df['morning_window'] | df['afternoon_window']) & (df['minute_dt'].dt.dayofweek != 4)
    
    # Sweep detection flags (per day)
    df['day_changed'] = df['day'].ne(df['day'].shift(1))
    df['sweptHigh'] = False
    df['sweptLow'] = False
    
    # Detect sweeps
    df['previousDayHighTaken'] = df['high'] > df['prev_day_high']
    df['previousDayLowTaken'] = df['low'] < df['prev_day_low']
    
    # Build boolean conditions
    cond_high_sweep = df['previousDayHighTaken'] & (df['current_4h_low'] > df['prev_day_low'])
    cond_low_sweep = df['previousDayLowTaken'] & (df['current_4h_high'] < df['prev_day_high'])
    
    # Iterate with per-day sweep flags
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        
        if row['day_changed']:
            sweptHigh = False
            sweptLow = False
        else:
            sweptHigh = prev_row['sweptHigh']
            sweptLow = prev_row['sweptLow']
        
        sweepHighNow = not sweptHigh and row['previousDayHighTaken']
        sweepLowNow = not sweptLow and row['previousDayLowTaken']
        
        if pd.isna(row['in_trading_window']) or not row['in_trading_window']:
            df.at[df.index[i], 'sweptHigh'] = sweptHigh
            df.at[df.index[i], 'sweptLow'] = sweptLow
            continue
        
        # Entry conditions
        long_cond = sweepHighNow and cond_high_sweep.iloc[i]
        short_cond = sweepLowNow and cond_low_sweep.iloc[i]
        
        if long_cond:
            ts = int(row['time'])
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            entry_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
            sweptHigh = True
        
        if short_cond:
            ts = int(row['time'])
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            entry_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
            sweptLow = True
        
        df.at[df.index[i], 'sweptHigh'] = sweptHigh
        df.at[df.index[i], 'sweptLow'] = sweptLow
    
    return entries