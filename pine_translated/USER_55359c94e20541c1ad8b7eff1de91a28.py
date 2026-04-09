import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    df['date'] = df['time'].dt.date
    
    daily_agg = df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).reset_index()
    daily_agg.columns = ['date', 'daily_high', 'daily_low']
    daily_agg['prev_day_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['daily_low'].shift(1)
    
    df = df.merge(daily_agg[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    df['current_day_high'] = df.groupby('date')['high'].cummax()
    df['current_day_low'] = df.groupby('date')['low'].cummin()
    
    hour = df['time'].dt.hour
    minute = df['time'].dt.minute
    
    in_window_1 = (hour >= 7) & ((hour < 10) | ((hour == 10) & (minute <= 59)))
    in_window_2 = (hour >= 15) & ((hour < 16) | ((hour == 16) & (minute <= 59)))
    in_trading_window = in_window_1 | in_window_2
    
    high_swept = df['high'] > df['prev_day_high']
    low_swept = df['low'] < df['prev_day_low']
    
    flagpdh = high_swept & (df['current_day_low'] > df['prev_day_low'])
    flagpdl = low_swept & (df['current_day_high'] < df['prev_day_high'])
    
    long_signal = in_trading_window & flagpdl
    short_signal = in_trading_window & flagpdh
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['prev_day_high'].iloc[i]):
            continue
        
        ts_val = df['time'].iloc[i]
        ts_int = int(ts_val.timestamp()) if isinstance(ts_val, pd.Timestamp) else int(ts_val)
        
        if long_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_int,
                'entry_time': datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_int,
                'entry_time': datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries