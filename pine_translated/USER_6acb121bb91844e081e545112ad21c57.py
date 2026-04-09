import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    is_friday = df['dayofweek'] == 4
    
    in_morning_window = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
                        (df['hour'] == 8) | \
                        ((df['hour'] == 9) & (df['minute'] <= 45))
    
    in_afternoon_window = ((df['hour'] == 15) & (df['minute'] >= 45)) | \
                          ((df['hour'] == 16) & (df['minute'] <= 45))
    
    in_trading_window = (in_morning_window | in_afternoon_window) & ~is_friday
    
    df['date'] = df['datetime'].dt.date
    daily_high = df.groupby('date')['high'].max().reset_index()
    daily_high.columns = ['date', 'daily_high']
    daily_low = df.groupby('date')['low'].min().reset_index()
    daily_low.columns = ['date', 'daily_low']
    daily_data = daily_high.merge(daily_low, on='date')
    daily_data['prev_day_high'] = daily_data['daily_high'].shift(1)
    daily_data['prev_day_low'] = daily_data['daily_low'].shift(1)
    df = df.merge(daily_data[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    high_swept = df['high'] > df['prev_day_high']
    low_swept = df['low'] < df['prev_day_low']
    
    long_entry = low_swept & in_trading_window
    short_entry = high_swept & in_trading_window
    
    entries = []
    trade_num = 1
    
    for idx in df.index:
        if long_entry[idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df.loc[idx, 'time']),
                'entry_time': datetime.fromtimestamp(df.loc[idx, 'time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': df.loc[idx, 'close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df.loc[idx, 'close'],
                'raw_price_b': df.loc[idx, 'close']
            })
            trade_num += 1
        elif short_entry[idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df.loc[idx, 'time']),
                'entry_time': datetime.fromtimestamp(df.loc[idx, 'time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': df.loc[idx, 'close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df.loc[idx, 'close'],
                'raw_price_b': df.loc[idx, 'close']
            })
            trade_num += 1
    
    return entries