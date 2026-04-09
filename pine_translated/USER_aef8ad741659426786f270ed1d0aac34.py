import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    # Convert Unix timestamp (seconds) to UTC datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # Adjust for UTC+1 timezone (ignore DST for simplicity)
    df['datetime_local'] = df['datetime'] + pd.Timedelta(hours=1)
    # Extract hour and minute from the local time
    df['hour_local'] = df['datetime_local'].dt.hour
    df['minute_local'] = df['datetime_local'].dt.minute

    # Trading window parameters
    start_hour = 7
    end_hour = 10
    end_minute = 59
    in_trading_window = (df['hour_local'] >= start_hour) & (df['hour_local'] <= end_hour) & ~((df['hour_local'] == end_hour) & (df['minute_local'] > end_minute))

    # Compute previous day's high and low
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily.rename(columns={'high': 'daily_high', 'low': 'daily_low'}, inplace=True)
    daily['prev_day_high'] = daily['daily_high'].shift(1)
    daily['prev_day_low'] = daily['daily_low'].shift(1)
    df = df.merge(daily[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')

    # Sweep flags: price crosses previous day's high/low
    df['flagpdh'] = df['close'] > df['prev_day_high']
    df['flagpdl'] = df['close'] < df['prev_day_low']

    # EMAs (Fast = 50, Slow = 200)
    df['ema_fast'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()

    # Entry conditions
    df['long_cond'] = df['flagpdh'] & in_trading_window & (df['ema_fast'] > df['ema_slow'])
    df['short_cond'] = df['flagpdl'] & in_trading_window & (df['ema_fast'] < df['ema_slow'])

    # Build entry list
    entries = []
    trade_num = 1
    for i, row in df.iterrows():
        # Skip bars where required indicators are NaN
        if (pd.isna(row['prev_day_high']) or pd.isna(row['prev_day_low'])
                or pd.isna(row['ema_fast']) or pd.isna(row['ema_slow'])):
            continue
        if row['long_cond']:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(int(row['time']), tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
        elif row['short_cond']:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(int(row['time']), tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
    return entries