import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure time column is datetime
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['time_dt'].dt.date

    # Compute daily high and low for each date
    daily = df.groupby('date')['high', 'low'].agg({'high': 'max', 'low': 'min'}).reset_index()
    daily.columns = ['date', 'daily_high', 'daily_low']
    # Shift to get previous day's high/low
    daily['prevDayHigh'] = daily['daily_high'].shift(1)
    daily['prevDayLow'] = daily['daily_low'].shift(1)

    # Merge back to original df
    df = df.merge(daily[['date', 'prevDayHigh', 'prevDayLow']], on='date', how='left')

    # Compute flagpdh and flagpdl per day
    # Determine if close > prevDayHigh or close < prevDayLow
    df['above_prev_high'] = df['close'] > df['prevDayHigh']
    df['below_prev_low'] = df['close'] < df['prevDayLow']

    # For each day, flag stays true after first occurrence
    df['flagpdh'] = df.groupby('date')['above_prev_high'].cummax()
    df['flagpdl'] = df.groupby('date')['below_prev_low'].cummax()

    # Time filter: extract hour and minute (assuming UTC)
    # Apply GMT+1 offset (adjust as needed)
    offset = 1  # GMT+1
    df['hour'] = (df['time_dt'].dt.hour + offset) % 24
    df['minute'] = df['time_dt'].dt.minute
    df['minute_of_day'] = df['hour'] * 60 + df['minute']

    # Define windows
    long_window = (df['minute_of_day'] >= 7*60) & (df['minute_of_day'] < 10*60)
    short_window = (df['minute_of_day'] >= 12*60) & (df['minute_of_day'] < 15*60)

    # Initialize flags for position
    in_long = False
    in_short = False
    trade_num = 1
    entries = []

    # Iterate over rows (skip first few days where prevDayHigh/Low are NaN)
    # We'll skip rows where prevDayHigh or prevDayLow is NaN
    for i, row in df.iterrows():
        if pd.isna(row['prevDayHigh']) or pd.isna(row['prevDayLow']):
            continue

        # Long entry condition
        if not in_long and row['flagpdh'] and long_window.iloc[i] if isinstance(long_window.iloc[i], (bool, np.bool_)) else False:
            # Enter long
            entry_price = row['close']
            entry_ts = row['time']
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            in_long = True

        # Short entry condition
        if not in_short and row['flagpdl'] and short_window.iloc[i] if isinstance(short_window.iloc[i], (bool, np.bool_)) else False:
            entry_price = row['close']
            entry_ts = row['time']
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            in_short = True

        # Reset position flags if we want to allow multiple entries? The script may only allow one entry per day.
        # But we can allow multiple entries after exit. However, we ignore exits.

    return entries