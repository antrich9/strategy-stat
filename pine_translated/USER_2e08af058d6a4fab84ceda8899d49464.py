import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    # Ensure the DataFrame is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Convert unix timestamps to datetime and extract the calendar date
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['dt'].dt.normalize()

    # Daily OHLC aggregates
    daily_ohlc = df.groupby('date').agg(
        daily_open=('open', 'first'),
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    )

    # Identify the first bar of each day (timestamp and its close price)
    first_bar_idx = df.groupby('date')['time'].idxmin()
    first_bar_df = df.loc[first_bar_idx, ['date', 'time', 'close']].copy()
    first_bar_df.columns = ['date', 'first_bar_time', 'first_bar_close']

    # Merge daily OHLC with the first bar information
    daily = daily_ohlc.join(first_bar_df.set_index('date'))

    # Previous day's high and low
    daily['prev_high'] = daily['daily_high'].shift(1)
    daily['prev_low'] = daily['daily_low'].shift(1)

    # Previous day's midpoint
    daily['prev_mid'] = daily['prev_low'] + (daily['prev_high'] - daily['prev_low']) * 0.5

    # Determine entry direction: long if today's open > prev_mid, short if < prev_mid
    daily['direction'] = pd.Series(index=daily.index, dtype=object)
    cond_long = daily['daily_open'] > daily['prev_mid']
    cond_short = daily['daily_open'] < daily['prev_mid']
    daily.loc[cond_long, 'direction'] = 'long'
    daily.loc[cond_short, 'direction'] = 'short'

    # Build the list of entry records
    entries = []
    trade_num = 1
    for row in daily.itertuples():
        if row.direction is None or pd.isna(row.direction):
            continue
        entry = {
            'trade_num': trade_num,
            'direction': row.direction,
            'entry_ts': int(row.first_bar_time),
            'entry_time': datetime.fromtimestamp(row.first_bar_time, tz=timezone.utc).isoformat(),
            'entry_price_guess': float(row.first_bar_close),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(row.first_bar_close),
            'raw_price_b': float(row.first_bar_close)
        }
        entries.append(entry)
        trade_num += 1

    return entries