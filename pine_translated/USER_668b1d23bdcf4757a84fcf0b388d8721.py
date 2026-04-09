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
    # Copy to avoid modifying the input DataFrame
    df = df.copy()

    # Convert timestamp to datetime and extract the calendar date
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date

    # Compute daily high and low for each day (max/min across the whole day)
    daily_agg = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min')
    ).reset_index()

    # Merge daily high/low back onto the intraday bars
    df = df.merge(daily_agg, on='date', how='left')

    # Entry offset: 3 pips expressed as 0.001 price units (typical for 5‑digit brokers)
    entry_pips = 3.0
    pip_offset = entry_pips * 0.001  # 0.003

    df['buy_point'] = df['daily_high'] + pip_offset
    df['sell_point'] = df['daily_low'] - pip_offset

    # Detect the first bar of a new day (same as Pine's change(time("D")))
    df['new_day'] = df['date'].ne(df['date'].shift(1))

    entries = []
    trade_num = 1

    for i, row in df.iterrows():
        if row['new_day']:
            # Skip bars where required price data is missing
            if pd.isna(row['buy_point']) or pd.isna(row['sell_point']):
                continue

            # Long entry (buy stop)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['buy_point']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['buy_point']),
                'raw_price_b': float(row['buy_point'])
            })
            trade_num += 1

            # Short entry (sell stop)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['sell_point']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['sell_point']),
                'raw_price_b': float(row['sell_point'])
            })
            trade_num += 1

    return entries