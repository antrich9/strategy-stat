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
    df = df.copy()

    required_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['ts'].dt.normalize()

    daily_agg = df.groupby('date').agg(
        day_high=('high', 'max'),
        day_low=('low', 'min')
    ).reset_index()

    daily_agg['prev_high'] = daily_agg['day_high'].shift(1)
    daily_agg['prev_low'] = daily_agg['day_low'].shift(1)

    df = df.merge(daily_agg[['date', 'prev_high', 'prev_low']], on='date', how='left')

    close = df['close']
    prev_high = df['prev_high']
    prev_low = df['prev_low']

    long_cond = (close > prev_high) & (close.shift(1) <= prev_high.shift(1))
    short_cond = (close < prev_low) & (close.shift(1) >= prev_low.shift(1))

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(prev_high.iloc[i]) or pd.isna(prev_low.iloc[i]):
            continue

        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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

        if short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
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

    return entries