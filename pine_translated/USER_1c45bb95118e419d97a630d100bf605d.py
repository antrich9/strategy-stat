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
    # Convert unix timestamp to date for daily grouping
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
    # Compute daily high and low
    daily_agg = df.groupby('date').agg(daily_high=('high', 'max'), daily_low=('low', 'min')).reset_index()
    # Shift to get previous day's high and low
    daily_agg['prev_daily_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_daily_low'] = daily_agg['daily_low'].shift(1)
    # Merge previous day's high/low back to the original dataframe
    df = df.merge(daily_agg[['date', 'prev_daily_high', 'prev_daily_low']], on='date', how='left')
    # Short signal: close > previous day's high
    df['short_signal'] = df['close'] > df['prev_daily_high']
    # Entry occurs on the bar immediately after the signal
    df['entry_signal'] = df['short_signal'].shift(1)
    # Build entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if df['entry_signal'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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