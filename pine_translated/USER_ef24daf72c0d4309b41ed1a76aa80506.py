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
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # Convert timestamps to calendar dates for day grouping
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date

    # Compute daily high, low, and open (first bar of the day)
    daily = df.groupby('date').agg(
        daily_high=('high', 'max'),
        daily_low=('low', 'min'),
        daily_open=('open', 'first')
    ).reset_index()

    # Previous day's high and low serve as d_info.ph and d_info.pl
    daily['prev_high'] = daily['daily_high'].shift(1)
    daily['prev_low'] = daily['daily_low'].shift(1)

    # Merge previous day values back onto the original dataframe
    df = df.merge(daily[['date', 'prev_high', 'prev_low']], on='date', how='left')

    # Entry condition: close below previous day's low while flat
    long_condition = (df['close'] < df['prev_low']) & df['prev_low'].notna()

    entries = []
    trade_num = 1
    position_state = 'flat'  # 'flat' or 'long'

    for i in range(len(df)):
        if position_state == 'flat' and long_condition.iloc[i]:
            entry_price = float(df['close'].iloc[i])
            entry_ts = int(df['time'].iloc[i])
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
            position_state = 'long'  # stay in position; further entries ignored

    return entries