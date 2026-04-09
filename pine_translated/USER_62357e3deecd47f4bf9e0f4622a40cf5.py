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
    close = df['close']
    time = df['time']

    cema9 = close.ewm(span=9, adjust=False).mean()
    cema18 = close.ewm(span=18, adjust=False).mean()

    condition_long = (close > cema9) & (close > cema18)
    condition_short = (close < cema9) & (close < cema18)

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if np.isnan(cema9.iloc[i]) or np.isnan(cema18.iloc[i]):
            continue

        entry_ts = int(time.iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = close.iloc[i]

        if condition_long.iloc[i]:
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

        if condition_short.iloc[i]:
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