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

    # Bullish Fair Value Gap (long entry): low > high 2 bars ago
    bfvg = (df['low'] > df['high'].shift(2)).fillna(False)

    # Bearish Fair Value Gap (short entry): high < low 2 bars ago
    sfvg = (df['high'] < df['low'].shift(2)).fillna(False)

    # The script defines optional filters (volume, ATR, trend) but they default to disabled.
    # Since we have no access to the input flags, we treat them as enabled (i.e., always True).
    # If you wish to respect those filters, compute them here and apply them to bfvg/sfvg.

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bfvg.iloc[i]:
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
        if sfvg.iloc[i]:
            entry_price = float(df['close'].iloc[i])
            entry_ts = int(df['time'].iloc[i])
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

    return entries