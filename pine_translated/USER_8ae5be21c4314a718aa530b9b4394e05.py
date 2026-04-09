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
    entries = []
    trade_num = 0

    if len(df) < 3:
        return entries

    close = df['close']
    high = df['high']
    low = df['low']

    thresholdPer = 0
    threshold = thresholdPer / 100

    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2)) & ((low - high.shift(2)) / high.shift(2) > threshold)
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2)) & ((low.shift(2) - high) / high > threshold)

    for i in range(2, len(df)):
        if np.isnan(close.iloc[i]) or np.isnan(high.iloc[i]) or np.isnan(low.iloc[i]):
            continue

        bull_cond = bull_fvg.iloc[i] if not np.isnan(bull_fvg.iloc[i]) else False
        bear_cond = bear_fvg.iloc[i] if not np.isnan(bear_fvg.iloc[i]) else False

        if bull_cond:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })

        if bear_cond:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })

    return entries