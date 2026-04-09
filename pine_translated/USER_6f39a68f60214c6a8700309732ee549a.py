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
    # Shifted price series for pattern detection
    low2 = df['low'].shift(2)
    open1 = df['open'].shift(1)
    high0 = df['high']
    close1 = df['close'].shift(1)
    close0 = df['close']
    low1 = df['low'].shift(1)

    high2 = df['high'].shift(2)
    low0 = df['low']
    high1 = df['high'].shift(1)

    # FVG size calculations
    top_size = low2 - high0
    bottom_size = low0 - high2

    # Top (bearish) FVG conditions
    top_imbalance_bway = (low2 <= open1) & (high0 >= close1) & (close0 < low1) & (top_size > 0)
    top_imbalance_xbway = (low2 <= open1) & (high0 >= close1) & (close0 > low1) & (top_size > 0)

    # Bottom (bullish) FVG conditions
    bottom_inbalance_bway = (high2 >= open1) & (low0 <= close1) & (close0 > high1) & (bottom_size > 0)
    bottom_inbalance_xbway = (high2 >= open1) & (low0 <= close1) & (close0 < high1) & (bottom_size > 0)

    # Combine entry signals
    short_entry = (top_imbalance_bway | top_imbalance_xbway)
    long_entry = (bottom_inbalance_bway | bottom_inbalance_xbway)

    # Ensure we only consider rows where all required shifted values exist (i.e., index >= 2)
    valid = low2.notna() & high2.notna() & open1.notna() & low1.notna() & high1.notna() & close1.notna()
    short_entry = short_entry & valid
    long_entry = long_entry & valid

    # Build entry list
    entries = []
    trade_num = 1

    for i in df.index:
        if short_entry.loc[i]:
            ts = int(df['time'].loc[i])
            entry_price = float(df['close'].loc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif long_entry.loc[i]:
            ts = int(df['time'].loc[i])
            entry_price = float(df['close'].loc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries