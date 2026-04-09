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
    # Series shortcuts
    open_s  = df['open']
    high_s  = df['high']
    low_s   = df['low']
    close_s = df['close']
    time_s  = df['time']

    # Bar 1 (previous) and Bar 2 (two bars back) shifted series
    open1  = open_s.shift(1)
    high1  = high_s.shift(1)
    low1   = low_s.shift(1)
    close1 = close_s.shift(1)

    open2  = open_s.shift(2)
    high2  = high_s.shift(2)
    low2   = low_s.shift(2)
    close2 = close_s.shift(2)

    # Body size for current bar and bar 1
    body   = (close_s - open_s).abs()
    body1  = body.shift(1)

    # Range of bar 1
    range1 = high1 - low1

    # Pattern parameters (defaults from Pine script)
    small_body_factor = 0.5
    wick_factor       = 2.0

    # Small body condition for bar 1
    is_small_body1 = body1 <= range1 * small_body_factor

    # Upper and lower wicks for bar 1
    max_oc1 = pd.concat([open1, close1], axis=1).max(axis=1)
    min_oc1 = pd.concat([open1, close1], axis=1).min(axis=1)
    u_wick1 = high1 - max_oc1
    l_wick1 = min_oc1 - low1

    is_long_upper1 = u_wick1 >= body1 * wick_factor
    is_long_lower1 = l_wick1 >= body1 * wick_factor

    # Bullish pattern: bearish(o2,c2) & small body & long lower wick & h0 > h2 & c0 < h2
    bull_pattern = (close2 < open2) & is_small_body1 & is_long_lower1 & (high_s > high2) & (close_s < high2)

    # Bearish pattern: bullish(o2,c2) & small body & long upper wick & l0 < l2 & c0 > l2
    bear_pattern = (close2 > open2) & is_small_body1 & is_long_upper1 & (low_s < low2) & (close_s > low2)

    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        if bear_pattern.iloc[i]:
            direction = 'short'
        elif bull_pattern.iloc[i]:
            direction = 'long'
        else:
            continue

        entry_price = close_s.iloc[i]
        ts = int(time_s.iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': ts,
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