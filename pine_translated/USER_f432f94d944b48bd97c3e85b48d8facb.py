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
    # Alias for clarity
    high = df['high']
    low = df['low']
    close = df['close']

    # Shifted series for higher‑timeframe references
    high_s1 = high.shift(1)
    high_s2 = high.shift(2)
    high_s3 = high.shift(3)
    high_s4 = high.shift(4)

    low_s1 = low.shift(1)
    low_s2 = low.shift(2)
    low_s3 = low.shift(3)
    low_s4 = low.shift(4)

    # Swing detection (dailyHigh2 / dailyLow2 are the reference)
    is_swing_high = (high_s1 < high_s2) & (high_s3 < high_s2) & (high_s4 < high_s2)
    is_swing_low  = (low_s1  > low_s2) & (low_s3  > low_s2) & (low_s4  > low_s2)

    # Track the most recent swing type
    last_swing_type = pd.Series("none", index=df.index)
    for i in range(len(df)):
        if is_swing_high.iloc[i]:
            last_swing_type.iloc[i] = "dailyHigh"
        elif is_swing_low.iloc[i]:
            last_swing_type.iloc[i] = "dailyLow"
        else:
            if i > 0:
                last_swing_type.iloc[i] = last_swing_type.iloc[i-1]

    # Basic FVG conditions (filters from inputs are ignored – treated as always true)
    bfvg = low > high_s2   # bullish FVG
    sfvg = high < low_s2   # bearish FVG

    # Entry signals
    long_cond  = bfvg & (last_swing_type == "dailyLow")
    short_cond = sfvg & (last_swing_type == "dailyHigh")

    # Build entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        # skip bars where required indicators are NaN (comparisons with NaN yield False)
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries