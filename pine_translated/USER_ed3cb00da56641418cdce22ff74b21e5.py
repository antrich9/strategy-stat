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
    # Validate required columns
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Series references
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Shifted series for FVG detection
    close_1 = close.shift(1)
    close_2 = close.shift(2)
    open_1 = open_.shift(1)
    open_2 = open_.shift(2)
    high_2 = high.shift(2)
    low_2 = low.shift(2)

    # Bullish Fair Value Gap (FVG) condition
    bull_fvg = (
        (low > high_2) &
        (close_1 > high_2) &
        (open_2 < close_2) &
        (open_1 < close_1) &
        (open_ < close)
    )

    # Bearish Fair Value Gap (FVG) condition
    bear_fvg = (
        (high < low_2) &
        (close_1 < low_2) &
        (open_2 > close_2) &
        (open_1 > close_1) &
        (open_ > close)
    )

    entries = []
    trade_num = 1

    # Iterate over bars and generate entries when condition is True and not NaN
    for i in range(len(df)):
        if pd.isna(bull_fvg.iloc[i]) or pd.isna(bear_fvg.iloc[i]):
            continue

        if bull_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1
        elif bear_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
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
            }
            entries.append(entry)
            trade_num += 1

    return entries