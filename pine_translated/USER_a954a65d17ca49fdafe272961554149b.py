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
    # Detect FVGs: low > high[2] (bullish) and high < low[2] (bearish)
    high_shift2 = df['high'].shift(2)
    low_shift2 = df['low'].shift(2)

    bullish_fvg = (df['low'] > high_shift2).fillna(False).astype(bool)
    bearish_fvg = (df['high'] < low_shift2).fillna(False).astype(bool)

    # Track last FVG type: 1 = bullish, -1 = bearish, 0 = none
    last_fvg_type = pd.Series(0, index=df.index, dtype=int)

    # Build state series bar‑by‑bar
    for i in range(2, len(df)):
        if bullish_fvg.iloc[i]:
            last_fvg_type.iloc[i] = 1
        elif bearish_fvg.iloc[i]:
            last_fvg_type.iloc[i] = -1
        else:
            last_fvg_type.iloc[i] = last_fvg_type.iloc[i - 1]

    # Entry signals: a sharp turn occurs when the current FVG is opposite the previous one
    entry_long = bullish_fvg & (last_fvg_type.shift(1) == -1)
    entry_short = bearish_fvg & (last_fvg_type.shift(1) == 1)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if entry_long.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif entry_short.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries