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
    # Validate columns
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Shortcuts
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    ts = df['time']

    # ----- Daily‑style shifted values (mimic request.security for 1D) -----
    high_sh1 = high.shift(1)
    high_sh2 = high.shift(2)
    high_sh3 = high.shift(3)
    high_sh4 = high.shift(4)
    low_sh1 = low.shift(1)
    low_sh2 = low.shift(2)
    low_sh3 = low.shift(3)
    low_sh4 = low.shift(4)

    # ----- FVG detection (filters disabled → always true) -----
    bfvg = (low > high_sh2).fillna(False)   # bullish FVG
    sfvg = (high < low_sh2).fillna(False)   # bearish FVG

    # ----- Swing detection (daily swing high/low) -----
    is_swing_high = (
        (high_sh1 < high_sh2) & (high_sh3 < high_sh2) & (high_sh4 < high_sh2)
    ).fillna(False)

    is_swing_low = (
        (low_sh1 > low_sh2) & (low_sh3 > low_sh2) & (low_sh4 > low_sh2)
    ).fillna(False)

    # ----- Last swing type series (path‑dependent) -----
    lastSwingType = pd.Series("none", index=df.index)
    for i in range(1, len(df)):
        if is_swing_high.iloc[i]:
            lastSwingType.iloc[i] = "dailyHigh"
        elif is_swing_low.iloc[i]:
            lastSwingType.iloc[i] = "dailyLow"
        else:
            lastSwingType.iloc[i] = lastSwingType.iloc[i - 1]

    # ----- Entry signals -----
    long_entry = bfvg & (lastSwingType == "dailyLow")
    short_entry = sfvg & (lastSwingType == "dailyHigh")

    # ----- Build entry list -----
    entries = []
    trade_num = 1

    for i in df.index:
        if long_entry.iloc[i]:
            entry_ts = int(ts.iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_entry.iloc[i]:
            entry_ts = int(ts.iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries