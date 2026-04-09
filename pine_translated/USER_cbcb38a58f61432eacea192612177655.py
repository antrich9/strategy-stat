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
    # Ensure required columns are present
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Sort by time to guarantee chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # ---- Fair Value Gap (FVG) detection ----
    # Bullish FVG: current low > high 2 bars ago
    bullFVG = df['low'] > df['high'].shift(2)
    # Bearish FVG: current high < low 2 bars ago
    bearFVG = df['high'] < df['low'].shift(2)

    # Any FVG on the bar
    fvgCreated = bullFVG | bearFVG

    # Two consecutive FVGs: current bar and previous bar both have an FVG
    twoConsecutiveFVGs = fvgCreated & fvgCreated.shift(1)

    entries = []
    trade_num = 1

    # Start from index 2 because we need at least two prior bars for the shift
    for i in range(2, len(df)):
        if not twoConsecutiveFVGs.iloc[i]:
            continue

        # Determine direction based on the type of the current FVG
        direction = 'long' if bullFVG.iloc[i] else 'short'

        entry_ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price,
        })
        trade_num += 1

    return entries