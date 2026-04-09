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
    results = []
    trade_num = 0

    n = len(df)
    if n < 5:
        return results

    # Initialize previous day high/low tracking
    prev_day_high = np.nan
    prev_day_low = np.nan
    flagpdl = False
    flagpdh = False

    # Detect new days (start of trading day)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    is_new_day = (times.dt.date != times.dt.date.shift(1)) & (times.dt.date.notna())

    # Calculate sweeps and flags
    swept_high = np.zeros(n, dtype=bool)
    swept_low = np.zeros(n, dtype=bool)

    # Helper functions for OB/FVG detection
    def is_up(idx):
        if idx < 0 or idx >= n:
            return False
        return df['close'].iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        if idx < 0 or idx >= n:
            return False
        return df['close'].iloc[idx] < df['open'].iloc[idx]

    def is_ob_up(idx):
        if idx - 1 < 0 or idx >= n:
            return False
        return is_down(idx - 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx - 1]

    def is_ob_down(idx):
        if idx - 1 < 0 or idx >= n:
            return False
        return is_up(idx - 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx - 1]

    def is_fvg_up(idx):
        if idx - 2 < 0 or idx >= n:
            return False
        return df['low'].iloc[idx] > df['high'].iloc[idx - 2]

    def is_fvg_down(idx):
        if idx - 2 < 0 or idx >= n:
            return False
        return df['high'].iloc[idx] < df['low'].iloc[idx - 2]

    # Track OB/FVG for current and previous bars
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)

    for i in range(n):
        if i >= 2:
            fvg_up[i] = is_fvg_up(i)
            fvg_down[i] = is_fvg_down(i)
        if i >= 1:
            ob_up[i] = is_ob_up(i)
            ob_down[i] = is_ob_down(i)

    # Process bars to detect entries
    for i in range(1, n):
        # Handle new day - find previous day high/low
        if is_new_day.iloc[i]:
            # Initialize with current day's values
            prev_day_high = df['high'].iloc[i]
            prev_day_low = df['low'].iloc[i]
            flagpdl = False
            flagpdh = False
        else:
            # Update prev day high/low looking back up to 500 bars
            lookback = min(500, i)
            for j in range(1, lookback + 1):
                if i - j >= 0 and is_new_day.iloc[i - j]:
                    break
                if i - j >= 0:
                    prev_day_high = max(prev_day_high, df['high'].iloc[i - j]) if not np.isnan(prev_day_high) else df['high'].iloc[i - j]
                    prev_day_low = min(prev_day_low, df['low'].iloc[i - j]) if not np.isnan(prev_day_low) else df['low'].iloc[i - j]

        # Detect sweeps on current bar
        if not np.isnan(prev_day_high) and not np.isnan(prev_day_low):
            swept_high[i] = (df['high'].iloc[i] > prev_day_high) and (df['close'].iloc[i] < prev_day_high)
            swept_low[i] = (df['low'].iloc[i] < prev_day_low) and (df['close'].iloc[i] > prev_day_low)

            # Update flags
            if swept_low[i]:
                flagpdl = True
            if swept_high[i]:
                flagpdh = True

        # Bullish entry: swept low + ob up + fvg up
        if swept_low[i] and ob_up[i] and fvg_up[i]:
            trade_num += 1
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        # Bearish entry: swept high + ob down + fvg down
        if swept_high[i] and ob_down[i] and fvg_down[i]:
            trade_num += 1
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return results