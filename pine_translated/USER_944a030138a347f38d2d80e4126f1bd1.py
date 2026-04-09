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
    # ---- 1. Baseline: Ehlers Two Pole Super Smoother (default baseline type) ----
    period = 20
    hl2 = (df['high'] + df['low']) / 2.0

    # Ehlers Two Pole Super Smoother coefficients
    pi = np.pi
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    n = len(df)
    filt2 = np.zeros(n, dtype=float)
    # first two bars are price source
    filt2[0] = hl2.iloc[0]
    if n > 1:
        filt2[1] = hl2.iloc[1]
    # subsequent bars use recurrence
    for i in range(2, n):
        filt2[i] = c1 * hl2.iloc[i] + c2 * filt2[i - 1] + c3 * filt2[i - 2]

    baseline_value = pd.Series(filt2, index=df.index)
    baseline_trigger = baseline_value.shift(1)

    # ---- 2. Baseline direction signals ----
    close = df['close']
    baseline_long = (close > baseline_value) & (baseline_value > baseline_trigger)
    baseline_short = (close < baseline_value) & (baseline_value < baseline_trigger)

    # ---- 3. Detect entry moments (crossovers) ----
    long_entry = baseline_long & ~baseline_long.shift(1).fillna(False)
    short_entry = baseline_short & ~baseline_short.shift(1).fillna(False)

    # ---- 4. Build entry list ----
    entries = []
    trade_num = 1
    for i in range(2, n):  # start after indicator warm‑up
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries