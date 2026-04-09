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
    # Pivot period (must match the Pine Script input)
    PP = 5

    close = df['close']
    high = df['high']
    low = df['low']

    # Compute pivot high and low flags using a rolling window
    n = len(df)
    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)

    # Only consider bars with enough history for the window
    for i in range(PP, n - PP):
        # Pivot high: high[i] is the maximum in the window [i-PP, i+PP]
        if high.iloc[i] == high.iloc[i - PP:i + PP + 1].max():
            pivot_high.iloc[i] = True
        # Pivot low: low[i] is the minimum in the window [i-PP, i+PP]
        if low.iloc[i] == low.iloc[i - PP:i + PP + 1].min():
            pivot_low.iloc[i] = True

    entries = []
    trade_num = 1

    # Previous close series for crossover detection
    prev_close = close.shift(1)

    # Track most recent swing high/low levels
    last_high_val = np.nan
    last_low_val = np.nan

    for i in range(1, n):
        # Update swing levels when a new pivot is detected
        if pivot_high.iloc[i]:
            last_high_val = high.iloc[i]
        if pivot_low.iloc[i]:
            last_low_val = low.iloc[i]

        # Long entry: price crosses above the most recent swing high
        if not np.isnan(last_high_val):
            if close.iloc[i] > last_high_val and prev_close.iloc[i] <= last_high_val:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1

        # Short entry: price crosses below the most recent swing low
        if not np.isnan(last_low_val):
            if close.iloc[i] < last_low_val and prev_close.iloc[i] >= last_low_val:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1

    return entries