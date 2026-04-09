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
    # Strategy parameters (must match Pine Script inputs)
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult = 2.0

    close = df['close']

    # MACD difference (fast EMA - slow EMA)
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()
    macd = fastEMA - slowEMA
    macd_prev = macd.shift(1)

    # T1: change in MACD * sensitivity
    t1 = (macd - macd_prev) * sensitivity

    # Trend up: non‑negative part of T1
    trendUp = t1.where(t1 >= 0, 0.0)

    # Bollinger Band width (difference between upper and lower bands)
    basis = close.rolling(window=channelLength).mean()
    stdev = close.rolling(window=channelLength).std()   # sample std (ddof=1)
    e1 = 2.0 * mult * stdev

    # Initialise series that tracks whether an entry has already been made
    entry_made = pd.Series(False, index=df.index)

    trade_num = 1
    entries = []

    # Iterate over bars starting from the second bar (need previous bar for shift)
    for i in range(1, len(df)):
        # Skip bars where indicators are not yet defined
        if pd.isna(trendUp.iloc[i]) or pd.isna(e1.iloc[i]):
            # Reset the flag on unknown bars
            entry_made.iloc[i] = False
            continue

        # If trendUp falls below the explosion line, reset the flag
        if trendUp.iloc[i] < e1.iloc[i]:
            entry_made.iloc[i] = False
        else:
            prev_entry_made = entry_made.iloc[i - 1]

            # Detect a crossover of trendUp above e1 while no entry has been taken
            if (not prev_entry_made) and (trendUp.iloc[i] > e1.iloc[i]) and (trendUp.iloc[i - 1] <= e1.iloc[i - 1]):
                ts = int(df['time'].iloc[i])
                entry_price = float(df['close'].iloc[i])
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
                entry_made.iloc[i] = True
            else:
                # Carry the flag forward
                entry_made.iloc[i] = prev_entry_made

    return entries