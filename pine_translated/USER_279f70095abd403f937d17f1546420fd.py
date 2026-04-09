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
    for col in ['time', 'open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['close']
    high = df['high']
    low = df['low']

    # 50‑period EMA
    ema = close.ewm(span=50, adjust=False).mean()

    # Swing high/low detection (5‑bar window)
    swing_high_update = high.rolling(window=5, min_periods=5).max() == high
    swing_low_update = low.rolling(window=5, min_periods=5).min() == low

    last_swing_high = high.where(swing_high_update).ffill()
    last_swing_low = low.where(swing_low_update).ffill()

    # Fibonacci retracement level (0.5)
    fib_level = 0.5

    pullback_long = last_swing_low + fib_level * (last_swing_high - last_swing_low)
    pullback_short = last_swing_high - fib_level * (last_swing_high - last_swing_low)

    # Trend filter
    long_condition = close > ema
    short_condition = close < ema

    # Entry signals
    long_entry = long_condition & (close > pullback_long)
    short_entry = short_condition & (close < pullback_short)

    entries = []
    trade_num = 1
    n = len(df)

    for i in range(n):
        # Skip bars where indicators are not yet defined
        if (pd.isna(ema.iloc[i]) or
            pd.isna(pullback_long.iloc[i]) or
            pd.isna(pullback_short.iloc[i])):
            continue

        if long_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i]),
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i]),
            })
            trade_num += 1

    return entries