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
    # Parameters
    pp = 5  # Pivot Period of Order Blocks Detector
    atr_length = 14
    atr_multiplier = 1.5

    high = df['high']
    low = df['low']
    close = df['close']

    # Wilder ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_length, adjust=False).mean()

    # Pivot detection (simple rolling window)
    window = 2 * pp + 1
    high_roll = high.rolling(window=window, center=True).max()
    low_roll = low.rolling(window=window, center=True).min()
    high_pivot = (high == high_roll) & high_roll.notna()
    low_pivot = (low == low_roll) & low_roll.notna()

    # Forward fill last pivot levels
    last_high_pivot = high.where(high_pivot).ffill()
    last_low_pivot = low.where(low_pivot).ffill()

    # Entry signals: price crosses above low pivot => long, below high pivot => short
    dt_signal = (close > last_low_pivot) & (close.shift(1) <= last_low_pivot.shift(1))
    db_signal = (close < last_high_pivot) & (close.shift(1) >= last_high_pivot.shift(1))

    entries = []
    trade_num = 1
    for i in range(len(df)):
        # skip bars where required indicators are NaN
        if pd.isna(last_low_pivot.iloc[i]) or pd.isna(last_high_pivot.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if dt_signal.iloc[i] or db_signal.iloc[i]:
            direction = 'long' if dt_signal.iloc[i] else 'short'
            entry_price = close.iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries