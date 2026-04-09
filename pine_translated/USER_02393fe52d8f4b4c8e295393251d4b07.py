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
    # Default settings from Pine script
    showBull = True
    showBear = True
    filterWidth = 0.0  # input float default 0

    # Compute True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR with length 200
    atr = tr.ewm(alpha=1.0/200, adjust=False).mean()

    # Shifted series for entry conditions
    low_s3 = df['low'].shift(3)
    high_s1 = df['high'].shift(1)
    low_s1 = df['low'].shift(1)
    high_s3 = df['high'].shift(3)
    close_s2 = df['close'].shift(2)
    close_cur = df['close']

    # Bullish entry condition
    bull = (
        showBull &
        (low_s3 > high_s1) &
        (close_s2 < low_s3) &
        (close_cur > low_s3) &
        ((low_s3 - high_s1) > atr * filterWidth)
    )

    # Bearish entry condition
    bear = (
        showBear &
        (low_s1 > high_s3) &
        (close_s2 > high_s3) &
        (close_cur < high_s3) &
        ((low_s1 - high_s3) > atr * filterWidth)
    )

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where required indicators are NaN
        if (pd.isna(atr.iloc[i]) or pd.isna(low_s3.iloc[i]) or pd.isna(high_s1.iloc[i]) or
            pd.isna(low_s1.iloc[i]) or pd.isna(high_s3.iloc[i]) or pd.isna(close_s2.iloc[i])):
            continue

        if bull.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
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
            trade_num += 1

        if bear.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
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
            trade_num += 1

    return entries