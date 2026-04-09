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
    length = 6

    # shift low/high by 1 to mimic Pine's low[1] and high[1]
    shift_low = df['low'].shift(1)
    shift_high = df['high'].shift(1)

    # rolling lowest/highest over `length` bars
    lower = shift_low.rolling(length).min()
    upper = shift_high.rolling(length).max()
    basis = (lower + upper) / 2.0

    # higher‑timeframe basis (if columns are present; otherwise fallback to `basis`)
    if 'htf_lower' in df.columns and 'htf_upper' in df.columns:
        htf_basis = (df['htf_upper'] + df['htf_lower']) / 2.0
    else:
        htf_basis = basis

    # medium‑timeframe basis (same fallback)
    if 'mtf_lower' in df.columns and 'mtf_upper' in df.columns:
        mtf_basis = (df['mtf_upper'] + df['mtf_lower']) / 2.0
    else:
        mtf_basis = basis

    # bullish entry condition
    bullish = (df['close'] > basis) & (df['close'] > htf_basis) & (df['close'] > mtf_basis)

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if bullish.iloc[i]:
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

    return entries