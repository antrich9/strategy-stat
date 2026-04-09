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
    barsback = 5

    high = df['high']
    close = df['close']
    open_ = df['open']

    # Initialize indicator series with NaN for bars where calculation isn’t possible
    etfSHHigh2 = pd.Series(np.nan, index=df.index)
    etfSclose2 = pd.Series(np.nan, index=df.index)
    etfSopen2 = pd.Series(np.nan, index=df.index)

    # Compute swing‑high based values (valuewhen of highest high)
    for i in range(barsback - 1, len(df)):
        window_start = i - barsback + 1
        window_high = high.iloc[window_start:i + 1]
        idx_max = window_high.idxmax()          # absolute index of the highest high in the window
        etfSHHigh2.iloc[i] = high.iloc[idx_max]
        etfSclose2.iloc[i] = close.iloc[idx_max]
        etfSopen2.iloc[i] = open_.iloc[idx_max]

    # Zone top: max(close, open) at the swing‑high bar
    etfSHTop2 = pd.Series(np.where(etfSclose2 > etfSopen2, etfSclose2, etfSopen2), index=df.index)

    # Entry conditions (showESD2 is always true in this strategy)
    long_cond = (close <= etfSHHigh2) & (close >= etfSHTop2)
    short_cond = (close >= etfSHHigh2) & (close <= etfSHTop2)

    entries = []
    trade_num = 1

    for i in df.index:
        if etfSHHigh2.isna().iloc[i]:
            continue

        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries