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
    # Compute EMAs on the same dataframe (simulating both chart and 60‑minute EMAs)
    close = df['close']
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema18 = close.ewm(span=18, adjust=False).mean()
    # cema9 / cema18 are the same as ema9/ema18 when using the same timeframe
    cema9  = ema9
    cema18 = ema18

    # Build entry conditions
    cond_long  = (close > ema9) & (close > ema18) & (close > cema9) & (close > cema18)
    cond_short = (close < ema9) & (close < ema18) & (close < cema9) & (close < cema18)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where any required indicator is NaN
        if (np.isnan(ema9.iloc[i]) or np.isnan(ema18.iloc[i]) or
            np.isnan(cema9.iloc[i]) or np.isnan(cema18.iloc[i])):
            continue

        if cond_long.iloc[i]:
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
        elif cond_short.iloc[i]:
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