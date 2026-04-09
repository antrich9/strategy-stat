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
    # Source for E.I.T. (hl2)
    source = (df['high'] + df['low']) / 2.0

    # Shifted sources required by the E.I.T. formula
    srcEIT = source.shift(1)          # srcEIT = source[1]
    srcEIT_1 = source.shift(2)       # srcEIT[1] = source[2]
    srcEIT_2 = source.shift(3)       # srcEIT[2] = source[3]

    alpha = 0.07
    n = len(df)

    # Allocate arrays for indicator values
    itrend = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    sigEIT = np.zeros(n)

    # Compute E.I.T. iteratively
    for i in range(n):
        # nz() replacement: treat NaN as 0.0
        src = 0.0 if pd.isna(srcEIT.iloc[i]) else srcEIT.iloc[i]
        src1 = 0.0 if pd.isna(srcEIT_1.iloc[i]) else srcEIT_1.iloc[i]
        src2 = 0.0 if pd.isna(srcEIT_2.iloc[i]) else srcEIT_2.iloc[i]

        itrend_prev1 = itrend[i - 1] if i - 1 >= 0 and not np.isnan(itrend[i - 1]) else 0.0
        itrend_prev2 = itrend[i - 2] if i - 2 >= 0 and not np.isnan(itrend[i - 2]) else 0.0

        if i < 7:
            itrend_i = (src + 2.0 * src1 + src2) / 4.0
        else:
            itrend_i = (
                (alpha - alpha ** 2 / 4.0) * src
                + 0.5 * alpha ** 2 * src1
                - (alpha - 0.75 * alpha ** 2) * src2
                + 2.0 * (1 - alpha) * itrend_prev1
                - (1 - alpha) ** 2 * itrend_prev2
            )

        itrend[i] = itrend_i

        # trigger = 2 * itrend - nz(itrend[2])
        trigger_i = 2.0 * itrend_i - itrend_prev2
        trigger[i] = trigger_i

        if trigger_i > itrend_i:
            sig = 1.0
        elif trigger_i < itrend_i:
            sig = -1.0
        else:
            sig = 0.0
        sigEIT[i] = sig

    # Convert to Series for crossover / crossunder detection
    sig_series = pd.Series(sigEIT, index=df.index)

    # Entry conditions
    long_cond = (sig_series > 0) & (sig_series.shift(1) <= 0)
    short_cond = (sig_series < 0) & (sig_series.shift(1) >= 0)

    entries = []
    trade_num = 1

    for i in df.index:
        if long_cond.loc[i]:
            entry_ts = int(df.loc[i, 'time'])
            entry_price = float(df.loc[i, 'close'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.loc[i]:
            entry_ts = int(df.loc[i, 'time'])
            entry_price = float(df.loc[i, 'close'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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