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
    # JMA parameters (default values from the Pine Script)
    length_jma = 7
    phase_jma = 50
    power_jma = 2
    src = df['close'].copy()

    # Compute beta, alpha, and phase ratio
    beta = 0.45 * (length_jma - 1) / (0.45 * (length_jma - 1) + 2)
    alpha = beta ** power_jma
    if phase_jma < -100:
        phase_ratio = 0.5
    elif phase_jma > 100:
        phase_ratio = 2.5
    else:
        phase_ratio = phase_jma / 100 + 1.5

    n = len(df)
    e0 = np.zeros(n)
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    jma = np.zeros(n)

    # Iterative JMA calculation
    for i in range(1, n):
        e0[i] = (1 - alpha) * src.iloc[i] + alpha * e0[i - 1]
        e1[i] = (src.iloc[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
        e2[i] = (e0[i] + phase_ratio * e1[i] - jma[i - 1]) * (1 - alpha) ** 2 + alpha ** 2 * e2[i - 1]
        jma[i] = e2[i] + jma[i - 1]

    jma_series = pd.Series(jma, index=df.index)

    # Use JMA and color confirmation flags (default true in Pine Script)
    use_jma = True
    use_color = True

    if use_jma:
        if use_color:
            long_cond = (jma_series > jma_series.shift(1)) & (df['close'] > jma_series)
            short_cond = (jma_series < jma_series.shift(1)) & (df['close'] < jma_series)
        else:
            long_cond = df['close'] > jma_series
            short_cond = df['close'] < jma_series
    else:
        long_cond = pd.Series(True, index=df.index)
        short_cond = pd.Series(True, index=df.index)

    # Replace NaNs with False
    long_cond = long_cond.fillna(False)
    short_cond = short_cond.fillna(False)

    # Build entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        if short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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