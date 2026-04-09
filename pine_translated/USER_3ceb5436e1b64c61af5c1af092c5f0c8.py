import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default parameters from Pine Script
    length = 7
    phase = 50
    power = 2
    usejma = True
    usecolor = True

    # Compute JMA
    src = df['close']
    n = len(src)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100 + 1.5)

    e0 = np.zeros(n)
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    jma = np.zeros(n)

    for i in range(n):
        if i == 0:
            e0_i = (1 - alpha) * src.iloc[i]
            e1_i = (src.iloc[i] - e0_i) * (1 - beta)
            e2_i = (e0_i + phase_ratio * e1_i) * (1 - alpha) ** 2
            jma_i = e2_i
        else:
            e0_i = (1 - alpha) * src.iloc[i] + alpha * e0[i-1]
            e1_i = (src.iloc[i] - e0_i) * (1 - beta) + beta * e1[i-1]
            e2_i = (e0_i + phase_ratio * e1_i - jma[i-1]) * (1 - alpha) ** 2 + alpha ** 2 * e2[i-1]
            jma_i = e2_i + jma[i-1]
        e0[i] = e0_i
        e1[i] = e1_i
        e2[i] = e2_i
        jma[i] = jma_i

    jma_series = pd.Series(jma, index=df.index)
    jma_prev = jma_series.shift(1)

    # Entry conditions
    if usejma:
        if usecolor:
            long_cond = (jma_series > jma_prev) & (df['close'] > jma_series)
            short_cond = (jma_series < jma_prev) & (df['close'] < jma_series)
        else:
            long_cond = df['close'] > jma_series
            short_cond = df['close'] < jma_series
    else:
        long_cond = pd.Series(True, index=df.index)
        short_cond = pd.Series(True, index=df.index)

    # Generate entries
    entries = []
    trade_num = 1
    for i in range(1, n):
        if long_cond.iloc[i]:
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
        elif short_cond.iloc[i]:
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