import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Price input for E2PSS (hl2)
    price = (df['high'] + df['low']) / 2.0

    # E2PSS parameters
    period = 15
    pi = np.pi
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / period)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1.0 - coef2 - coef3

    n = len(df)
    Filt2 = np.zeros(n, dtype=float)
    for i in range(n):
        if i < 3:
            Filt2[i] = price.iloc[i]
        else:
            Filt2[i] = coef1 * price.iloc[i] + coef2 * Filt2[i - 1] + coef3 * Filt2[i - 2]

    Filt2_series = pd.Series(Filt2)
    TriggerE2PSS = Filt2_series.shift(1).fillna(0)

    # Long/Short signals (useE2PSS = true, inverseE2PSS = false)
    long_signal = Filt2_series > TriggerE2PSS
    short_signal = Filt2_series < TriggerE2PSS

    entries = []
    trade_num = 1

    # Start from bar 1 to avoid undefined previous values; can use 0 if desired
    for i in range(1, n):
        if long_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries