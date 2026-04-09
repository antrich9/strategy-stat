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
    # ── Parameters (matching Pine Script inputs) ──────────────────────────────
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    atrLength = 14

    # ── Ehlers Two‑Pole Supersmoother (E2PSS) ──────────────────────────────────
    price_e2pss = (df['high'] + df['low']) / 2.0

    pi = np.pi
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1.0 - coef2 - coef3

    n = len(df)
    filt2 = np.full(n, np.nan)
    trigger = np.full(n, np.nan)

    for i in range(n):
        if i < 2:
            filt2[i] = price_e2pss.iloc[i]
        else:
            filt2[i] = coef1 * price_e2pss.iloc[i] + coef2 * filt2[i - 1] + coef3 * filt2[i - 2]
        if i > 0:
            trigger[i] = filt2[i - 1]

    if useE2PSS:
        signal_long_e2pss = filt2 > trigger
        signal_short_e2pss = filt2 < trigger
    else:
        signal_long_e2pss = np.ones(n, dtype=bool)
        signal_short_e2pss = np.ones(n, dtype=bool)

    if inverseE2PSS:
        signal_long = signal_short_e2pss
        signal_short = signal_long_e2pss
    else:
        signal_long = signal_long_e2pss
        signal_short = signal_short_e2pss

    # ── Trendilo ────────────────────────────────────────────────────────────────
    close = df['close']
    pct_change = close.diff(trendilo_smooth) / close * 100.0

    # ALMA (Arnaud Legoux Moving Average) implementation
    def alma(series, length, offset, sigma):
        k = np.arange(length)
        m = (length - 1) * offset
        w = np.exp(-((k - m) ** 2) / (2 * sigma ** 2))
        w_sum = w.sum()

        def roll_func(x):
            return np.dot(x, w) / w_sum

        return series.rolling(length).apply(roll_func, raw=True)

    avg_pct = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)

    rms = trendilo_bmult * np.sqrt(avg_pct.pow(2).rolling(trendilo_length).mean())

    trendilo_dir = np.where(avg_pct > rms, 1, np.where(avg_pct < -rms, -1, 0))

    # ── Entry conditions ────────────────────────────────────────────────────────
    long_cond = signal_long & (trendilo_dir == 1)
    short_cond = signal_short & (trendilo_dir == -1)

    # Guard against NaNs
    long_cond = long_cond.fillna(False)
    short_cond = short_cond.fillna(False)

    # Transition detection: entry only on the first bar of a consecutive run
    entry_long = long_cond & (~long_cond.shift(1).fillna(False))
    entry_short = short_cond & (~short_cond.shift(1).fillna(False))

    # ── Build output list ──────────────────────────────────────────────────────
    entries = []
    trade_num = 1

    for i in range(n):
        if entry_long.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif entry_short.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries