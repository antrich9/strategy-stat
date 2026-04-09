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
    # T3 parameters
    length_t3 = 5
    factor = 0.7

    close = df['close']

    # T3 calculation
    def compute_t3(series, length, factor):
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema_ema1 = ema1.ewm(span=length, adjust=False).mean()
        gd1 = ema1 * (1 + factor) - ema_ema1 * factor

        ema2 = gd1.ewm(span=length, adjust=False).mean()
        ema_ema2 = ema2.ewm(span=length, adjust=False).mean()
        gd2 = ema2 * (1 + factor) - ema_ema2 * factor

        ema3 = gd2.ewm(span=length, adjust=False).mean()
        ema_ema3 = ema3.ewm(span=length, adjust=False).mean()
        gd3 = ema3 * (1 + factor) - ema_ema3 * factor

        return gd3

    t3 = compute_t3(close, length_t3, factor)

    # Trendilo parameters
    smooth = 1
    length_trendilo = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0

    # Percent change
    pch = close.diff(smooth) / close.shift(smooth) * 100.0

    # ALMA
    def alma(series, length, offset, sigma):
        if length <= 0:
            return series
        m = (length - 1) * offset
        s = (length - 1) / sigma if sigma != 0 else 0.0
        x = np.arange(length)
        w = np.exp(-((x - m) ** 2) / (2 * s ** 2))
        w = w / w.sum()
        w_rev = w[::-1]
        result = series.rolling(length).apply(lambda x_vals: np.dot(x_vals.values, w_rev), raw=True)
        return result

    avpch = alma(pch, length_trendilo, offset, sigma)

    # RMS
    blen = length_trendilo  # cblen = false
    rms = ((avpch ** 2).rolling(blen).mean()) ** 0.5 * bmult

    # Trend direction
    cdir = pd.Series(
        np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0)),
        index=avpch.index
    )

    # Entry conditions
    long_cond = (close > t3) & (cdir == 1)
    short_cond = (close < t3) & (cdir == -1)

    # Simulate position state
    entries = []
    trade_num = 1
    position_open = False

    for i in range(len(df)):
        if pd.isna(close.iloc[i]) or pd.isna(t3.iloc[i]) or pd.isna(cdir.iloc[i]):
            continue

        if long_cond.iloc[i] and not position_open:
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            position_open = True

        elif short_cond.iloc[i] and not position_open:
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            position_open = True

    return entries