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
    # Parameters
    length_t3 = 5
    factor_t3 = 0.7
    length_trendilo = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    smooth = 1

    # T3 calculation (triple smoothing)
    src = df['close']
    e1 = src.ewm(span=length_t3, adjust=False).mean()
    e2 = e1.ewm(span=length_t3, adjust=False).mean()
    e3 = e2.ewm(span=length_t3, adjust=False).mean()
    gd = e1 * (1 + factor_t3) - e2 * factor_t3
    gd1 = gd.ewm(span=length_t3, adjust=False).mean()
    gd2 = gd1.ewm(span=length_t3, adjust=False).mean()
    gd3 = gd2.ewm(span=length_t3, adjust=False).mean()
    t3 = gd3

    # Trendilo components
    pch = src.pct_change(smooth) * 100  # (close - close.shift(smooth)) / close * 100

    # ALMA function
    def alma_series(series, length, offset, sigma):
        k = np.arange(length)
        m = offset * (length - 1)
        w = np.exp(-((k - m) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        def dot_weight(x):
            return np.dot(x, w)
        return series.rolling(length).apply(dot_weight, raw=True)

    avpch = alma_series(pch, length_trendilo, offset, sigma)

    # blength = length_trendilo (cblen is false)
    blength = length_trendilo

    # RMS of avpch over blength
    rms = np.sqrt(avpch.pow(2).rolling(blength).mean())

    # Direction: 1 = bullish, -1 = bearish, 0 = neutral
    cdir = pd.Series(
        np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0)),
        index=df.index
    )

    # Entry conditions (signalStiffness is always true)
    long_cond = (df['close'] > t3) & (cdir == 1)
    short_cond = (df['close'] < t3) & (cdir == -1)

    # Generate entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        # Skip bars where any indicator is NaN
        if pd.isna(long_cond.iloc[i]) or pd.isna(short_cond.iloc[i]):
            continue
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
        elif short_cond.iloc[i]:
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