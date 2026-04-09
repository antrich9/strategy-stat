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
    # ── Trendilo parameters (hard‑coded from the script) ─────────────────────
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6.0
    trendilo_bmult = 1.0

    close = df['close']

    # ── pct_change = ta.change(close, trendilo_smooth) / close * 100 ────────
    pct_change = close.diff(trendilo_smooth) / close * 100.0

    # ── ALMA (Arnaud Legoux Moving Average) ──────────────────────────────────
    length = trendilo_length
    offset = trendilo_offset
    sigma = trendilo_sigma

    k = np.arange(length)
    m = offset * (length - 1)
    w = np.exp(-((k - m) ** 2) / (2 * sigma ** 2))
    w = w / w.sum()   # normalise weights

    def alma_series(series: pd.Series) -> pd.Series:
        return series.rolling(length).apply(
            lambda x: np.dot(x.values, w), raw=True
        )

    avg_pct_change = alma_series(pct_change)

    # ── RMS = trendilo_bmult * sqrt( sum(avg_pct_change^2) / length ) ─────────
    rms = trendilo_bmult * np.sqrt(
        avg_pct_change.rolling(length).apply(lambda x: (x ** 2).sum(), raw=True)
        / length
    )

    # ── Trend direction: 1 = long, -1 = short, 0 = neutral ──────────────────
    trendilo_dir = np.where(
        avg_pct_change > rms, 1,
        np.where(avg_pct_change < -rms, -1, 0)
    )
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)

    # ── Generate entry list ──────────────────────────────────────────────────
    entries = []
    trade_num = 1
    in_position = False  # True after the first entry (simulates position size == 0 guard)

    for i in range(len(df)):
        if pd.isna(trendilo_dir.iloc[i]):
            continue

        direction = None
        if not in_position:
            if trendilo_dir.iloc[i] == 1:
                direction = 'long'
            elif trendilo_dir.iloc[i] == -1:
                direction = 'short'

        if direction is None:
            continue

        entry_price = float(df['close'].iloc[i])
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        raw_price_a = raw_price_b = entry_price

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': raw_price_a,
            'raw_price_b': raw_price_b,
        })

        trade_num += 1
        in_position = True  # subsequent bars are ignored until an exit (which we ignore)

    return entries