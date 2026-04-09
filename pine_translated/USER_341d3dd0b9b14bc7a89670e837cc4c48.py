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
    # ----- Input parameters (defaults from Pine Script) -----
    use_hull_ma = True          # input.bool(true, 'Use Hull MA?')
    use_color_hull_ma = True    # input.bool(true, 'Use color confirmation? - Hull MA')
    length_hull_ma = 9          # input.int(9, minval=1, title="Length - Hull MA")
    src_hull_ma = df['close']   # using close as source

    # ----- Weighted Moving Average (WMA) helper -----
    def wma(series: pd.Series, length: int) -> pd.Series:
        if length <= 0:
            return series
        weights = np.arange(1, length + 1)
        def weighted_avg(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(length).apply(weighted_avg, raw=True)

    # ----- Hull MA calculation -----
    half_len = length_hull_ma // 2
    sqrt_len = int(np.sqrt(length_hull_ma))

    wma_half = wma(src_hull_ma, half_len)
    wma_full = wma(src_hull_ma, length_hull_ma)
    hull_ma = wma(2 * wma_half - wma_full, sqrt_len)

    # ----- Hull MA signal (up = 1, down = -1) -----
    hull_up = hull_ma > hull_ma.shift(1)
    sig_hull = hull_up.astype(int).replace(0, -1)

    # ----- Entry condition based on Hull MA -----
    if use_hull_ma:
        if use_color_hull_ma:
            entry_cond = (sig_hull > 0) & (df['close'] > hull_ma)
        else:
            entry_cond = df['close'] > hull_ma
    else:
        # When Hull MA is disabled, always trigger entry
        entry_cond = pd.Series(True, index=df.index)

    # ----- Generate entries -----
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where the required indicator is NaN
        if pd.isna(hull_ma.iloc[i]):
            continue

        if entry_cond.iloc[i]:
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

    return entries