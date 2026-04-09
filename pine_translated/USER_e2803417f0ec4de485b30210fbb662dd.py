import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    Generate entry signals for the Hull MA + T3 + Volume strategy.
    """
    # Copy to avoid mutating input
    df = df.copy()

    # Strategy parameters (default values from Pine script)
    length_hull = 9
    half_hull = length_hull // 2                     # 4
    sqrt_hull = int(np.floor(np.sqrt(length_hull)))  # 3
    length_t3 = 5
    factor_t3 = 0.7
    vol_period = 20

    # ---------- helper functions ----------
    def wma(series: pd.Series, length: int) -> pd.Series:
        """Weighted moving average."""
        weights = np.arange(1, length + 1)
        def weighted_sum(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=length).apply(weighted_sum, raw=True)

    # ---------- Hull MA ----------
    wma_half = wma(df['close'], half_hull)
    wma_full = wma(df['close'], length_hull)
    hullma = wma(2 * wma_half - wma_full, sqrt_hull)

    # Hull MA direction (rising) and price > hullma
    hullma_rising = hullma > hullma.shift(1)
    signal_hull_long = hullma_rising & (df['close'] > hullma)

    # ---------- T3 (triple‑EMA smoothing) ----------
    t3 = df['close'].copy()
    for _ in range(3):
        e1 = t3.ewm(span=length_t3, adjust=False).mean()
        e2 = e1.ewm(span=length_t3, adjust=False).mean()
        t3 = e1 * (1 + factor_t3) - e2 * factor_t3

    t3_rising = t3 > t3.shift(1)
    basic_long_cond = t3_rising & (df['close'] > t3)

    # Crossover of the T3 long condition
    t3_cross = (~basic_long_cond.shift(1).fillna(False)) & basic_long_cond

    # ---------- Volume condition ----------
    vol_ma = df['volume'].rolling(window=vol_period).mean()
    norm_vol = df['volume'] / vol_ma
    vol_cond = norm_vol > 1

    # ---------- Combined entry ----------
    entry_cond = signal_hull_long & t3_cross & vol_cond

    # ---------- Build entry list ----------
    entries = []
    trade_num = 1
    for i in df.index:
        if entry_cond.loc[i]:
            ts = int(df.loc[i, 'time'])
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price,
            })
            trade_num += 1

    return entries