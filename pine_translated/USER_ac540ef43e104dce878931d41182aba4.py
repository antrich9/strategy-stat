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
    # Strategy parameters (hardcoded as per default inputs)
    use_hull_ma = True
    usecolor_hull_ma = True
    length_hull_ma = 9
    src_hull_ma = df['close']

    use_t3 = True
    cross_t3 = True
    inverse_t3 = False
    length_t3 = 5
    factor_t3 = 0.7
    highlight_movements_t3 = True
    src_t3 = df['close']

    # Helper: Weighted Moving Average
    def wma(series: pd.Series, period: int) -> pd.Series:
        if period <= 0:
            return series
        weights = np.arange(1, period + 1)
        def weighted_sum(x):
            return np.dot(x, weights[:len(x)])
        result = series.rolling(period).apply(lambda x: weighted_sum(x), raw=True)
        denominator = period * (period + 1) // 2
        return result / denominator

    # Hull MA
    half_len = length_hull_ma // 2
    sqrt_len = int(np.sqrt(length_hull_ma))

    wma_half = wma(src_hull_ma, half_len)
    wma_full = wma(src_hull_ma, length_hull_ma)
    hull_ma = wma(2 * wma_half - wma_full, sqrt_len)

    # Hull MA direction signal (+1 for up, -1 for down)
    sig_hull_ma = np.where(hull_ma > hull_ma.shift(1), 1, -1)

    # Hull MA long condition
    if use_hull_ma:
        if usecolor_hull_ma:
            signal_hull_ma_long = (sig_hull_ma > 0) & (src_hull_ma > hull_ma)
        else:
            signal_hull_ma_long = src_hull_ma > hull_ma
    else:
        signal_hull_ma_long = pd.Series(True, index=df.index)

    # T3 (GD)
    def gd_t3(src: pd.Series, length: int, factor: float) -> pd.Series:
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = gd_t3(gd_t3(gd_t3(src_t3, length_t3, factor_t3), length_t3, factor_t3), length_t3, factor_t3)

    # T3 direction signal
    t3_signals = np.where(t3 > t3.shift(1), 1, -1)

    # T3 long condition
    basic_long_cond = (t3_signals > 0) & (src_t3 > t3)

    if use_t3:
        if highlight_movements_t3:
            t3_signals_long = basic_long_cond
        else:
            t3_signals_long = src_t3 > t3
    else:
        t3_signals_long = pd.Series(True, index=df.index)

    # Cross confirmation
    if cross_t3:
        t3_signals_long_cross = (~t3_signals_long.shift(1).fillna(False).astype(bool)) & t3_signals_long.astype(bool)
    else:
        t3_signals_long_cross = t3_signals_long.astype(bool)

    # Inverse option
    if inverse_t3:
        t3_signals_long_final = ~t3_signals_long_cross
    else:
        t3_signals_long_final = t3_signals_long_cross

    # Combined entry condition
    entry_condition = signal_hull_ma_long & t3_signals_long_final
    entry_condition = entry_condition.fillna(False).astype(bool)

    # Generate entries
    entries = []
    trade_num = 1
    for i in df.index:
        if entry_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries