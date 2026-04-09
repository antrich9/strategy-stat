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

    # Input parameters from Pine Script
    use_hull = True
    use_color = True
    hull_len = 9
    use_t3 = True
    cross_t3 = True
    inverse_t3 = False
    t3_len = 5
    t3_factor = 0.7
    highlight_movements = True
    stc_len = 10
    fast_len = 23
    slow_len = 50
    stc_factor = 0.5
    stc_threshold = 25.0

    # Hull MA calculation
    half_len = int(hull_len / 2)
    sqrt_len = int(np.sqrt(hull_len))

    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    hull_ma = wma(2 * wma(df['close'], half_len) - wma(df['close'], hull_len), sqrt_len)
    hull_prev = hull_ma.shift(1)
    sig_hull = np.where(hull_ma > hull_prev, 1.0, -1.0)

    # T3 calculation
    ema1 = df['close'].ewm(span=t3_len, adjust=False).mean()
    ema2 = ema1.ewm(span=t3_len, adjust=False).mean()
    t3 = ema1 * (1 + t3_factor) - ema2 * t3_factor
    t3_prev = t3.shift(1)
    t3_signals = np.where(t3 > t3_prev, 1.0, -1.0)

    # STC calculation
    ema_fast = df['close'].ewm(span=fast_len, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_len, adjust=False).mean()
    macd = ema_fast - ema_slow

    stc_arr = np.zeros(len(df))
    v2_arr = np.full(len(df), np.nan)
    prev_fract = 50.0
    prev_k = 50.0
    d = 50.0
    v2 = np.nan

    for i in range(len(df)):
        macd_val = macd.iloc[i] if i < len(macd) else macd.iloc[-1]

        if i >= stc_len:
            window_macd = macd.iloc[i - stc_len + 1:i + 1].values
            max_val = np.nanmax(window_macd)
            min_val = np.nanmin(window_macd)
            k_val = ((macd_val - min_val) / (max_val - min_val) * 100) if max_val != min_val else prev_k
            k_val = k_val if not np.isnan(k_val) else prev_k
        else:
            k_val = prev_k
        prev_k = k_val

        d = stc_factor * k_val + (1 - stc_factor) * d if i >= stc_len else d
        kd = k_val - d
        numerator = abs(kd)

        if i >= stc_len:
            denom = np.nanmax(numerator_arr[max(0, i - stc_len + 1):i + 1]) if i > 0 else numerator
        else:
            denom = numerator if i == 0 else np.nanmax(numerator_arr[:i + 1])
        fract = (numerator / denom * 100) if denom != 0 else prev_fract
        fract = fract if not np.isnan(fract) else prev_fract
        prev_fract = fract

        if i == 0 or np.isnan(v2_arr[i - 1]):
            v2 = fract * stc_factor
        else:
            v2 = v2_arr[i - 1] + stc_factor * (fract - v2_arr[i - 1])
        v2_arr[i] = v2
        stc_arr[i] = 100 * v2

        numerator_arr = np.zeros(len(df)) if i == 0 else numerator_arr
        numerator_arr[i] = numerator

    stc = pd.Series(stc_arr, index=df.index)

    # Entry conditions
    hull_long = (
        (~use_hull) |
        ((use_hull & use_color) & (sig_hull > 0) & (df['close'] > hull_ma)) |
        ((use_hull & ~use_color) & (df['close'] > hull_ma))
    )

    basic_long = (t3_signals > 0) & (df['close'] > t3)
    t3_long = (
        (~use_t3) |
        ((use_t3 & highlight_movements) & basic_long) |
        ((use_t3 & ~highlight_movements) & (df['close'] > t3))
    )

    t3_long_series = pd.Series(t3_long, index=df.index)
    t3_long_prev = t3_long_series.shift(1).fillna(False).astype(bool)
    t3_long_curr = t3_long_series.astype(bool)

    t3_long_cross = (
        (~cross_t3) |
        ((cross_t3) & (~t3_long_prev) & t3_long_curr)
    )

    t3_final = (
        (~inverse_t3) |
        (~t3_long_cross)
    )

    long_condition = hull_long & t3_final & (stc > stc_threshold)

    # Generate entries
    entries = []
    trade_num = 1
    position_size = 0

    for i in range(1, len(df)):
        if position_size == 0 and long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry)
            trade_num += 1
            position_size = 1

    return entries