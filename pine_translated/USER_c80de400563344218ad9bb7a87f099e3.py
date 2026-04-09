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
    lengthT3 = 5
    factorT3 = 0.7
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    ff_up_tolerance = 2.5
    ff_down_tolerance = 2.5
    di_len = 14
    adx_len = 14

    def alma(arr, length, offset, sigma):
        window = length
        m = offset * (window - 1)
        s = sigma * (window - 1) / 6
        k = np.arange(window)
        weights = np.exp(-((k - m) ** 2) / (2 * s * s))
        weights = weights / weights.sum()
        return pd.Series(np.convolve(arr, weights, mode='valid'), index=arr.index[window-1:])

    def wilder_ma(series, period):
        result = series.copy()
        result.iloc[:period] = series.iloc[:period].mean()
        result.iloc[period:] = series.iloc[period:].ewm(alpha=1.0/period, adjust=False).mean()
        return result

    def calc_t3(src, length, factor):
        def gd(s):
            ema1 = s.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema2 * (1 + factor) - ema2.ewm(span=length, adjust=False).mean() * factor
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = gd(ema1)
        ema3 = gd(ema2)
        return gd(ema3)

    def calc_adx(high, low, close, di_len, adx_len):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = tr1.where(tr1 >= tr2, tr2).where(tr1 >= tr3, tr3)
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        tr_s = wilder_ma(tr, di_len)
        plus = 100 * wilder_ma(plus_dm, di_len) / tr_s
        minus = 100 * wilder_ma(minus_dm, di_len) / tr_s
        dx = 100 * wilder_ma(abs(plus - minus) / (plus + minus).where(plus + minus != 0, 1), adx_len)
        return dx

    t3 = calc_t3(df['close'], lengthT3, factorT3)
    t3_signals = pd.Series(np.where(t3 > t3.shift(1), 1, -1), index=df.index)

    bar_upper = df['close'].shift(1).where(df['close'].shift(1) >= df['open'].shift(1), df['open'].shift(1))
    bar_lower = df['close'].shift(1).where(df['close'].shift(1) <= df['open'].shift(1), df['open'].shift(1))

    ff_up = ((df['high'] - bar_upper) / bar_upper * 100).clip(lower=0)
    ff_down = ((df['low'] - bar_lower) / bar_lower * 100).clip(lower=0)

    ff_up_signal = ff_up >= ff_up_tolerance
    ff_down_signal = ff_down >= ff_down_tolerance

    pct_change = df['close'].diff(trendilo_smooth) / df['close'] * 100
    avg_pct_change = alma(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = pd.Series(avg_pct_change, index=df.index[trendilo_length-1:])
    rms = trendilo_sigma * 1.0 * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    adx_value = calc_adx(df['high'], df['low'], df['close'], di_len, adx_len)

    basic_long = (t3_signals > 0) & (df['close'] > t3)
    basic_short = (t3_signals < 0) & (df['close'] < t3)

    long_condition = basic_long & ff_up_signal & (trendilo_dir == 1) & (adx_value > 20)
    short_condition = basic_short & ff_down_signal & (trendilo_dir == -1) & (adx_value > 20)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(adx_value.iloc[i]):
            continue
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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