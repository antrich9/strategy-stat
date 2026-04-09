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
    # ---------- ALMA ----------
    def alma(series: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
        arr = np.arange(length, dtype=float)
        m = (length - 1) * offset
        w = np.exp(-((arr - m) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        # rolling weighted average
        result = series.rolling(window=length, min_periods=length).apply(lambda x: np.dot(x, w), raw=True)
        return result

    # ---------- Trendilo ----------
    length_trendilo = 50
    alma_offset = 0.85
    alma_sigma = 6.0
    bmult = 1.0
    smooth = 1

    pch = df['close'].pct_change(smooth) * 100.0
    avpch = alma(pch, length_trendilo, alma_offset, alma_sigma)
    rms = bmult * np.sqrt(avpch.pow(2).rolling(length_trendilo).mean())
    cdir = np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0))

    # ---------- T3 ----------
    factor_t3 = 0.7
    length_t3 = 5

    def gd(src: pd.Series, length: int, factor: float) -> pd.Series:
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = gd(gd(gd(df['close'], length_t3, factor_t3), length_t3, factor_t3), length_t3, factor_t3)

    t3_up = t3 > t3.shift(1)
    t3_down = t3 < t3.shift(1)

    close_series = df['close']
    t3_bull_state = t3_up & (close_series > t3)
    t3_bear_state = t3_down & (close_series < t3)

    t3_bull_cross = t3_bull_state & (~t3_bull_state.shift(1).fillna(False))

    t3_long_signal = t3_bull_cross          # inverse false
    t3_short_signal = t3_bear_state         # persistent bear

    # ---------- WAE ----------
    fast_length = 20
    slow_length = 40
    sensitivity = 150
    channel_length = 20
    bb_mult = 2.0

    wae_macd = close_series.ewm(span=fast_length, adjust=False).mean() - close_series.ewm(span=slow_length, adjust=False).mean()
    t1 = (wae_macd - wae_macd.shift(1)) * sensitivity

    sma = close_series.rolling(channel_length).mean()
    stdev = close_series.rolling(channel_length).std()   # sample stdev (ddof=1)

    e1 = (sma + bb_mult * stdev) - (sma - bb_mult * stdev)

    wae_bullish = (t1 > e1) & (t1 > 0)
    wae_bearish = (t1 < e1) & (t1 < 0)

    # first candle filter (default disabled)
    wae_bull_first = wae_bullish
    wae_bear_first = wae_bearish

    # ---------- Entry Conditions ----------
    cdir_series = pd.Series(cdir, index=df.index)

    long_condition  = t3_long_signal & (cdir_series == 1) & wae_bull_first
    short_condition = t3_short_signal & (cdir_series == -1) & wae_bear_first

    # ---------- Generate Entries ----------
    # mask out bars where any component is NaN
    valid = ~(long_condition.isna() | short_condition.isna())

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if not valid.iloc[i]:
            continue
        long_ok  = long_condition.iloc[i]
        short_ok = short_condition.iloc[i]

        if long_ok:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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

        if short_ok:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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