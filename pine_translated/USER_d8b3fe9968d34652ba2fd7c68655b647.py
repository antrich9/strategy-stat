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
    # Parameters from Pine Script
    lengthT3 = 5
    factorT3 = 0.7
    ff_up_tolerance = 2.5
    ff_down_tolerance = 2.5
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    adx_len = 14
    di_len = 14

    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']

    # T3 Indicator
    def gdT3(src, length, factor):
        ema = src.ewm(span=length, adjust=False).mean()
        return src.ewm(span=length, adjust=False).mean() * (1 + factor) - ema.ewm(span=length, adjust=False).mean() * factor

    t3 = gdT3(gdT3(gdT3(close, lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    t3_signals = (t3 > t3.shift(1)).astype(int).replace(0, -1)

    basic_long = (t3_signals > 0) & (close > t3)
    basic_short = (t3_signals < 0) & (close < t3)

    # Fat Fingers
    bar_upper = close.shift(1).where(close.shift(1) >= open_price.shift(1), open_price.shift(1))
    bar_lower = close.shift(1).where(close.shift(1) <= open_price.shift(1), open_price.shift(1))

    ff_up = ((high - bar_upper) / bar_upper * 100).fillna(0)
    ff_up = ff_up.where(high > bar_upper, 0)
    ff_down = ((bar_lower - low) / bar_lower * 100).fillna(0)
    ff_down = ff_down.where(low < bar_lower, 0)

    ff_up_signal = ff_up >= ff_up_tolerance
    ff_down_signal = ff_down >= ff_down_tolerance

    # Trendilo
    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(src, length, offset, sigma):
        window = length
        m = (window - 1) * offset
        s = sigma * np.sqrt(2 * np.log(window))
        k = np.arange(window)
        weights = np.exp(-0.5 * ((k - m) / s) ** 2)
        weights = weights / weights.sum()

        def roll_alma(x):
            if len(x) < window or np.any(np.isnan(x)):
                return np.nan
            return np.sum(weights * x)

        return pct_change.rolling(window).apply(roll_alma, raw=True)

    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())

    trendilo_dir = pd.Series(0, index=close.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    # ADX (Wilder)
    plusDM = high.diff()
    minusDM = -low.diff()
    plusDM = plusDM.where((plusDM > minusDM) & (plusDM > 0), 0)
    minusDM = minusDM.where((minusDM > plusDM) & (minusDM > 0), 0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    truerange = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def wilder_smooth(series, period):
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    atr = wilder_smooth(truerange, di_len)
    plusDI = 100 * wilder_smooth(plusDM, di_len) / atr
    minusDI = 100 * wilder_smooth(minusDM, di_len) / atr
    dx = 100 * (plusDI - minusDI).abs() / (plusDI + minusDI)
    adx_value = wilder_smooth(dx, adx_len)

    # Entry conditions (from Pine Script - only the active ones)
    long_condition = basic_long & ff_up_signal & (trendilo_dir == 1) & (adx_value > 20)
    short_condition = basic_short & ff_down_signal & (trendilo_dir == -1) & (adx_value > 20)

    # Generate entries
    entries = []
    trade_num = 1
    in_position = False

    for i in range(len(df)):
        if i == 0:
            continue
        if in_position:
            in_position = False
            continue

        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
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
            in_position = True
        elif short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
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
            in_position = True

    return entries