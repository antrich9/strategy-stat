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
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    time = df['time'].copy()

    # JMA parameters
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True

    # Trendilo parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    # JMA calculation
    phasejmaJMARatiojmaJMA = np.where(phasejmaJMA < -100, 0.5,
                            np.where(phasejmaJMA > 100, 2.5, phasejmaJMA / 100 + 1.5))
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = np.power(betajmaJMA, powerjmaJMA)

    # Recursive JMA calculation using loop
    jmaJMA = pd.Series(np.zeros(len(df)), index=df.index)
    e0JMA = 0.0
    e1JMA = 0.0
    e2JMA = 0.0

    for i in range(1, len(df)):
        e0JMA = (1 - alphajmaJMA) * close.iloc[i] + alphajmaJMA * e0JMA
        e1JMA = (close.iloc[i] - e0JMA) * (1 - betajmaJMA) + betajmaJMA * e1JMA
        e2JMA = (e0JMA + phasejmaJMARatiojmaJMA * e1JMA - jmaJMA.iloc[i-1]) * np.power(1 - alphajmaJMA, 2) + np.power(alphajmaJMA, 2) * e2JMA
        jmaJMA.iloc[i] = e2JMA + jmaJMA.iloc[i-1]

    jmaJMA_prev = jmaJMA.shift(1)

    # JMA signals
    if usejmaJMA:
        if usecolorjmaJMA:
            signalmaJMALong = (jmaJMA > jmaJMA_prev) & (close > jmaJMA)
            signalmaJMAShort = (jmaJMA < jmaJMA_prev) & (close < jmaJMA)
        else:
            signalmaJMALong = close > jmaJMA
            signalmaJMAShort = close < jmaJMA
    else:
        signalmaJMALong = pd.Series(True, index=df.index)
        signalmaJMAShort = pd.Series(True, index=df.index)

    # Final signals with inverse
    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort

    # Trendilo calculation
    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(arr, length, offset, sigma):
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * length / 6
        w = np.exp(-((window - m) ** 2) / (2 * s ** 2))
        w = w / w.sum()
        return pd.Series(arr).rolling(length).apply(lambda x: np.dot(x, w), raw=True)

    avg_pct_change = alma(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = pd.Series(avg_pct_change, index=df.index)

    rms_vals = np.sqrt(pd.Series(avg_pct_change * avg_pct_change).rolling(trendilo_length).mean() * trendilo_bmult)
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir = np.where(avg_pct_change > rms_vals, 1, np.where(avg_pct_change < -rms_vals, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)

    # Time window (afternoon: 14:45 - 16:45 London)
    def is_in_afternoon_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        current_mins = hour * 60 + minute
        start_mins = 14 * 60 + 45  # 14:45
        end_mins = 16 * 60 + 45    # 16:45
        return start_mins <= current_mins < end_mins

    in_trading_window = time.apply(is_in_afternoon_window)

    # Entry conditions
    long_condition = finalLongSignalJMA & (trendilo_dir == 1)
    short_condition = finalShortSignalJMA & (trendilo_dir == -1)

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(jmaJMA.iloc[i]) or pd.isna(trendilo_dir.iloc[i]):
            continue

        entry_price = close.iloc[i]
        ts = int(time.iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_condition.iloc[i] and in_trading_window.iloc[i]:
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

        if short_condition.iloc[i] and in_trading_window.iloc[i]:
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