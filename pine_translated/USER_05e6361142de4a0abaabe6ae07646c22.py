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
    useTimeFilter = False
    useAllHours = False
    useLondon = False
    useNYAM = False
    useNYPM = False

    useSweepFilter = False
    sweepMode = "None"

    lookback_bars = 12
    threshold = 0.0

    superTrendMultiplier = 3
    superTrendPeriod = 10

    trade_num = 1
    entries = []

    n = len(df)

    df = df.copy()
    df['hour'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour

    in_time_window = pd.Series([True] * n, index=df.index)

    ny_hour = df['hour'].values

    in_asian_session = ny_hour >= 19

    asian_high_arr = np.full(n, np.nan)
    asian_low_arr = np.full(n, np.nan)
    asian_swept_high = np.full(n, False)
    asian_swept_low = np.full(n, False)

    tmp_ah = np.nan
    tmp_al = np.nan
    prev_in_asian = False

    for i in range(n):
        if in_asian_session[i]:
            if not prev_in_asian:
                tmp_ah = df['high'].iloc[i]
                tmp_al = df['low'].iloc[i]
            else:
                tmp_ah = max(tmp_ah, df['high'].iloc[i]) if not np.isnan(tmp_ah) else df['high'].iloc[i]
                tmp_al = min(tmp_al, df['low'].iloc[i]) if not np.isnan(tmp_al) else df['low'].iloc[i]

            asian_high_arr[i] = tmp_ah
            asian_low_arr[i] = tmp_al

            if i > 0 and not np.isnan(tmp_ah) and df['high'].iloc[i] > tmp_ah:
                asian_swept_high[i] = True
            else:
                asian_swept_high[i] = asian_swept_high[i-1] if i > 0 else False

            if i > 0 and not np.isnan(tmp_al) and df['low'].iloc[i] < tmp_al:
                asian_swept_low[i] = True
            else:
                asian_swept_low[i] = asian_swept_low[i-1] if i > 0 else False
        else:
            if prev_in_asian and not np.isnan(tmp_ah):
                asian_high_arr[i] = tmp_ah
                asian_low_arr[i] = tmp_al

            if i > 0:
                asian_swept_high[i] = asian_swept_high[i-1]
                asian_swept_low[i] = asian_swept_low[i-1]

        prev_in_asian = in_asian_session[i]

    df['asian_swept_high'] = asian_swept_high
    df['asian_swept_low'] = asian_swept_low

    days = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    new_day = days.diff().fillna(False)

    pd_high_arr = np.full(n, np.nan)
    pd_low_arr = np.full(n, np.nan)
    pd_swept_high = np.full(n, False)
    pd_swept_low = np.full(n, False)

    tmp_ph = np.nan
    tmp_pl = np.nan

    for i in range(n):
        if new_day.iloc[i]:
            if not np.isnan(tmp_ph):
                pd_high_arr[i] = tmp_ph
                pd_low_arr[i] = tmp_pl
            tmp_ph = df['high'].iloc[i]
            tmp_pl = df['low'].iloc[i]
            pd_swept_high[i] = False
            pd_swept_low[i] = False
        else:
            tmp_ph = max(tmp_ph, df['high'].iloc[i]) if not np.isnan(tmp_ph) else df['high'].iloc[i]
            tmp_pl = min(tmp_pl, df['low'].iloc[i]) if not np.isnan(tmp_pl) else df['low'].iloc[i]

            if not pd_swept_high[i-1] and i > 0 and not np.isnan(pd_high_arr[i-1]) and df['high'].iloc[i] > pd_high_arr[i-1]:
                pd_swept_high[i] = True
            else:
                pd_swept_high[i] = pd_swept_high[i-1] if i > 0 else False

            if not pd_swept_low[i-1] and i > 0 and not np.isnan(pd_low_arr[i-1]) and df['low'].iloc[i] < pd_low_arr[i-1]:
                pd_swept_low[i] = True
            else:
                pd_swept_low[i] = pd_swept_low[i-1] if i > 0 else False

    df['pd_swept_high'] = pd_swept_high
    df['pd_swept_low'] = pd_swept_low

    c1_high = df['high'].shift(2)
    c1_low = df['low'].shift(2)
    c2_close = df['close'].shift(1)
    c2_high = df['high'].shift(1)
    c2_low = df['low'].shift(1)

    bullish_bias = (c2_close > c1_high) | ((c2_low < c1_low) & (c2_close > c1_low))
    bearish_bias = (c2_close < c1_low) | ((c2_high > c1_high) & (c2_close < c2_high))

    df['bullish_bias'] = bullish_bias
    df['bearish_bias'] = bearish_bias

    asian_long_ok = df['asian_swept_low'] & (~df['asian_swept_high'])
    asian_short_ok = df['asian_swept_high'] & (~df['asian_swept_low'])
    pd_long_ok = df['pd_swept_low'] & (~df['pd_swept_high'])
    pd_short_ok = df['pd_swept_high'] & (~df['pd_swept_low'])
    bias_long_ok = df['bullish_bias']
    bias_short_ok = df['bearish_bias']

    raw_long_sweep = pd.Series([True] * n, index=df.index)
    raw_short_sweep = pd.Series([True] * n, index=df.index)

    if sweepMode == "Asian Only":
        raw_long_sweep = asian_long_ok
        raw_short_sweep = asian_short_ok
    elif sweepMode == "PD Only":
        raw_long_sweep = pd_long_ok
        raw_short_sweep = pd_short_ok
    elif sweepMode == "Bias Only":
        raw_long_sweep = bias_long_ok
        raw_short_sweep = bias_short_ok
    elif sweepMode == "Asian + Bias":
        raw_long_sweep = asian_long_ok & bias_long_ok
        raw_short_sweep = asian_short_ok & bias_short_ok
    elif sweepMode == "PD + Bias":
        raw_long_sweep = pd_long_ok & bias_long_ok
        raw_short_sweep = pd_short_ok & bias_short_ok
    elif sweepMode == "All Three":
        raw_long_sweep = asian_long_ok & pd_long_ok & bias_long_ok
        raw_short_sweep = asian_short_ok & pd_short_ok & bias_short_ok

    long_sweep_ok = (~useSweepFilter) | raw_long_sweep
    short_sweep_ok = (~useSweepFilter) | raw_short_sweep

    df['long_sweep_ok'] = long_sweep_ok
    df['short_sweep_ok'] = short_sweep_ok

    alpha = 1.0 / superTrendPeriod
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()

    hl2 = (df['high'] + df['low']) / 2
    up = hl2 + (superTrendMultiplier * atr)
    dn = hl2 - (superTrendMultiplier * atr)

    supertrend_up = pd.Series(np.nan, index=df.index)
    supertrend_down = pd.Series(np.nan, index=df.index)
    trend = pd.Series(0, index=df.index)

    for i in range(1, n):
        if np.isnan(supertrend_up.iloc[i-1]):
            trend.iloc[i] = 1
            supertrend_up.iloc[i] = up.iloc[i]
            supertrend_down.iloc[i] = dn.iloc[i]
        else:
            if df['close'].iloc[i] > supertrend_up.iloc[i-1]:
                trend.iloc[i] = 1
                supertrend_up.iloc[i] = max(up.iloc[i], supertrend_up.iloc[i-1])
                supertrend_down.iloc[i] = dn.iloc[i]
            elif df['close'].iloc[i] < supertrend_down.iloc[i-1]:
                trend.iloc[i] = -1
                supertrend_up.iloc[i] = up.iloc[i]
                supertrend_down.iloc[i] = min(dn.iloc[i], supertrend_down.iloc[i-1])
            else:
                trend.iloc[i] = trend.iloc[i-1]
                supertrend_up.iloc[i] = supertrend_up.iloc[i-1]
                supertrend_down.iloc[i] = supertrend_down.iloc[i-1]

    df['supertrend_trend'] = trend

    for i in range(lookback_bars, n):
        if pd.isna(df['supertrend_trend'].iloc[i]) or df['supertrend_trend'].iloc[i] == 0:
            continue

        if df['supertrend_trend'].iloc[i] > 0 and in_time_window.iloc[i] and df['long_sweep_ok'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()

            entries.append({
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
            })
            trade_num += 1

        elif df['supertrend_trend'].iloc[i] < 0 and in_time_window.iloc[i] and df['short_sweep_ok'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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