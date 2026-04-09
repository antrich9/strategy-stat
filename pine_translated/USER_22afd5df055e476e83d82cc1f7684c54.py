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
    # Strategy parameters
    L = 30
    vzLen = 60
    vzThr = -0.5
    rLen = 60
    fadeNoLow = 0.01
    longThr = 1.0
    shortThr = -1.0
    useHTF = True
    adxLen = 14
    adxThr = 25.0

    # HTF bias (use_htf=True requires HTF data - we default to True since we don't have HTF data)
    htfBias = True if not useHTF else True

    # Calculate volume shock components
    volTrend = df['volume'].rolling(L).mean()
    vs = np.where(
        (df['volume'] > 0) & (volTrend > 0),
        np.log(df['volume']) - np.log(volTrend),
        np.nan
    )
    vs = pd.Series(vs, index=df.index)

    # zScore function
    def zscore(series, length):
        m = series.rolling(length).mean()
        d = series.rolling(length).std()
        return np.where(d == 0.0, 0.0, (series - m) / d)

    vsZ = zscore(vs, vzLen)
    lowVS = vsZ <= vzThr

    # Return z-score
    ret1 = np.log(df['close'] / df['close'].shift(1))
    retZ = zscore(pd.Series(ret1), rLen)

    # Liquidity Void Oscillator
    base = retZ
    weight = np.where(lowVS, 1.0, fadeNoLow)
    osc = base * weight

    # ADX calculation (Wilder smoothing)
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low),
                       np.maximum(high - high.shift(1), 0), 0)
    minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)),
                        np.maximum(low.shift(1) - low, 0), 0)

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)
    tr_series = pd.Series(tr, index=df.index)

    # Wilder smoothing
    def wilder_smooth(series, length):
        result = pd.Series(index=df.index)
        result.iloc[length - 1] = series.iloc[:length].sum()
        alpha = 1.0 / length
        for i in range(length, len(df)):
            result.iloc[i] = result.iloc[i - 1] * (1 - alpha) + series.iloc[i]
        return result

    atr = wilder_smooth(tr_series, adxLen)
    dip = wilder_smooth(plus_dm_series, adxLen)
    dim = wilder_smooth(minus_dm_series, adxLen)

    dips = dip / atr * 100
    dims = dim / atr * 100
    dx = np.abs(dips - dims) / (dips + dims) * 100
    adx = wilder_smooth(pd.Series(dx, index=df.index), adxLen)

    trendStrong = adx > adxThr

    # Time window (London time)
    def to_london_ts(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.replace(tzinfo=None)

    morning_start = datetime(2000, 1, 1, 6, 45)
    morning_end = datetime(2000, 1, 1, 9, 45)
    afternoon_start = datetime(2000, 1, 1, 14, 45)
    afternoon_end = datetime(2000, 1, 1, 16, 45)

    in_trading_window = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        dt = to_london_ts(df['time'].iloc[i])
        time_only = dt.time()
        m_start = morning_start.time()
        m_end = morning_end.time()
        a_start = afternoon_start.time()
        a_end = afternoon_end.time()
        is_morning = m_start <= time_only < m_end
        is_afternoon = a_start <= time_only < a_end
        in_trading_window[i] = is_morning or is_afternoon

    # Define conditions
    VoidAbove = (osc > longThr) & (osc.shift(1) <= longThr)
    VoidBelow = (osc < shortThr) & (osc.shift(1) >= shortThr)

    # Build valid mask (skip bars where required indicators are NaN)
    valid_mask = ~(osc.isna() | adx.isna() | vsZ.isna() | retZ.isna())
    valid_indices = df.index[valid_mask]

    entries = []
    trade_num = 1

    for i in valid_indices:
        if isinstance(i, int):
            idx = i
        else:
            idx = i

        osc_val = osc.iloc[idx]
        htf_allow_long = htfBias if useHTF else True
        htf_allow_short = (not htfBias) if useHTF else True

        # Long entry: VoidBelow + HTF bias + trend strong + no position + time window
        longCond = (VoidBelow.iloc[idx] and htf_allow_long and trendStrong.iloc[idx]
                    and in_trading_window[idx])

        # Short entry: VoidAbove + (!htfBias or !useHTF) + trend strong + no position + time window
        shortCond = (VoidAbove.iloc[idx] and htf_allow_short and trendStrong.iloc[idx]
                     and in_trading_window[idx])

        if longCond:
            entry_price = df['close'].iloc[idx]
            entry_ts = int(df['time'].iloc[idx])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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

        if shortCond:
            entry_price = df['close'].iloc[idx]
            entry_ts = int(df['time'].iloc[idx])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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