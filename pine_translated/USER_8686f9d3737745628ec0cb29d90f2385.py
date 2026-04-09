import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default parameters from the Pine Script
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    useKalmanFilter = True
    q = 0.001
    r = 0.001
    adxLength = 14
    adxThreshold = 25

    # ── Time‑of‑day filter ──────────────────────────────────────────────────
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = dt.dt.hour
    isValidTradeTime = ((hour >= 2) & (hour < 5)) | ((hour >= 10) & (hour < 12))

    # ── Helper: Weighted Moving Average ─────────────────────────────────────
    def wma(series, window):
        weights = np.arange(1, window + 1)
        def weighted(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window).apply(weighted, raw=True)

    # ── Helper: Exponential Moving Average ─────────────────────────────────
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    # ── Hull MA ─────────────────────────────────────────────────────────────
    half_len = lengthHullMA // 2
    sqrt_len = int(np.sqrt(lengthHullMA))
    wma_half = wma(df['close'], half_len)
    wma_full = wma(df['close'], lengthHullMA)
    hullma = wma(2 * wma_half - wma_full, sqrt_len)

    hullma_prev = hullma.shift(1)
    sigHullMA = pd.Series(np.where(hullma > hullma_prev, 1, -1), index=df.index)

    if useHullMA:
        if usecolorHullMA:
            signalHullLong = (sigHullMA > 0) & (df['close'] > hullma)
        else:
            signalHullLong = df['close'] > hullma
    else:
        signalHullLong = pd.Series(True, index=df.index)

    # ── T3 ──────────────────────────────────────────────────────────────────
    ema1 = ema(df['close'], lengthT3)
    ema2 = ema(ema1, lengthT3)
    gd1 = ema1 * (1 + factorT3) - ema2 * factorT3

    ema3 = ema(gd1, lengthT3)
    ema4 = ema(ema3, lengthT3)
    gd2 = ema3 * (1 + factorT3) - ema4 * factorT3

    ema5 = ema(gd2, lengthT3)
    ema6 = ema(ema5, lengthT3)
    t3 = ema5 * (1 + factorT3) - ema6 * factorT3

    t3_prev = t3.shift(1)
    t3Signals = pd.Series(np.where(t3 > t3_prev, 1, -1), index=df.index)

    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)
    if useT3:
        if highlightMovementsT3:
            t3SignalsLong = basicLongCondition
        else:
            t3SignalsLong = df['close'] > t3
    else:
        t3SignalsLong = pd.Series(True, index=df.index)

    if crossT3:
        t3SignalsLongCross = t3SignalsLong & ~t3SignalsLong.shift(1).fillna(False)
    else:
        t3SignalsLongCross = t3SignalsLong

    if inverseT3:
        t3SignalsLongFinal = ~t3SignalsLongCross
    else:
        t3SignalsLongFinal = t3SignalsLongCross

    # ── Kalman Filter ───────────────────────────────────────────────────────
    def kalman_filter_series(series, q, r):
        n = len(series)
        x = np.nan
        p = 1.0
        out = np.empty(n)
        for i in range(n):
            if np.isnan(x):
                x = series.iloc[i]
            x_pred = x
            p_pred = p + q
            k = p_pred / (p_pred + r)
            x = x_pred + k * (series.iloc[i] - x_pred)
            p = (1 - k) * p_pred
            out[i] = x
        return pd.Series(out, index=series.index)

    kalmanPrice = kalman_filter_series(df['close'], q, r)
    if useKalmanFilter:
        kalmanLongCondition = df['close'] > kalmanPrice
    else:
        kalmanLongCondition = pd.Series(True, index=df.index)

    # ── ADX (Wilder smoothed) ────────────────────────────────────────────────
    def calc_adx(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        high_diff = high.diff()
        low_diff = -low.diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

        def wilder(s):
            return s.ewm(alpha=1.0 / period, adjust=False).mean()

        tr_s = wilder(tr)
        plus_dm_s = wilder(plus_dm)
        minus_dm_s = wilder(minus_dm)

        plus_di = 100 * (plus_dm_s / tr_s)
        minus_di = 100 * (minus_dm_s / tr_s)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = wilder(dx)
        return adx

    adx = calc_adx(df, adxLength)

    # ── Combined entry condition ────────────────────────────────────────────
    entry_cond = (
        signalHullLong &
        t3SignalsLongFinal &
        kalmanLongCondition &
        (adx > adxThreshold) &
        isValidTradeTime
    )

    # ── Build entry list ─────────────────────────────────────────────────────
    entries = []
    trade_num = 1
    for i in df.index:
        if entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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