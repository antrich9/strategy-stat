import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # JMA Parameters
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    srcjmaJMA = df['close']
    highlightMovementsjmaJMA = True
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True

    # JMA calculation
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA

    n = len(df)
    e0JMA = np.zeros(n)
    e1JMA = np.zeros(n)
    e2JMA = np.zeros(n)
    jmaJMA = np.zeros(n)

    for i in range(1, n):
        e0JMA[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i] + alphajmaJMA * e0JMA[i-1]
        e1JMA[i] = (srcjmaJMA.iloc[i] - e0JMA[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA[i-1]
        e2JMA[i] = (e0JMA[i] + phasejmaJMARatiojmaJMA * e1JMA[i] - jmaJMA[i-1]) * (1 - alphajmaJMA) ** 2 + alphajmaJMA ** 2 * e2JMA[i-1]
        jmaJMA[i] = e2JMA[i] + jmaJMA[i-1]

    jmaJMA_series = pd.Series(jmaJMA, index=df.index)

    # signalmaJMA
    if usejmaJMA:
        if usecolorjmaJMA:
            signalmaJMALong = (jmaJMA_series > jmaJMA_series.shift(1)) & (df['close'] > jmaJMA_series)
            signalmaJMAShort = (jmaJMA_series < jmaJMA_series.shift(1)) & (df['close'] < jmaJMA_series)
        else:
            signalmaJMALong = df['close'] > jmaJMA_series
            signalmaJMAShort = df['close'] < jmaJMA_series
    else:
        signalmaJMALong = pd.Series(True, index=df.index)
        signalmaJMAShort = pd.Series(True, index=df.index)

    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = df['close'].diff(trendilo_smooth) / df['close'] * 100
    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)

    # ALMA approximation
    def alma_approx(series, length, offset, sigma):
        window = min(length, len(series))
        if window < 2:
            return series.rolling(window, min_period=1).mean()
        k = np.arange(window)
        w = np.exp(-((k - np.floor(offset * (window - 1))) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        return series.rolling(window, min_period=1).apply(lambda x: np.convolve(x, w, mode='valid')[0] if len(x) >= window else np.nan, raw=True)

    avg_pct_change = alma_approx(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = pd.Series(avg_pct_change, index=df.index)

    rms_vals = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        window = avg_pct_change.iloc[i - trendilo_length + 1:i + 1]
        rms_vals[i] = trendilo_bmult * np.sqrt((window ** 2).mean())
    rms = pd.Series(rms_vals, index=df.index)

    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    # HawkEye Volume
    length_hev = 200
    range_1 = df['high'] - df['low']
    rangeAvg = range_1.rolling(window=length_hev).mean()
    HV_ma = 20
    durchschnitt = df['volume'].rolling(window=HV_ma).mean()

    volumeA = df['volume'].rolling(window=length_hev).mean()
    divisor = 3.6

    high1 = df['high'].shift(1)
    low1 = df['low'].shift(1)
    mid1 = ((df['high'] + df['low']) / 2).shift(1)

    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor

    r_enabled1 = (range_1 > rangeAvg) & (df['close'] < d1) & (df['volume'] > volumeA)
    r_enabled2 = df['close'] < mid1
    r_enabled = r_enabled1 | r_enabled2

    g_enabled1 = df['close'] > mid1
    g_enabled2 = (range_1 > rangeAvg) & (df['close'] > u1) & (df['volume'] > volumeA)
    g_enabled3 = (df['high'] > high1) & (range_1 < rangeAvg / 1.5) & (df['volume'] < volumeA)
    g_enabled4 = (df['low'] < low1) & (range_1 < rangeAvg / 1.5) & (df['volume'] > volumeA)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4

    basicLongHEVCondition = g_enabled & (df['volume'] > durchschnitt)
    basicShorHEVondition = r_enabled & (df['volume'] > durchschnitt)

    crossHEV = True
    inverseHEV = False

    HEVSignalsLong = basicLongHEVCondition
    HEVSignalsShort = basicShorHEVondition

    HEVSignalsLongCross = HEVSignalsLong & ~HEVSignalsLong.shift(1).fillna(False) if crossHEV else HEVSignalsLong
    HEVSignalsShorHEVross = HEVSignalsShort & ~HEVSignalsShort.shift(1).fillna(False) if crossHEV else HEVSignalsShort

    HEVSignalsLongFinal = HEVSignalsShorHEVross if inverseHEV else HEVSignalsLongCross
    HEVSignalsShortFinal = HEVSignalsLongCross if inverseHEV else HEVSignalsShorHEVross

    # Entry conditions
    long_condition = signalmaJMAShort & (trendilo_dir == 1) & basicLongHEVCondition
    short_condition = signalmaJMALong & (trendilo_dir == -1) & basicShorHEVondition

    # Build entries
    entries = []
    trade_num = 1

    for i in range(2, n):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries