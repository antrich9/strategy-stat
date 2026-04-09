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
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    jmaJMA = np.zeros(len(df))
    e0JMA = np.zeros(len(df))
    e1JMA = np.zeros(len(df))
    e2JMA = np.zeros(len(df))
    
    srcjmaJMA = close.values
    for i in range(1, len(df)):
        e0JMA[i] = (1 - alphajmaJMA) * srcjmaJMA[i] + alphajmaJMA * e0JMA[i-1]
        e1JMA[i] = (srcjmaJMA[i] - e0JMA[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA[i-1]
        e2JMA[i] = (e0JMA[i] + phasejmaJMARatiojmaJMA * e1JMA[i] - jmaJMA[i-1]) * ((1 - alphajmaJMA) ** 2) + (alphajmaJMA ** 2) * e2JMA[i-1]
        jmaJMA[i] = e2JMA[i] + jmaJMA[i-1]
    
    jmaJMA_series = pd.Series(jmaJMA)
    
    signalmaJMALong = (jmaJMA_series > jmaJMA_series.shift(1)) & (close > jmaJMA_series)
    signalmaJMAShort = (jmaJMA_series < jmaJMA_series.shift(1)) & (close < jmaJMA_series)
    
    inverseJMA = True
    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = (close - close.shift(trendilo_smooth)) / close * 100
    
    def alma(arr, length, offset, sigma):
        w = np.arange(length) + 1
        m = (offset * (length - 1))
        s = sigma * (length - 1) / 6
        w = np.exp(-((w - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        return pd.Series(w).rolling(length).apply(lambda x: (x * arr.iloc[-len(x):][::-1].values).sum(), raw=True).fillna(method='bfill')
    
    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms_arr = np.sqrt(avg_pct_change.rolling(trendilo_length).apply(lambda x: (x * x).sum() / trendilo_length, raw=True))
    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms_arr * trendilo_bmult] = 1
    trendilo_dir[avg_pct_change < -rms_arr * trendilo_bmult] = -1
    
    length_hev = 200
    range_1 = high - low
    rangeAvg = range_1.rolling(length_hev).mean()
    HV_ma = 20
    durchschnitt = volume.rolling(HV_ma).mean()
    volumeA = volume.rolling(length_hev).mean()
    divisor = 3.6
    
    high1 = high.shift(1)
    low1 = low.shift(1)
    mid1 = ((high + low) / 2).shift(1)
    
    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor
    
    r_enabled1 = (range_1 > rangeAvg) & (close < d1) & (volume > volumeA)
    r_enabled2 = close < mid1
    r_enabled = r_enabled1 | r_enabled2
    
    g_enabled1 = close > mid1
    g_enabled2 = (range_1 > rangeAvg) & (close > u1) & (volume > volumeA)
    g_enabled3 = (high > high1) & (range_1 < rangeAvg / 1.5) & (volume < volumeA)
    g_enabled4 = (low < low1) & (range_1 < rangeAvg / 1.5) & (volume > volumeA)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4
    
    basicLongHEVCondition = g_enabled & (volume > durchschnitt)
    basicShorHEVondition = r_enabled & (volume > durchschnitt)
    
    useHEV = True
    highlightMovementsHEV = True
    HEVSignalsLong = basicLongHEVCondition if useHEV else pd.Series(True, index=df.index)
    HEVSignalsShort = basicShorHEVondition if useHEV else pd.Series(True, index=df.index)
    
    crossHEV = True
    HEVSignalsLongCross = HEVSignalsLong & ~HEVSignalsLong.shift(1).fillna(False) if crossHEV else HEVSignalsLong
    HEVSignalsShorHEVross = HEVSignalsShort & ~HEVSignalsShort.shift(1).fillna(False) if crossHEV else HEVSignalsShort
    
    inverseHEV = False
    HEVSignalsLongFinal = HEVSignalsShorHEVross if inverseHEV else HEVSignalsLongCross
    HEVSignalsShortFinal = HEVSignalsLongCross if inverseHEV else HEVSignalsShorHEVross
    
    long_condition = finalLongSignalJMA & (trendilo_dir == 1) & HEVSignalsLongFinal
    short_condition = finalShortSignalJMA & (trendilo_dir == -1) & HEVSignalsShortFinal
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
    
    trade_num = 1
    for i in range(1, len(df)):
        if short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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