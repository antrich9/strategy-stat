import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    useHEV = True
    crossHEV = True
    inverseHEV = False
    highlightMovementsHEV = True
    length = 200
    HV_ma = 20
    divisor = 3.6
    
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    jmaJMA = pd.Series(0.0, index=df.index)
    e0JMA = pd.Series(0.0, index=df.index)
    e1JMA = pd.Series(0.0, index=df.index)
    e2JMA = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        e0JMA.iloc[i] = (1 - alphajmaJMA) * close.iloc[i] + alphajmaJMA * e0JMA.iloc[i-1]
        e1JMA.iloc[i] = (close.iloc[i] - e0JMA.iloc[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA.iloc[i-1]
        e2JMA.iloc[i] = (e0JMA.iloc[i] + phasejmaJMARatiojmaJMA * e1JMA.iloc[i] - jmaJMA.iloc[i-1]) * (1 - alphajmaJMA)**2 + alphajmaJMA**2 * e2JMA.iloc[i-1]
        jmaJMA.iloc[i] = e2JMA.iloc[i] + jmaJMA.iloc[i-1]
    
    pct_change = close.pct_change(trendilo_smooth) * 100
    window = pd.Series(range(trendilo_length), dtype=float)
    m = (trendilo_offset * (trendilo_length - 1))
    s = (trendilo_sigma * pd.Series(range(trendilo_length), dtype=float) / trendilo_length)
    alma_weights = np.exp(-np.square(window - m) / (2 * np.square(s)))
    alma_weights = alma_weights / alma_weights.sum()
    
    avg_pct_change = pct_change.rolling(trendilo_length).apply(lambda x: np.sum(x * alma_weights), raw=True)
    rms = trendilo_bmult * np.sqrt((avg_pct_change**2).rolling(trendilo_length).mean())
    trendilo_dir = pd.Series(np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0)), index=df.index)
    
    range_1 = high - low
    rangeAvg = range_1.rolling(length).mean()
    durchschnitt = volume.rolling(HV_ma).mean()
    volumeA = volume.rolling(length).mean()
    
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
    
    gr_enabled1 = (range_1 > rangeAvg) & (close > d1) & (close < u1) & (volume > volumeA) & (volume < volumeA * 1.5) & (volume > volume.shift(1))
    gr_enabled2 = (range_1 < rangeAvg / 1.5) & (volume < volumeA / 1.5)
    gr_enabled3 = (close > d1) & (close < u1)
    gr_enabled = gr_enabled1 | gr_enabled2 | gr_enabled3
    
    basicLongHEVCondition = g_enabled & (volume > durchschnitt)
    basicShorHEVondition = r_enabled & (volume > durchschnitt)
    
    HEVSignalsLongBase = basicLongHEVCondition if highlightMovementsHEV else g_enabled
    HEVSignalsShortBase = basicShorHEVondition if highlightMovementsHEV else r_enabled
    
    HEVSignalsLongCross = (~HEVSignalsLongBase.shift(1).fillna(False)) & HEVSignalsLongBase if crossHEV else HEVSignalsLongBase
    HEVSignalsShorHEVross = (~HEVSignalsShortBase.shift(1).fillna(False)) & HEVSignalsShortBase if crossHEV else HEVSignalsShortBase
    
    HEVSignalsLongFinal = HEVSignalsShorHEVross if inverseHEV else HEVSignalsLongCross
    HEVSignalsShortFinal = HEVSignalsLongCross if inverseHEV else HEVSignalsShorHEVross
    
    jmaJMA_prev = jmaJMA.shift(1)
    close_series = close
    
    if usejmaJMA:
        if usecolorjmaJMA:
            signalmaJMALong = (jmaJMA > jmaJMA_prev) & (close_series > jmaJMA)
            signalmaJMAShort = (jmaJMA < jmaJMA_prev) & (close_series < jmaJMA)
        else:
            signalmaJMALong = close_series > jmaJMA
            signalmaJMAShort = close_series < jmaJMA
    else:
        signalmaJMALong = pd.Series(True, index=df.index)
        signalmaJMAShort = pd.Series(True, index=df.index)
    
    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort
    
    long_condition = finalLongSignalJMA & (trendilo_dir == 1) & basicLongHEVCondition
    short_condition = finalShortSignalJMA & (trendilo_dir == -1) & basicShorHEVondition
    
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pd.isna(jmaJMA.iloc[i]) or pd.isna(avg_pct_change.iloc[i]) or pd.isna(rangeAvg.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries