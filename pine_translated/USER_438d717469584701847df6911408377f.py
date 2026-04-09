import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    factorT3 = 0.7
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    length_hev = 200
    HV_ma = 20
    divisor = 3.6
    
    def calc_t3(src, length):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        c1 = -factorT3 * factorT3 * factorT3
        c2 = 3 * factorT3 * factorT3 + 3 * factorT3 * factorT3 * factorT3
        c3 = -6 * factorT3 * factorT3 - 3 * factorT3 - 3 * factorT3 * factorT3 * factorT3
        c4 = 1 + 3 * factorT3 + factorT3 * factorT3 + 3 * factorT3 * factorT3
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    
    def alma(arr, length, offset, sigma):
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * length / 6.0
        w = np.exp(-((window - m) ** 2) / (2 * s ** 2))
        w = w / w.sum()
        return pd.Series(np.convolve(arr, w, mode='same'), index=arr.index)
    
    t3_25 = calc_t3(close, 25)
    t3_100 = calc_t3(close, 100)
    t3_200 = calc_t3(close, 200)
    
    longConditionIndiT3 = (close > t3_25) & (close > t3_100) & (close > t3_200)
    shortConditionIndiT3 = (close < t3_25) & (close < t3_100) & (close < t3_200)
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)
    
    signalTypeT3 = 'MA + Price'
    crossOnlyT3 = True
    inverseT3 = False
    
    if signalTypeT3 == 'MA + Price':
        signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
        signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA
    elif signalTypeT3 == 'MA Only':
        signalEntryLongT3 = longConditionIndiT3MA
        signalEntryShortT3 = shortConditionIndiT3MA
    else:
        signalEntryLongT3 = longConditionIndiT3
        signalEntryShortT3 = shortConditionIndiT3
    
    if crossOnlyT3:
        prev_long = signalEntryLongT3.shift(1).fillna(False)
        prev_short = signalEntryShortT3.shift(1).fillna(False)
        signalEntryLongT3 = signalEntryLongT3 & ~prev_long
        signalEntryShortT3 = signalEntryShortT3 & ~prev_short
    
    finalLongSignalT3 = signalEntryShortT3 if inverseT3 else signalEntryLongT3
    finalShortSignalT3 = signalEntryLongT3 if inverseT3 else signalEntryShortT3
    
    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    avg_pct_change_alma = alma(avg_pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change_alma = pd.Series(avg_pct_change_alma, index=avg_pct_change.index)
    rms = trendilo_bmult * np.sqrt((avg_pct_change_alma ** 2).rolling(trendilo_length).mean())
    trendilo_dir = pd.Series(0, index=close.index)
    trendilo_dir[avg_pct_change_alma > rms] = 1
    trendilo_dir[avg_pct_change_alma < -rms] = -1
    
    range_1 = high - low
    rangeAvg = range_1.rolling(length_hev).mean()
    durchschnitt = volume.rolling(HV_ma).mean()
    volumeA = volume.rolling(length_hev).mean()
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
    crossHEV = True
    inverseHEV = False
    highlightMovementsHEV = True
    
    HEVSignalsLong = basicLongHEVCondition if highlightMovementsHEV else g_enabled
    HEVSignalsShort = basicShorHEVondition if highlightMovementsHEV else r_enabled
    
    if crossHEV:
        HEVSignalsLong = HEVSignalsLong & (~HEVSignalsLong.shift(1).fillna(False))
        HEVSignalsShort = HEVSignalsShort & (~HEVSignalsShort.shift(1).fillna(False))
    
    HEVSignalsLongFinal = HEVSignalsShort if inverseHEV else HEVSignalsLong
    HEVSignalsShortFinal = HEVSignalsLong if inverseHEV else HEVSignalsShort
    
    long_condition = finalLongSignalT3 & (trendilo_dir == 1) & basicLongHEVCondition
    short_condition = finalShortSignalT3 & (trendilo_dir == -1) & basicShorHEVondition
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries