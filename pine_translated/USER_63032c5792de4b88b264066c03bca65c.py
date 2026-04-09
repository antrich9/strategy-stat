import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wma(series, length):
    weights = np.arange(1, length + 1)
    def weighted_avg(window):
        return np.sum(window * weights) / weights.sum()
    return series.rolling(window=length).apply(weighted_avg, raw=True)

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 0
    
    lengthHullMA = 9
    half_length = lengthHullMA // 2
    sqrt_length = int(np.sqrt(lengthHullMA))
    
    lengthT3 = 5
    factorT3 = 0.7
    braid_length = 9
    braid_mult = 1.0
    
    useHullMA = True
    usecolorHullMA = True
    useT3 = True
    crossT3 = True
    inverseT3 = False
    highlightMovementsT3 = True
    
    wma_half = wma(df['close'], half_length)
    wma_full = wma(df['close'], lengthHullMA)
    hullma = wma(2 * wma_half - wma_full, sqrt_length)
    hullma_diff = hullma.diff()
    
    def gd_t3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    
    t3 = gd_t3(gd_t3(gd_t3(df['close'], lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    t3Change = t3.diff()
    
    close_prev = df['close'].shift(1).fillna(df['close'])
    high_low = df['high'] - df['low']
    high_close = (df['high'] - close_prev).abs()
    low_close = (df['low'] - close_prev).abs()
    braid_truerange = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    braid_atr = braid_truerange.rolling(braid_length).mean()
    braid_upper = (df['high'] + df['low']) / 2 + braid_atr * braid_mult
    braid_lower = (df['high'] + df['low']) / 2 - braid_atr * braid_mult
    
    if useHullMA:
        if usecolorHullMA:
            signalHullMALong = (hullma_diff > 0) & (df['close'] > hullma)
        else:
            signalHullMALong = df['close'] > hullma
    else:
        signalHullMALong = pd.Series(True, index=df.index)
    
    basicLongCondition = (t3Change > 0) & (df['close'] > t3)
    if useT3:
        if highlightMovementsT3:
            t3SignalsLong = basicLongCondition
        else:
            t3SignalsLong = df['close'] > t3
    else:
        t3SignalsLong = pd.Series(True, index=df.index)
    
    if crossT3:
        t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong
    else:
        t3SignalsLongCross = t3SignalsLong
    
    if inverseT3:
        t3SignalsLongFinal = ~t3SignalsLongCross
    else:
        t3SignalsLongFinal = t3SignalsLongCross
    
    braidFilterLong = df['close'] > braid_upper
    
    fullCondition = signalHullMALong & t3SignalsLongFinal & braidFilterLong
    
    for i in range(1, len(df)):
        if i < sqrt_length or i < lengthT3 or i < braid_length:
            continue
        if np.isnan(hullma.iloc[i]) or np.isnan(t3.iloc[i]) or np.isnan(braid_upper.iloc[i]):
            continue
        if fullCondition.iloc[i] and (i == 0 or not fullCondition.iloc[i-1]):
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries