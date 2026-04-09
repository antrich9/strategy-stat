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
    
    # Default parameters from Pine Script
    lengthHullMA = 9
    srcHullMA = df['close']
    useHullMA = True
    usecolorHullMA = True
    useT3 = True
    crossT3 = True
    inverseT3 = False
    highlightMovementsT3 = True
    lengthT3 = 5
    factorT3 = 0.7
    
    # Hull MA calculation: ta.wma(2*ta.wma(src, len/2) - ta.wma(src, len), sqrt(len))
    half_length = int(lengthHullMA / 2)
    sqrt_length = int(np.floor(np.sqrt(lengthHullMA)))
    
    wma_half = srcHullMA.ewm(span=half_length, adjust=False).mean()
    wma_full = srcHullMA.ewm(span=lengthHullMA, adjust=False).mean()
    hullma = (2 * wma_half - wma_full).ewm(span=sqrt_length, adjust=False).mean()
    
    # T3 calculation: gdT3(gdT3(gdT3(src, len), len), len) where gdT3(x) = ema(x)*(1+f) - ema(ema(x), len)*f
    def gd_t3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    
    t3 = gd_t3(gd_t3(gd_t3(df['close'], lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    
    # Signal calculations
    sigHullMA = (hullma > hullma.shift(1)).astype(int)
    sigHullMA = sigHullMA.replace(0, -1)
    
    t3Signals = (t3 > t3.shift(1)).astype(int)
    t3Signals = t3Signals.replace(0, -1)
    
    # Long entry conditions
    signalHullMALong = (useHullMA and usecolorHullMA) and (sigHullMA > 0) & (df['close'] > hullma)
    signalHullMALong = signalHullMALong | ((useHullMA and not usecolorHullMA) and (df['close'] > hullma))
    signalHullMALong = signalHullMALong | (~useHullMA)
    
    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)
    
    if useT3 and highlightMovementsT3:
        t3SignalsLong = basicLongCondition
    elif useT3:
        t3SignalsLong = df['close'] > t3
    else:
        t3SignalsLong = pd.Series(True, index=df.index)
    
    if crossT3:
        t3SignalsLongCross = t3SignalsLong & ~t3SignalsLong.shift(1)
    else:
        t3SignalsLongCross = t3SignalsLong
    
    if inverseT3:
        t3SignalsLongFinal = ~t3SignalsLongCross
    else:
        t3SignalsLongFinal = t3SignalsLongCross
    
    long_condition = signalHullMALong & t3SignalsLongFinal
    
    # Short entry conditions
    signalHullMAShort = (useHullMA and usecolorHullMA) and (sigHullMA < 0) & (df['close'] < hullma)
    signalHullMAShort = signalHullMAShort | ((useHullMA and not usecolorHullMA) and (df['close'] < hullma))
    signalHullMAShort = signalHullMAShort | (~useHullMA)
    
    basicShortCondition = (t3Signals < 0) & (df['close'] < t3)
    
    if useT3 and highlightMovementsT3:
        t3SignalsShort = basicShortCondition
    elif useT3:
        t3SignalsShort = df['close'] < t3
    else:
        t3SignalsShort = pd.Series(True, index=df.index)
    
    if crossT3:
        t3SignalsShortCross = t3SignalsShort & ~t3SignalsShort.shift(1)
    else:
        t3SignalsShortCross = t3SignalsShort
    
    if inverseT3:
        t3SignalsShortFinal = ~t3SignalsShortCross
    else:
        t3SignalsShortFinal = t3SignalsShortCross
    
    short_condition = signalHullMAShort & t3SignalsShortFinal
    
    # Build entry list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(hullma.iloc[i]) or pd.isna(t3.iloc[i]):
            continue
        
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