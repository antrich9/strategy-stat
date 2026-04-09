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
    
    # Hull MA parameters
    lengthHullMA = 9
    srcHullMA = df['close']
    
    # Hull MA calculation
    half_length = int(lengthHullMA / 2)
    sqrt_length = int(np.floor(np.sqrt(lengthHullMA)))
    
    wma1 = srcHullMA.rolling(window=half_length).mean()
    wma2 = srcHullMA.rolling(window=lengthHullMA).mean()
    hullmaHullMA = (2 * wma1 - wma2).rolling(window=sqrt_length).mean()
    
    # T3 calculation
    lengthT3 = 5
    factorT3 = 0.7
    srcT3 = df['close']
    
    def gdT3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    
    t3 = gdT3(gdT3(gdT3(srcT3, lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    
    # T3 signals (positive when t3 > t3[1])
    t3Signals = (t3 > t3.shift(1)).astype(int)
    t3Signals = t3Signals.replace(0, -1)
    
    # Hull MA signals (positive when hullmaHullMA > hullmaHullMA[1])
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int)
    sigHullMA = sigHullMA.replace(0, -1)
    
    # Entry conditions based on default inputs
    # useHullMA = true, usecolorHullMA = true, useT3 = true, crossT3 = true, inverseT3 = false, highlightMovementsT3 = true
    signalHullMALong = (sigHullMA > 0) & (df['close'] > hullmaHullMA)
    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)
    t3SignalsLong = basicLongCondition
    t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong
    t3SignalsLongFinal = t3SignalsLongCross
    
    # Stiffness parameters
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    # Stiffness calculation
    boundStiffness = df['close'].rolling(window=maLengthStiffness).mean() - 0.2 * df['close'].rolling(window=maLengthStiffness).std()
    sumAboveStiffness = (df['close'] > boundStiffness).rolling(window=stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    # Combined entry condition
    entryCondition = t3SignalsLongFinal & (stiffness > thresholdStiffness)
    
    # Generate entries
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        if i < stiffLength or i < maLengthStiffness or i < sqrt_length or i < lengthT3:
            continue
        if np.isnan(hullmaHullMA.iloc[i]) or np.isnan(t3.iloc[i]) or np.isnan(stiffness.iloc[i]):
            continue
        if entryCondition.iloc[i]:
            trade_num += 1
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
    
    return entries