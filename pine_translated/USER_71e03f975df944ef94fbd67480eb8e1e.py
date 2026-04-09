import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    lengthHullMA = 9
    srcHullMA = df['close']
    half_length = lengthHullMA // 2
    
    hullmaHullMA = (2 * srcHullMA.ewm(span=half_length, adjust=False).mean() - srcHullMA.ewm(span=lengthHullMA, adjust=False).mean()).ewm(span=int(np.sqrt(lengthHullMA)), adjust=False).mean()
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int) - (hullmaHullMA < hullmaHullMA.shift(1)).astype(int)
    
    lengthT3 = 5
    factorT3 = 0.7
    srcT3 = df['close']
    
    def gdT3(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factorT3) - ema2 * factorT3
    
    t3 = gdT3(gdT3(gdT3(srcT3, lengthT3), lengthT3), lengthT3)
    t3Signals = (t3 > t3.shift(1)).astype(int) - (t3 < t3.shift(1)).astype(int)
    
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    boundStiffness = df['close'].rolling(maLengthStiffness).mean() - 0.2 * df['close'].rolling(maLengthStiffness).std()
    sumAboveStiffness = (df['close'] > boundStiffness).rolling(stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalHullMALong = (sigHullMA > 0) & (df['close'] > hullmaHullMA)
    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)
    t3SignalsLong = basicLongCondition
    t3SignalsLongCross = t3SignalsLong & ~t3SignalsLong.shift(1).fillna(False)
    t3SignalsLongFinal = t3SignalsLongCross
    
    signalStiffness = stiffness > thresholdStiffness
    
    entryCondition = t3SignalsLongFinal & signalStiffness
    
    signalHullMAShort = (sigHullMA < 0) & (df['close'] < hullmaHullMA)
    basicShortCondition = (t3Signals < 0) & (df['close'] < t3)
    t3SignalsShort = basicShortCondition
    t3SignalsShortCross = t3SignalsShort & ~t3SignalsShort.shift(1).fillna(False)
    t3SignalsShortFinal = t3SignalsShortCross
    
    entryShortCondition = t3SignalsShortFinal & (stiffness < thresholdStiffness)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if entryCondition.iloc[i]:
            ts = df['time'].iloc[i]
            price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                'raw_price_a': price, 'raw_price_b': price
            })
            trade_num += 1
        
        if entryShortCondition.iloc[i]:
            ts = df['time'].iloc[i]
            price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                'raw_price_a': price, 'raw_price_b': price
            })
            trade_num += 1
    
    return entries