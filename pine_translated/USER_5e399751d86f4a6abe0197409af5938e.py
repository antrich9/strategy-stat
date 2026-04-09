import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    
    # Hull MA calculation
    lengthHullMA = 9
    half_length = int(lengthHullMA / 2)
    wma1 = df['close'].rolling(window=half_length, min_periods=half_length).mean()
    wma2 = df['close'].rolling(window=lengthHullMA, min_periods=lengthHullMA).mean()
    hullmaHullMA = (2 * wma1 - wma2).rolling(window=int(np.sqrt(lengthHullMA)), min_periods=int(np.sqrt(lengthHullMA))).mean()
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int).replace(0, -1)
    
    # T3 calculation
    lengthT3 = 5
    factorT3 = 0.7
    def calc_t3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    t3 = calc_t3(calc_t3(calc_t3(df['close'], lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    t3Signals = (t3 > t3.shift(1)).astype(int).replace(0, -1)
    
    # Stiffness calculation
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    boundStiffness = close.rolling(window=maLengthStiffness, min_periods=maLengthStiffness).mean() - 0.2 * close.rolling(window=maLengthStiffness, min_periods=maLengthStiffness).std()
    sumAboveStiffness = (close > boundStiffness).rolling(window=stiffLength, min_periods=stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(alpha=1/stiffSmooth, adjust=False).mean()
    
    # Long signals
    signalHullMALong = (sigHullMA > 0) & (close > hullmaHullMA)
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = basicLongCondition
    t3SignalsLongCross = (~t3SignalsLong.shift(1).astype(bool)) & t3SignalsLong.astype(bool)
    t3SignalsLongFinal = ~t3SignalsLongCross
    
    # Long entry condition
    longEntryCondition = signalHullMALong & t3SignalsLongFinal & (stiffness > thresholdStiffness)
    
    # Short signals
    signalHullMAShort = (sigHullMA < 0) & (close < hullmaHullMA)
    basicShortCondition = (t3Signals < 0) & (close < t3)
    t3SignalsShort = basicShortCondition
    t3SignalsShortCross = (~t3SignalsShort.shift(1).astype(bool)) & t3SignalsShort.astype(bool)
    t3SignalsShortFinal = ~t3SignalsShortCross
    
    # Short entry condition
    shortEntryCondition = signalHullMAShort & t3SignalsShortFinal & (stiffness < thresholdStiffness)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.notna(longEntryCondition.iloc[i]) and longEntryCondition.iloc[i]:
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
            trade_num += 1
        
        if pd.notna(shortEntryCondition.iloc[i]) and shortEntryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries