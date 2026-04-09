import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_prices = df['open']
    volume = df['volume']
    ts = df['time']
    
    # JMA Parameters
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    srcjmaJMA = close
    
    phasejmaJMARatiojmaJMA = np.where(phasejmaJMA < -100, 0.5, 
                             np.where(phasejmaJMA > 100, 2.5, phasejmaJMA / 100 + 1.5))
    
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = np.pow(betajmaJMA, powerjmaJMA)
    
    # JMA calculation (recursive)
    jmaJMA = np.zeros(len(df))
    e0JMA = np.zeros(len(df))
    e1JMA = np.zeros(len(df))
    e2JMA = np.zeros(len(df))
    
    for i in range(1, len(df)):
        e0JMA[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i] + alphajmaJMA * e0JMA[i-1]
        e1JMA[i] = (srcjmaJMA.iloc[i] - e0JMA[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA[i-1]
        e2JMA[i] = (e0JMA[i] + phasejmaJMARatiojmaJMA * e1JMA[i] - jmaJMA[i-1]) * np.pow(1 - alphajmaJMA, 2) + np.pow(alphajmaJMA, 2) * e2JMA[i-1]
        jmaJMA[i] = e2JMA[i] + jmaJMA[i-1]
    
    jmaJMA = pd.Series(jmaJMA, index=df.index)
    e0JMA = pd.Series(e0JMA, index=df.index)
    e1JMA = pd.Series(e1JMA, index=df.index)
    e2JMA = pd.Series(e2JMA, index=df.index)
    
    # JMA conditions (inverseJMA = true)
    signalmaJMALong = (jmaJMA > jmaJMA.shift(1)) & (close > jmaJMA)
    signalmaJMAShort = (jmaJMA < jmaJMA.shift(1)) & (close < jmaJMA)
    
    finalLongSignalJMA = signalmaJMAShort
    finalShortSignalJMA = signalmaJMALong
    
    # TDFI Parameters
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    mmaModeTDFI = 'ema'
    smmaLengthTDFI = 13
    smmaModeTDFI = 'ema'
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    priceTDFI = close * 1000
    
    # TEMA function
    def tema_tdfi(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    # MA function
    def ma_tdfi(mode, src, length):
        if mode == 'ema':
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'wma':
            weights = np.arange(1, len(src) + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        elif mode == 'swma':
            return src.rolling(8).mean()
        elif mode == 'vwma':
            return (src * volume).rolling(length).sum() / volume.rolling(length).sum()
        elif mode == 'hull':
            half_len = int(length / 2)
            wma_half = src.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / (len(x)*(len(x)+1)/2), raw=True)
            wma_full = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / (len(x)*(len(x)+1)/2), raw=True)
            hull = 2 * wma_half - wma_full
            sqrt_len = int(np.sqrt(length))
            return hull.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / (len(x)*(len(x)+1)/2), raw=True)
        elif mode == 'tema':
            return tema_tdfi(src, length)
        else:
            return src.rolling(length).mean()
    
    # TDFI calculation
    mmaTDFI = ma_tdfi(mmaModeTDFI, priceTDFI, mmaLengthTDFI)
    smmaTDFI = ma_tdfi(smmaModeTDFI, mmaTDFI, smmaLengthTDFI)
    impetmmaTDFI = mmaTDFI - mmaTDFI.shift(1)
    impetsmmaTDFI = smmaTDFI - smmaTDFI.shift(1)
    divmaTDFI = np.abs(mmaTDFI - smmaTDFI)
    averimpetTDFI = (impetmmaTDFI + impetsmmaTDFI) / 2
    tdfTDFI = np.pow(divmaTDFI, 1) * np.pow(averimpetTDFI, nLengthTDFI)
    
    highest_tdf = tdfTDFI.abs().rolling(lookbackTDFI * nLengthTDFI).max()
    signalTDFI = tdfTDFI / highest_tdf
    
    # TDFI conditions (inverseTDFI = true, crossTDFI = true)
    signalLongTDFI = signalTDFI > filterHighTDFI
    signalShortTDFI = (signalTDFI.shift(1) <= filterLowTDFI) & (signalTDFI > filterLowTDFI)
    
    finalLongSignalTDFI = signalShortTDFI
    finalShortSignalTDFI = signalLongTDFI
    
    # Stiffness Parameters
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    boundStiffness = close.rolling(maLengthStiffness).mean() - 0.2 * close.rolling(maLengthStiffness).std()
    sumAboveStiffness = (close > boundStiffness).rolling(stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = stiffness > thresholdStiffness
    
    # Track TDFI state and trade open status
    lastTDFIGreen = False
    tradeOpen = False
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if np.isnan(jmaJMA.iloc[i]) or np.isnan(signalTDFI.iloc[i]) or np.isnan(stiffness.iloc[i]):
            continue
        
        # Update TDFI state
        if signalTDFI.iloc[i] > filterHighTDFI:
            lastTDFIGreen = True
        elif signalTDFI.iloc[i] < filterLowTDFI:
            lastTDFIGreen = False
        
        # Entry condition: bullishEntryCondition AND lastTDFIGreen AND not tradeOpen
        bullishEntryCondition = finalLongSignalJMA.iloc[i] and signalStiffness.iloc[i] and (signalTDFI.iloc[i] > filterHighTDFI)
        
        if bullishEntryCondition and lastTDFIGreen and not tradeOpen:
            entry_ts = int(ts.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
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
            tradeOpen = True
        
        if tradeOpen and close.iloc[i] != close.iloc[i]:
            pass
        else:
            pass
        
        if i < len(df) - 1:
            if pd.isna(close.iloc[i+1]):
                if tradeOpen:
                    tradeOpen = False
            else:
                pass
    
    return entries