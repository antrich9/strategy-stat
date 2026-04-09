import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    
    n = len(df)
    PP = 7
    
    in_time_window = np.ones(n, dtype=bool)
    
    useSweepFilter = False
    sweepMode = "None"
    
    asianHigh = np.nan * np.ones(n)
    asianLow = np.nan * np.ones(n)
    asianSweptHigh = np.zeros(n, dtype=bool)
    asianSweptLow = np.zeros(n, dtype=bool)
    
    wasInAsian = False
    tmpAH = np.nan
    tmpAL = np.nan
    
    for i in range(n):
        nyHour = df['hour'].iloc[i]
        inAsianSession = nyHour >= 19
        
        if inAsianSession and not wasInAsian:
            tmpAH = df['high'].iloc[i]
            tmpAL = df['low'].iloc[i]
        elif inAsianSession:
            tmpAH = max(tmpAH, df['high'].iloc[i]) if not np.isnan(tmpAH) else df['high'].iloc[i]
            tmpAL = min(tmpAL, df['low'].iloc[i]) if not np.isnan(tmpAL) else df['low'].iloc[i]
        
        if not inAsianSession and wasInAsian:
            asianHigh[i] = tmpAH
            asianLow[i] = tmpAL
            asianSweptHigh[i] = False
            asianSweptLow[i] = False
        elif inAsianSession:
            asianSweptHigh[i] = asianSweptHigh[i-1] if i > 0 else False
            asianSweptLow[i] = asianSweptLow[i-1] if i > 0 else False
        else:
            asianSweptHigh[i] = asianSweptHigh[i-1] if i > 0 else False
            asianSweptLow[i] = asianSweptLow[i-1] if i > 0 else False
            
            if not asianSweptHigh[i] and not np.isnan(asianHigh[i-1]) and df['high'].iloc[i] > asianHigh[i-1]:
                asianSweptHigh[i] = True
            if not asianSweptLow[i] and not np.isnan(asianLow[i-1]) and df['low'].iloc[i] < asianLow[i-1]:
                asianSweptLow[i] = True
        
        wasInAsian = inAsianSession
    
    df['asianHigh'] = asianHigh
    df['asianLow'] = asianLow
    df['asianSweptHigh'] = asianSweptHigh
    df['asianSweptLow'] = asianSweptLow
    
    pdHigh = np.nan * np.ones(n)
    pdLow = np.nan * np.ones(n)
    tmpPH = np.nan
    tmpPL = np.nan
    pdSweptHigh = np.zeros(n, dtype=bool)
    pdSweptLow = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        current_date = df['date'].iloc[i]
        prev_date = df['date'].iloc[i-1]
        newDay = current_date != prev_date
        
        if newDay:
            pdHigh[i] = tmpPH
            pdLow[i] = tmpPL
            tmpPH = df['high'].iloc[i]
            tmpPL = df['low'].iloc[i]
            pdSweptHigh[i] = False
            pdSweptLow[i] = False
        else:
            tmpPH = df['high'].iloc[i] if np.isnan(tmpPH) else max(tmpPH, df['high'].iloc[i])
            tmpPL = df['low'].iloc[i] if np.isnan(tmpPL) else min(tmpPL, df['low'].iloc[i])
            pdSweptHigh[i] = pdSweptHigh[i-1]
            pdSweptLow[i] = pdSweptLow[i-1]
            
            if not pdSweptHigh[i] and not np.isnan(pdHigh[i-1]) and df['high'].iloc[i] > pdHigh[i-1]:
                pdSweptHigh[i] = True
            if not pdSweptLow[i] and not np.isnan(pdLow[i-1]) and df['low'].iloc[i] < pdLow[i-1]:
                pdSweptLow[i] = True
    
    df['pdHigh'] = pdHigh
    df['pdLow'] = pdLow
    df['pdSweptHigh'] = pdSweptHigh
    df['pdSweptLow'] = pdSweptLow
    
    bullishBias = np.zeros(n, dtype=bool)
    bearishBias = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        c1High_val = df['high'].iloc[i-2] if i >= 2 else np.nan
        c1Low_val = df['low'].iloc[i-2] if i >= 2 else np.nan
        c2Close_val = df['close'].iloc[i-1] if i >= 1 else np.nan
        c2High_val = df['high'].iloc[i-1] if i >= 1 else np.nan
        c2Low_val = df['low'].iloc[i-1] if i >= 1 else np.nan
        
        bullishBias[i] = (c2Close_val > c1High_val) or (c2Low_val < c1Low_val and c2Close_val > c1Low_val)
        bearishBias[i] = (c2Close_val < c1Low_val) or (c2High_val > c1High_val and c2Close_val < c1High_val)
    
    df['bullishBias'] = bullishBias
    df['bearishBias'] = bearishBias
    
    asianLongOk = df['asianSweptLow'] & (~df['asianSweptHigh'])
    asianShortOk = df['asianSweptHigh'] & (~df['asianSweptLow'])
    pdLongOk = df['pdSweptLow'] & (~df['pdSweptHigh'])
    pdShortOk = df['pdSweptHigh'] & (~df['pdSweptLow'])
    biasLongOk = df['bullishBias']
    biasShortOk = df['bearishBias']
    
    rawLongSweep = np.ones(n, dtype=bool)
    rawShortSweep = np.ones(n, dtype=bool)
    
    if sweepMode == "Asian Only":
        rawLongSweep = asianLongOk
        rawShortSweep = asianShortOk
    elif sweepMode == "PD Only":
        rawLongSweep = pdLongOk
        rawShortSweep = pdShortOk
    elif sweepMode == "Bias Only":
        rawLongSweep = biasLongOk
        rawShortSweep = biasShortOk
    elif sweepMode == "Asian + Bias":
        rawLongSweep = asianLongOk & biasLongOk
        rawShortSweep = asianShortOk & biasShortOk
    elif sweepMode == "PD + Bias":
        rawLongSweep = pdLongOk & biasLongOk
        rawShortSweep = pdShortOk & biasShortOk
    elif sweepMode == "All Three":
        rawLongSweep = asianLongOk & pdLongOk & biasLongOk
        rawShortSweep = asianShortOk & pdShortOk & biasShortOk
    
    longSweepOk = (~useSweepFilter) | rawLongSweep
    shortSweepOk = (~useSweepFilter) | rawShortSweep
    
    df['longSweepOk'] = longSweepOk
    df['shortSweepOk'] = shortSweepOk
    
    swingHigh = np.nan * np.ones(n)
    swingLow = np.nan * np.ones(n)
    lastSwingHigh = np.nan
    lastSwingLow = np.nan
    lastSwingHighIdx = -1
    lastSwingLowIdx = -1
    
    for i in range(PP, n - PP):
        isPivotHigh = True
        isPivotLow = True
        for j in range(1, PP + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                isPivotHigh = False
            if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                isPivotLow = False
        
        if isPivotHigh:
            lastSwingHigh = df['high'].iloc[i]
            lastSwingHighIdx = i
        if isPivotLow:
            lastSwingLow = df['low'].iloc[i]
            lastSwingLowIdx = i
        
        swingHigh[i] = lastSwingHigh
        swingLow[i] = lastSwingLow
    
    df['swingHigh'] = swingHigh
    df['swingLow'] = swingLow
    
    bosBullish = np.zeros(n, dtype=bool)
    bosBearish = np.zeros(n, dtype=bool)
    
    for i in range(PP + 1, n - PP):
        prevSwingHigh = df['swingHigh'].iloc[i - 1]
        prevSwingLow = df['swingLow'].iloc[i - 1]
        
        if not np.isnan(prevSwingHigh) and not np.isnan(prevSwingLow):
            if prevSwingHigh > prevSwingLow:
                if df['close'].iloc[i] > prevSwingHigh:
                    bosBullish[i] = True
                if df['close'].iloc[i] < prevSwingLow:
                    bosBearish[i] = True
    
    df['bosBullish'] = bosBullish
    df['bosBearish'] = bosBearish
    
    longEntry = df['bosBullish'] & df['longSweepOk'] & in_time_window
    shortEntry = df['bosBearish'] & df['shortSweepOk'] & in_time_window
    
    entries = []
    trade_num = 1
    
    for i in range(n):
        if longEntry.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
        elif shortEntry.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
    
    return entries