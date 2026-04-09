import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    timestamps = df['time']
    
    fastLen, medLen, slowLen = 8, 20, 50
    dojiPerc = 0.30
    sweepMode = "None"
    offset = 0.0004
    
    useAllHours, useLondon, useNYAM, useNYPM = False, True, True, False
    
    ema8 = close.ewm(span=fastLen, adjust=False).mean()
    ema20 = close.ewm(span=medLen, adjust=False).mean()
    ema50 = close.ewm(span=slowLen, adjust=False).mean()
    
    n = len(df)
    asianHigh_vals = pd.Series(np.nan, index=df.index)
    asianLow_vals = pd.Series(np.nan, index=df.index)
    pdHigh_vals = pd.Series(np.nan, index=df.index)
    pdLow_vals = pd.Series(np.nan, index=df.index)
    bullishBias_vals = pd.Series(False, index=df.index)
    bearishBias_vals = pd.Series(False, index=df.index)
    
    asianHigh = asianLow = pdHigh = pdLow = np.nan
    tmpAH = tmpAL = tmpPH = tmpPL = np.nan
    asianSweptHigh = asianSweptLow = pdSweptHigh = pdSweptLow = False
    wasInAsian = False
    newDay = False
    prev_in_time_window = False
    
    entries = []
    trade_num = 0
    
    pendingLong = pendingShort = False
    pendLongEntry = pendShortEntry = pendLongStop = pendShortStop = np.nan
    pendSignalBar = pendSignalTime = 0
    
    for i in range(n):
        ts = timestamps.iloc[i]
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour = dt.hour
        
        in_time_window = useAllHours or (useLondon and 7 <= hour < 10) or (useNYAM and 13 <= hour < 16) or (useNYPM and 19 <= hour < 21)
        
        nyHour = (dt.hour - 5) % 24
        inAsianSession = nyHour >= 19
        
        sessionJustStarted = inAsianSession and not wasInAsian
        sessionEnded = not inAsianSession and wasInAsian and i > 0 and ((df['time'].iloc[i-1] / 1000) - (df['time'].iloc[i] / 1000)) < 43200000
        
        if sessionJustStarted:
            tmpAH = high.iloc[i]
            tmpAL = low.iloc[i]
        elif inAsianSession and not pd.isna(tmpAH):
            tmpAH = max(tmpAH, high.iloc[i])
            tmpAL = min(tmpAL, low.iloc[i])
        
        if sessionEnded or (i > 0 and inAsianSession and not ((datetime.fromtimestamp(df['time'].iloc[i-1] / 1000, tz=timezone.utc).hour - 5) % 24 >= 19)):
            if not inAsianSession and wasInAsian:
                asianHigh = tmpAH
                asianLow = tmpAL
                asianSweptHigh = asianSweptLow = False
        
        if not asianSweptHigh and not pd.isna(asianHigh) and high.iloc[i] > asianHigh:
            asianSweptHigh = True
        if not asianSweptLow and not pd.isna(asianLow) and low.iloc[i] < asianLow:
            asianSweptLow = True
        
        newDay = i > 0 and dt.date() != datetime.fromtimestamp(timestamps.iloc[i-1] / 1000, tz=timezone.utc).date()
        
        if newDay:
            pdHigh = tmpPH
            pdLow = tmpPL
            tmpPH = high.iloc[i]
            tmpPL = low.iloc[i]
            pdSweptHigh = pdSweptLow = False
        else:
            tmpPH = high.iloc[i] if pd.isna(tmpPH) else max(tmpPH, high.iloc[i])
            tmpPL = low.iloc[i] if pd.isna(tmpPL) else min(tmpPL, low.iloc[i])
        
        if not pdSweptHigh and not pd.isna(pdHigh) and high.iloc[i] > pdHigh:
            pdSweptHigh = True
        if not pdSweptLow and not pd.isna(pdLow) and low.iloc[i] < pdLow:
            pdSweptLow = True
        
        asianLongOk = asianSweptLow and not asianSweptHigh
        asianShortOk = asianSweptHigh and not asianSweptLow
        pdLongOk = pdSweptLow and not pdSweptHigh
        pdShortOk = pdSweptHigh and not pdSweptLow
        
        bullishBias = bullishBias_vals.iloc[i] if not pd.isna(bullishBias_vals.iloc[i]) else False
        bearishBias = bearishBias_vals.iloc[i] if not pd.isna(bearishBias_vals.iloc[i]) else False
        biasLongOk = bullishBias
        biasShortOk = bearishBias
        
        if sweepMode == "Asian Only":
            longSweepOk = asianLongOk
            shortSweepOk = asianShortOk
        elif sweepMode == "PD Only":
            longSweepOk = pdLongOk
            shortSweepOk = pdShortOk
        elif sweepMode == "Bias Only":
            longSweepOk = biasLongOk
            shortSweepOk = biasShortOk
        elif sweepMode == "Asian + Bias":
            longSweepOk = asianLongOk and biasLongOk
            shortSweepOk = asianShortOk and biasShortOk
        elif sweepMode == "PD + Bias":
            longSweepOk = pdLongOk and biasLongOk
            shortSweepOk = pdShortOk and biasShortOk
        elif sweepMode == "All Three":
            longSweepOk = asianLongOk and pdLongOk and biasLongOk
            shortSweepOk = asianShortOk and pdShortOk and biasShortOk
        else:
            longSweepOk = shortSweepOk = True
        
        if pd.isna(ema8.iloc[i]) or pd.isna(ema20.iloc[i]) or pd.isna(ema50.iloc[i]):
            wasInAsian = inAsianSession
            prev_in_time_window = in_time_window
            continue
        
        rng = high.iloc[i] - low.iloc[i]
        body = abs(close.iloc[i] - open_price.iloc[i])
        isDoji = rng > 0 and (body / rng) <= dojiPerc
        
        emaBull = ema8.iloc[i] > ema20.iloc[i] and ema20.iloc[i] > ema50.iloc[i]
        emaBear = ema8.iloc[i] < ema20.iloc[i] and ema20.iloc[i] < ema50.iloc[i]
        
        if rng > 0:
            bodyInUpper33 = open_price.iloc[i] >= low.iloc[i] + rng * 0.67 and close.iloc[i] >= low.iloc[i] + rng * 0.67
            bodyInLower33 = open_price.iloc[i] <= low.iloc[i] + rng * 0.33 and close.iloc[i] <= low.iloc[i] + rng * 0.33
            closeInUpper33 = (close.iloc[i] - low.iloc[i]) / rng >= 0.67
            closeInLower33 = (high.iloc[i] - close.iloc[i]) / rng >= 0.67
        else:
            bodyInUpper33 = bodyInLower33 = closeInUpper33 = closeInLower33 = False
        
        bullSignal = emaBull and isDoji and low.iloc[i] <= ema8.iloc[i] and closeInUpper33 and bodyInUpper33
        bearSignal = emaBear and isDoji and high.iloc[i] >= ema8.iloc[i] and closeInLower33 and bodyInLower33
        
        if not pendingLong and not pendingShort:
            if bullSignal and not bearSignal and longSweepOk and in_time_window:
                pendingLong = True
                pendLongEntry = high.iloc[i] + offset
                pendLongStop = low.iloc[i] - offset
                pendSignalBar = i
                pendSignalTime = ts
            elif bearSignal and not bullSignal and shortSweepOk and in_time_window:
                pendingShort = True
                pendShortEntry = low.iloc[i] - offset
                pendShortStop = high.iloc[i] + offset
                pendSignalBar = i
                pendSignalTime = ts
        
        if pendingLong and high.iloc[i] > pendLongEntry:
            trade_num += 1
            entries.append({
                'trade_num': trade_num, 'direction': 'long',
                'entry_ts': int(ts), 'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': pendLongEntry, 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                'raw_price_a': pendLongEntry, 'raw_price_b': pendLongEntry
            })
            pendingLong = False
        
        if pendingShort and low.iloc[i] < pendShortEntry:
            trade_num += 1
            entries.append({
                'trade_num': trade_num, 'direction': 'short',
                'entry_ts': int(ts), 'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': pendShortEntry, 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                'raw_price_a': pendShortEntry, 'raw_price_b': pendShortEntry
            })
            pendingShort = False
        
        wasInAsian = inAsianSession
        prev_in_time_window = in_time_window
    
    return entries