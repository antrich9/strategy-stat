import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    zigzagPeriod = 13
    zigzagThreshold = 0.02
    useOptimizedZone = True
    fibLevel = 0.5
    dtdbLookback = 10
    dtdbTolerance = 0.001
    atrLength = 14
    atrMult = 2.0
    minRR = 1.5
    skipOnCHoCH = True
    
    n = len(df)
    minChange = df['close'] * zigzagThreshold
    
    # Zigzag arrays
    zigzag = []
    highIndex = []
    lowIndex = []
    
    # ATR
    atr = pd.Series(index=df.index, dtype=float)
    prevClose = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], 
                   (df['high'] - prevClose).abs(), 
                   (df['low'] - prevClose).abs()], axis=1).max(axis=1)
    atr.iloc[0] = tr.iloc[0]
    alpha = 1.0 / atrLength
    for i in range(1, n):
        atr.iloc[i] = (atr.iloc[i-1] * (atrLength - 1) + tr.iloc[i]) / atrLength
    
    # State variables
    upTrend = True
    lastPivot = np.nan
    bullishBoS = False
    bearishBoS = False
    bullishCHoCH = False
    bearishCHoCH = False
    
    bullZoneTop = np.nan
    bullZoneBottom = np.nan
    bearZoneTop = np.nan
    bearZoneBottom = np.nan
    bullStopLevel = np.nan
    bullTargetLevel = np.nan
    bearStopLevel = np.nan
    bearTargetLevel = np.nan
    bullishSetup = False
    bearishSetup = False
    
    dtPriceLevel = np.nan
    dtBarIndex = 0
    dtDetected = False
    dbPriceLevel = np.nan
    dbBarIndex = 0
    dbDetected = False
    
    entries = []
    trade_num = 1
    
    zigzagBuffer = df['close'].iloc[0]
    highBuffer = df['high'].iloc[0]
    lowBuffer = df['low'].iloc[0]
    
    for i in range(n):
        if upTrend:
            if df['high'].iloc[i] > highBuffer:
                highBuffer = df['high'].iloc[i]
                lastPivot = highBuffer
            if df['low'].iloc[i] < lowBuffer - minChange.iloc[i]:
                zigzag.append(highBuffer)
                highIndex.append(i - 1)
                upTrend = False
                lowBuffer = df['low'].iloc[i]
                lastPivot = lowBuffer
        else:
            if df['low'].iloc[i] < lowBuffer:
                lowBuffer = df['low'].iloc[i]
                lastPivot = lowBuffer
            if df['high'].iloc[i] > highBuffer + minChange.iloc[i]:
                zigzag.append(lowBuffer)
                lowIndex.append(i - 1)
                upTrend = True
                highBuffer = df['high'].iloc[i]
                lastPivot = highBuffer
        
        swingHigh = zigzag[-1] if len(highIndex) > 0 else np.nan
        swingLow = zigzag[-1] if len(lowIndex) > 0 else np.nan
        prevSwingHigh = zigzag[-2] if len(highIndex) > 1 else np.nan
        prevSwingLow = zigzag[-2] if len(lowIndex) > 1 else np.nan
        
        if not np.isnan(swingHigh) and not np.isnan(prevSwingHigh):
            if df['high'].iloc[i] > swingHigh:
                if swingHigh > prevSwingHigh:
                    bullishBoS = True
                    bearishBoS = False
                else:
                    bullishCHoCH = True
        
        if not np.isnan(swingLow) and not np.isnan(prevSwingLow):
            if df['low'].iloc[i] < swingLow:
                if swingLow < prevSwingLow:
                    bearishBoS = True
                    bullishBoS = False
                else:
                    bearishCHoCH = True
        
        if bullishBoS and not np.isnan(swingLow):
            initialBottom = swingLow
            initialTop = df['low'].iloc[i]
            bullStopLevel = swingLow - (atr.iloc[i] * atrMult)
            bullTargetLevel = swingHigh if not np.isnan(swingHigh) else df['high'].iloc[i]
            if useOptimizedZone:
                totalRange = initialTop - initialBottom
                bullZoneBottom = initialBottom
                bullZoneTop = initialBottom + (totalRange * fibLevel)
            else:
                bullZoneBottom = initialBottom
                bullZoneTop = initialTop
            bullishSetup = True
            bearishSetup = False
        
        if bearishBoS and not np.isnan(swingHigh):
            initialBottom = df['high'].iloc[i]
            initialTop = swingHigh
            bearStopLevel = swingHigh + (atr.iloc[i] * atrMult)
            bearTargetLevel = swingLow if not np.isnan(swingLow) else df['low'].iloc[i]
            if useOptimizedZone:
                totalRange = initialTop - initialBottom
                bearZoneBottom = initialTop - (totalRange * fibLevel)
                bearZoneTop = initialTop
            else:
                bearZoneBottom = initialBottom
                bearZoneTop = initialTop
            bearishSetup = True
            bullishSetup = False
        
        inBullZone = False
        inBearZone = False
        if bullishSetup and not np.isnan(bullZoneTop) and not np.isnan(bullZoneBottom):
            if df['high'].iloc[i] >= bullZoneBottom and df['low'].iloc[i] <= bullZoneTop:
                inBullZone = True
        if bearishSetup and not np.isnan(bearZoneTop) and not np.isnan(bearZoneBottom):
            if df['high'].iloc[i] >= bearZoneBottom and df['low'].iloc[i] <= bearZoneTop:
                inBearZone = True
        
        tolerance = df['close'].iloc[i] * dtdbTolerance
        
        if inBearZone:
            if np.isnan(dtPriceLevel) or (i - dtBarIndex) > dtdbLookback:
                dtPriceLevel = df['high'].iloc[i]
                dtBarIndex = i
                dtDetected = False
            elif (i - dtBarIndex) > 0 and (i - dtBarIndex) <= dtdbLookback:
                if abs(df['high'].iloc[i] - dtPriceLevel) <= tolerance:
                    dtDetected = True
                elif df['high'].iloc[i] > dtPriceLevel + tolerance:
                    dtPriceLevel = df['high'].iloc[i]
                    dtBarIndex = i
                    dtDetected = False
        
        if inBullZone:
            if np.isnan(dbPriceLevel) or (i - dbBarIndex) > dtdbLookback:
                dbPriceLevel = df['low'].iloc[i]
                dbBarIndex = i
                dbDetected = False
            elif (i - dbBarIndex) > 0 and (i - dbBarIndex) <= dtdbLookback:
                if abs(df['low'].iloc[i] - dbPriceLevel) <= tolerance:
                    dbDetected = True
                elif df['low'].iloc[i] < dbPriceLevel - tolerance:
                    dbPriceLevel = df['low'].iloc[i]
                    dbBarIndex = i
                    dbDetected = False
        
        bullRR_OK = False
        if not np.isnan(bullStopLevel) and not np.isnan(bullTargetLevel):
            bullRisk = abs(df['close'].iloc[i] - bullStopLevel)
            bullReward = abs(bullTargetLevel - df['close'].iloc[i])
            if bullRisk > 0:
                bullRR_OK = (bullReward / bullRisk) >= minRR
        
        bearRR_OK = False
        if not np.isnan(bearStopLevel) and not np.isnan(bearTargetLevel):
            bearRisk = abs(df['close'].iloc[i] - bearStopLevel)
            bearReward = abs(bearTargetLevel - df['close'].iloc[i])
            if bearRisk > 0:
                bearRR_OK = (bearReward / bearRisk) >= minRR
        
        tradeInvalidated = False
        if skipOnCHoCH:
            if bullishCHoCH or bearishCHoCH:
                tradeInvalidated = True
                bullishSetup = False
                bearishSetup = False
                dtDetected = False
                dbDetected = False
        
        longEntry = bullishSetup and inBullZone and dbDetected and bullRR_OK and not tradeInvalidated
        shortEntry = bearishSetup and inBearZone and dtDetected and bearRR_OK and not tradeInvalidated
        
        if longEntry:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat() if ts > 10000000000 else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            dbDetected = False
            dbPriceLevel = np.nan
        
        if shortEntry:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat() if ts > 10000000000 else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            dtDetected = False
            dtPriceLevel = np.nan
    
    return entries