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
    
    # Default params from script
    pivotLen = 2
    bosWindow = 30
    sweepMaxBars = 30
    fvgMaxBars = 30
    fibLevel = 0.71
    minScore = 50
    useHTFBias = True
    useLongs = True
    useShorts = True
    useSweep = True
    useFVG = True
    requireFibInFVG = False
    
    # Initialize state
    lastHighPrice = np.nan
    lastHighBar = -1
    lastLowPrice = np.nan
    lastLowBar = -1
    bosBullBar = -1
    bosBullHigh = np.nan
    bosBearBar = -1
    bosBearLow = np.nan
    sweptLowBar = -1
    sweptHighBar = -1
    bullFvgLow = np.nan
    bullFvgHigh = np.nan
    bullFvgBar = -1
    bearFvgLow = np.nan
    bearFvgHigh = np.nan
    bearFvgBar = -1
    
    n = len(df)
    entries = []
    trade_num = 0
    
    for i in range(n):
        high_i = df['high'].iloc[i]
        low_i = df['low'].iloc[i]
        close_i = df['close'].iloc[i]
        open_i = df['open'].iloc[i]
        
        # Pivot detection
        ph = np.nan
        pl = np.nan
        
        if i >= pivotLen:
            window_high = df['high'].iloc[i-pivotLen+1:i+1].max()
            window_low = df['low'].iloc[i-pivotLen+1:i+1].min()
            
            if high_i == window_high and (i == pivotLen or df['high'].iloc[i-pivotLen] >= window_high):
                ph = high_i
            if low_i == window_low and (i == pivotLen or df['low'].iloc[i-pivotLen] <= window_low):
                pl = low_i
        
        if not np.isnan(ph):
            lastHighPrice = ph
            lastHighBar = i - pivotLen
        
        if not np.isnan(pl):
            lastLowPrice = pl
            lastLowBar = i - pivotLen
        
        # BOS detection
        if not np.isnan(lastHighPrice) and close_i > lastHighPrice:
            bosBullBar = i
            bosBullHigh = close_i
        
        if not np.isnan(lastLowPrice) and close_i < lastLowPrice:
            bosBearBar = i
            bosBearLow = close_i
        
        # Sweep detection
        if i > 0:
            prev_low = df['low'].iloc[i-1]
            prev_high = df['high'].iloc[i-1]
            
            if not np.isnan(lastLowPrice) and low_i < lastLowPrice and prev_low >= lastLowPrice:
                sweptLowBar = i
            
            if not np.isnan(lastHighPrice) and high_i > lastHighPrice and prev_high <= lastHighPrice:
                sweptHighBar = i
        
        # FVG detection
        if i >= 2:
            low_prev = df['low'].iloc[i-1]
            low_2prev = df['low'].iloc[i-2]
            high_prev = df['high'].iloc[i-1]
            high_2prev = df['high'].iloc[i-2]
            
            if low_prev > high_2prev:
                bullFvgLow = high_2prev
                bullFvgHigh = low_prev
                bullFvgBar = i
            
            if high_prev < low_2prev:
                bearFvgLow = low_2prev
                bearFvgHigh = high_prev
                bearFvgBar = i
        
        # Score calculation
        bullScore = 0
        bearScore = 0
        
        if not np.isnan(lastHighPrice) and lastHighPrice > 0:
            bullScore += 25
        if not np.isnan(lastLowPrice) and lastLowPrice > 0:
            bullScore += 25
        if bosBullBar >= 0:
            bullScore += 25
        if bullFvgBar >= 0:
            bullScore += 25
        
        if not np.isnan(lastLowPrice) and lastLowPrice > 0:
            bearScore += 25
        if not np.isnan(lastHighPrice) and lastHighPrice > 0:
            bearScore += 25
        if bosBearBar >= 0:
            bearScore += 25
        if bearFvgBar >= 0:
            bearScore += 25
        
        # Entry signals
        if bullScore >= minScore and useLongs:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': bullFvgLow,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': bullFvgLow,
                'raw_price_b': bullFvgHigh
            })
        
        if bearScore >= minScore and useShorts:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': bearFvgHigh,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': bearFvgLow,
                'raw_price_b': bearFvgHigh
            })
        
        # Reset conditions
        if bosBullBar >= 0 and i - bosBullBar > bosWindow:
            bosBullBar = -1
            bosBullHigh = np.nan
        
        if bosBearBar >= 0 and i - bosBearBar > bosWindow:
            bosBearBar = -1
            bosBearLow = np.nan
        
        if bullFvgBar >= 0 and i - bullFvgBar > fvgMaxBars:
            bullFvgLow = np.nan
            bullFvgHigh = np.nan
            bullFvgBar = -1
        
        if bearFvgBar >= 0 and i - bearFvgBar > fvgMaxBars:
            bearFvgLow = np.nan
            bearFvgHigh = np.nan
            bearFvgBar = -1
    
    return entries