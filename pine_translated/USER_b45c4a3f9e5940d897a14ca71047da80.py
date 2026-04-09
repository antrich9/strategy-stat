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
    
    swingHighLength = 10
    swingLowLength = 10
    fvgMinRange = 0.5
    fvgMaxRange = 3.0
    
    n = len(df)
    
    # swingHigh: pivot high with swingHighLength bars to left and right
    swingHigh = pd.Series(np.nan, index=df.index)
    for i in range(swingHighLength, n - swingHighLength):
        isPivot = True
        for j in range(1, swingHighLength + 1):
            if df['high'].iloc[i - j] >= df['high'].iloc[i] or df['high'].iloc[i + j] >= df['high'].iloc[i]:
                isPivot = False
                break
        if isPivot:
            swingHigh.iloc[i] = df['high'].iloc[i]
    
    # swingLow: pivot low with swingLowLength bars to left and right
    swingLow = pd.Series(np.nan, index=df.index)
    for i in range(swingLowLength, n - swingLowLength):
        isPivot = True
        for j in range(1, swingLowLength + 1):
            if df['low'].iloc[i - j] <= df['low'].iloc[i] or df['low'].iloc[i + j] <= df['low'].iloc[i]:
                isPivot = False
                break
        if isPivot:
            swingLow.iloc[i] = df['low'].iloc[i]
    
    # Bullish/Bearish Market Structure Breaks
    bullishBreak = pd.Series(False, index=df.index)
    bearishBreak = pd.Series(False, index=df.index)
    
    for i in range(swingLowLength * 2, n):
        if pd.notna(swingLow.iloc[i]):
            low_swing = df['low'].iloc[i]
            low_prev = df['low'].iloc[i - swingLowLength]
            low_prev2 = df['low'].iloc[i - swingLowLength * 2]
            if low_prev < low_prev2:
                bullishBreak.iloc[i] = True
    
    for i in range(swingHighLength * 2, n):
        if pd.notna(swingHigh.iloc[i]):
            high_swing = df['high'].iloc[i]
            high_prev = df['high'].iloc[i - swingHighLength]
            high_prev2 = df['high'].iloc[i - swingHighLength * 2]
            if high_prev > high_prev2:
                bearishBreak.iloc[i] = True
    
    # FVG Detection
    rangePercent = (df['close'] - df['open']).abs() / df['close'] * 100
    priceRange = (rangePercent >= fvgMinRange) & (rangePercent <= fvgMaxRange)
    
    fvgUpperRange = pd.Series(np.nan, index=df.index)
    fvgLowerRange = pd.Series(np.nan, index=df.index)
    
    for i in range(n):
        if priceRange.iloc[i]:
            fvgUpperRange.iloc[i] = max(df['high'].iloc[i], df['open'].iloc[i])
            fvgLowerRange.iloc[i] = min(df['low'].iloc[i], df['open'].iloc[i])
    
    # Entry Conditions
    longCondition = bullishBreak & priceRange & (df['close'] > fvgUpperRange)
    shortCondition = bearishBreak & priceRange & (df['close'] < fvgLowerRange)
    
    entries = []
    trade_num = 1
    
    # Need to skip bars where required indicators are NaN
    for i in range(n):
        if pd.isna(swingLow.iloc[i]) and i < swingLowLength * 2:
            continue
        if pd.isna(swingHigh.iloc[i]) and i < swingHighLength * 2:
            continue
        if pd.isna(fvgUpperRange.iloc[i]) and longCondition.iloc[i]:
            continue
        if pd.isna(fvgLowerRange.iloc[i]) and shortCondition.iloc[i]:
            continue
            
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            
        if shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries