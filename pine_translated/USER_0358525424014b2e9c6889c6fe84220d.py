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
    
    # Parameters
    length = 65
    minCupLength = 7
    handleLength = 4
    minHandleLength = 1
    depth = 0.33
    handleDepth = 0.12
    volumeMultiplier = 1.4
    
    # Calculate indicators
    averageVolume = df['volume'].rolling(50).mean()
    
    # Calculate highest and lowest over the lookback period
    lookback = length * 5
    highest_high = df['high'].rolling(lookback).max()
    lowest_low = df['low'].rolling(lookback).min()
    
    # Initialize state variables
    inCup = False
    inHandle = False
    cupLow = np.nan
    cupHigh = np.nan
    cupStart = np.nan
    cupEnd = np.nan
    handleLow = np.nan
    handleHigh = np.nan
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_volume = df['volume'].iloc[i]
        
        # Calculate durations
        if inCup:
            cupDuration = (i - cupStart) / 5
        else:
            cupDuration = np.nan
            
        if inHandle:
            handleDuration = (i - cupEnd) / 5
        else:
            handleDuration = np.nan
        
        # Check for new cup formation
        if not inCup and not inHandle:
            if current_low < lowest_low.iloc[i] * (1 + depth):
                cupLow = current_low
                cupStart = i
                inCup = True
        
        # Check for cup completion and handle start
        if inCup:
            if current_high > highest_high.iloc[i] * (1 - depth):
                cupHigh = current_high
                cupEnd = i
                inCup = False
                inHandle = True
                handleLow = current_low
                handleHigh = current_high
        
        # Check for handle timeout
        if inHandle:
            if i - cupEnd > handleLength * 5:
                inHandle = False
        
        # Update handle range
        if inHandle:
            if current_low < handleLow:
                handleLow = current_low
            if current_high > handleHigh:
                handleHigh = current_high
            
            # Check for handle depth violation
            if current_low < cupLow * (1 - handleDepth):
                inHandle = False
            
            # Check for breakout
            buyPoint = handleHigh + 0.10
            
            if (current_close > buyPoint and 
                current_volume > averageVolume.iloc[i] * volumeMultiplier and
                cupDuration >= minCupLength and
                handleDuration >= minHandleLength):
                
                entry_time = datetime.fromtimestamp(current_time / 1000, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': current_time,
                    'entry_time': entry_time,
                    'entry_price_guess': buyPoint,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': buyPoint,
                    'raw_price_b': buyPoint
                })
                
                trade_num += 1
                inHandle = False
    
    return entries