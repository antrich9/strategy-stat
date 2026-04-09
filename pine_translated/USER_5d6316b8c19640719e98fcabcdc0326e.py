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
    
    # Strategy parameters
    length = 65
    minCupLength = 7
    handleLength = 4
    minHandleLength = 1
    depth = 0.33
    handleDepth = 0.12
    volumeMultiplier = 1.4
    
    # Calculate indicators
    averageVolume = df['volume'].rolling(50).mean()
    lookback = length * 5
    
    entries = []
    trade_num = 0
    
    # State variables
    cupLow = np.nan
    cupHigh = np.nan
    cupStart = -1
    cupEnd = -1
    inCup = False
    inHandle = False
    handleLow = np.nan
    handleHigh = np.nan
    
    for i in range(lookback, len(df)):
        # Condition 1: Start cup
        lowest_low = df['low'].iloc[i-lookback:i+1].min()
        if not inCup and not inHandle and df['low'].iloc[i] < lowest_low * (1 + depth):
            cupLow = df['low'].iloc[i]
            cupStart = i
            inCup = True
        
        # Condition 2: Cup complete, start handle
        highest_high = df['high'].iloc[i-lookback:i+1].max()
        if inCup and df['high'].iloc[i] > highest_high * (1 - depth):
            cupHigh = df['high'].iloc[i]
            cupEnd = i
            inCup = False
            inHandle = True
            handleLow = df['low'].iloc[i]
            handleHigh = df['high'].iloc[i]
        
        # Condition 3: Handle timeout
        if inHandle and i - cupEnd > handleLength * 5:
            inHandle = False
        
        # Condition 4: Update handleLow
        if inHandle and df['low'].iloc[i] < handleLow:
            handleLow = df['low'].iloc[i]
        
        # Condition 5: Update handleHigh
        if inHandle and df['high'].iloc[i] > handleHigh:
            handleHigh = df['high'].iloc[i]
        
        # Condition 6: Handle depth exceeded
        if inHandle and df['low'].iloc[i] < cupLow * (1 - handleDepth):
            inHandle = False
        
        # Breakout logic
        if inHandle:
            cupDuration = (i - cupStart) / 5
            handleDuration = (i - cupEnd) / 5
            buyPoint = handleHigh + 0.10
            
            if (df['close'].iloc[i] > buyPoint and 
                df['volume'].iloc[i] > averageVolume.iloc[i] * volumeMultiplier and
                cupDuration >= minCupLength and 
                handleDuration >= minHandleLength):
                
                trade_num += 1
                ts = df['time'].iloc[i]
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
                inHandle = False
    
    return entries