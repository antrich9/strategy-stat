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
    
    length = 65
    minCupLength = 7
    handleLength = 4
    minHandleLength = 1
    depth = 0.33
    handleDepth = 0.12
    volumeMultiplier = 1.4
    
    avg_vol = df['volume'].rolling(50).mean()
    lowest_low = df['low'].rolling(length * 5).min()
    highest_high = df['high'].rolling(length * 5).max()
    
    entries = []
    trade_num = 1
    
    inCup = False
    inHandle = False
    cupLow = np.nan
    cupHigh = np.nan
    cupStart = np.nan
    cupEnd = np.nan
    handleLow = np.nan
    handleHigh = np.nan
    
    for i in range(len(df)):
        if pd.isna(avg_vol.iloc[i]) or pd.isna(lowest_low.iloc[i]) or pd.isna(highest_high.iloc[i]):
            continue
        
        cupDuration = (i - cupStart) / 5.0 if inCup and not pd.isna(cupStart) else np.nan
        handleDuration = (i - cupEnd) / 5.0 if inHandle and not pd.isna(cupEnd) else np.nan
        
        if not inCup and not inHandle and df['low'].iloc[i] < lowest_low.iloc[i] * (1 + depth):
            cupLow = df['low'].iloc[i]
            cupStart = i
            inCup = True
        
        if inCup and df['high'].iloc[i] > highest_high.iloc[i] * (1 - depth):
            cupHigh = df['high'].iloc[i]
            cupEnd = i
            inCup = False
            inHandle = True
            handleLow = df['low'].iloc[i]
            handleHigh = df['high'].iloc[i]
        
        if inHandle and i - cupEnd > handleLength * 5:
            inHandle = False
        
        if inHandle and df['low'].iloc[i] < handleLow:
            handleLow = df['low'].iloc[i]
        
        if inHandle and df['high'].iloc[i] > handleHigh:
            handleHigh = df['high'].iloc[i]
        
        if inHandle and df['low'].iloc[i] < cupLow * (1 - handleDepth):
            inHandle = False
        
        buyPoint = handleHigh + 0.10
        breakoutCondition = inHandle and df['close'].iloc[i] > buyPoint and df['volume'].iloc[i] > avg_vol.iloc[i] * volumeMultiplier and (not pd.isna(cupDuration) and cupDuration >= minCupLength) and (not pd.isna(handleDuration) and handleDuration >= minHandleLength)
        
        if breakoutCondition:
            inHandle = False
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[i],
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': buyPoint,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': buyPoint,
                'raw_price_b': buyPoint
            }
            entries.append(entry)
            trade_num += 1
    
    return entries