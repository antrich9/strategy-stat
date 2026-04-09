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
    length = 100
    maxCupDepth = 0.30
    minHandleDepth = 0.08
    maxHandleDepth = 0.12
    minHandleDuration = 5
    handleLength = 20
    volumeMultiplier = 1.5
    proximityTo52WeekHigh = 0.80
    minHandleDurationInWeeks = 1
    
    # Date range
    startDate = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    endDate = int(datetime(2023, 12, 31, tzinfo=timezone.utc).timestamp())
    
    # Indicators
    averageVolume = df['volume'].rolling(50).mean()
    fiftyTwoWeekHigh = df['high'].rolling(252).max()
    fiftyDayMA = df['close'].rolling(50).mean()
    lowestLow = df['low'].rolling(length).min()
    
    # State variables
    inCup = False
    inHandle = False
    tradeEntered = False
    cupLow = np.nan
    cupStart = 0
    cupHigh = np.nan
    cupEnd = 0
    handleStart = 0
    buyPoint = np.nan
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['time'].iloc[i] < startDate or df['time'].iloc[i] > endDate:
            continue
            
        low_val = df['low'].iloc[i]
        high_val = df['high'].iloc[i]
        close_val = df['close'].iloc[i]
        volume_val = df['volume'].iloc[i]
        avgVol = averageVolume.iloc[i]
        high52 = fiftyTwoWeekHigh.iloc[i]
        ma50 = fiftyDayMA.iloc[i]
        lowMin = lowestLow.iloc[i]
        
        # Skip if any indicator is NaN
        if pd.isna(avgVol) or pd.isna(high52) or pd.isna(ma50) or pd.isna(lowMin):
            continue
        
        # Condition 1: Start Cup formation
        if (not inCup and not inHandle and not tradeEntered and 
            low_val < lowMin * (1 + maxCupDepth) and 
            close_val > high52 * proximityTo52WeekHigh):
            inCup = True
            cupLow = low_val
            cupStart = i
        
        # Condition 2: Complete Cup (handle the cup formation logic)
        if inCup:
            # Assume cup completes when a higher high forms after the cup low
            if high_val > cupLow * 1.05:  # Basic cup completion heuristic
                inCup = False
                cupHigh = high_val
                cupEnd = i
                inHandle = True
                handleStart = i
                buyPoint = cupHigh * 1.02  # Entry slightly above cup high
        
        # Condition 3: Handle formation and entry
        if inHandle:
            # Handle forms as a consolidation below cup high
            handle_depth = (buyPoint - low_val) / buyPoint
            
            # Check if handle meets criteria
            if handle_depth <= maxHandleDepth and handle_depth >= minHandleDepth:
                # Volume confirmation
                if volume_val > averageVolume * volumeMultiplier:
                    # Entry signal
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': df['time'].iloc[i],
                        'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                        'entry_price': close_val,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price': 0.0
                    })
                    trade_num += 1
                    inHandle = False
                    tradeEntered = True
        
        # Reset if trade was entered
        if tradeEntered:
            inCup = False
            inHandle = False
            tradeEntered = False
    
    return entries