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
    
    # Parameters from Pine Script
    length = 100
    maxCupDepth = 0.30
    maxHandleDepth = 0.12
    minHandleDuration = 5
    handleLength = 20
    volumeMultiplier = 1.5
    proximityTo52WeekHigh = 0.80
    minHandleDurationInWeeks = 1
    
    # Backtest Date Range
    start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    
    # Calculate indicators
    averageVolume = df['volume'].rolling(50).mean()
    fiftyTwoWeekHigh = df['high'].rolling(252).max()
    fiftyDayMA = df['close'].rolling(50).mean()
    
    # State variables
    cupLow = np.nan
    cupStart = -1
    inCup = False
    inHandle = False
    tradeEntered = False
    cupHigh = np.nan
    cupEnd = -1
    handleStart = -1
    buyPoint = np.nan
    
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        
        # Date condition
        dateCondition = (ts >= start_ts and ts <= end_ts)
        
        if not dateCondition:
            continue
            
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        close = df['close'].iloc[i]
        volume = df['volume'].iloc[i]
        
        # Calculate lowest low over length period
        if i >= length:
            lowest_low_length = df['low'].iloc[i-length+1:i+1].min()
        else:
            lowest_low_length = df['low'].iloc[:i+1].min()
        
        # Cup formation start condition
        if (low < lowest_low_length * (1 + maxCupDepth) and not inCup and not inHandle):
            if close > fiftyTwoWeekHigh.iloc[i] * proximityTo52WeekHigh:
                cupLow = low
                cupStart = i
                inCup = True
                tradeEntered = False
        
        # Cup high detection and handle formation
        if inCup:
            # Detect cup high - when price makes a significant recovery from cup low
            # This marks the end of the cup formation and start of handle
            if high > cupLow * 1.1:  # 10% recovery from cup low
                cupHigh = high
                cupEnd = i
                inCup = False
                inHandle = True
                handleStart = i
                buyPoint = cupHigh
        
        # Handle phase entry logic
        if inHandle and low > fiftyDayMA.iloc[i]:
            if i - cupEnd > handleLength:
                inHandle = False
            if low < cupHigh * (1 - maxHandleDepth):
                inHandle = False
            if i - cupEnd >= minHandleDuration:
                if not tradeEntered and (i - handleStart) >= minHandleDurationInWeeks * 5:
                    if close > buyPoint and volume > averageVolume.iloc[i] * volumeMultiplier:
                        trade_num += 1
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': ts,
                            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': close,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': close,
                            'raw_price_b': close
                        })
                        tradeEntered = True
                        inHandle = False
    
    return entries