import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters from Pine Script inputs
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
    end_ts = int(datetime(2023, 12, 31, tzinfo=timezone.utc).timestamp())
    
    # Calculate indicators
    df = df.copy()
    df['avg_volume'] = df['volume'].rolling(50).mean()
    df['fifty_two_week_high'] = df['high'].rolling(252).max()
    df['fifty_day_ma'] = df['close'].rolling(50).mean()
    df['lowest_low'] = df['low'].rolling(length).min()
    
    # State variables
    inCup = False
    inHandle = False
    trade_entered = False
    cup_low = np.nan
    cup_start = 0
    cup_high = np.nan
    cup_end = 0
    handle_start = 0
    entry_price = np.nan
    buy_point = np.nan
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        date_condition = start_ts <= ts <= end_ts
        
        # Cup detection
        if (df['low'].iloc[i] < df['lowest_low'].iloc[i] * (1 + maxCupDepth) and 
            not inCup and not inHandle and 
            df['close'].iloc[i] > df['fifty_two_week_high'].iloc[i] * proximityTo52WeekHigh and 
            date_condition):
            cup_low = df['low'].iloc[i]
            cup_start = i
            inCup = True
            trade_entered = False
        
        # Cup end detection
        if inCup:
            if df['low'].iloc[i] >= cup_low:
                cup_high = df['high'].iloc[i]
                cup_end = i
                inCup = False
                inHandle = True
                handle_start = i
                buy_point = cup_high
        
        # Handle logic
        if inHandle and df['low'].iloc[i] > df['fifty_day_ma'].iloc[i] and date_condition:
            if i - cup_end > handleLength:
                inHandle = False
            if df['low'].iloc[i] < cup_high * (1 - maxHandleDepth):
                inHandle = False
            if i - cup_end >= minHandleDuration:
                if (i - handle_start) >= minHandleDurationInWeeks * 5:
                    if df['close'].iloc[i] > buy_point and df['volume'].iloc[i] > df['avg_volume'].iloc[i] * volumeMultiplier:
                        entry_price = df['close'].iloc[i]
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': ts,
                            'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': entry_price,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': entry_price,
                            'raw_price_b': entry_price
                        })
                        trade_num += 1
                        trade_entered = True
                        inHandle = False
    
    return entries