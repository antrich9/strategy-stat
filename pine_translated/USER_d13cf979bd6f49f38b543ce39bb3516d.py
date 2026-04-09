import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    
    entries = []
    trade_num = 0
    
    flagpdh = False
    flagpdl = False
    waitingForEntry = False
    waitingForShortEntry1 = False
    consecutiveBfvg = 0
    consecutiveSfvg = 0
    
    # Calculate daily high/low for previous day
    df['day'] = df['time'].dt.date
    daily_agg = df.groupby('day').agg({'high': 'max', 'low': 'min'}).shift(1)
    df['prevDayHigh'] = df['day'].map(daily_agg['high'])
    df['prevDayLow'] = df['day'].map(daily_agg['low'])
    
    # FVG conditions
    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)
    
    # OB conditions (shifted by 1 for current bar)
    isDown_1 = df['close'].shift(1) < df['open'].shift(1)
    isUp_0 = df['close'] > df['open']
    obUp = isDown_1 & isUp_0 & (df['close'] > df['high'].shift(1))
    
    isUp_1 = df['close'].shift(1) > df['open'].shift(1)
    isDown_0 = df['close'] < df['open']
    obDown = isUp_1 & isDown_0 & (df['close'] < df['low'].shift(1))
    
    # Time filter: 0700-0959 and 1200-1459 in GMT+1
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    time_cond1 = ((df['hour'] == 7) | ((df['hour'] == 8) | ((df['hour'] == 9) & (df['minute'] <= 59))))
    time_cond2 = ((df['hour'] == 12) | ((df['hour'] == 13) | ((df['hour'] == 14) & (df['minute'] <= 59))))
    time_filter = time_cond1 | time_cond2
    
    for i in range(1, len(df)):
        ts = int(df['time'].iloc[i].timestamp())
        entry_time = df['time'].iloc[i].isoformat()
        
        # Reset on new day
        if i > 0 and df['day'].iloc[i] != df['day'].iloc[i-1]:
            flagpdh = False
            flagpdl = False
            waitingForEntry = False
            waitingForShortEntry1 = False
            consecutiveBfvg = 0
            consecutiveSfvg = 0
        
        # Sweep conditions
        if df['close'].iloc[i] > df['prevDayHigh'].iloc[i]:
            flagpdh = True
        if df['close'].iloc[i] < df['prevDayLow'].iloc[i]:
            flagpdl = True
        
        # Set waiting flags on sweep
        if flagpdl:
            waitingForEntry = True
        if flagpdh:
            waitingForShortEntry1 = True
        
        # Consecutive FVG tracking
        if fvgUp.iloc[i]:
            consecutiveBfvg += 1
        else:
            consecutiveBfvg = 0
        
        if fvgDown.iloc[i]:
            consecutiveSfvg += 1
        else:
            consecutiveSfvg = 0
        
        # Long entry conditions
        if (waitingForEntry and obUp.iloc[i] and fvgUp.iloc[i] and 
            consecutiveBfvg >= 2 and time_filter.iloc[i]):
            trade_num += 1
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
            waitingForEntry = False
        
        # Short entry conditions
        if (waitingForShortEntry1 and obDown.iloc[i] and fvgDown.iloc[i] and 
            consecutiveSfvg >= 2 and time_filter.iloc[i]):
            trade_num += 1
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
            waitingForShortEntry1 = False
    
    return entries