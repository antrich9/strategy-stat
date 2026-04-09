import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Shifted values for bar offsets
    open_1 = df['open'].shift(1)
    high_1 = df['high'].shift(1)
    high_2 = df['high'].shift(2)
    low_1 = df['low'].shift(1)
    low_2 = df['low'].shift(2)
    close_1 = df['close'].shift(1)
    
    # Time filter (hour extraction from unix timestamp)
    hours = pd.to_datetime(df['time'], unit='s').dt.hour
    isValidTradeTime = ((hours >= 2) & (hours < 5)) | ((hours >= 10) & (hours < 12))
    
    # Indicators
    sma54 = df['close'].rolling(54).mean()
    
    # Wilder ATR(20)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr20 = true_range.ewm(alpha=1/20, adjust=False).mean()
    
    # Volume filter
    vol_sma9 = df['volume'].rolling(9).mean()
    
    # Bar direction conditions
    isUp_curr = df['close'] > df['open']
    isDown_curr = df['close'] < df['open']
    isUp_1 = close_1 > open_1
    isDown_1 = close_1 < open_1
    
    # Order Block conditions
    isObUp = isDown_1 & isUp_curr & (df['close'] > high_1)
    isObDown = isUp_1 & isDown_curr & (df['close'] < low_1)
    
    # FVG conditions
    isFvgUp = df['low'] > high_2
    isFvgDown = df['high'] < low_2
    
    # Filters
    volfilt = (df['volume'].shift(1) > 1.5 * vol_sma9)
    atrfilt = ((df['low'] - high_2 > atr20/1.5) | (low_2 - df['high'] > atr20/1.5))
    locfiltb = (sma54 > sma54.shift(1))
    locfilts = (sma54 <= sma54.shift(1))
    
    # Stacked OB+FVG entry conditions
    bullishStack = isFvgUp & isObUp & volfilt & atrfilt & locfiltb
    bearishStack = isFvgDown & isObDown & volfilt & atrfilt & locfilts
    
    # Combined conditions with time filter
    conditions = isValidTradeTime & (bullishStack | bearishStack)
    conditions = conditions & ~(bullishStack & bearishStack)  # prevent conflicting signals
    conditions = conditions & ~conditions.isna()  # skip NaN bars
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if conditions.iloc[i]:
            direction = 'long' if bullishStack.iloc[i] else 'short'
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
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