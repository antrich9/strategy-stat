import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    zigzag_period = 13
    zigzag_threshold = 0.02
    minRR = 1.0
    requireReversalCandle = True
    useRSI = False
    rsiLength = 14
    rsiOverbought = 70
    rsiOversold = 30
    atrLength = 14
    atrMultiplier = 1.5
    targetFrontRun = 2
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsiLength, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsiLength, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # ZigZag state
    zigzag_buffer = close.iloc[0]
    high_buffer = high.iloc[0]
    low_buffer = low.iloc[0]
    up_trend = True
    last_pivot = close.iloc[0]
    
    min_change = close * zigzag_threshold
    
    zigzag_points = []
    
    for i in range(1, len(df)):
        if pd.isna(zigzag_buffer):
            zigzag_buffer = close.iloc[i]
            high_buffer = high.iloc[i]
            low_buffer = low.iloc[i]
        
        if up_trend:
            if high.iloc[i] > high_buffer:
                high_buffer = high.iloc[i]
                last_pivot = high_buffer
            if low.iloc[i] < low_buffer - min_change.iloc[i]:
                zigzag_points.append((i-1, high_buffer))
                up_trend = False
                low_buffer = low.iloc[i]
                last_pivot = low_buffer
        else:
            if low.iloc[i] < low_buffer:
                low_buffer = low.iloc[i]
                last_pivot = low_buffer
            if high.iloc[i] > high_buffer + min_change.iloc[i]:
                zigzag_points.append((i-1, low_buffer))
                up_trend = True
                high_buffer = high.iloc[i]
                last_pivot = high_buffer
    
    return zigzag_points