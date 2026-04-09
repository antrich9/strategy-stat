import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # Detect FVGs
    # Bullish FVG: high < low[2] and close[1] < low[2]
    # Bearish FVG: low > high[2] and close[1] > high[2]
    
    bull_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    bear_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    
    entries = []
    trade_num = 1
    
    # Track active FVGs: (bar_index, price_level)
    bull_fvg_tops = []
    bear_fvg_bottoms = []
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        # Detect new FVGs at current bar
        if bull_fvg.iloc[i]:
            # Bullish FVG top is current low
            bull_fvg_tops.append((i, low.iloc[i]))
        
        if bear_fvg.iloc[i]:
            # Bearish FVG bottom is current high
            bear_fvg_bottoms.append((i, high.iloc[i]))
        
        # Check for entries: price enters FVG zone
        # Long entry: low goes below bullish FVG top (price enters from above)
        for idx, top in bull_fvg_tops[:]:
            if low.iloc[i] < top:
                ts = int(time.iloc[i])