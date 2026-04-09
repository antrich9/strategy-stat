import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Calculate required shifted columns
    high_1 = df['high'].shift(1)
    high_2 = df['high'].shift(2)
    low_1 = df['low'].shift(1)
    low_2 = df['low'].shift(2)
    open_1 = df['open'].shift(1)
    close_0 = df['close']
    
    # Bearish FVG conditions (for short entries)
    top_imbalance_bway = (
        (low_2 <= open_1) & 
        (df['high'] >= df['close'].shift(1)) & 
        (df['close'] < low_1) & 
        ((low_2 - df['high']) > 0)
    )
    
    # Bullish FVG conditions (for long entries)
    bottom_imbalance_bway = (
        (high_2 >= open_1) & 
        (df['low'] <= df['close'].shift(1)) & 
        (df['close'] > high_1) & 
        ((df['low'] - high_2) > 0)
    )