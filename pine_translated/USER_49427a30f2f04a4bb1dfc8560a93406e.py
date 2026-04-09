import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    bb = 20  # lookback
    input_retSince = 2
    input_retValid = 2
    atrLength = 14
    
    # Calculate pivots
    def pivotlow(low, left_bars, right_bars):
        result = pd.Series(np.nan, index=low.index)
        for i in range(right_bars, len(low) - left_bars):
            if all(low.iloc[i - right_bars:i + 1] >= low.iloc[i]) and all(low.iloc[i:i + left_bars + 1] >= low.iloc[i]):
                result.iloc[i] = low.iloc[i]
        return result
    
    def pivothigh(high, left_bars, right_bars):
        result = pd.Series(np.nan, index=high.index)
        for i in range(right_bars, len(high) - left_bars):
            if all(high.iloc[i - right_bars:i + 1] <= high.iloc[i]) and all(high.iloc[i:i + left_bars + 1] <= high.iloc[i]):
                result.iloc[i] = high.iloc[i]
        return result
    
    pl = pivotlow(df['low'], bb, bb).ffill().dropna()
    ph = pivothigh(df['high'], bb, bb).ffill().dropna()