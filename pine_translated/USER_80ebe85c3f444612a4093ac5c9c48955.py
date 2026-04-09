import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Convert time to datetime if needed
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to daily to get previous day's high/low
    # For each daily bar, we want high and low
    daily = df.resample('D', on='datetime').agg({
        'high': 'max',
        'low': 'min',
        'open': 'first',
        'close': 'last'
    }).dropna()
    
    # Get previous day's high and low
    # Shift to get previous day values
    prev_day_high = daily['high'].shift(1)
    prev_day_low = daily['low'].shift(1)
    prev_day_mid = (prev_day_high + prev_day_low) / 2
    
    # Now I need to map these back to the original dataframe
    # Each bar should have the previous day's levels
    # Forward fill the previous day values
    df['prev_day_high'] = prev_day_high.reindex(df['datetime'].dt.date, method='ffill')
    df['prev_day_low'] = prev_day_low.reindex(df['datetime'].dt.date, method='ffill')
    df['prev_day_mid'] = prev_day_mid.reindex(df['datetime'].dt.date, method='ffill')