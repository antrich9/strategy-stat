import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Convert timestamp to datetime
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Extract time components (assuming UTC or converting to London time)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # London windows: 07:45-11:45 and 14:00-14:45
    window1_start = 7 * 60 + 45  # 465 minutes
    window1_end = 11 * 60 + 45   # 705 minutes
    window2_start = 14 * 60      # 840 minutes
    window2_end = 14 * 60 + 45   # 885 minutes
    
    in_window1 = (df['time_minutes'] >= window1_start) & (df['time_minutes'] < window1_end)
    in_window2 = (df['time_minutes'] >= window2_start) & (df['time_minutes'] < window2_end)
    in_trading_window = in_window1 | in_window2