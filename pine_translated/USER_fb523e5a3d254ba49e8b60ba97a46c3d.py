import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    
    df['time_numeric'] = df['hour'] * 60 + df['minute']
    morning_window = (df['time_numeric'] >= 7 * 60 + 45) & (df['time_numeric'] <= 9 * 60 + 45)
    afternoon_window = (df['time_numeric'] >= 14 * 60 + 45) & (df['time_numeric'] <= 16 * 60 + 45)
    in_trading_window = morning_window | afternoon_window