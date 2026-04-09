import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    n = len(df)
    if n < 3:
        return []
    
    entries = []
    trade_num = 0
    
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    local_hour = (ts.dt.hour + 1) % 24
    local_minute = ts.dt.minute
    current_time = local_hour * 60 + local_minute
    
    session1_start = 7 * 60
    session1_end = 10 * 60
    session2_start = 12 * 60
    session2_end = 15 * 60
    
    in_session = ((session1_start <= current_time) & (current_time < session1_end)) | \
                  ((session2_start <= current_time) & (current_time < session2_end))
    
    prevDayHigh = df['high'].shift(1)
    prevDayLow = df['low'].shift(1)
    
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    
    for i in range(1, n):
        if df['close'].iloc[i] > prevDayHigh.iloc[i]:
            flagpd