import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['ts'] = df['time']
    df = df.set_index('time')
    
    daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    htf_4h = df.resample('240min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    
    htf_close = htf_4h['close']
    htf_open = htf_4h['open']
    htf_high = htf_4h['high']
    htf_low = htf_4h['low']
    
    prev_day = daily.shift(1)
    pdh = prev_day['high']
    pdl = prev_day['low']