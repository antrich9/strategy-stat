import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # Convert timestamps to datetime (UTC)
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['dt'].dt.date

    # ---- Daily bar aggregation ----
    daily = df.groupby('date').agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    )
    # Daily EMAs (9 and 18 periods)
    daily['ema9'] = daily['close'].ewm(span=9, adjust=False).mean()
    daily['ema18'] = daily['close'].ewm(span=18, adjust=False).mean()

    # Previous day high / low (shift by one day)
    daily['prevDayHigh'] = daily['high'].shift(1)
    daily['prevDayLow'] = daily['low'].shift(1)

    # Detect sweep of previous day high / low within the day
    daily['sweepPDH'] = daily['high']