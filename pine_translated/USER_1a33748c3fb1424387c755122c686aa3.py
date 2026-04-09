import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    ts_arr = df['time'].values

    PeriodE2PSS = 15
    pi = 2 * np.arcsin(1.0)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.full(len(df), np.nan)
    Filt2[0] = np.nan
    Filt2[1] = np.nan

    for i in range(2, len(df)):
        prev1 = Filt2[i-1] if not np.isnan(Filt2[i-1]) else (high.iloc[i] + low.iloc[i]) / 2
        prev2 = Filt2[i-2] if not np.isnan(Filt2[i-2]) else (high.iloc[i-2] + low.iloc[i-2]) / 2
        Filt2[i]