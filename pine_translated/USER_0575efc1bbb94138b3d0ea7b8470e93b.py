import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    def alma(data, length, offset=0.85, sigma=6):
        result = np.full(len(data), np.nan)
        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            positions = np.arange(length)
            weights = np.exp(-((positions - offset * (length - 1)) ** 2) / (2 * sigma ** 2 * length ** 2))
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result

    def wilder_smooth(series, period):
        result = np.full(len(series), np.nan)
        alpha = 1.0 / period
        result[period - 1] = series.iloc[:period].mean()
        for i in range(period, len(series)):
            result[i] = result[i - 1] + alpha * (series.iloc[i] - result[i - 1])
        return pd.Series(result, index=series.index)

    PeriodE2PSS = 15
    pi = 2 * np.pi
    alpha = np.exp(-1.414 * pi / PeriodE2PSS)
    beta = 2 * alpha * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = beta
    coef3 = -alpha * alpha
    coef1 = 1 - coef2 - coef3

    hl2 = (df['high'] + df['low']) / 2
    Filt2 = np.zeros(len(df))
    Filt2[0] = hl2.iloc[0]
    Filt2[1] = hl2.iloc[1]
    for i in range(2, len(df)):
        Filt2[i] = coef1 * hl2.iloc[i] + coef2 * Filt2[i - 1] + coef3 * Filt2[i - 2]
    Filt2 = pd.Series(Filt2, index=df.index)
    TriggerE2PSS = Filt2.shift(1)
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85