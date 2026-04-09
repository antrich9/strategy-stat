import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    length_T3 = 5
    factor_T3 = 0.7
    adx_length = 14
    adx_threshold = 25.0
    dmi_period = 14

    # ---- T3 indicator ----
    def t3_func(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = df['close'].copy()
    for _ in range(3):
        t3 = t3_func(t3, length_T3, factor_T3)

    # t3Signals: 1 when rising, -1 when falling
    t3Signals = pd.Series(np.where(t3 > t3.shift(1), 1, -1), index=df.index)

    # basicLongCondition: t3Signals > 0 and close > t3
    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)

    # t3SignalsLong = basicLongCondition (highlightMovementsT3 is true)
    t3SignalsLong = basicLongCondition

    # t3SignalsLongCross: crossT3 == true