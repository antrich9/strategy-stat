import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_prices = df['open']
    volume = df['volume']

    # ============ JMA ============
    lengthjma = 7
    phasejma = 50
    powerjma = 2

    beta = 0.45 * (lengthjma - 1) / (0.45 * (lengthjma - 1) + 2)
    alpha = beta ** powerjma
    phase_ratio = 0.5 if phasejma < -100 else 2.5 if phasejma > 100 else phasejma / 100 + 1.5

    jma = pd.Series(0.0, index=df.index)
    e0 = pd.Series(0.0, index=df.index)
    e1 = pd.Series(0.0, index=df.index)
    e2 = pd.Series(0.0, index=df.index)

    e0_prev = 0.0
    e1_prev = 0.0
    e2_prev = 0.0
    jma_prev = 0.0

    for i in range(len(df)):
        src_val = close.iloc[i]
        e0_curr = (1 - alpha) * src_val + alpha * e0_prev
        e1_curr = (src_val - e0_curr) * (1 - beta) + beta * e1_prev
        e2_curr = (e0_curr + phase_ratio * e1_curr - jma_prev) * (1 - alpha) ** 2 + alpha ** 2 * e2_prev
        jma_curr = e2_curr + jma_prev

        e0.iloc[i] = e0_curr
        e1.iloc[i] = e1_curr
        e2.iloc[i] = e2_curr
        jma.iloc[i] = jma_curr

        e0_prev = e0_curr
        e1_prev = e1_curr
        e2_prev = e2_curr
        jma_prev = jma_curr

    jma = jma

    # ============ TDFI ============
    lookback = 13
    mma_length = 13
    smma_length = 13
    n_length = 3
    filter_high = 0.05
    filter_low = -0.05

    def ema(s, l):
        return s.ewm(span=l, adjust=False).mean()

    def wma(s, l):
        w = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    def smma(s, l):
        return s.ewm(span=l, adjust=False).mean()

    def vwma(s, l, v):
        return (s * v).rolling(l).sum() / v.rolling(l).sum()

    def hull(s, l):
        return wma(2 * wma(s, int(l/2)) - wma(s, l), int(np.sqrt(l)))

    def tema(s, l):
        e1 = ema(s, l)
        e2 = ema(e1, l)
        e3 = ema(e2, l)
        return 3*e1 - 3*e2 + e3

    def ma(mode, s, l):
        if mode == 'ema': return ema(s, l)
        elif mode == 'wma': return wma(s, l)
        elif mode == 'swma': return smma(s, l)
        elif mode == 'vwma': return vwma(s, l, volume)
        elif mode == 'hull': return hull(s, l)
        elif mode == 'tema': return tema(s, l)
        else: return s.rolling(l).mean()

    mma = ma('ema', close * 1000, mma_length)
    smma_result = ma('ema', mma, smma_length)
    impetmma = mma.diff()
    impetsmma = smma_result.diff()
    divma = (mma - smma_result).abs()
    averimpet = (impetmma + impetsmma) / 2
    tdf = (divma ** 1) * (averimpet ** n_length)
    max_tdf = tdf.abs().rolling(window=lookback * n_length).max()
    tdfi_signal = tdf / max_tdf

    signal_long_tdfi_base = tdfi_signal > filter_high
    signal_short_tdfi_base = tdfi_signal < filter_low
    signal_long_tdfi_cross = (tdfi_signal > filter_high) & (tdfi_signal.shift(1) <= filter_high)
    signal_short_tdfi_cross = (tdfi_signal < filter_low) & (tdfi_signal.shift(1) >= filter_low)

    # ============ Stiffness