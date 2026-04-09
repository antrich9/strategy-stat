import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    timestamps = df['time'].values
    n = len(df)

    length_AMOM = 8
    alpha_AMOM = 0.7
    poles_AMOM = 3

    def get2PoleSSF_AMOM(src, length):
        arg = np.sqrt(2) * np.pi / length
        a1 = np.exp(-arg)
        b1 = 2 * a1 * np.cos(arg)
        c2 = b1
        c3 = -(a1 ** 2)
        c1 = 1 - c2 - c3
        ssf = np.zeros(n)
        ssf[0] = src[0]
        if n > 1:
            ssf[1] = c1 * src[1] + c2 * ssf[0]
        for i in range(2, n):
            ssf[i] = c1 * src[i] + c2 * ssf[i-1] + c3 * ssf[i-2]
        return ssf

    def get3PoleSSF_AMOM(src, length):
        arg = np.pi / length
        a1 = np.exp(-arg)
        b1 = 2 * a1 * np.cos(1.738 * arg)
        c1 = a1 ** 2
        coef2 = b1 + c1
        coef3 = -(c1 + b1 * c1)
        coef4 = c1 ** 2
        coef1 = 1 - coef2 - coef3 - coef4
        ssf = np.zeros(n)
        ssf[0] = src[0]
        if n > 1:
            ssf[1] = coef1 * src[1] + coef2 * ssf[0]
        if n > 2:
            ssf[2] = coef1 * src[2] + coef2 * ssf[1] + coef