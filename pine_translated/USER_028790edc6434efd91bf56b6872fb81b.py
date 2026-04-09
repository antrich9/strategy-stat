import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure numeric types
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # ----- True Range -----
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).fillna(hl)

    # ----- Bollinger Bands (using multKC) -----
    length1 = 20
    mult = 2.0
    multKC = 1.5
    source = df['close']
    basis = source.rolling(length1).mean()
    dev = multKC * source.rolling(length1).std(ddof=0)
    upperBB = basis + dev
    lowerBB = basis - dev

    # ----- Keltner Channel -----
    lengthKC = 20
    ma = source.rolling(lengthKC).mean()
    priceRange = df['tr']          # useTrueRange = true
    rangeMa = priceRange.rolling(lengthKC).mean()
    upperKC = ma + rangeMa * multKC
    lowerKC = ma - rangeMa * multKC

    # ----- Squeeze conditions -----
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)

    # ----- Squeeze Momentum (linreg of deviation) -----
    highest_high = df['high'].rolling(lengthKC).max()
    lowest_low = df['low'].rolling(lengthKC).min()
    mid = (highest_high + lowest_low) / 2
    sma_close = source.rolling(lengthKC).mean()
    avg = (mid + sma_close) / 2
    deviation = source - avg

    def linreg(y, length):
        n = length
        x = np.arange(n, dtype=float)
        sum_x = x.sum()
        sum_x2 = (x**2).sum()
        denom = n * sum_x2 - sum_x**2
        def _calc(vals):
            sum_y = vals.sum()
            sum_xy = (vals * x).sum()
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            return intercept + slope * (n - 1)
        return y.rolling(window=n).apply(_calc, raw=True)

    val