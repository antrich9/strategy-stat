import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default parameter values from Pine Script inputs
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    useT3 = True
    highlightMovementsT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    useStiffness = False
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90

    close = df['close']
    high = df['high']
    low = df['low']

    min_len = max(lengthHullMA, lengthT3, maLengthStiffness + stiffLength, stiffSmooth) + 1

    # Hull MA calculation
    half_length = int(lengthHullMA // 2)
    sqrt_length = int(np.sqrt(lengthHullMA))
    wma_half = close.rolling(half_length).apply(lambda x: np.dot(x, np.arange(1, half_length + 1)) / half_length, raw=True)
    wma_full = close.rolling(lengthHullMA).apply(lambda x: np.dot(x, np.arange(1, lengthHullMA + 1)) / lengthHullMA, raw=True)
    hullma = (2 * wma_half - wma_full).rolling(sqrt_length).apply(lambda x: np.dot(x, np.arange(1, sqrt_length + 1)) / sqrt_length, raw=True)
    sigHullMA = (hullma > hullma.shift(1)).astype(int).replace(0, -1)

    # T3 calculation
    ema1 = close.ewm(span=lengthT3, adjust=False).mean()
    ema2 = ema1.ewm(span=lengthT3, adjust=False).mean()
    gdT3 = ema1 * (1 + factorT3) - ema2 * factorT3
    t3 = gdT3.ewm(span=lengthT3, adjust=False).mean()
    t3Signals = (t3 > t3.shift(1)).astype(int).replace(0, -1)

    # Stiffness calculation
    sma_bound = close.rolling(maLengthStiffness).mean()
    std_bound = close.rolling(maLengthStiffness).std()
    boundStiffness = sma_bound - 0.2 * std_bound
    sumAboveStiffness = (close > boundStiffness).rolling(window=stiffLength).sum()
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()

    # Build entry condition series
    signalHullMALong = (~useHullMA) | (usecolorHullMA & (sigHullMA > 0) & (close > hullma)) | ((~usecolorHullMA) & (close > hullma))
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = (~useT3) | (highlightMovementsT3 & basicLongCondition) | ((~highlightMovementsT3) & (close > t3))
    t3SignalsLongShift = t3SignalsLong.shift(1).fillna(0).astype(bool)
    t3SignalsLongSeries = t3SignalsLong.fillna(0).astype(bool)
    t3SignalsLongCross = (~crossT3) | ((~t3SignalsLongShift) & t3SignalsLongSeries)
    t3SignalsLongFinal = (~inverseT3) & t3SignalsLongCross if inverseT3 else t3SignalsLongCross
    signalStiffness = (~useStiffness) | (stiffness > thresholdStiffness)
    entryCondition = signalHullMALong & t3SignalsLongFinal & signalStiffness

    entries = []
    trade_num = 1

    for i in range(min_len, len(df)):
        if pd.isna(stiffness.iloc[i]):
            continue
        if entryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1

    return entries