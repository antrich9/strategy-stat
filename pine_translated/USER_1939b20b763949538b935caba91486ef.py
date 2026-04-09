import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    if len(df) < 2:
        return []

    alphaLength = 20
    gammaLength = 20

    alpha = 2 / (alphaLength + 1)
    gamma = 2 / (gammaLength + 1)

    n = len(df)
    hema = np.zeros(n)
    b = np.zeros(n)

    src = df['close'].values

    for i in range(1, n):
        hema[i] = (1 - alpha) * (hema[i-1] + b[i-1]) + alpha * src[i]
        b[i] = (1 - gamma) * b[i-1] + gamma * (hema[i] - hema[i-1])

    hemaColorGreen = np.zeros(n, dtype=bool)
    hemaColorRed = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if hema[i] > hema[i-1]:
            hemaColorGreen[i] = True
        else:
            hemaColorRed[i] = True

    per_tbi = 14
    per2_tbi = 14
    length50line = 50

    loc_tbi = np.zeros(n, dtype=bool)
    loc2_tbi = np.zeros(n, dtype=bool)
    bottom_tbi = np.zeros(n)
    top_tbi = np.zeros(n)

    for i in range(2, n):
        lowest_prev = np.min(df['low'].iloc[max(0, i-per_tbi):i].values)
        lowest_curr = np.min(df['low'].iloc[max(0, i-per_tbi+1):i+1].values)
        loc_tbi[i] = df['low'].iloc[i] < lowest_prev and df['low'].iloc[i] <= lowest_curr

        highest_prev = np.max(df['high'].iloc[max(0, i-per2_tbi):i].values)
        highest_curr = np.max(df['high'].iloc[max(0, i-per2_tbi+1):i+1].values)
        loc2_tbi[i] = df['high'].iloc[i] > highest_prev and df['high'].iloc[i] >= highest_curr

    for i in range(n):
        if loc_tbi[i]:
            bottom_tbi[i] = 0
        else:
            bars = 0
            for j in range(i-1, -1, -1):
                if loc_tbi[j]:
                    break
                bars += 1
            bottom_tbi[i] = bars

        if loc2_tbi[i]:
            top_tbi[i] = 0
        else:
            bars = 0
            for j in range(i-1, -1, -1):
                if loc2_tbi[j]:
                    break
                bars += 1
            top_tbi[i] = bars

    condtion50Long = bottom_tbi < length50line
    condtion50Short = top_tbi < length50line

    basicLongConditionTBI = (top_tbi < bottom_tbi) & condtion50Long
    basicShortConditionTBI = (top_tbi > bottom_tbi) & condtion50Short

    TBISignalsLong = basicLongConditionTBI
    TBISignalsShort = basicShortConditionTBI

    TBISignalsLongCross = (~TBISignalsLong[:-1].values) & TBISignalsLong[1:].values
    TBISignalsLongCross = np.concatenate([[False], TBISignalsLongCross])
    TBISignalsShortCross = (~TBISignalsShort[:-1].values) & TBISignalsShort[1:].values
    TBISignalsShortCross = np.concatenate([[False], TBISignalsShortCross])

    TBISignalsLongFinal = TBISignalsLongCross
    TBISignalsShortFinal = TBISignalsShortCross

    lookbackTDFI = 13
    mmaLengthTDFI = 13
    smmaLengthTDFI = 13
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05

    priceTDFI = df['close'].values * 1000

    def ema(src, length):
        return pd.Series(src).ewm(span=length, adjust=False).mean().values

    def smma(src, length):
        result = np.zeros_like(src)
        result[0] = src[0]
        for i in range(1, len(src)):
            result[i] = (result[i-1] * (length - 1) + src[i]) / length
        return result

    def wma(src, length):
        result = np.zeros_like(src)
        weights = np.arange(1, length + 1)
        for i in range(length - 1, len(src)):
            result[i] = np.sum(src[i-length+1:i+1] * weights) / np.sum(weights)
        return result

    def swma(src):
        n = len(src)
        result = np.zeros_like(src)
        for i in range(4, n):
            result[i] = (src[i] + src[i-2] + 2*src[i-1] + 2*src[i-3] + src[i-4]) / 6
        return result

    def vwma(src, length, volume):
        result = np.zeros_like(src)
        for i in range(length - 1, len(src)):
            numerator = np.sum(src[i-length+1:i+1] * volume[i-length+1:i+1])
            denominator = np.sum(volume[i-length+1:i+1])
            if denominator != 0:
                result[i] = numerator / denominator
            else:
                result[i] = src[i]
        return result

    def hull(src, length):
        half = int(length / 2)
        wma1 = wma(src, half)
        wma2 = wma(src, length)
        diff = 2 * wma1 - wma2
        sqrt_len = int(np.sqrt(length))
        result = wma(diff, sqrt_len)
        return result

    def tema(src, length):
        ema1 = ema(src, length)
        ema2 = ema(ema1, length)
        ema3 = ema(ema2, length)
        return 3 * ema1 - 3 * ema2 + ema3

    mmaTDFI = tema(priceTDFI, mmaLengthTDFI)
    smmaTDFI = smma(mmaTDFI, smmaLengthTDFI)

    impetmmaTDFI = np.zeros(n)
    impetsmmaTDFI = np.zeros(n)
    for i in range(1, n):
        impetmmaTDFI[i] = mmaTDFI[i] - mmaTDFI[i-1]
        impetsmmaTDFI[i] = smmaTDFI[i] - smmaTDFI[i-1]

    divmaTDFI = np.abs(mmaTDFI - smmaTDFI)
    averimpetTDFI = (impetmmaTDFI + impetsmmaTDFI) / 2
    tdfTDFI = np.power(divmaTDFI, 1) * np.power(averimpetTDFI, nLengthTDFI)

    lookback = lookbackTDFI * nLengthTDFI
    highest_tdf = np.zeros(n)
    for i in range(lookback, n):
        highest_tdf[i] = np.max(np.abs(tdfTDFI[i-lookback+1:i+1]))

    signalTDFI = np.zeros(n)
    for i in range(lookback, n):
        if highest_tdf[i] != 0:
            signalTDFI[i] = tdfTDFI[i] / highest_tdf[i]

    signalLongTDFI = signalTDFI > filterHighTDFI
    signalShortTDFI = signalTDFI < filterLowTDFI

    finalLongSignalTDFI = signalLongTDFI
    finalShortSignalTDFI = signalShortTDFI

    longCondition = hemaColorGreen & (signalTDFI > 0) & TBISignalsLongFinal
    shortCondition = hemaColorRed & (signalTDFI < 0) & TBISignalsShortFinal

    entries = []
    trade_num = 1

    for i in range(1, n):
        if np.isnan(signalTDFI[i]):
            continue

        direction = None
        if longCondition[i]:
            direction = 'long'
        elif shortCondition[i]:
            direction = 'short'

        if direction is not None:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries