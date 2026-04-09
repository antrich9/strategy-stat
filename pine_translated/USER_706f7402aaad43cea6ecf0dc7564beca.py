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
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    highlightMovementsjmaJMA = True

    useTDFI = True
    crossTDFI = True
    inverseTDFI = True
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    mmaModeTDFI = 'ema'
    smmaLengthTDFI = 13
    smmaModeTDFI = 'ema'
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    priceTDFI = df['close']

    useHEV = True
    crossHEV = True
    inverseHEV = False
    highlightMovementsHEV = True
    length = 200
    HV_ma = 20
    divisor = 3.6

    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = np.power(betajmaJMA, powerjmaJMA)

    jmaJMA = pd.Series(index=df.index, dtype=float)
    e0JMA = pd.Series(0.0, index=df.index)
    e1JMA = pd.Series(0.0, index=df.index)
    e2JMA = pd.Series(0.0, index=df.index)

    srcjmaJMA = df['close']
    for i in range(1, len(df)):
        e0JMA.iloc[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i] + alphajmaJMA * e0JMA.iloc[i-1]
        e1JMA.iloc[i] = (srcjmaJMA.iloc[i] - e0JMA.iloc[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA.iloc[i-1]
        e2JMA.iloc[i] = (e0JMA.iloc[i] + phasejmaJMARatiojmaJMA * e1JMA.iloc[i] - jmaJMA.iloc[i-1]) * np.power(1 - alphajmaJMA, 2) + np.power(alphajmaJMA, 2) * e2JMA.iloc[i-1]
        jmaJMA.iloc[i] = e2JMA.iloc[i] + jmaJMA.iloc[i-1]

    signalmaJMALong = (~usejmaJMA) | ((~usecolorjmaJMA) & (df['close'] > jmaJMA)) | (usecolorjmaJMA & (jmaJMA > jmaJMA.shift(1)) & (df['close'] > jmaJMA))
    signalmaJMAShort = (~usejmaJMA) | ((~usecolorjmaJMA) & (df['close'] < jmaJMA)) | (usecolorjmaJMA & (jmaJMA < jmaJMA.shift(1)) & (df['close'] < jmaJMA))

    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort

    def tema_tdfi(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    def ma_tdfi(mode, src, length):
        if mode == 'ema':
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'wma':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
        elif mode == 'swma':
            return src.rolling(8).mean()
        elif mode == 'vwma':
            return pd.Series((src * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum())
        elif mode == 'hull':
            wma_half = src.rolling(int(length/2)).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
            hull = 2 * wma_half - src.rolling(length).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
            return hull.rolling(int(np.sqrt(length))).mean()
        elif mode == 'tema':
            return tema_tdfi(src, length)
        else:
            return src.rolling(length).mean()

    mma_tdfi = ma_tdfi(mmaModeTDFI, priceTDFI * 1000, mmaLengthTDFI)
    smma_tdfi = ma_tdfi(smmaModeTDFI, mma_tdfi, smmaLengthTDFI)
    impetmma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impetsmma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    divma_tdfi = np.abs(mma_tdfi - smma_tdfi)
    averimpet_tdfi = (impetmma_tdfi + impetsmma_tdfi) / 2
    tdf_tdfi = np.power(divma_tdfi, 1) * np.power(averimpet_tdfi, nLengthTDFI)
    highest_tdfi = tdf_tdfi.abs().rolling(lookbackTDFI * nLengthTDFI).max()
    signal_tdfi = tdf_tdfi / highest_tdfi

    signalLongTDFI = (~useTDFI) | ((~crossTDFI) & (signal_tdfi > filterHighTDFI)) | (crossTDFI & (signal_tdfi > signal_tdfi.shift(1)) & (signal_tdfi.shift(1) <= filterHighTDFI))
    signalShortTDFI = (~useTDFI) | ((~crossTDFI) & (signal_tdfi < filterLowTDFI)) | (crossTDFI & (signal_tdfi < signal_tdfi.shift(1)) & (signal_tdfi.shift(1) >= filterLowTDFI))

    finalLongSignalTDFI = signalShortTDFI if inverseTDFI else signalLongTDFI
    finalShortSignalTDFI = signalLongTDFI if inverseTDFI else signalShortTDFI

    range_1 = df['high'] - df['low']
    rangeAvg = range_1.rolling(length).mean()
    durchschnitt = df['volume'].rolling(HV_ma).mean()
    volumeA = df['volume'].rolling(length).mean()

    high1 = df['high'].shift(1)
    low1 = df['low'].shift(1)
    mid1 = ((df['high'] + df['low']) / 2).shift(1)

    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor

    r_enabled1 = (range_1 > rangeAvg) & (df['close'] < d1) & (df['volume'] > volumeA)
    r_enabled2 = df['close'] < mid1
    r_enabled = r_enabled1 | r_enabled2

    g_enabled1 = df['close'] > mid1
    g_enabled2 = (range_1 > rangeAvg) & (df['close'] > u1) & (df['volume'] > volumeA)
    g_enabled3 = (df['high'] > high1) & (range_1 < rangeAvg / 1.5) & (df['volume'] < volumeA)
    g_enabled4 = (df['low'] < low1) & (range_1 < rangeAvg / 1.5) & (df['volume'] > volumeA)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4

    gr_enabled1 = (range_1 > rangeAvg) & (df['close'] > d1) & (df['close'] < u1) & (df['volume'] > volumeA) & (df['volume'] < volumeA * 1.5) & (df['volume'] > df['volume'].shift(1))
    gr_enabled2 = (range_1 < rangeAvg / 1.5) & (df['volume'] < volumeA / 1.5)
    gr_enabled3 = (df['close'] > d1) & (df['close'] < u1)
    gr_enabled = gr_enabled1 | gr_enabled2 | gr_enabled3

    basicLongHEVCondition = g_enabled & (df['volume'] > durchschnitt)
    basicShorHEVondition = r_enabled & (df['volume'] > durchschnitt)

    HEVSignalsLong = (~useHEV) | ((~highlightMovementsHEV) & g_enabled) | (highlightMovementsHEV & basicLongHEVCondition)
    HEVSignalsShort = (~useHEV) | ((~highlightMovementsHEV) & r_enabled) | (highlightMovementsHEV & basicShorHEVondition)

    HEVSignalsLongCross = (~crossHEV) | ((~HEVSignalsLong.shift(1)) & HEVSignalsLong)
    HEVSignalsShorHEVross = (~crossHEV) | ((~HEVSignalsShort.shift(1)) & HEVSignalsShort)

    HEVSignalsLongFinal = HEVSignalsShorHEVross if inverseHEV else HEVSignalsLongCross
    HEVSignalsShortFinal = HEVSignalsLongCross if inverseHEV else HEVSignalsShorHEVross

    long_condition = signalmaJMALong & finalLongSignalTDFI & basicLongHEVCondition
    short_condition = signalmaJMAShort & finalShortSignalTDFI & basicShorHEVondition

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        close_price = df['close'].iloc[i]

        if long_condition.iloc[i]:
            if pd.notna(jmaJMA.iloc[i]) and pd.notna(signal_tdfi.iloc[i]):
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_price,
                    'raw_price_b': close_price
                })
                trade_num += 1

        if short_condition.iloc[i]:
            if pd.notna(jmaJMA.iloc[i]) and pd.notna(signal_tdfi.iloc[i]):
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close_price,
                    'raw_price_b': close_price
                })
                trade_num += 1

    return entries