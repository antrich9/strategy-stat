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
    # Input parameters from Pine Script
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    pinBarSize = 30

    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']

    # Hull MA calculation
    half_length = int(lengthHullMA / 2)
    sqrt_length = int(np.floor(np.sqrt(lengthHullMA)))

    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    hullmaHullMA = wma(2 * wma(close, half_length) - wma(close, lengthHullMA), sqrt_length)
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int) - (hullmaHullMA <= hullmaHullMA.shift(1)).astype(int)

    # T3 calculation
    def gdT3(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factorT3) - ema2 * factorT3

    t3 = gdT3(gdT3(gdT3(close, lengthT3), lengthT3), lengthT3)
    t3Signals = (t3 > t3.shift(1)).astype(int) - (t3 <= t3.shift(1)).astype(int)

    # Long entry conditions
    if useHullMA:
        if usecolorHullMA:
            signalHullMALong = (sigHullMA > 0) & (close > hullmaHullMA)
        else:
            signalHullMALong = close > hullmaHullMA
    else:
        signalHullMALong = pd.Series(True, index=df.index)

    basicLongCondition = (t3Signals > 0) & (close > t3)

    if useT3:
        if highlightMovementsT3:
            t3SignalsLong = basicLongCondition
        else:
            t3SignalsLong = close > t3
    else:
        t3SignalsLong = pd.Series(True, index=df.index)

    if crossT3:
        t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong
    else:
        t3SignalsLongCross = t3SignalsLong

    if inverseT3:
        t3SignalsLongFinal = ~t3SignalsLongCross
    else:
        t3SignalsLongFinal = t3SignalsLongCross

    # Short entry conditions
    if useHullMA:
        if usecolorHullMA:
            signalHullMAShort = (sigHullMA < 0) & (close < hullmaHullMA)
        else:
            signalHullMAShort = close < hullmaHullMA
    else:
        signalHullMAShort = pd.Series(True, index=df.index)

    basicShortCondition = (t3Signals < 0) & (close < t3)

    if useT3:
        if highlightMovementsT3:
            t3SignalsShort = basicShortCondition
        else:
            t3SignalsShort = close < t3
    else:
        t3SignalsShort = pd.Series(True, index=df.index)

    if crossT3:
        t3SignalsShortCross = (~t3SignalsShort.shift(1).fillna(False)) & t3SignalsShort
    else:
        t3SignalsShortCross = t3SignalsShort

    if inverseT3:
        t3SignalsShortFinal = ~t3SignalsShortCross
    else:
        t3SignalsShortFinal = t3SignalsShortCross

    # Pin bar calculation
    bodySize = (close - open_price).abs()
    lowerWickSize = np.where(open_price > close, close - low, open_price - low)
    upperWickSize = np.where(open_price > close, high - open_price, high - close)
    lowerWickSize = pd.Series(lowerWickSize, index=df.index)
    upperWickSize = pd.Series(upperWickSize, index=df.index)

    isBullishPinBar = (close > open_price) & (lowerWickSize > bodySize * pinBarSize / 100) & (lowerWickSize > upperWickSize * 2)
    isBearishPinBar = (close < open_price) & (upperWickSize > bodySize * pinBarSize / 100) & (upperWickSize > lowerWickSize * 2)

    # Combined entry conditions
    entryCondition = signalHullMALong & t3SignalsLongFinal & isBullishPinBar
    entryConditionShort = signalHullMAShort & t3SignalsShortFinal & isBearishPinBar

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(hullmaHullMA.iloc[i]) or pd.isna(t3.iloc[i]):
            continue

        if entryCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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

        if entryConditionShort.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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