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
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    useT3 = True
    crossT3 = True
    inverseT3 = False
    highlightMovementsT3 = True
    lengthT3 = 5
    factorT3 = 0.7
    pinBarSize = 30

    close = df['close']
    open_prices = df['open']
    high = df['high']
    low = df['low']

    # Hull MA calculation
    half_length = int(lengthHullMA / 2)
    sqrt_length = int(np.floor(np.sqrt(lengthHullMA)))

    wma_half = close.rolling(half_length).apply(lambda x: np.dot(x, np.arange(half_length) + 1) / (half_length * (half_length + 1) / 2), raw=True)
    wma_full = close.rolling(lengthHullMA).apply(lambda x: np.dot(x, np.arange(lengthHullMA) + 1) / (lengthHullMA * (lengthHullMA + 1) / 2), raw=True)
    hullmaHullMA = 2 * wma_half - wma_full
    hullmaHullMA = hullmaHullMA.rolling(sqrt_length).apply(lambda x: np.dot(x, np.arange(sqrt_length) + 1) / (sqrt_length * (sqrt_length + 1) / 2), raw=True)

    hullmaHullMA_prev = hullmaHullMA.shift(1)
    sigHullMA = (hullmaHullMA > hullmaHullMA_prev).astype(int) - (hullmaHullMA < hullmaHullMA_prev).astype(int)

    # T3 calculation
    def calc_t3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = calc_t3(calc_t3(calc_t3(close, lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    t3_prev = t3.shift(1)
    t3Signals = (t3 > t3_prev).astype(int) - (t3 < t3_prev).astype(int)

    # Pin bar calculation
    bodySize = (close - open_prices).abs()
    lowerWickSize = np.where(open_prices > close, close - low, open_prices - low)
    upperWickSize = np.where(open_prices > close, high - open_prices, high - close)
    totalRange = high - low

    isBullishPinBar = (close > open_prices) & (lowerWickSize > bodySize * pinBarSize / 100) & (lowerWickSize > upperWickSize * 2)
    isBearishPinBar = (close < open_prices) & (upperWickSize > bodySize * pinBarSize / 100) & (upperWickSize > lowerWickSize * 2)

    # Long conditions
    signalHullMALong = ~useHullMA | (~usecolorHullMA | ((sigHullMA > 0) & (close > hullmaHullMA)))
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = ~useT3 | (~highlightMovementsT3 | basicLongCondition)
    t3SignalsLong_prev = t3SignalsLong.shift(1)
    t3SignalsLongCross = (~crossT3 | (~t3SignalsLong_prev & t3SignalsLong))
    t3SignalsLongFinal = (~inverseT3 | ~t3SignalsLongCross)
    entryCondition = signalHullMALong & t3SignalsLongFinal & isBullishPinBar

    # Short conditions
    signalHullMAShort = ~useHullMA | (~usecolorHullMA | ((sigHullMA < 0) & (close < hullmaHullMA)))
    basicShortCondition = (t3Signals < 0) & (close < t3)
    t3SignalsShort = ~useT3 | (~highlightMovementsT3 | basicShortCondition)
    t3SignalsShort_prev = t3SignalsShort.shift(1)
    t3SignalsShortCross = (~crossT3 | (~t3SignalsShort_prev & t3SignalsShort))
    t3SignalsShortFinal = (~inverseT3 | ~t3SignalsShortCross)
    entryConditionShort = signalHullMAShort & t3SignalsShortFinal & isBearishPinBar

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if np.isnan(hullmaHullMA.iloc[i]) or np.isnan(t3.iloc[i]):
            continue

        if entryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if entryConditionShort.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries