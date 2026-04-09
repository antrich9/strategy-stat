import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Extract price data
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    ts = df['time']

    # Default input parameters
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    srcHullMA = close

    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    srcT3 = close

    pinBarSize = 30

    # Hull MA calculation
    half_len = int(lengthHullMA / 2)
    sqrt_len = int(np.sqrt(lengthHullMA))
    wma_half = srcHullMA.ewm(span=half_len, adjust=False).mean()
    wma_full = srcHullMA.ewm(span=lengthHullMA, adjust=False).mean()
    hullmaHullMA = (2 * wma_half - wma_full).ewm(span=sqrt_len, adjust=False).mean()

    # T3 calculation
    def gd_t3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = gd_t3(gd_t3(gd_t3(srcT3, lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)

    # Signal calculations
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int) - (hullmaHullMA <= hullmaHullMA.shift(1)).astype(int)
    t3Signals = (t3 > t3.shift(1)).astype(int) - (t3 <= t3.shift(1)).astype(int)

    # Long entry conditions
    signalHullMALong = (~useHullMA) | (not usecolorHullMA) | ((sigHullMA > 0) & (close > hullmaHullMA))
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = (~useT3) | (not highlightMovementsT3) | basicLongCondition | (close > t3)
    t3SignalsLong = pd.Series(np.where(~useT3, True, np.where(highlightMovementsT3, basicLongCondition, close > t3)), index=close.index)
    t3SignalsLongCross = pd.Series(np.where(crossT3, (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong, t3SignalsLong), index=close.index)
    t3SignalsLongFinal = pd.Series(np.where(inverseT3, ~t3SignalsLongCross, t3SignalsLongCross), index=close.index)

    # Pin bar calculation
    bodySize = (close - open_price).abs()
    lowerWickSize = np.where(open_price > close, close - low, open_price - low)
    upperWickSize = np.where(open_price > close, high - open_price, high - close)
    isBullishPinBar = (close > open_price) & (lowerWickSize > bodySize * pinBarSize / 100) & (lowerWickSize > upperWickSize * 2)

    # Combined long entry
    entryCondition = signalHullMALong & t3SignalsLongFinal & isBullishPinBar

    # Short entry conditions
    pinBarSizeShort = 30
    isBearishPinBar = (close < open_price) & (upperWickSize > bodySize * pinBarSizeShort / 100) & (upperWickSize > lowerWickSize * 2)
    signalHullMAShort = (~useHullMA) | (not usecolorHullMA) | ((sigHullMA < 0) & (close < hullmaHullMA))
    basicShortCondition = (t3Signals < 0) & (close < t3)
    t3SignalsShort = pd.Series(np.where(~useT3, True, np.where(highlightMovementsT3, basicShortCondition, close < t3)), index=close.index)
    t3SignalsShortCross = pd.Series(np.where(crossT3, (~t3SignalsShort.shift(1).fillna(False)) & t3SignalsShort, t3SignalsShort), index=close.index)
    t3SignalsShortFinal = pd.Series(np.where(inverseT3, ~t3SignalsShortCross, t3SignalsShortCross), index=close.index)

    # Combined short entry
    shortEntryCondition = signalHullMAShort & t3SignalsShortFinal & isBearishPinBar

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        entry_price = close.iloc[i]

        if entryCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts.iloc[i]),
                'entry_time': datetime.fromtimestamp(ts.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if shortEntryCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts.iloc[i]),
                'entry_time': datetime.fromtimestamp(ts.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries