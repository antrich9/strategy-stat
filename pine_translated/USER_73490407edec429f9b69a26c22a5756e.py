import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    ...
    """
    # Constants (default values from script)
    lengthHullMA = 9
    useHullMA = True
    usecolorHullMA = True
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    crossT3 = True
    inverseT3 = False
    ttfLength = 15
    ttfFactor = 2.0
    # Ensure required columns exist
    close = df['close']
    high = df['high']
    low = df['low']
    # Hull MA
    def wma(series, length):
        weights = np.arange(1, length + 1, dtype=float)
        def weighted_sum(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=length).apply(weighted_sum, raw=True)
    half_len = lengthHullMA // 2
    sqrt_len = int(np.sqrt(lengthHullMA))
    wma_half = wma(close, half_len)
    wma_full = wma(close, lengthHullMA)
    hullma = wma(2 * wma_half - wma_full, sqrt_len)
    sigHullMA = pd.Series(np.where(hullma > hullma.shift(1), 1, -1), index=close.index)
    # HullMA condition
    signalHullMALong = ((sigHullMA > 0) & (close > hullma)) if useHullMA else pd.Series(True, index=close.index)
    if useHullMA and not usecolorHullMA:
        signalHullMALong = close > hullma
    # T3
    def gd3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    t3_step1 = gd3(close, lengthT3, factorT3)
    t3_step2 = gd3(t3_step1, lengthT3, factorT3)
    t3 = gd3(t3_step2, lengthT3, factorT3)
    t3Signals = pd.Series(np.where(t3 > t3.shift(1), 1, -1), index=close.index)
    # Long condition based on T3
    basicLongCondition = (t3Signals > 0) & (close > t3)
    if highlightMovementsT3:
        t3SignalsLong = basicLongCondition
    else:
        t3SignalsLong = close > t3
    # Cross condition
    if crossT3:
        prev = t3SignalsLong.shift(1).fillna(False)
        t3SignalsLongCross = (~prev) & t3SignalsLong
    else:
        t3SignalsLongCross = t3SignalsLong
    # Inverse (not used)
    t3SignalsLongFinal = t3SignalsLongCross if not inverseT3 else ~t3SignalsLongCross
    # TTF
    close_prev = close.shift(1)
    hi = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    hi_sum = hi.rolling(window=ttfLength).sum()
    bull = ((close > close_prev).astype(float) * hi).rolling(window=ttfLength).sum()
    bear = ((close < close_prev).astype(float) * hi).rolling(window=ttfLength).sum()
    buy = bull * 100 / hi_sum
    sell = bear * 100 / hi_sum
    v1 = pd.Series(np.where(buy > buy.shift(1), buy - buy.shift(1), 0.0), index=close.index)
    v2 = pd.Series(np.where(sell > sell.shift(1), sell - sell.shift(1), 0.0), index=close.index)
    vb = v1.rolling(window=ttfLength).sum()
    vs = v2.rolling(window=ttfLength).sum()
    ttfValue = pd.Series(np.where(vb + vs != 0, 100 * (vb - vs) / (vb + vs), 0.0), index=close.index)
    ttfBuySignal = ttfValue > 0
    # Combined entry condition
    entry_condition = signalHullMALong & t3SignalsLongFinal & ttfBuySignal
    # Ensure all required indicators are not NaN
    valid = hullma.notna() & t3.notna() & ttfValue.notna()
    entry_condition = entry_condition & valid
    # Build entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_condition.iloc[i]:
            ts = df['time'].iloc[i]
            entry_price = df['close'].iloc[i]
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    return entries