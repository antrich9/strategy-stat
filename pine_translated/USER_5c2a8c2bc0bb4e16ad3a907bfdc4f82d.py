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
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Default parameters
    lengthHullMA = 9
    srcHullMA_col = close
    
    lengthT3 = 5
    factorT3 = 0.7
    
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    # Hull MA calculation
    halfLength = int(lengthHullMA / 2)
    wma_half = srcHullMA_col.rolling(halfLength).apply(lambda x: np.dot(x, np.arange(halfLength)) / np.arange(halfLength).sum(), raw=True)
    wma_full = srcHullMA_col.rolling(lengthHullMA).apply(lambda x: np.dot(x, np.arange(lengthHullMA)) / np.arange(lengthHullMA).sum(), raw=True)
    hullmaHullMA = (2 * wma_half - wma_full).rolling(int(np.floor(np.sqrt(lengthHullMA)))).apply(
        lambda x: np.dot(x, np.arange(int(np.floor(np.sqrt(lengthHullMA))))) / np.arange(int(np.floor(np.sqrt(lengthHullMA)))).sum(), raw=True)
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int) - (hullmaHullMA <= hullmaHullMA.shift(1)).astype(int)
    signalHullMALong = ((sigHullMA > 0) & (close > hullmaHullMA))
    
    # T3 calculation
    def gdT3(src, length, factor):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor
    
    t3 = gdT3(gdT3(gdT3(close, lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    t3Signals = (t3 > t3.shift(1)).astype(int) - (t3 <= t3.shift(1)).astype(int)
    
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = basicLongCondition
    
    # crossT3 is True by default
    t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong
    
    # inverseT3 is False by default
    t3SignalsLongFinal = t3SignalsLongCross
    
    # Stiffness calculation
    boundStiffness = close.rolling(maLengthStiffness).mean() - 0.2 * close.rolling(maLengthStiffness).std(ddof=0)
    
    def rolling_sum_above(close_col, bound_col, stiff_len):
        result = pd.Series(np.nan, index=close_col.index)
        for i in range(stiff_len - 1, len(close_col)):
            window_close = close_col.iloc[i - stiff_len + 1:i + 1]
            window_bound = bound_col.iloc[i - stiff_len + 1:i + 1]
            result.iloc[i] = (window_close > window_bound).sum()
        return result
    
    sumAboveStiffness = rolling_sum_above(close, boundStiffness, stiffLength)
    stiffness = (sumAboveStiffness * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = stiffness > thresholdStiffness
    
    # Combined entry condition
    entryCondition = t3SignalsLongFinal & signalStiffness
    
    # Generate entries
    entries = []
    trade_num = 0
    in_position = False
    
    for i in range(len(df)):
        if entryCondition.iloc[i] and not in_position:
            if not pd.isna(hullmaHullMA.iloc[i]) and not pd.isna(t3.iloc[i]) and not pd.isna(stiffness.iloc[i]):
                trade_num += 1
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
                in_position = True
    
    return entries