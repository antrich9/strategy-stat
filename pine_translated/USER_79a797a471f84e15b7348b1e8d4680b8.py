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
    
    lengthCoppockCurve = 10
    longRocLengthCoppockCurve = 14
    shortRocLengthCoppockCurve = 11
    lengthCoppockCurveMA = 10
    lengthTC1 = 5
    factorTC1 = 0.7
    lengthTC2 = 18
    factorTC2 = 0.7
    atrLength = 14
    
    def compute_wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr
    
    def compute_wma(series, length):
        weights = np.arange(1, length + 1)
        def wma_func(x):
            if len(x) < length or np.any(np.isnan(x)):
                return np.nan
            return np.dot(x, weights) / weights.sum()
        return series.rolling(length).apply(wma_func, raw=True)
    
    def compute_roc(series, length):
        return ((series / series.shift(length)) - 1) * 100
    
    def gdTC(src, lengthTC, factorTC):
        ema1 = src.ewm(span=lengthTC, adjust=False).mean()
        ema2 = ema1.ewm(span=lengthTC, adjust=False).mean()
        tc = ema1 * (1 + factorTC) - ema2 * factorTC
        return tc
    
    roc_long = compute_roc(close, longRocLengthCoppockCurve)
    roc_short = compute_roc(close, shortRocLengthCoppockCurve)
    roc_sum = roc_long + roc_short
    coppock = compute_wma(roc_sum, lengthCoppockCurve)
    coppockMA = coppock.ewm(span=lengthCoppockCurveMA, adjust=False).mean()
    
    TC1 = gdTC(gdTC(gdTC(close, lengthTC1, factorTC1), lengthTC1, factorTC1), lengthTC1, factorTC1)
    TC2 = gdTC(gdTC(gdTC(close, lengthTC2, factorTC2), lengthTC2, factorTC2), lengthTC2, factorTC2)
    
    basicLongCondition = (TC1 > TC1.shift(1)) & (TC2 > TC2.shift(1)) & (TC1 > TC2)
    basicShortCondition = (TC1 < TC1.shift(1)) & (TC2 < TC2.shift(1)) & (TC1 < TC2)
    
    coppock_long = coppock > 0
    coppock_short = coppock < 0
    
    entrySignalLong = coppock_long & basicLongCondition
    entrySignalShort = coppock_short & basicShortCondition
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(coppock.iloc[i]):
            continue
        
        if entrySignalLong.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif entrySignalShort.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries