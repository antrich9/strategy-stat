import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wma(series, length):
    w = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hull_ma(src, length):
    half_length = int(length / 2)
    sqrt_length = int(np.floor(np.sqrt(length)))
    wma1 = wma(src, half_length)
    wma2 = wma(src, length)
    hull = wma(2 * wma1 - wma2, sqrt_length)
    return hull

def t3_ema(src, length, factor):
    ema1 = src.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return ema1 * (1 + factor) - ema2 * factor

def gd_t3(src, length, factor):
    ema1 = src.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return ema1 * (1 + factor) - ema2 * factor

def t3(src, length, factor):
    e1 = gd_t3(src, length, factor)
    e2 = gd_t3(e1, length, factor)
    e3 = gd_t3(e2, length, factor)
    return e3

def wilder_rsi(src, length):
    delta = src.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def wilder_atr(high, low, close, length):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))

def generate_entries(df: pd.DataFrame) -> list:
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    srcHullMA = df['close']
    
    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    srcT3 = df['close']
    
    pinBarSize = 30
    
    hullmaHullMA = hull_ma(srcHullMA, lengthHullMA)
    sigHullMA = (hullmaHullMA > hullmaHullMA.shift(1)).astype(int) - (hullmaHullMA < hullmaHullMA.shift(1)).astype(int)
    
    t3_val = t3(srcT3, lengthT3, factorT3)
    t3Signals = (t3_val > t3_val.shift(1)).astype(int) - (t3_val < t3_val.shift(1)).astype(int)
    
    signalHullMALong_series = hullmaHullMA.notna() & (sigHullMA > 0) & (df['close'] > hullmaHullMA)
    basicLongCondition_series = (t3Signals > 0) & (df['close'] > t3_val)
    t3SignalsLong_series = (t3_val > t3_val.shift(1)) & (df['close'] > t3_val)
    t3SignalsLongCross_series = ~t3SignalsLong_series.shift(1, fill_value=False) & t3SignalsLong_series
    t3SignalsLongFinal_series = ~t3SignalsLongCross_series if inverseT3 else t3SignalsLongCross_series
    
    bodySize = (df['close'] - df['open']).abs()
    lowerWickSize = df[['close', 'open']].max(axis=1) - df['low']
    upperWickSize = df['high'] - df[['close', 'open']].min(axis=1)
    
    isBullishPinBar = (df['close'] > df['open']) & (lowerWickSize > bodySize * pinBarSize / 100) & (lowerWickSize > upperWickSize * 2)
    isBearishPinBar = (df['close'] < df['open']) & (upperWickSize > bodySize * pinBarSize / 100) & (upperWickSize > lowerWickSize * 2)
    
    entryCondition = signalHullMALong_series & t3SignalsLongFinal_series & isBullishPinBar
    shortEntryCondition_series = (sigHullMA < 0) & (df['close'] < hullmaHullMA) & (t3Signals < 0) & (df['close'] < t3_val) & isBearishPinBar
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if entryCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        if shortEntryCondition_series.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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