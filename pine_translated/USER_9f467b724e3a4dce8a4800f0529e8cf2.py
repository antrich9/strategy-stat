import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

def calculate_hull_ma(close: pd.Series, length: int) -> pd.Series:
    half_length = int(length / 2)
    sqrt_length = int(np.floor(np.sqrt(length)))
    wma_half = close.rolling(half_length).mean()
    wma_full = close.rolling(length).mean()
    hull = (2 * wma_half - wma_full).rolling(sqrt_length).mean()
    return hull

def calculate_t3(close: pd.Series, length: int, factor: float) -> tuple:
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    gd1 = ema1 * (1 + factor) - ema2 * factor
    ema3 = gd1.ewm(span=length, adjust=False).mean()
    ema4 = ema3.ewm(span=length, adjust=False).mean()
    gd2 = ema3 * (1 + factor) - ema4 * factor
    ema5 = gd2.ewm(span=length, adjust=False).mean()
    ema6 = ema5.ewm(span=length, adjust=False).mean()
    t3 = ema6 * (1 + factor) - ema6.ewm(span=length, adjust=False).mean() * factor
    return t3, t3.shift(1)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr = wilder_smooth(tr, length)
    plus_di = 100 * wilder_smooth(plus_dm, length) / atr
    minus_di = 100 * wilder_smooth(minus_dm, length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder_smooth(dx, length)
    return adx

def kalman_filter_single(source: float, x: float, p: float, q: float, r: float) -> tuple:
    x_predicted = x
    p_predicted = p + q
    k = p_predicted / (p_predicted + r)
    x_new = x_predicted + k * (source - x_predicted)
    p_new = (1 - k) * p_predicted
    return x_new, p_new

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    
    adxLength = 14
    adxThreshold = 25
    lengthHullMA = 9
    lengthT3 = 5
    factorT3 = 0.7
    q = 0.001
    r = 0.001
    
    hullma = calculate_hull_ma(close, lengthHullMA)
    hullma_prev = hullma.shift(1)
    sigHullMA = (hullma > hullma_prev).astype(int).replace(0, -1)
    
    t3, t3_prev = calculate_t3(close, lengthT3, factorT3)
    t3Signals = (t3 > t3_prev).astype(int).replace(0, -1)
    
    adx = calculate_adx(high, low, close, adxLength)
    
    kalmanPrice = np.full(len(df), np.nan)
    x_kalman = np.nan
    p_kalman = 1.0
    for i in range(len(df)):
        if np.isnan(x_kalman):
            x_kalman = close.iloc[i]
        else:
            x_kalman, p_kalman = kalman_filter_single(close.iloc[i], x_kalman, p_kalman, q, r)
        kalmanPrice[i] = x_kalman
    kalmanPrice = pd.Series(kalmanPrice, index=df.index)
    
    signalHullMALong = (sigHullMA > 0) & (close > hullma)
    
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = basicLongCondition
    
    t3SignalsLongPrev = t3SignalsLong.shift(1)
    t3SignalsLongCross = (~t3SignalsLongPrev) & t3SignalsLong
    
    t3SignalsLongFinal = t3SignalsLongCross
    
    kalmanLongCondition = close > kalmanPrice
    
    entryCondition = signalHullMALong & t3SignalsLongFinal & kalmanLongCondition & (adx > adxThreshold)
    
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entryCondition.iloc[i] and not pd.isna(hullma.iloc[i]) and not pd.isna(t3.iloc[i]) and not pd.isna(adx.iloc[i]) and not pd.isna(kalmanPrice.iloc[i]):
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    return entries