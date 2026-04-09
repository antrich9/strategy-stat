import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    def weighted_sum(x):
        if len(x) < length:
            return np.nan
        return np.dot(x, weights[:len(x)]) / weights[:len(x)].sum()
    return series.rolling(length, min_periods=length).apply(weighted_sum, raw=True)

def wilder_ema(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def wilders_rsi(prices: pd.Series, length: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = wilder_ema(gain, length)
    avg_loss = wilder_ema(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def wilders_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = wilder_ema(tr, length)
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)

    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7

    hullma = wma(
        2 * wma(df['close'], lengthHullMA // 2) - wma(df['close'], lengthHullMA),
        int(np.sqrt(lengthHullMA))
    )

    def gdT3(src, length):
        ema1 = wilder_ema(src, length)
        ema2 = wilder_ema(ema1, length)
        return ema1 * (1 + factorT3) - ema2 * factorT3

    t3 = gdT3(gdT3(gdT3(df['close'], lengthT3), lengthT3), lengthT3)

    sigHullMA = (hullma > hullma.shift(1)).astype(int) * 2 - 1
    t3Signals = (t3 > t3.shift(1)).astype(int) * 2 - 1

    signalHullMALong = (sigHullMA > 0) & (df['close'] > hullma)
    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)

    if useT3:
        if True:
            t3SignalsLong = basicLongCondition
        else:
            t3SignalsLong = df['close'] > t3
    else:
        t3SignalsLong = pd.Series(True, index=df.index)

    if crossT3:
        t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False).astype(bool)) & t3SignalsLong.astype(bool)
    else:
        t3SignalsLongCross = t3SignalsLong

    if inverseT3:
        entryCondition = ~t3SignalsLongCross.astype(bool)
    else:
        entryCondition = t3SignalsLongCross

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pd.isna(hullma.iloc[i]) or pd.isna(t3SignalsLongFinal.iloc[i]):
            continue
        if entryCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    return entries