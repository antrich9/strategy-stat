import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # ---------- Hull MA ----------
    lengthHullMA = 9
    half_len = lengthHullMA // 2
    sqrt_len = int(np.sqrt(lengthHullMA))

    def wma(series, length):
        weights = np.arange(1, length + 1)
        def weighted_avg(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(length).apply(weighted_avg, raw=True)

    hullma_half = wma(df['close'], half_len)
    hullma_full = wma(df['close'], lengthHullMA)
    hullma_raw = 2 * hullma_half - hullma_full
    hullma = wma(hullma_raw, sqrt_len)

    sigHullMA = (hullma > hullma.shift(1)).astype(int)
    # useHullMA = true, usecolorHullMA = true
    signalHullMALong = ((sigHullMA > 0) & (df['close'] > hullma)).fillna(False)

    # ---------- T3 ----------
    lengthT3 = 5
    factorT3 = 0.7

    def ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    def gd(src, length):
        e1 = ema(src, length)
        e2 = ema(e1, length)
        return e1 * (1 + factorT3) - e2 * factorT3

    t3 = gd(gd(gd(df['close'], lengthT3), lengthT3), lengthT3)

    t3Signals = (t3 > t3.shift(1)).astype(int)
    basicLongCondition = ((t3Signals > 0) & (df['close'] > t3)).fillna(False)

    # useT3 = true, highlightMovementsT3 = true
    t3SignalsLong = basicLongCondition
    # crossT3 = true
    t3SignalsLongCross = ((~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong).fillna(False)
    # inverseT3 = false
    t3SignalsLongFinal = t3SignalsLongCross

    # ---------- Fisher Transform ----------
    fisherLength = 14
    fisherThreshold = 0.0

    hl2 = (df['high'] + df['low']) / 2.0
    lowest_low = df['low'].rolling(fisherLength).min()
    highest_high = df['high'].rolling(fisherLength).max()

    value = pd.Series(0.0, index=df.index)
    for i in range(fisherLength, len(df)):
        denom = highest_high.iloc[i] - lowest_low.iloc[i]
        ratio = (hl2.iloc[i] - lowest_low.iloc[i]) / denom if denom != 0 else 0.0
        value.iloc[i] = 0.66 * (ratio - 0.5) + 0.67 * value.iloc[i - 1]

    value = value.clip(lower=-0.999, upper=0.999)
    fisher = 0.5 * np.log((1 + value) / (1 - value))

    fisherBuyCondition = ((fisher > fisherThreshold) & (fisher.shift(1).fillna(0) <= fisherThreshold)).fillna(False)

    # ---------- Combined entry condition ----------
    entry_condition = (signalHullMALong & t3SignalsLongFinal & fisherBuyCondition).fillna(False)

    # ---------- Generate entries ----------
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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

    return entries