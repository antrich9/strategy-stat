import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']

    lengthHullMA = 9
    srcHullMA = close
    lengthT3 = 5
    factorT3 = 0.7
    pinBarSize = 30
    pinBarSizeShort = 30
    stcLength = 10
    stcFastLength = 23
    stcSlowLength = 50
    atrPeriod = 14

    ema200 = close.ewm(span=200, adjust=False).mean()

    wma_half = srcHullMA.rolling(lengthHullMA // 2).mean()
    wma_full = srcHullMA.rolling(lengthHullMA).mean()
    hullmaHullMA = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(lengthHullMA)), adjust=False).mean()

    hullmaHullMA_prev = hullmaHullMA.shift(1)
    sigHullMA = (hullmaHullMA > hullmaHullMA_prev).astype(int) - (hullmaHullMA <= hullmaHullMA_prev).astype(int)

    def calc_gd_t3(src):
        ema1 = src.ewm(span=lengthT3, adjust=False).mean()
        ema2 = ema1.ewm(span=lengthT3, adjust=False).mean()
        return ema1 * (1 + factorT3) - ema2 * factorT3

    t3 = calc_gd_t3(calc_gd_t3(calc_gd_t3(srcHullMA)))
    t3_prev = t3.shift(1)
    t3Signals = (t3 > t3_prev).astype(int) - (t3 <= t3_prev).astype(int)

    bodySize = (close - open_arr).abs()
    lowerWickSize = np.where(open_arr > close, close - low, open_arr - low)
    upperWickSize = np.where(open_arr > close, high - open_arr, high - close)

    isBullishPinBar = (close > open_arr) & (lowerWickSize > bodySize * pinBarSize / 100) & (lowerWickSize > upperWickSize * 2)
    isBearishPinBar = (close < open_arr) & (upperWickSize > bodySize * pinBarSizeShort / 100) & (upperWickSize > lowerWickSize * 2)

    stcMacd = close.ewm(span=stcFastLength, adjust=False).mean() - close.ewm(span=stcSlowLength, adjust=False).mean()
    stcMacd_min = stcMacd.rolling(stcLength).min()
    stcMacd_max = stcMacd.rolling(stcLength).max()
    stcValue = 100 * (stcMacd - stcMacd_min) / (stcMacd_max - stcMacd_min)
    stcK = stcValue.ewm(span=3, adjust=False).mean()
    stcD = stcK.ewm(span=3, adjust=False).mean()

    obv = (np.sign(close.diff()) * df['volume']).fillna(0).cumsum()

    signalHullMALong = (sigHullMA > 0) & (close > hullmaHullMA)
    basicLongCondition = (t3Signals > 0) & (close > t3)
    t3SignalsLong = basicLongCondition
    t3SignalsLong_prev = t3SignalsLong.shift(1).fillna(False).astype(bool)
    t3SignalsLongCross = (~t3SignalsLong_prev) & t3SignalsLong
    t3SignalsLongFinal = t3SignalsLongCross

    longEntryCondition = (close > ema200) & signalHullMALong & t3SignalsLongFinal & isBullishPinBar & (stcK > stcD) & (stcK > 50) & (obv > obv.shift(1))
    shortEntryCondition = (close < ema200) & (~signalHullMALong) & (~t3SignalsLongFinal) & isBearishPinBar & (stcK < stcD) & (stcK < 50) & (obv < obv.shift(1))

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if np.isnan(hullmaHullMA.iloc[i]) or np.isnan(t3.iloc[i]) or np.isnan(stcK.iloc[i]) or np.isnan(obv.iloc[i]):
            continue

        ts = int(df['time'].iloc[i])
        price = close.iloc[i]

        if longEntryCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

        if shortEntryCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries