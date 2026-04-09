import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math

def linear_weighted_ma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    def weighted_avg(x):
        return np.sum(x * weights) / weights.sum()
    return series.rolling(window).apply(weighted_avg, raw=True)

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    # Hull MA
    length_hull = 9
    half_len = length_hull // 2
    sqrt_len = int(math.sqrt(length_hull))
    hull_half = linear_weighted_ma(close, half_len)
    hull_full = linear_weighted_ma(close, length_hull)
    hull_raw = 2 * hull_half - hull_full
    hullMA = linear_weighted_ma(hull_raw, sqrt_len)
    sigHullMA = pd.Series(np.where(hullMA > hullMA.shift(1), 1, -1), index=close.index)
    signalHullMALong = (sigHullMA > 0) & (close > hullMA)
    signalHullMAShort = (sigHullMA < 0) & (close < hullMA)

    # T3
    length_t3 = 5
    factor_t3 = 0.7
    def gd(s):
        ema = s.ewm(span=length_t3, adjust=False).mean()
        ema_ema = ema.ewm(span=length_t3, adjust=False).mean()
        return ema * (1 + factor_t3) - ema_ema * factor_t3
    t3 = gd(gd(gd(close)))
    t3Signals = pd.Series(np.where(t3 > t3.shift(1), 1, -1), index=close.index)
    basicLongCondition = (t3Signals > 0) & (close > t3)
    basicShortCondition = (t3Signals < 0) & (close < t3)
    t3SignalsLong = basicLongCondition
    # cross confirmation
    t3SignalsLongCross = t3SignalsLong & (~t3SignalsLong.shift(1).fillna(False))
    t3SignalsShort = basicShortCondition
    t3LongFinal = t3SignalsLongCross
    t3ShortFinal = t3SignalsShort

    # Zero-Lag MACD
    fast_len = 12
    slow_len = 26
    signal_len = 9
    macd_threshold = 0
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=slow_len, adjust=False).mean()
    zl_macd = ema_fast - ema_slow
    zl_signal = zl_macd.ewm(span=signal_len, adjust=False).mean()
    zlmacdLong = (zl_macd > zl_signal) & (zl_macd > macd_threshold)
    zlmacdShort = (zl_macd < zl_signal) & (zl_macd < -macd_threshold)

    # Final conditions
    longCondition = signalHullMALong & t3LongFinal & zlmacdLong
    shortCondition = signalHullMAShort & t3ShortFinal & zlmacdShort

    # Fill NaNs with False
    longCondition = longCondition.fillna(False)
    shortCondition = shortCondition.fillna(False)

    # Generate entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    return entries