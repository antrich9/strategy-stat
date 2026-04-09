import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    lookback = 3
    pivotType = "Close"
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    time_col = df['time']
    
    def calc_pivothigh(src, lb):
        result = np.full(len(src), np.nan)
        for i in range(lb, len(src) - lb):
            is_high = True
            for j in range(1, lb + 1):
                if src.iloc[i] <= src.iloc[i - j] or src.iloc[i] <= src.iloc[i + j]:
                    is_high = False
                    break
            if is_high:
                result[i] = src.iloc[i]
        return pd.Series(result, index=src.index)
    
    def calc_pivotlow(src, lb):
        result = np.full(len(src), np.nan)
        for i in range(lb, len(src) - lb):
            is_low = True
            for j in range(1, lb + 1):
                if src.iloc[i] >= src.iloc[i - j] or src.iloc[i] >= src.iloc[i + j]:
                    is_low = False
                    break
            if is_low:
                result[i] = src.iloc[i]
        return pd.Series(result, index=src.index)
    
    if pivotType == "Close":
        ph = calc_pivothigh(close, lookback)
        pl = calc_pivotlow(close, lookback)
    else:
        ph = calc_pivothigh(high, lookback)
        pl = calc_pivotlow(low, lookback)
    
    highLevel = pd.Series(np.nan, index=close.index)
    lowLevel = pd.Series(np.nan, index=close.index)
    
    for i in range(len(close)):
        if not np.isnan(ph.iloc[i]) and i + lookback < len(close):
            highLevel.iloc[i + lookback] = close.iloc[i + lookback]
    
    for i in range(len(close)):
        if not np.isnan(pl.iloc[i]) and i + lookback < len(close):
            lowLevel.iloc[i + lookback] = close.iloc[i + lookback]
    
    highLevel = highLevel.ffill().bfill()
    lowLevel = lowLevel.ffill().bfill()
    
    uptrendSignal = high > highLevel
    downtrendSignal = low < lowLevel
    
    inUptrend = pd.Series(False, index=close.index)
    inDowntrend = pd.Series(False, index=close.index)
    
    for i in range(1, len(close)):
        if uptrendSignal.iloc[i-1]:
            inUptrend.iloc[i] = True
        elif downtrendSignal.iloc[i-1]:
            inUptrend.iloc[i] = False
        else:
            inUptrend.iloc[i] = inUptrend.iloc[i-1]
        inDowntrend.iloc[i] = not inUptrend.iloc[i]
    
    bull_candle = close > open_
    prev_trend = pd.Series(0, index=close.index)
    prev_trend[bull_candle] = 1
    prev_trend[~bull_candle] = -1
    
    htf_trend_1 = prev_trend.copy()
    htf_trend_2 = prev_trend.copy()
    htf_trend_3 = prev_trend.copy()
    
    bull_1 = htf_trend_1 == 1
    bull_2 = htf_trend_2 == 1
    bull_3 = htf_trend_3 == 1
    
    all_bullish = bull_1 & bull_2 & bull_3
    all_bearish = (~bull_1) & (~bull_2) & (~bull_3)
    
    long_cond = inUptrend & all_bullish
    long_cond = long_cond & (~long_cond.shift(1).fillna(False))
    
    short_cond = inDowntrend & all_bearish
    short_cond = short_cond & (~short_cond.shift(1).fillna(False))
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_cond.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(time_col.iloc[i])
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
        elif short_cond.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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