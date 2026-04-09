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
    
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maLength = 20
    maLengthB = 8
    maReaction = 1
    maReactionB = 1
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    
    prevBarsCount = previousBarsCount
    
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = pd.Series(np.nan, index=tr.index)
        atr.iloc[length - 1] = tr.iloc[:length].mean()
        alpha = 1.0 / length
        for i in range(length, len(tr)):
            atr.iloc[i] = atr.iloc[i-1] * (1 - alpha) + tr.iloc[i] * alpha
        return atr
    
    highestHigh = high.rolling(window=donchLength).max()
    lowestLow = low.rolling(window=donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2
    
    atr = wilder_atr(high, low, close, prevBarsCount)
    
    body = np.abs(close - open_price)
    range_hl = high - low
    
    isGreenElephant = close > open_price
    isRedElephant = close < open_price
    
    isGreenElephantValid = isGreenElephant & (body * 100 / range_hl >= minBodyPercentage)
    isRedElephantValid = isRedElephant & (body * 100 / range_hl >= minBodyPercentage)
    
    isGreenElephantStrong = isGreenElephantValid & (body >= atr.shift(1) * searchFactor)
    isRedElephantStrong = isRedElephantValid & (body >= atr.shift(1) * searchFactor)
    
    slowMA = close.ewm(span=maLength, adjust=False).mean()
    fastMA = close.ewm(span=maLengthB, adjust=False).mean()
    
    slowMATrend = pd.Series(0, index=close.index)
    fastMATrend = pd.Series(0, index=close.index)
    
    for i in range(maReaction, len(close)):
        if slowMA.iloc[i] > slowMA.iloc[i - maReaction]:
            slowMATrend.iloc[i] = 1
        elif slowMA.iloc[i] < slowMA.iloc[i - maReaction]:
            slowMATrend.iloc[i] = -1
        else:
            slowMATrend.iloc[i] = slowMATrend.iloc[i - 1] if not pd.isna(slowMATrend.iloc[i - 1]) else 0
    
    for i in range(maReactionB, len(close)):
        if fastMA.iloc[i] > fastMA.iloc[i - maReactionB]:
            fastMATrend.iloc[i] = 1
        elif fastMA.iloc[i] < fastMA.iloc[i - maReactionB]:
            fastMATrend.iloc[i] = -1
        else:
            fastMATrend.iloc[i] = fastMATrend.iloc[i - 1] if not pd.isna(fastMATrend.iloc[i - 1]) else 0
    
    isBullishTrend = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBearishTrend = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    
    longCondition = isGreenElephantStrong & isBullishTrend
    shortCondition = isRedElephantStrong & isBearishTrend
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < prevBarsCount:
            continue
        if pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries