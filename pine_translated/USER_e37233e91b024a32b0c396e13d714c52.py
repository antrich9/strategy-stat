import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maType = 'SMA'
    maLength = 20
    maReaction = 1
    maTypeB = 'SMA'
    maLengthB = 8
    maReactionB = 1
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    smooth_param = 1
    length = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    cblen = False
    blen = 20
    
    highestHigh = df['high'].rolling(window=donchLength).max()
    lowestLow = df['low'].rolling(window=donchLength).min()
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/previousBarsCount, adjust=False).mean()
    
    isGreenElephantCandle = df['close'] > df['open']
    isRedElephantCandle = df['close'] < df['open']
    body = (df['close'] - df['open']).abs()
    range_ = df['high'] - df['low']
    body_percentage = body * 100 / range_
    isGreenElephantCandleValid = isGreenElephantCandle & (body_percentage >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_percentage >= minBodyPercentage)
    atr_prev = atr.shift(1)
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr_prev * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr_prev * searchFactor)
    
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        v9 = pd.Series(np.nan, index=src.index)
        v9.iloc[0] = c1 * (src.iloc[0] + src.iloc[0]) / 2
        for i in range(1, len(src)):
            prev_v9 = 0 if pd.isna(v9.iloc[i-1]) else v9.iloc[i-1]
            prev_v9_2 = 0 if pd.isna(v9.iloc[i-2]) else v9.iloc[i-2]
            v9.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * prev_v9 + c3 * prev_v9_2
        return v9
    
    def variant_smoothed(src, length):
        v5 = pd.Series(np.nan, index=src.index)
        v5.iloc[0] = src.iloc[0]
        for i in range(1, len(src)):
            if pd.isna(v5.iloc[i-1]):
                v5.iloc[i] = src.iloc[:i+1].mean()
            else:
                v5.iloc[i] = (v5.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return v5
    
    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2
    
    def variant_doubleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2
    
    def variant_tripleema(src, length):
        ema1 = src.ew