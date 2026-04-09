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
    
    # QQE Parameters
    length = 14
    SSF = 5
    
    # Elephant candle parameters
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    
    # Moving average parameters
    maLength = 20
    maLengthB = 8
    
    # Donchian channel
    donchLength = 20
    
    # Filter settings
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    maReaction = 1
    maReactionB = 1
    
    close = df['close']
    open_vals = df['open']
    high = df['high']
    low = df['low']
    
    # Wilder RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # QQE calculation
    QQEF = rsi.ewm(span=SSF, adjust=False).mean()
    
    TR = np.abs(QQEF - QQEF.shift(1))
    wwalpha = 1 / length
    
    WWMA = np.zeros(len(df))
    ATRRSI = np.zeros(len(df))
    WWMA[0] = TR.iloc[0]
    ATRRSI[0] = TR.iloc[0]
    
    for i in range(1, len(df)):
        WWMA[i] = wwalpha * TR.iloc[i] + (1 - wwalpha) * WWMA[i-1]
        ATRRSI[i] = wwalpha * WWMA[i] + (1 - wwalpha) * ATRRSI[i-1]
    
    QUP = QQEF + ATRRSI * 4.236
    QDN = QQEF - ATRRSI * 4.236
    
    QQES = np.zeros(len(df))
    QQES[0] = QUP.iloc[0]
    
    for i in range(1, len(df)):
        if QUP.iloc[i] < QQES[i-1]:
            QQES[i] = QUP.iloc[i]
        elif QQEF.iloc[i] > QQES[i-1] and QQEF.iloc[i-1] < QQES[i-1]:
            QQES[i] = QDN.iloc[i]
        elif QDN.iloc[i] > QQES[i-1]:
            QQES[i] = QDN.iloc[i]
        elif QQEF.iloc[i] < QQES[i-1] and QQEF.iloc[i-1] > QQES[i-1]:
            QQES[i] = QUP.iloc[i]
        else:
            QQES[i] = QQES[i-1]
    
    buySignalQQE = (QQEF > QQES) & (QQEF.shift(1) <= QQES.shift(1))
    sellSignalQQE = (QQEF < QQES) & (QQEF.shift(1) >= QQES.shift(1))
    
    # ATR for elephant candles (uses previousBarsCount)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = tr.ewm(alpha=1/previousBarsCount, adjust=False).mean()
    
    # Elephant candles
    body = np.abs(close - open_vals)
    full_range = np.abs(high - low)
    
    isGreenElephant = close > open_vals
    isRedElephant = close < open_vals
    
    body_pct = (body * 100) / full_range
    
    isGreenValid = isGreenElephant & (body_pct >= minBodyPercentage)
    isRedValid = isRedElephant & (body_pct >= minBodyPercentage)
    
    isGreenStrong = isGreenValid & (body >= atr.shift(1) * searchFactor)
    isRedStrong = isRedValid & (body >= atr.shift(1) * searchFactor)
    
    # Donchian Channel
    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2
    
    # Moving averages (SMA default per inputs)
    slowMA = close.rolling(maLength).mean()
    fastMA = close.rolling(maLengthB).mean()
    
    # Trend direction
    slowMATrend_vals = np.zeros(len(df))
    fastMATrend_vals = np.zeros(len(df))
    
    for i in range(maReaction, len(df)):
        if slowMA.iloc[i] > slowMA.iloc[i - maReaction]:
            slowMATrend_vals[i] = 1
        elif slowMA.iloc[i] < slowMA.iloc[i - maReaction]:
            slowMATrend_vals[i] = -1
        else:
            slowMATrend_vals[i] = slowMATrend_vals[i-1] if i > 0 else 0
    
    for i in range(maReactionB, len(df)):
        if fastMA.iloc[i] > fastMA.iloc[i - maReactionB]:
            fastMATrend_vals[i] = 1
        elif fastMA.iloc[i] < fastMA.iloc[i - maReactionB]:
            fastMATrend_vals[i] = -1
        else:
            fastMATrend_vals[i] = fastMATrend_vals[i-1] if i > 0 else 0
    
    slowMATrend = pd.Series(slowMATrend_vals, index=df.index)
    fastMATrend = pd.Series(fastMATrend_vals, index=df.index)
    
    # Trend conditions based on bullishTrendCondition and bearishTrendCondition
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close > slowMA) & (close > fastMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close > slowMA) & (close > fastMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMAT