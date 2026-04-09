import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters (from inputs)
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maType = "SMA"
    maLength = 20
    maTypeB = "SMA"
    maLengthB = 8
    maReaction = 1
    maReactionB = 1
    bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    bearishTrendCondition = "DIRECCION MEDIA RAPIDA BAJISTA"
    filterType = "CON FILTRADO DE TENDENCIA"
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    almaLength = 9
    offset = 0.85
    sigma = 6
    
    # ATR (Wilder) - length 100
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/previousBarsCount, min_periods=previousBarsCount, adjust=False).mean()
    
    # Elephant candles
    body = (df['close'] - df['open']).abs()
    range_ = df['high'] - df['low']
    
    isGreenElephantCandle = df['close'] > df['open']
    isRedElephantCandle = df['close'] < df['open']
    
    isGreenElephantCandleValid = isGreenElephantCandle & (body * 100 / range_ >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body * 100 / range_ >= minBodyPercentage)
    
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    
    # Moving Averages (SMA for both as per defaults)
    slowMA = df['close'].rolling(maLength).mean()
    fastMA = df['close'].rolling(maLengthB).mean()
    
    # MA Trends
    def calc_rising(series, length):
        rising = pd.Series(True, index=series.index)
        for i in range(length):
            rising &= (series.shift(i) > series.shift(i + 1))
        return rising
    
    def calc_falling(series, length):
        falling = pd.Series(True, index=series.index)
        for i in range(length):
            falling &= (series.shift(i) < series.shift(i + 1))
        return falling
    
    rising_slow = calc_rising(slowMA, maReaction)
    falling_slow = calc_falling(slowMA, maReaction)
    rising_fast = calc_rising(fastMA, maReactionB)
    falling_fast = calc_falling(fastMA, maReactionB)
    
    slowMATrend = pd.Series(0, index=df.index)
    fastMATrend = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        if rising_slow.iloc[i]:
            slowMATrend.iloc[i] = 1
        elif falling_slow.iloc[i]:
            slowMATrend.iloc[i] = -1
        else:
            slowMATrend.iloc[i] = slowMATrend.iloc[i-1]
    
    for i in range(1, len(df)):
        if rising_fast.iloc[i]:
            fastMATrend.iloc[i] = 1
        elif falling_fast.iloc[i]:
            fastMATrend.iloc[i] = -1
        else:
            fastMATrend.iloc[i] = fastMATrend.iloc[i-1]
    
    # Trend conditions based on inputs
    # bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    # This means: fastMATrend > 0
    
    isFastMATrendBullish = fastMATrend > 0
    isFastMATrendBearish = fastMATrend < 0
    
    # Final conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & isFastMATrendBullish
    finalRedElephantCandle = isRedElephantCandleStrong & isFastMATrendBearish
    
    # Apply filters
    if filterType == 'CON FILTRADO DE TENDENCIA':
        resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles
        resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles
    else: # SIN FILTRADO DE TENDENCIA
        resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & isGreenElephantCandleStrong
        resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & isRedElephantCandleStrong
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        direction = None
        if resultGreenElephantCandle.iloc[i]:
            direction = 'long'
        elif resultRedElephantCandle.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries