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
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Parameters (hardcoded from script)
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maLength = 20
    maLengthB = 8
    maReaction = 1
    maReactionB = 1
    
    # ATR Calculation (Wilder)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/previousBarsCount, adjust=False).mean()
    atr = atr.bfill()
    
    # Donchian Channel
    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2
    
    # Elephant Candle Conditions
    body = np.abs(close - open_)
    bodyPct = body * 100 / (high - low + 1e-10)
    
    isGreenElephantCandle = close > open_
    isRedElephantCandle = close < open_
    
    isGreenElephantCandleValid = isGreenElephantCandle & (bodyPct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (bodyPct >= minBodyPercentage)
    
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    
    # Moving Averages (SMA)
    slowMA = close.rolling(maLength).mean()
    fastMA = close.rolling(maLengthB).mean()
    
    # Trend Direction
    slowMATrend = pd.Series(0.0, index=df.index)
    fastMATrend = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if slowMA.iloc[i] > slowMA.iloc[i - maReaction]:
            slowMATrend.iloc[i] = 1
        elif slowMA.iloc[i] < slowMA.iloc[i - maReaction]:
            slowMATrend.iloc[i] = -1
        else:
            slowMATrend.iloc[i] = slowMATrend.iloc[i - 1]
    
    for i in range(1, len(df)):
        if fastMA.iloc[i] > fastMA.iloc[i - maReactionB]:
            fastMATrend.iloc[i] = 1
        elif fastMA.iloc[i] < fastMA.iloc[i - maReactionB]:
            fastMATrend.iloc[i] = -1
        else:
            fastMATrend.iloc[i] = fastMATrend.iloc[i - 1]
    
    # Bullish Trend Conditions (PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA)
    priceAboveFastMAWithBullishTrend = (close > fastMA) & (fastMATrend > 0)
    
    # Bearish Trend Conditions (PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA)
    priceBelowFastMAWithBearishTrend = (close < fastMA) & (fastMATrend < 0)
    
    # Final Elephant Candle Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & priceAboveFastMAWithBullishTrend
    finalRedElephantCandle = isRedElephantCandleStrong & priceBelowFastMAWithBearishTrend
    
    # Filter settings
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    filterType = 'CON FILTRADO DE TENDENCIA'
    
    # Final results
    resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') | 
        ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong)
    )
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') | 
        ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong)
    )
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if resultGreenElephantCandle.iloc[i]:
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
        
        if resultRedElephantCandle.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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