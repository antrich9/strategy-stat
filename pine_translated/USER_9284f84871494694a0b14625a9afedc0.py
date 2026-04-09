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
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    # Parameters (from inputs)
    donchLength = 20
    donchianFilterMode = "BREAKOUT"
    atrLength = 14
    atrMultiplier = 1.0
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maType = "SMA"
    maLength = 20
    maSource = close
    maReaction = 1
    maTypeB = "SMA"
    maLengthB = 8
    maSourceB = close
    maReactionB = 1
    bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    bearishTrendCondition = "DIRECCION MEDIA RAPIDA BAJISTA"
    
    # Wilder RSI implementation
    def wilders_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilders_atr(h, l, c, length):
        prev_c = c.shift(1).fillna(0)
        tr1 = h - l
        tr2 = (h - prev_c).abs()
        tr3 = (l - prev_c).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Donchian Channel
    highestHigh = high.rolling(window=donchLength).max()
    lowestLow = low.rolling(window=donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2
    
    # ATR
    atrValue = wilders_atr(high, low, close, atrLength)
    elephantAtr = wilders_atr(high, low, close, previousBarsCount)
    
    # Elephant Candle Conditions
    body = (close - open_price).abs()
    candleRange = (high - low).replace(0, np.nan)
    bodyPct = body / candleRange * 100
    
    isGreenElephantCandleValid = (close > open_price) & (bodyPct >= minBodyPercentage)
    isRedElephantCandleValid = (close < open_price) & (bodyPct >= minBodyPercentage)
    
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= elephantAtr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= elephantAtr.shift(1) * searchFactor)
    
    # Moving Averages (SMA)
    slowMA = maSource.rolling(window=maLength).mean()
    fastMA = maSourceB.rolling(window=maLengthB).mean()
    
    # Trend Direction (rising/falling)
    slowMATrend = pd.Series(0.0, index=df.index)
    fastMATrend = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if pd.isna(slowMA.iloc[i]) or pd.isna(slowMA.iloc[i-1]):
            slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if not pd.isna(slowMATrend.iloc[i-1]) else 0
        elif i >= maReaction and all(slowMA.iloc[i-j+1] > slowMA.iloc[i-j] for j in range(1, maReaction+1)):
            slowMATrend.iloc[i] = 1
        elif i >= maReaction and all(slowMA.iloc[i-j+1] < slowMA.iloc[i-j] for j in range(1, maReaction+1)):
            slowMATrend.iloc[i] = -1
        else:
            slowMATrend.iloc[i] = slowMATrend.iloc[i-1]
            
        if pd.isna(fastMA.iloc[i]) or pd.isna(fastMA.iloc[i-1]):
            fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if not pd.isna(fastMATrend.iloc[i-1]) else 0
        elif i >= maReactionB and all(fastMA.iloc[i-j+1] > fastMA.iloc[i-j] for j in range(1, maReactionB+1)):
            fastMATrend.iloc[i] = 1
        elif i >= maReactionB and all(fastMA.iloc[i-j+1] < fastMA.iloc[i-j] for j in range(1, maReactionB+1)):
            fastMATrend.iloc[i] = -1
        else:
            fastMATrend.iloc[i] = fastMATrend.iloc[i-1]
    
    # Bullish Trend Conditions
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close > slowMA) & (close > fastMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close > slowMA) & (close > fastMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'
    
    # Bearish Trend Conditions
    isPriceBelowFastMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA') & (close < fastMA)
    isPriceBelowSlowMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA') & (close < slowMA)
    isPriceBelowBothMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close < slowMA) & (close < fastMA)
    isPriceBelowSlowMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close < slowMA) & (slowMATrend < 0)
    isPriceBelowFastMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close < fastMA) & (fastMATrend < 0)
    isPriceBelowBothMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close < slowMA) & (close < fastMA) & (slowMATrend < 0) & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'
    
    # Final Elephant Candle Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & (
        isPriceAboveFastMA | isPriceAboveSlowMA | isPriceAboveBothMA |
        isPriceAboveSlowMAWithBullishTrend | isPriceAboveFastMAWithBullishTrend |
        isPriceAboveBothMAWithBullishTrend | isSlowMATrendBullish |
        isFastMATrendBullish | isBothMATrendBullish | noBullishCondition)
    
    finalRedElephantCandle = isRedElephantCandleStrong & (
        isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA |
        isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend |
        isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish |
        isFastMATrendBearish | isBothMATrendBearish | noBearishCondition)
    
    # Apply Donchian Filter
    if donchianFilterMode == "BREAKOUT":
        donchianBullishFilter = close > highestHigh
        donchianBearishFilter = close < lowestLow
    elif donchianFilterMode == "TREND (MIDDLE BAND)":
        donchianBullishFilter = close > middleBand
        donchianBearishFilter = close < middleBand
    else:
        donchianBullishFilter = pd.Series(True, index=df.index)
        donchianBearishFilter = pd.Series(True, index=df.index)
    
    finalLongEntry = finalGreenElephantCandle & donchianBullishFilter
    finalShortEntry = finalRedElephantCandle & donchianBearishFilter
    
    # Iterate and generate entries
    entries = []
    trade_num = 0
    
    for i in range(len(df)):
        if i < previousBarsCount:
            continue
        if pd.isna(fastMA.iloc[i]) or pd.isna(slowMA.iloc[i]) or pd.isna(atrValue.iloc[i]) or pd.isna(elephantAtr.iloc[i]):
            continue
        if pd.isna(highestHigh.iloc[i]) or pd.isna(lowestLow.iloc[i]):
            continue
        if finalLongEntry.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
        if finalShortEntry.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
    
    return entries