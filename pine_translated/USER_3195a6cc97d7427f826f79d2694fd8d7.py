import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters (matching Pine Script defaults)
    lengthVIDYA = 14
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    
    # MA parameters (defaults)
    maType = 'SMA'
    maLength = 20
    maSource = close
    maReaction = 1
    maTypeB = 'SMA'
    maLengthB = 8
    maSourceB = close
    maReactionB = 1
    
    # Trend conditions (defaults)
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    
    # Flags
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    filterType = 'CON FILTRADO DE TENDENCIA'
    
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    timestamps = df['time'].values
    n = len(df)
    
    entries = []
    trade_num = 1
    
    # Wilder ATR
    tr = np.maximum(high_prices - low_prices, np.maximum(np.abs(high_prices - np.roll(close_prices, 1)), np.abs(low_prices - np.roll(close_prices, 1))))
    tr[0] = high_prices[0] - low_prices[0]
    atr = np.zeros(n)
    atr[previousBarsCount-1] = np.mean(tr[:previousBarsCount])
    multiplier = 1 - 1/previousBarsCount
    for i in range(previousBarsCount, n):
        atr[i] = atr[i-1] * multiplier + tr[i]
    
    # VIDYA calculation
    srcVIDYA = close_prices
    cmo = np.zeros(n)
    for i in range(lengthVIDYA, n):
        upSum = 0.0
        downSum = 0.0
        for j in range(lengthVIDYA):
            mom = srcVIDYA[i-j] - srcVIDYA[i-1-j]
            if mom > 0:
                upSum += mom
            else:
                downSum -= mom
        denom = upSum + downSum
        if denom != 0:
            cmo[i] = abs((upSum - downSum) / denom)
    
    alpha = 2 / (lengthVIDYA + 1)
    vidya = np.zeros(n)
    for i in range(n):
        if i == 0:
            vidya[i] = srcVIDYA[i] * alpha * cmo[i]
        else:
            vidya[i] = srcVIDYA[i] * alpha * cmo[i] + vidya[i-1] * (1 - alpha * cmo[i])
    
    # Donchian Channel (not used in entry but needed for completeness)
    highestHigh = np.zeros(n)
    lowestLow = np.zeros(n)
    for i in range(donchLength-1, n):
        highestHigh[i] = max(high_prices[i-donchLength+1:i+1])
        lowestLow[i] = min(low_prices[i-donchLength+1:i+1])
    
    # Elephant Candle Detection
    bodySize = np.abs(close_prices - open_prices)
    rangeSize = high_prices - low_prices
    
    isGreenElephantCandleValid = (close_prices > open_prices) & (bodySize / np.where(rangeSize != 0, rangeSize, 1) * 100 >= minBodyPercentage)
    isRedElephantCandleValid = (close_prices < open_prices) & (bodySize / np.where(rangeSize != 0, rangeSize, 1) * 100 >= minBodyPercentage)
    
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (bodySize >= atr * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (bodySize >= atr * searchFactor)
    
    # Moving Averages (using SMA as default)
    def calc_sma(src, length):
        result = np.zeros(len(src))
        for i in range(length-1, len(src)):
            result[i] = np.mean(src[i-length+1:i+1])
        return result
    
    slowMA = calc_sma(close_prices, maLength)
    fastMA = calc_sma(close_prices, maLengthB)
    
    # Trend Detection
    slowMATrend = np.zeros(n)
    fastMATrend = np.zeros(n)
    
    for i in range(1, n):
        rising_count = 0
        for k in range(1, maReaction + 1):
            if i >= k and slowMA[i] > slowMA[i-k]:
                rising_count += 1
        falling_count = 0
        for k in range(1, maReaction + 1):
            if i >= k and slowMA[i] < slowMA[i-k]:
                falling_count += 1
        if rising_count == maReaction:
            slowMATrend[i] = 1
        elif falling_count == maReaction:
            slowMATrend[i] = -1
        else:
            slowMATrend[i] = slowMATrend[i-1] if not np.isnan(slowMATrend[i-1]) else 0
    
    for i in range(1, n):
        rising_count = 0
        for k in range(1, maReactionB + 1):
            if i >= k and fastMA[i] > fastMA[i-k]:
                rising_count += 1
        falling_count = 0
        for k in range(1, maReactionB + 1):
            if i >= k and fastMA[i] < fastMA[i-k]:
                falling_count += 1
        if rising_count == maReactionB:
            fastMATrend[i] = 1
        elif falling_count == maReactionB:
            fastMATrend[i] = -1
        else:
            fastMATrend[i] = fastMATrend[i-1] if not np.isnan(fastMATrend[i-1]) else 0
    
    # Trend Conditions
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close_prices > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close_prices > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close_prices > fastMA) & (close_prices > slowMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close_prices > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close_prices > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close_prices > fastMA) & (close_prices > slowMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'
    
    isPriceBelowFastMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA') & (close_prices < fastMA)
    isPriceBelowSlowMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA') & (close_prices < slowMA)
    isPriceBelowBothMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close_prices < fastMA) & (close_prices < slowMA)
    isPriceBelowSlowMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close_prices < slowMA) & (slowMATrend < 0)
    isPriceBelowFastMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close_prices < fastMA) & (fastMATrend < 0)
    isPriceBelowBothMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close_prices < fastMA) & (close_prices < slowMA) & (slowMATrend < 0) & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'
    
    # Final Elephant Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & (
        isPriceAboveFastMA | isPriceAboveSlowMA | isPriceAboveBothMA |
        isPriceAboveSlowMAWithBullishTrend | isPriceAboveFastMAWithBullishTrend |
        isPriceAboveBothMAWithBullishTrend | isSlowMATrendBullish |
        isFastMATrendBullish | isBothMATrendBullish | noBullishCondition
    )
    
    finalRedElephantCandle = isRedElephantCandleStrong & (
        isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA |
        isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend |
        isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish |
        isFastMATrendBearish | isBothMATrendBearish | noBearishCondition
    )
    
    # Result conditions
    resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') |
        ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong)
    )
    
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') |
        ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong)
    )
    
    # Entry conditions
    longCondition = resultGreenElephantCandle & (close_prices > vidya)
    shortCondition = resultRedElephantCandle & (close_prices < vidya)
    
    # Generate entries
    for i in range(n):
        if i < max(previousBarsCount, lengthVIDYA, donchLength, maLength, maLengthB):
            continue
        
        entry_price = close_prices[i]
        
        if longCondition.iloc[i] if hasattr(longCondition, 'iloc') else longCondition[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if shortCondition.iloc[i] if hasattr(shortCondition, 'iloc') else shortCondition[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries