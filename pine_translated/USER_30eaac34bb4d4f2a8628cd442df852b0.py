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
    
    # Strategy parameters (from Pine Script inputs)
    donchLength = 20
    operationMode = True
    filterType = "CON FILTRADO DE TENDENCIA"
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    
    # Moving Average Parameters (Slow)
    maType = "SMA"
    maLength = 20
    maSource = df['close']
    maReaction = 1
    
    # Moving Average Parameters (Fast)
    maTypeB = "SMA"
    maLengthB = 8
    maSourceB = df['close']
    maReactionB = 1
    
    # Trend Configuration
    bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    bearishTrendCondition = "DIRECCION MEDIA RAPIDA BAJISTA"
    
    # TDFI Parameters
    trendPeriod = 20
    maType1 = "JMA"
    maPeriod = 8
    triggerUp = 0.05
    triggerDown = -0.05
    
    # Prepare data
    close = df['close']
    open_prices = df['open']
    high = df['high']
    low = df['low']
    
    # Donchian Channel
    highestHigh = high.rolling(window=donchLength).max()
    lowestLow = low.rolling(window=donchLength).min()
    
    # ATR Calculation (Wilder)
    tr = pd.concat([high - low, 
                    (high - close.shift(1)).abs(), 
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/previousBarsCount, adjust=False).mean()
    
    # Elephant Candle Conditions
    body = (close - open_prices).abs()
    full_range = (high - low).replace(0, np.nan)
    bodyPercentage = body / full_range * 100
    
    isGreenElephantCandle = close > open_prices
    isRedElephantCandle = close < open_prices
    
    isGreenElephantCandleValid = isGreenElephantCandle & (bodyPercentage >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (bodyPercentage >= minBodyPercentage)
    
    # Strong elephant candles require body >= atr[1] * searchFactor
    atr_prev = atr.shift(1)
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr_prev * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr_prev * searchFactor)
    
    # Variant function for MA calculations
    def variant_sma(src, length):
        return src.rolling(length).mean()
    
    def variant_ema(src, length):
        return src.ewm(span=length, adjust=False).mean()
    
    def variant_wma(src, length):
        weights = np.arange(1, length + 1)
        return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
    
    def variant_vwma(src, length):
        return (df['close'] * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
    
    def variant_smoothed(src, length):
        result = pd.Series(index=src.index, dtype=float)
        result.iloc[0] = src.iloc[0]
        for i in range(1, len(src)):
            if pd.isna(result.iloc[i-1]):
                result.iloc[i] = src.iloc[:i+1].mean()
            else:
                result.iloc[i] = (result.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return result
    
    def variant_doubleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema2
    
    def variant_tripleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * (ema1 - ema2) + ema3
    
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        result = pd.Series(index=src.index, dtype=float)
        result.iloc[0] = src.iloc[0]
        prev1 = src.iloc[0]
        prev2 = src.iloc[0]
        
        for i in range(1, len(src)):
            val = c1 * (src.iloc[i] + prev1) / 2 + c2 * prev1 + c3 * prev2
            result.iloc[i] = val
            prev2 = prev1
            prev1 = val
        return result
    
    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2
    
    def variant_hullma(src, length):
        half_length = int(length / 2)
        sqrt_length = int(np.round(np.sqrt(length)))
        wma_half = variant_wma(src, half_length)
        wma_full = variant_wma(src, length)
        return variant_wma(2 * wma_half - wma_full, sqrt_length)
    
    def variant(type_str, src, length):
        if type_str == 'EMA':
            return variant_ema(src, length)
        elif type_str == 'WMA':
            return variant_wma(src, length)
        elif type_str == 'VWMA':
            return variant_vwma(src, length)
        elif type_str == 'SMMA':
            return variant_smoothed(src, length)
        elif type_str == 'DEMA':
            return variant_doubleema(src, length)
        elif type_str == 'TEMA':
            return variant_tripleema(src, length)
        elif type_str == 'HullMA':
            return variant_hullma(src, length)
        elif type_str == 'SSMA':
            return variant_supersmoother(src, length)
        elif type_str == 'ZEMA':
            return variant_zerolagema(src, length)
        elif type_str == 'TMA':
            return variant_sma(src, length).rolling(length).mean()
        else:
            return variant_sma(src, length)
    
    # Calculate Moving Averages
    slowMA = variant(maType, maSource, maLength)
    fastMA = variant(maTypeB, maSourceB, maLengthB)
    
    # Trend Direction Calculation (ta.rising/falling)
    def calc_trend(ma, reaction):
        trend = pd.Series(0, index=ma.index)
        for i in range(reaction, len(ma)):
            if ma.iloc[i] > ma.iloc[i-reaction]:
                trend.iloc[i] = 1
            elif ma.iloc[i] < ma.iloc[i-reaction]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1] if pd.notna(trend.iloc[i-1]) else 0
        return trend
    
    slowMATrend = calc_trend(slowMA, maReaction)
    fastMATrend = calc_trend(fastMA, maReactionB)
    
    # Build trend condition masks
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'
    
    # Bullish conditions
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close > slowMA) & (close > fastMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close > slowMA) & (close > fastMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    
    # Combine bullish conditions
    bullTrendConditionMet = (
        isPriceAboveFastMA | isPriceAboveSlowMA | isPriceAboveBothMA |
        isPriceAboveSlowMAWithBullishTrend | isPriceAboveFastMAWithBullishTrend |
        isPriceAboveBothMAWithBullishTrend | isSlowMATrendBullish |
        isFastMATrendBullish | isBothMATrendBullish | noBullishCondition
    )
    
    # Bearish conditions
    isPriceBelowFastMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA') & (close < fastMA)
    isPriceBelowSlowMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA') & (close < slowMA)
    isPriceBelowBothMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close < slowMA) & (close < fastMA)
    isPriceBelowSlowMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close < slowMA) & (slowMATrend < 0)
    isPriceBelowFastMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close < fastMA) & (fastMATrend < 0)
    isPriceBelowBothMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close < slowMA) & (close < fastMA) & (slowMATrend < 0) & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    
    # Combine bearish conditions
    bearTrendConditionMet = (
        isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA |
        isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend |
        isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish |
        isFastMATrendBearish | isBothMATrendBearish | noBearishCondition
    )
    
    # TDFI Calculation (simplified for long entries)
    # Using CRYPTOCAP:TOTAL equivalent - approximate with close price
    symbolData = close  # Using main chart data as proxy for CRYPTOCAP:TOTAL
    
    mma = symbolData.ewm(span=trendPeriod, adjust=False).mean()
    smma = mma.ewm(span=trendPeriod, adjust=False).mean()
    
    impetmma = mma.diff()
    impetsmma = smma.diff()
    
    min_tick = 0.01  # Approximate minimum tick
    divma = (mma - smma).abs() / min_tick
    averimpet = ((impetmma + impetsmma) / 2) / (2 * min_tick)
    
    tdfRaw = divma * (averimpet ** 3)
    tdfAbsRaw = tdfRaw.abs()
    
    # Rolling max of tdfAbsRaw over 3*trendPeriod-1 bars
    roll_window = 3 * trendPeriod - 1
    tdfAbsRaw_smooth = tdfAbsRaw.rolling(window=roll_window, min_periods=1).max()
    
    # Avoid division by zero
    tdfAbsRaw_smooth = tdfAbsRaw_smooth.replace(0, np.nan)
    ratio = tdfRaw / tdfAbsRaw_smooth
    
    # Apply smoothing based on maType1
    if maType1 == 'JMA':
        # JMA approximation using triple EMA with smoothing
        smooth = ratio.ewm(span=maPeriod, adjust=False).mean()
        smooth = smooth.ewm(span=maPeriod, adjust=False).mean()
        smooth = smooth.ewm(span=maPeriod, adjust=False).mean()
    elif maType1 == 'EMA':
        smooth = ratio.ewm(span=maPeriod, adjust=False).mean()
    elif maType1 == 'DEMA':
        ema1 = ratio.ewm(span=maPeriod, adjust=False).mean()
        ema2 = ema1.ewm(span=maPeriod, adjust=False).mean()
        smooth = 2 * ema1 - ema2
    elif maType1 == 'HMA':
        half_len = int(maPeriod / 2)
        sqrt_len = int(np.round(np.sqrt(maPeriod)))
        wma1 = ratio.rolling(half_len).mean()
        wma2 = ratio.rolling(maPeriod).mean()
        hull = (2 * wma1 - wma2).rolling(sqrt_len).mean()
        smooth = hull
    elif maType1 == 'SMA':
        smooth = ratio.rolling(maPeriod).mean()
    elif maType1 == 'SMMA':
        smooth = ratio.ewm(alpha=1/maPeriod, adjust=False).mean()
    else:
        smooth = ratio.ewm(span=maPeriod, adjust=False).mean()
    
    # TDFI conditions
    tdfiBullish = smooth > triggerUp
    tdfiBearish = smooth < triggerDown
    
    # Entry Conditions
    # Long: price crosses above highestHigh + bullish trend + elephant candle
    longCondition = (
        operationMode &
        (close > highestHigh) &
        (close.shift(1) <= highestHigh.shift(1)) &
        bullTrendConditionMet &
        isGreenElephantCandleStrong &
        activateElephantCandles &
        activateGreenElephantCandles &
        tdfiBullish
    )
    
    # Short: price crosses below lowestLow + bearish trend + red elephant candle
    shortCondition = (
        operationMode &
        (close < lowestLow) &
        (close.shift(1) >= lowestLow.shift(1)) &
        bearTrendConditionMet &
        isRedElephantCandleStrong &
        activateElephantCandles &
        activateRedElephantCandles &
        tdfiBearish
    )
    
    # Filter with filterType
    if filterType == "CON FILTRADO DE TENDENCIA":
        longCondition = longCondition & (tdfiBullish | noBullishCondition)
        shortCondition = shortCondition & (tdfiBearish | noBearishCondition)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(highestHigh.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue
        if pd.isna(smooth.iloc[i]):
            continue
            
        if longCondition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            
        elif shortCondition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries