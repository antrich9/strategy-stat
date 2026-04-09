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
    
    # Configuration defaults (matching Pine Script inputs)
    donchLength = 20
    minBodyPercentage = 70
    searchFactor = 1.3
    maType = 'SMA'
    maLength = 20
    maSource = df['close']
    maReaction = 1
    maTypeB = 'SMA'
    maLengthB = 8
    maSourceB = df['close']
    maReactionB = 1
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    
    n = len(df)
    
    # Donchian Channel
    highestHigh = df['high'].rolling(window=donchLength).max()
    lowestLow = df['low'].rolling(window=donchLength).min()
    
    # Wilder ATR (14)
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Elephant Candle Conditions
    isGreenElephantCandle = df['close'] > df['open']
    isRedElephantCandle = df['close'] < df['open']
    
    body = np.abs(df['open'] - df['close'])
    candle_range = np.abs(df['high'] - df['low'])
    body_pct = body * 100 / candle_range.replace(0, np.nan)
    
    isGreenElephantCandleValid = isGreenElephantCandle & (body_pct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_pct >= minBodyPercentage)
    
    atr_prev = atr.shift(1)
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr_prev * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr_prev * searchFactor)
    
    # MA variant functions
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        result = pd.Series(0.0, index=src.index)
        prev_val = 0.0
        prev_prev_val = 0.0
        prev_src = 0.0
        for i in range(len(src)):
            if i == 0:
                result.iloc[i] = c1 * (src.iloc[i] + prev_src) / 2 + c2 * prev_val + c3 * prev_prev_val
            else:
                result.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * result.iloc[i-1] + c3 * (result.iloc[i-2] if i >= 2 else prev_prev_val)
        return result
    
    def variant_smoothed(src, length):
        result = pd.Series(0.0, index=src.index)
        prev_val = src.iloc[0] if not pd.isna(src.iloc[0]) else 0.0
        for i in range(len(src)):
            if i == 0 or pd.isna(result.iloc[i-1]):
                result.iloc[i] = src.rolling(length).mean().iloc[i] if not pd.isna(src.rolling(length).mean().iloc[i]) else 0.0
            else:
                result.iloc[i] = (result.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return result
    
    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2
    
    def variant_doubleema(src, length):
        v2 = src.ewm(span=length, adjust=False).mean()
        return 2 * v2 - v2.ewm(span=length, adjust=False).mean()
    
    def variant_tripleema(src, length):
        v2 = src.ewm(span=length, adjust=False).mean()
        v_temp = v2.ewm(span=length, adjust=False).mean()
        return 3 * (v2 - v_temp) + v_temp
    
    def variant_hull(src, length):
        half_length = int(length / 2)
        sqrt_length = int(np.round(np.sqrt(length)))
        wma1 = src.rolling(half_length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)) if len(x) == half_length else np.nan, raw=False)
        wma2 = src.rolling(length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)) if len(x) == length else np.nan, raw=False)
        hull = 2 * wma1 - wma2
        return hull.rolling(sqrt_length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)) if len(x) == sqrt_length else np.nan, raw=False)
    
    def variant(src, length, ma_type):
        if ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'WMA':
            return src.rolling(length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)) if len(x) == length else np.nan, raw=False)
        elif ma_type == 'VWMA':
            return src.rolling(length).mean()
        elif ma_type == 'SMMA':
            return variant_smoothed(src, length)
        elif ma_type == 'DEMA':
            return variant_doubleema(src, length)
        elif ma_type == 'TEMA':
            return variant_tripleema(src, length)
        elif ma_type == 'HullMA':
            return variant_hull(src, length)
        elif ma_type == 'SSMA':
            return variant_supersmoother(src, length)
        elif ma_type == 'ZEMA':
            return variant_zerolagema(src, length)
        elif ma_type == 'TMA':
            return src.rolling(length).mean().rolling(length).mean()
        else:  # SMA
            return src.rolling(length).mean()
    
    # Moving Averages
    slowMA = variant(maSource, maLength, maType)
    fastMA = variant(maSourceB, maLengthB, maTypeB)
    
    # Trend direction
    slowMATrend = pd.Series(0, index=df.index)
    for i in range(1, n):
        if i < maReaction:
            slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if not pd.isna(slowMATrend.iloc[i-1]) else 0
        else:
            if slowMA.iloc[i] > slowMA.iloc[i-maReaction]:
                slowMATrend.iloc[i] = 1
            elif slowMA.iloc[i] < slowMA.iloc[i-maReaction]:
                slowMATrend.iloc[i] = -1
            else:
                slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if not pd.isna(slowMATrend.iloc[i-1]) else 0
    
    fastMATrend = pd.Series(0, index=df.index)
    for i in range(1, n):
        if i < maReactionB:
            fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if not pd.isna(fastMATrend.iloc[i-1]) else 0
        else:
            if fastMA.iloc[i] > fastMA.iloc[i-maReactionB]:
                fastMATrend.iloc[i] = 1
            elif fastMA.iloc[i] < fastMA.iloc[i-maReactionB]:
                fastMATrend.iloc[i] = -1
            else:
                fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if not pd.isna(fastMATrend.iloc[i-1]) else 0
    
    # Trend Conditions - Bullish
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (df['close'] > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (df['close'] > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (df['close'] > slowMA) & (df['close'] > fastMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (df['close'] > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (df['close'] > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (df['close'] > slowMA) & (df['close'] > fastMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'
    
    # Trend Conditions - Bearish
    isPriceBelowFastMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA') & (df['close'] < fastMA)
    isPriceBelowSlowMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA') & (df['close'] < slowMA)
    isPriceBelowBothMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (df['close'] < slowMA) & (df['close'] < fastMA)
    isPriceBelowSlowMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (df['close'] < slowMA) & (slowMATrend < 0)
    isPriceBelowFastMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (df['close'] < fastMA) & (fastMATrend < 0)
    isPriceBelowBothMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (df['close'] < slowMA) & (df['close'] < fastMA) & (slowMATrend < 0) & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'
    
    # Final Elephant Candle Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & (df['close'] > slowMA) & (df['close'] > fastMA) & activateGreenElephantCandles
    
    finalRedElephantCandle = isRedElephantCandleStrong & (
        isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA | 
        isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend | 
        isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish | 
        isFastMATrendBearish | isBothMATrendBearish | noBearishCondition
    ) & activateRedElephantCandles
    
    # Result conditions
    resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') | 
        ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong)
    )
    
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') | 
        ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong)
    )
    
    # Build entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        if i < donchLength:
            continue
        if pd.isna(highestHigh.iloc[i]) or pd.isna(lowestLow.iloc[i]):
            continue
        if pd.isna(atr.iloc[i]):
            continue
        if pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue
            
        if resultGreenElephantCandle.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            
        if resultRedElephantCandle.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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