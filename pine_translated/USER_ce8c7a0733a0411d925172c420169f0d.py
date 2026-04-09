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
    # Configuration inputs (matching Pine Script defaults)
    donchLength = 20
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    filterType = 'CON FILTRADO DE TENDENCIA'
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    paintElephantCandles = True
    showElephantLabel = True
    movingAverageConfig = True
    showSlowMA = True
    maType = 'SMA'
    maLength = 20
    maSource = 'close'
    maReaction = 1
    showFastMA = True
    maTypeB = 'SMA'
    maLengthB = 8
    maSourceB = 'close'
    maReactionB = 1
    trendConfig = True
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'

    # Helper functions for MA variants
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        result = pd.Series(np.zeros(len(src)), index=src.index)
        for i in range(2, len(src)):
            prev_result = result.iloc[i-1] if i-1 >= 0 else 0
            prev_src = src.iloc[i-1] if i-1 >= 0 else 0
            prev_prev_result = result.iloc[i-2] if i-2 >= 0 else 0
            result.iloc[i] = c1 * (src.iloc[i] + prev_src) / 2 + c2 * prev_result + c3 * prev_prev_result
        return result

    def variant_smoothed(src, length):
        result = pd.Series(np.zeros(len(src)), index=src.index)
        prev_result = None
        for i in range(len(src)):
            if pd.isna(prev_result):
                result.iloc[i] = src.iloc[:i+1].mean()
            else:
                result.iloc[i] = (prev_result * (length - 1) + src.iloc[i]) / length
            prev_result = result.iloc[i]
        return result

    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2

    def variant_doubleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema1.ewm(span=length, adjust=False).mean()

    def variant_tripleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * (ema1 - ema2) + ema3

    def variant(src, ma_type, length):
        if ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        elif ma_type == 'VWMA':
            return pd.Series(index=src.index)
        elif ma_type == 'SMMA':
            return variant_smoothed(src, length)
        elif ma_type == 'DEMA':
            return variant_doubleema(src, length)
        elif ma_type == 'TEMA':
            return variant_tripleema(src, length)
        elif ma_type == 'HullMA':
            half_length = int(length / 2)
            wma1 = src.rolling(half_length).apply(lambda x: np.dot(x, np.arange(1, half_length + 1)) / np.arange(1, half_length + 1).sum(), raw=True)
            wma2 = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(), raw=True)
            sqrt_len = int(np.round(np.sqrt(length)))
            return (2 * wma1 - wma2).rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len + 1)) / np.arange(1, sqrt_len + 1).sum(), raw=True)
        elif ma_type == 'SSMA':
            return variant_supersmoother(src, length)
        elif ma_type == 'ZEMA':
            return variant_zerolagema(src, length)
        elif ma_type == 'TMA':
            return src.rolling(length).mean().rolling(length).mean()
        else:
            return src.rolling(length).mean()

    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(high, low, close, length):
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']

    atr = wilder_atr(high_series, low_series, close_series, 14)

    # Elephant Candle Conditions
    isGreenElephantCandle = close_series > open_series
    isRedElephantCandle = close_series < open_series

    body = (close_series - open_series).abs()
    range_val = high_series - low_series

    isGreenElephantCandleValid = isGreenElephantCandle & (body * 100 / range_val >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body * 100 / range_val >= minBodyPercentage)

    atr_prev = atr.shift(1)
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr_prev * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr_prev * searchFactor)

    # Moving Averages
    if maSource == 'close':
        source_slow = close_series
    else:
        source_slow = close_series

    if maSourceB == 'close':
        source_fast = close_series
    else:
        source_fast = close_series

    slowMA = variant(source_slow, maType, maLength)
    fastMA = variant(source_fast, maTypeB, maLengthB)

    # Trend Direction
    slowMATrend = pd.Series(np.zeros(len(df)), index=df.index)
    slowMATrend.iloc[0] = 0
    for i in range(1, len(df)):
        if pd.isna(slowMA.iloc[i]):
            slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if not pd.isna(slowMATrend.iloc[i-1]) else 0
        else:
            rising = all(slowMA.iloc[max(1, i-maReaction+1):i+1].diff().dropna() > 0) if i >= maReaction else False
            falling = all(slowMA.iloc[max(1, i-maReaction+1):i+1].diff().dropna() < 0) if i >= maReaction else False
            if rising:
                slowMATrend.iloc[i] = 1
            elif falling:
                slowMATrend.iloc[i] = -1
            else:
                slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if not pd.isna(slowMATrend.iloc[i-1]) else 0

    fastMATrend = pd.Series(np.zeros(len(df)), index=df.index)
    fastMATrend.iloc[0] = 0
    for i in range(1, len(df)):
        if pd.isna(fastMA.iloc[i]):
            fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if not pd.isna(fastMATrend.iloc[i-1]) else 0
        else:
            rising = all(fastMA.iloc[max(1, i-maReactionB+1):i+1].diff().dropna() > 0) if i >= maReactionB else False
            falling = all(fastMA.iloc[max(1, i-maReactionB+1):i+1].diff().dropna() < 0) if i >= maReactionB else False
            if rising:
                fastMATrend.iloc[i] = 1
            elif falling:
                fastMATrend.iloc[i] = -1
            else:
                fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if not pd.isna(fastMATrend.iloc[i-1]) else 0

    # Trend Conditions
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close_series > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close_series > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close_series > slowMA) & (close_series > fastMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close_series > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close_series > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close_series > slowMA) & (close_series > fastMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'

    isPriceBelowFastMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA') & (close_series < fastMA)
    isPriceBelowSlowMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA') & (close_series < slowMA)
    isPriceBelowBothMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close_series < slowMA) & (close_series < fastMA)
    isPriceBelowSlowMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close_series < slowMA) & (slowMATrend < 0)
    isPriceBelowFastMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close_series < fastMA) & (fastMATrend < 0)
    isPriceBelowBothMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close_series < slowMA) & (close_series < fastMA) & (slowMATrend < 0) & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'

    # Final Elephant Candle Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & (isPriceAboveFastMA | isPriceAboveSlowMA | isPriceAboveBothMA | isPriceAboveSlowMAWithBullishTrend | isPriceAboveFastMAWithBullishTrend | isPriceAboveBothMAWithBullishTrend | isSlowMATrendBullish | isFastMATrendBullish | isBothMATrendBullish | noBullishCondition)
    finalRedElephantCandle = isRedElephantCandleStrong & (isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA | isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend | isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish | isFastMATrendBearish | isBothMATrendBearish | noBearishCondition)

    # Result
    resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & ((filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong))
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & ((filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong))

    # Build entries list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        if resultGreenElephantCandle.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
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
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries