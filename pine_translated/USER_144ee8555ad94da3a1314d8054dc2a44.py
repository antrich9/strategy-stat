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
    entries = []
    trade_num = 0

    # Parameters (from Pine Script inputs)
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
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    filterType = "CON FILTRADO DE TENDENCIA"
    bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    bearishTrendCondition = "DIRECCION MEDIA RAPIDA BAJISTA"

    # Trendilo parameters
    src = df['close'].copy()
    smooth = 1
    length = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    blen = 20

    # Calculate Wilder ATR
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/previousBarsCount, adjust=False).mean()

    # Elephant Candle base conditions
    isGreenElephantCandle = close > df['open']
    isRedElephantCandle = close < df['open']

    body = abs(df['open'] - close)
    range_ = abs(high - low)
    body_pct = body / range_ * 100

    isGreenElephantCandleValid = isGreenElephantCandle & (body_pct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_pct >= minBodyPercentage)

    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr.shift(1) * searchFactor)

    # SuperSmoother variant
    def variant_supersmoother(s, length):
        result = pd.Series(index=s.index, dtype=float)
        a1 = np.exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        v9 = 0.0
        prev_v9_1 = 0.0
        prev_v9_2 = 0.0
        prev_s_1 = 0.0
        for i in range(len(s)):
            if i == 0:
                v9 = c1 * (s.iloc[i] + prev_s_1) / 2
            else:
                v9 = c1 * (s.iloc[i] + s.iloc[i-1]) / 2 + c2 * prev_v9_1 + c3 * prev_v9_2
            result.iloc[i] = v9
            prev_v9_2 = prev_v9_1
            prev_v9_1 = v9
            prev_s_1 = s.iloc[i]
        return result

    # Smoothed variant
    def variant_smoothed(s, length):
        result = pd.Series(index=s.index, dtype=float)
        v5_prev = np.nan
        for i in range(len(s)):
            if pd.isna(v5_prev):
                v5 = s.iloc[:i+1].mean()
            else:
                v5 = (v5_prev * (length - 1) + s.iloc[i]) / length
            result.iloc[i] = v5
            v5_prev = v5
        return result

    # Zero lag EMA variant
    def variant_zerolagema(s, length):
        ema1 = s.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2

    # Double EMA variant
    def variant_doubleema(s, length):
        v2 = s.ewm(span=length, adjust=False).mean()
        return 2 * v2 - v2.ewm(span=length, adjust=False).mean()

    # Triple EMA variant
    def variant_tripleema(s, length):
        v2 = s.ewm(span=length, adjust=False).mean()
        v2_ema = v2.ewm(span=length, adjust=False).mean()
        return 3 * (v2 - v2_ema) + v2_ema

    # Hull MA
    def variant_hullma(s, length):
        half = int(length / 2)
        sqrt_len = int(np.round(np.sqrt(length)))
        wma1 = s.rolling(half).apply(lambda x: np.dot(x, np.arange(half)) / np.sum(np.arange(half)), raw=True)
        wma2 = s.rolling(length).apply(lambda x: np.dot(x, np.arange(length)) / np.sum(np.arange(length)), raw=True)
        return (2 * wma1 - wma2).rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(sqrt_len)) / np.sum(np.arange(sqrt_len)), raw=True)

    # TMA
    def variant_tma(s, length):
        return s.rolling(length).mean().rolling(length).mean()

    # Generic variant function
    def variant(ma_type, source, length):
        if ma_type == 'EMA':
            return source.ewm(span=length, adjust=False).mean()
        elif ma_type == 'WMA':
            return source.rolling(length).apply(lambda x: np.dot(x, np.arange(length)) / np.sum(np.arange(length)), raw=True)
        elif ma_type == 'VWMA':
            return (source * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
        elif ma_type == 'SMMA':
            return variant_smoothed(source, length)
        elif ma_type == 'DEMA':
            return variant_doubleema(source, length)
        elif ma_type == 'TEMA':
            return variant_tripleema(source, length)
        elif ma_type == 'HullMA':
            return variant_hullma(source, length)
        elif ma_type == 'SSMA':
            return variant_supersmoother(source, length)
        elif ma_type == 'ZEMA':
            return variant_zerolagema(source, length)
        elif ma_type == 'TMA':
            return variant_tma(source, length)
        else:  # SMA
            return source.rolling(length).mean()

    # Moving Averages
    slowMA = variant(maType, close, maLength)
    fastMA = variant(maTypeB, close, maLengthB)

    # Trend direction calculation
    slowMATrend = pd.Series(0, index=close.index)
    for i in range(len(close)):
        if i < maReaction:
            continue
        if slowMA.iloc[i] > slowMA.iloc[i - maReaction]:
            slowMATrend.iloc[i] = 1
        elif slowMA.iloc[i] < slowMA.iloc[i - maReaction]:
            slowMATrend.iloc[i] = -1
        else:
            slowMATrend.iloc[i] = slowMATrend.iloc[i - 1]

    fastMATrend = pd.Series(0, index=close.index)
    for i in range(len(close)):
        if i < maReactionB:
            continue
        if fastMA.iloc[i] > fastMA.iloc[i - maReactionB]:
            fastMATrend.iloc[i] = 1
        elif fastMA.iloc[i] < fastMA.iloc[i - maReactionB]:
            fastMATrend.iloc[i] = -1
        else:
            fastMATrend.iloc[i] = fastMATrend.iloc[i - 1]

    # Trend conditions - bullish
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

    # Trend conditions - bearish
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

    # Final Elephant Candle conditions
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
        (filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong)
    )
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & (
        (filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong)
    )

    # Trendilo Indicator
    pch = close.diff(smooth) / close * 100

    # ALMA implementation
    def alma(series, length, offset, sigma):
        w = np.zeros(length)
        m = offset * (length - 1)
        s = length / sigma
        for i in range(length):
            w[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
        w_sum = w.sum()
        if w_sum == 0:
            return pd.Series(np.nan, index=series.index)
        w = w / w_sum
        result = series.rolling(length).apply(lambda x: np.sum(w * x), raw=True)
        return result

    avpch = alma(pch, length, offset, sigma)
    avpch_filled = avpch.fillna(0)
    blength = blen

    rms = bmult * np.sqrt((avpch_filled ** 2).rolling(blength).sum() / blength)
    cdir = pd.Series(0, index=close.index)
    cdir[avpch > rms] = 1
    cdir[avpch < -rms] = -1

    # Donchian Channel for stop loss (used in entry but not needed for entries list)
    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()

    # Generate entries
    for i in range(len(df)):
        if i < donchLength:
            continue
        if pd.isna(highestHigh.iloc[i]) or pd.isna(lowestLow.iloc[i]):
            continue
        if pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue
        if pd.isna(atr.iloc[i]):
            continue
        if pd.isna(cdir.iloc[i]) or cdir.iloc[i] == 0:
            continue

        entry_price_guess = close.iloc[i]

        # Long entry
        if resultGreenElephantCandle.iloc[i] and cdir.iloc[i] > 0:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })

        # Short entry
        if resultRedElephantCandle.iloc[i] and cdir.iloc[i] < 0:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })

    return entries