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
    # Parameters (matching Pine Script defaults)
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maLength = 20
    maSource_col = 'close'
    maReaction = 1
    maLengthB = 8
    maSourceB_col = 'close'
    maReactionB = 1
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    smooth = 1
    length = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    blen = 20

    open_col = df['open']
    high = df['high']
    low = df['low']
    close_col = df['close']

    # Donchian Channel
    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()

    # ATR (Wilder)
    tr1 = high - low
    tr2 = (high - close_col.shift(1)).abs()
    tr3 = (low - close_col.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/previousBarsCount, adjust=False).mean()

    # Elephant Candle Conditions
    isGreenElephantCandle = close_col > open_col
    isRedElephantCandle = close_col < open_col

    body = (close_col - open_col).abs()
    range_ = high - low
    body_pct = body / range_ * 100

    isGreenElephantCandleValid = isGreenElephantCandle & (body_pct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_pct >= minBodyPercentage)

    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr.shift(1) * searchFactor)

    # Moving Averages (EMA for simplicity - most common default)
    slowMA = close_col.ewm(span=maLength, adjust=False).mean()
    fastMA = close_col.ewm(span=maLengthB, adjust=False).mean()

    # Trend Direction
    def rolling_sign_diff(series, n):
        return (series > series.shift(n)).astype(int) - (series < series.shift(n)).astype(int)

    slowMATrend = (rolling_sign_diff(slowMA, maReaction) > 0).astype(int) - (rolling_sign_diff(slowMA, maReaction) < 0).astype(int)
    slowMATrend = slowMATrend.replace(0, np.nan).ffill().fillna(0)

    fastMATrend = (rolling_sign_diff(fastMA, maReactionB) > 0).astype(int) - (rolling_sign_diff(fastMA, maReactionB) < 0).astype(int)
    fastMATrend = fastMATrend.replace(0, np.nan).ffill().fillna(0)

    # Bullish Trend Conditions
    isPriceAboveFastMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close_col > fastMA)
    isPriceAboveSlowMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close_col > slowMA)
    isPriceAboveBothMA = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close_col > slowMA) & (close_col > fastMA)
    isPriceAboveSlowMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close_col > slowMA) & (slowMATrend > 0)
    isPriceAboveFastMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close_col > fastMA) & (fastMATrend > 0)
    isPriceAboveBothMAWithBullishTrend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close_col > slowMA) & (close_col > fastMA) & (slowMATrend > 0) & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'

    # Bearish Trend Conditions
    isPriceBelowFastMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA') & (close_col < fastMA)
    isPriceBelowSlowMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA') & (close_col < slowMA)
    isPriceBelowBothMA = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close_col < slowMA) & (close_col < fastMA)
    isPriceBelowSlowMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close_col < slowMA) & (slowMATrend < 0)
    isPriceBelowFastMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close_col < fastMA) & (fastMATrend < 0)
    isPriceBelowBothMAWithBearishTrend = (bearishTrendCondition == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close_col < slowMA) & (close_col < fastMA) & (slowMATrend < 0) & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'

    # Final Elephant Candle Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & (isPriceAboveFastMA | isPriceAboveSlowMA | isPriceAboveBothMA | isPriceAboveSlowMAWithBullishTrend | isPriceAboveFastMAWithBullishTrend | isPriceAboveBothMAWithBullishTrend | isSlowMATrendBullish | isFastMATrendBullish | isBothMATrendBullish | noBullishCondition)
    finalRedElephantCandle = isRedElephantCandleStrong & (isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA | isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend | isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish | isFastMATrendBearish | isBothMATrendBearish | noBearishCondition)

    # Result conditions
    resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & ((filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong))
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & ((filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong))

    # Trendilo Indicator (cdir)
    src = close_col
    pch = src.pct_change(smooth) * 100

    # ALMA approximation
    def alma_approx(series, n, offset, sigma):
        k = np.arange(n)
        w = np.exp(-((k - offset * (n - 1)) ** 2) / (2 * sigma ** 2 * (n / 2) ** 2))
        w = w / w.sum()
        return series.rolling(n).apply(lambda x: np.dot(x, w), raw=True)

    avpch = alma_approx(pch, length, offset, sigma)
    blength = blen
    rms = bmult * np.sqrt(avpch.rolling(blength).apply(lambda x: (x ** 2).sum() / blength, raw=True))
    cdir = pd.Series(np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0)), index=df.index)

    entries = []
    trade_num = 1

    # Entry Logic: resultGreenElephantCandle and cdir > 0
    for i in range(1, len(df)):
        if pd.isna(highestHigh.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]) or pd.isna(avpch.iloc[i]) or pd.isna(rms.iloc[i]):
            continue

        if resultGreenElephantCandle.iloc[i] and cdir.iloc[i] > 0:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close_col.iloc[i]

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

    return entries