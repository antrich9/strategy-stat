import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    donchLength = 20
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maType = "SMA"
    maLength = 20
    maSource = df['close']
    maReaction = 1
    maTypeB = "SMA"
    maLengthB = 8
    maSourceB = df['close']
    maReactionB = 1
    filterType = "CON FILTRADO DE TENDENCIA"
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    bearishTrendCondition = "DIRECCION MEDIA RAPIDA BAJISTA"

    def calc_ema(src, length):
        return src.ewm(span=length, adjust=False).mean()

    def calc_sma(src, length):
        return src.rolling(length).mean()

    def calc_wma(src, length):
        weights = np.arange(1, length + 1)
        return src.rolling(length).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)

    def calc_vwma(src, length, volume):
        return (src * volume).rolling(length).sum() / volume.rolling(length).sum()

    def calc_smma(src, length):
        smma = pd.Series(index=src.index, dtype=float)
        smma.iloc[length-1] = src.iloc[:length].mean()
        for i in range(length, len(src)):
            smma.iloc[i] = (smma.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return smma

    def calc_dema(src, length):
        ema1 = calc_ema(src, length)
        ema2 = calc_ema(ema1, length)
        return 2 * ema1 - ema2

    def calc_tema(src, length):
        ema1 = calc_ema(src, length)
        ema2 = calc_ema(ema1, length)
        ema3 = calc_ema(ema2, length)
        return 3 * ema1 - 3 * ema2 + ema3

    def calc_hullma(src, length):
        half_len = int(length / 2)
        sqrt_len = int(np.round(np.sqrt(length)))
        wma1 = calc_wma(src, half_len)
        wma2 = calc_wma(src, length)
        hull = calc_wma(2 * wma1 - wma2, sqrt_len)
        return hull

    def calc_ssma(src, length):
        ssma = pd.Series(0.0, index=src.index)
        a1 = np.exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        for i in range(2, len(src)):
            if i < 2:
                continue
            prev_src = src.iloc[i-1] if i > 0 else src.iloc[0]
            prev_ssma1 = ssma.iloc[i-1] if i > 0 else 0
            prev_ssma2 = ssma.iloc[i-2] if i > 1 else prev_ssma1
            ssma.iloc[i] = c1 * (src.iloc[i] + prev_src) / 2 + c2 * prev_ssma1 + c3 * prev_ssma2
        return ssma

    def calc_zema(src, length):
        ema1 = calc_ema(src, length)
        ema2 = calc_ema(ema1, length)
        return 2 * ema1 - ema2

    def calc_tma(src, length):
        sma1 = calc_sma(src, length)
        return calc_sma(sma1, length)

    def variant_ma(type_name, src, length):
        if type_name == 'EMA':
            return calc_ema(src, length)
        elif type_name == 'WMA':
            return calc_wma(src, length)
        elif type_name == 'VWMA':
            return calc_vwma(src, length, df['volume'])
        elif type_name == 'SMMA':
            return calc_smma(src, length)
        elif type_name == 'DEMA':
            return calc_dema(src, length)
        elif type_name == 'TEMA':
            return calc_tema(src, length)
        elif type_name == 'HullMA':
            return calc_hullma(src, length)
        elif type_name == 'SSMA':
            return calc_ssma(src, length)
        elif type_name == 'ZEMA':
            return calc_zema(src, length)
        elif type_name == 'TMA':
            return calc_tma(src, length)
        else:
            return calc_sma(src, length)

    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0/previousBarsCount, adjust=False).mean()

    isGreenElephantCandle = df['close'] > df['open']
    isRedElephantCandle = df['close'] < df['open']
    body = np.abs(df['open'] - df['close'])
    range_hl = df['high'] - df['low']
    body_pct = body * 100 / range_hl
    isGreenElephantCandleValid = isGreenElephantCandle & (body_pct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_pct >= minBodyPercentage)
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr.shift(1) * searchFactor)

    slowMA = variant_ma(maType, maSource, maLength)
    fastMA = variant_ma(maTypeB, maSourceB, maLengthB)

    def calc_trend(ma_series, reaction):
        trend = pd.Series(0, index=ma_series.index)
        for i in range(reaction, len(ma_series)):
            if ma_series.iloc[i] > ma_series.iloc[i - reaction]:
                trend.iloc[i] = 1
            elif ma_series.iloc[i] < ma_series.iloc[i - reaction]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i - 1]
        return trend

    slowMATrend = calc_trend(slowMA, maReaction)
    fastMATrend = calc_trend(fastMA, maReactionB)

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

    finalGreenElephantCandle = isGreenElephantCandleStrong & (isPriceAboveFastMA | isPriceAboveSlowMA | isPriceAboveBothMA | isPriceAboveSlowMAWithBullishTrend | isPriceAboveFastMAWithBullishTrend | isPriceAboveBothMAWithBullishTrend | isSlowMATrendBullish | isFastMATrendBullish | isBothMATrendBullish | noBullishCondition)
    finalRedElephantCandle = isRedElephantCandleStrong & (isPriceBelowFastMA | isPriceBelowSlowMA | isPriceBelowBothMA | isPriceBelowSlowMAWithBearishTrend | isPriceBelowFastMAWithBearishTrend | isPriceBelowBothMAWithBearishTrend | isSlowMATrendBearish | isFastMATrendBearish | isBothMATrendBearish | noBearishCondition)

    resultGreenElephantCandle = finalGreenElephantCandle & activateGreenElephantCandles & ((filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isGreenElephantCandleStrong))
    resultRedElephantCandle = finalRedElephantCandle & activateRedElephantCandles & ((filterType == 'CON FILTRADO DE TENDENCIA') | ((filterType == 'SIN FILTRADO DE TENDENCIA') & isRedElephantCandleStrong))

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i == 0:
            continue
        if pd.isna(atr.iloc[i]) or pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue
        if resultGreenElephantCandle.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif resultRedElephantCandle.iloc[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries