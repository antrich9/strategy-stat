import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # CMF Parameters
    cmfLength = 20
    cmfThreshold = 0.05
    cmfThresholdShort = -0.05
    
    # Donchian Parameters
    donchLength = 20
    
    # Elephant Candle Parameters
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    
    # MA Parameters
    maType = "SMA"
    maSource = df['close']
    maLength = 20
    maReaction = 1
    maTypeB = "SMA"
    maSourceB = df['close']
    maLengthB = 8
    maReactionB = 1
    
    # Trend Conditions
    bullishTrendCondition = "DIRECCION MEDIA RAPIDA ALCISTA"
    bearishTrendCondition = "DIRECCION MEDIA RAPIDA BAJISTA"
    
    # Filter Type
    filterType = "CON FILTRADO DE TENDENCIA"
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    
    # CMF Calculation
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfm * df['volume']
    ad = mfv.cumsum() - df['volume'].cumsum()
    cmf = ad.rolling(cmfLength).mean() / df['volume'].rolling(cmfLength).mean()
    
    # CMF Filter
    isCmfValidForLong = cmf > cmfThreshold
    isCmfValidForShort = cmf < cmfThresholdShort
    
    # Donchian Channel
    highestHigh = df['high'].rolling(donchLength).max()
    lowestLow = df['low'].rolling(donchLength).min()
    
    # ATR Calculation (Wilder)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/previousBarsCount, min_periods=previousBarsCount, adjust=False).mean()
    
    # Variant functions
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        v9 = pd.Series(0.0, index=src.index)
        for i in range(1, len(src)):
            prev = src.iloc[i-1] if i > 0 else 0
            v9.iloc[i] = c1 * (src.iloc[i] + prev) / 2 + c2 * v9.iloc[i-1] + c3 * (v9.iloc[i-2] if i > 1 else 0)
        return v9
    
    def variant_smoothed(src, length):
        result = pd.Series(0.0, index=src.index)
        result.iloc[length-1] = src.iloc[:length].mean()
        for i in range(length, len(src)):
            result.iloc[i] = (result.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return result
    
    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2
    
    def variant_doubleema(src, length):
        ema = src.ewm(span=length, adjust=False).mean()
        return 2 * ema - ema.ewm(span=length, adjust=False).mean()
    
    def variant_tripleema(src, length):
        ema = src.ewm(span=length, adjust=False).mean()
        return 3 * (ema - ema.ewm(span=length, adjust=False).mean()) + ema.ewm(span=length, adjust=False).mean().ewm(span=length, adjust=False).mean()
    
    def variant(type_str, src, length):
        if type_str == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif type_str == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        elif type_str == 'VWMA':
            return (src * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
        elif type_str == 'SMMA':
            return variant_smoothed(src, length)
        elif type_str == 'DEMA':
            return variant_doubleema(src, length)
        elif type_str == 'TEMA':
            return variant_tripleema(src, length)
        elif type_str == 'HullMA':
            wma1 = df['close'].rolling(int(length/2)).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / len(x)/(len(x)+1)*2, raw=True)
            wma2 = df['close'].rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / len(x)/(len(x)+1)*2, raw=True)
            return 2 * wma1 - wma2
        elif type_str == 'SSMA':
            return variant_supersmoother(src, length)
        elif type_str == 'ZEMA':
            return variant_zerolagema(src, length)
        elif type_str == 'TMA':
            return src.rolling(length).mean().rolling(length).mean()
        else:
            return src.rolling(length).mean()
    
    slowMA = variant(maType, maSource, maLength)
    fastMA = variant(maTypeB, maSourceB, maLengthB)
    
    # Trend Direction
    slowMATrend = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if pd.notna(slowMA.iloc[i]):
            rising = all(slowMA.iloc[max(0,i-maReaction+1):i+1].diff().dropna() > 0) if i >= maReaction else False
            falling = all(slowMA.iloc[max(0,i-maReaction+1):i+1].diff().dropna() < 0) if i >= maReaction else False
            if rising:
                slowMATrend.iloc[i] = 1
            elif falling:
                slowMATrend.iloc[i] = -1
            else:
                slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if i > 0 else 0
    
    fastMATrend = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if pd.notna(fastMA.iloc[i]):
            rising = all(fastMA.iloc[max(0,i-maReactionB+1):i+1].diff().dropna() > 0) if i >= maReactionB else False
            falling = all(fastMA.iloc[max(0,i-maReactionB+1):i+1].diff().dropna() < 0) if i >= maReactionB else False
            if rising:
                fastMATrend.iloc[i] = 1
            elif falling:
                fastMATrend.iloc[i] = -1
            else:
                fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if i > 0 else 0
    
    # Trend Conditions
    def check_bullish_condition(trend_cond, close, slow_ma, fast_ma, slow_trend, fast_trend):
        if trend_cond == 'NINGUNA CONDICION':
            return True
        elif trend_cond == 'PRECIO MAYOR A MEDIA RAPIDA':
            return close > fast_ma
        elif trend_cond == 'PRECIO MAYOR A MEDIA LENTA':
            return close > slow_ma
        elif trend_cond == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA':
            return (close > slow_ma) & (close > fast_ma)
        elif trend_cond == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA':
            return (close > slow_ma) & (slow_trend > 0)
        elif trend_cond == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA':
            return (close > fast_ma) & (fast_trend > 0)
        elif trend_cond == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA':
            return (close > slow_ma) & (close > fast_ma) & (slow_trend > 0) & (fast_trend > 0)
        elif trend_cond == 'DIRECCION MEDIA LENTA ALCISTA':
            return slow_trend > 0
        elif trend_cond == 'DIRECCION MEDIA RAPIDA ALCISTA':
            return fast_trend > 0
        elif trend_cond == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA':
            return (slow_trend > 0) & (fast_trend > 0)
        return False
    
    def check_bearish_condition(trend_cond, close, slow_ma, fast_ma, slow_trend, fast_trend):
        if trend_cond == 'NINGUNA CONDICION':
            return True
        elif trend_cond == 'PRECIO MENOR A MEDIA RAPIDA':
            return close < fast_ma
        elif trend_cond == 'PRECIO MENOR A MEDIA LENTA':
            return close < slow_ma
        elif trend_cond == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA':
            return (close < slow_ma) & (close < fast_ma)
        elif trend_cond == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA':
            return (close < slow_ma) & (slow_trend < 0)
        elif trend_cond == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA':
            return (close < fast_ma) & (fast_trend < 0)
        elif trend_cond == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA':
            return (close < slow_ma) & (close < fast_ma) & (slow_trend < 0) & (fast_trend < 0)
        elif trend_cond == 'DIRECCION MEDIA LENTA BAJISTA':
            return slow_trend < 0
        elif trend_cond == 'DIRECCION MEDIA RAPIDA BAJISTA':
            return fast_trend < 0
        elif trend_cond == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA':
            return (slow_trend < 0) & (fast_trend < 0)
        return False
    
    bullishTrendValid = check_bullish_condition(bullishTrendCondition, df['close'], slowMA, fastMA, slowMATrend, fastMATrend)
    bearishTrendValid = check_bearish_condition(bearishTrendCondition, df['close'], slowMA, fastMA, slowMATrend, fastMATrend)
    
    # Elephant Candle Conditions
    body = np.abs(df['close'] - df['open'])
    range_size = df['high'] - df['low']
    body_ratio = (body / range_size) * 100
    
    isGreenElephantCandleStrong = (df['close'] > df['open']) & (body_ratio >= minBodyPercentage) & (body >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = (df['close'] < df['open']) & (body_ratio >= minBodyPercentage) & (body >= atr.shift(1) * searchFactor)
    
    # Final conditions
    greenElephantValid = isGreenElephantCandleStrong & bullishTrendValid & activateGreenElephantCandles
    redElephantValid = isRedElephantCandleStrong & bearishTrendValid & activateRedElephantCandles
    
    if filterType == 'CON FILTRADO DE TENDENCIA':
        resultGreenElephantCandle = greenElephantValid
        resultRedElephantCandle = redElephantValid
    else:
        resultGreenElephantCandle = greenElephantValid & isGreenElephantCandleStrong
        resultRedElephantCandle = redElephantValid & isRedElephantCandleStrong
    
    longCondition = resultGreenElephantCandle & isCmfValidForLong
    shortCondition = resultRedElephantCandle & isCmfValidForShort
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(highestHigh.iloc[i]) or pd.isna(lowestLow.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue
        
        if longCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if shortCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries