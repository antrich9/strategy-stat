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
    # Input parameters
    donchLength = 20
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateElephantCandles = True
    activateRedElephantCandles = True
    activateGreenElephantCandles = True
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    maType = 'SMA'
    maLength = 20
    maSource = 'close'
    maReaction = 1
    maTypeB = 'SMA'
    maLengthB = 8
    maSourceB = 'close'
    maReactionB = 1
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'

    # Calculate Donchian Channel
    highestHigh = df['high'].rolling(window=donchLength).max()
    lowestLow = df['low'].rolling(window=donchLength).min()

    # Wilder RSI implementation
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        return atr

    # ATR calculation
    atr = wilder_atr(df['high'], df['low'], df['close'], previousBarsCount)

    # Elephant Candle Conditions
    body = np.abs(df['close'] - df['open'])
    candle_range = np.abs(df['high'] - df['low'])
    body_pct = body / candle_range * 100

    isGreenElephantCandle = df['close'] > df['open']
    isRedElephantCandle = df['close'] < df['open']

    isGreenElephantCandleValid = isGreenElephantCandle & (body_pct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_pct >= minBodyPercentage)

    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body.shift(1) >= atr.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body.shift(1) >= atr.shift(1) * searchFactor)

    # MA variant functions
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        v9 = pd.Series(0.0, index=src.index)
        for i in range(1, len(src)):
            prev_v9 = v9.iloc[i-1] if i >= 1 else 0
            prev2_v9 = v9.iloc[i-2] if i >= 2 else 0
            prev_src = src.iloc[i-1] if i >= 1 else 0
            v9.iloc[i] = c1 * (src.iloc[i] + prev_src) / 2 + c2 * prev_v9 + c3 * prev2_v9
        return v9

    def variant_smoothed(src, length):
        v5 = pd.Series(0.0, index=src.index)
        v5.iloc[length-1] = src.iloc[:length].mean()
        for i in range(length, len(src)):
            v5.iloc[i] = (v5.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return v5

    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        v10 = ema1 + ema1 - ema2
        return v10

    def variant_doubleema(src, length):
        v2 = src.ewm(span=length, adjust=False).mean()
        v6 = 2 * v2 - v2.ewm(span=length, adjust=False).mean()
        return v6

    def variant_tripleema(src, length):
        v2 = src.ewm(span=length, adjust=False).mean()
        v7 = 3 * (v2 - v2.ewm(span=length, adjust=False).mean()) + v2.ewm(span=length, adjust=False).mean().ewm(span=length, adjust=False).mean()
        return v7

    def variant_hullma(src, length):
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma1 = src.rolling(window=half_len).apply(lambda x: np.sum(x * np.arange(1, half_len+1)) / np.sum(np.arange(1, half_len+1)), raw=True)
        wma2 = src.rolling(window=length).apply(lambda x: np.sum(x * np.arange(1, length+1)) / np.sum(np.arange(1, length+1)), raw=True)
        hull = 2 * wma1 - wma2
        result = hull.rolling(window=sqrt_len).apply(lambda x: np.sum(x * np.arange(1, sqrt_len+1)) / np.sum(np.arange(1, sqrt_len+1)), raw=True)
        return result

    def variant(type, src, length):
        if type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif type == 'WMA':
            return src.rolling(window=length).apply(lambda x: np.sum(x * np.arange(1, length+1)) / np.sum(np.arange(1, length+1)), raw=True)
        elif type == 'VWMA':
            return (df['close'] * df['volume']).rolling(window=length).mean() / df['volume'].rolling(window=length).mean()
        elif type == 'SMMA':
            return variant_smoothed(src, length)
        elif type == 'DEMA':
            return variant_doubleema(src, length)
        elif type == 'TEMA':
            return variant_tripleema(src, length)
        elif type == 'HullMA':
            return variant_hullma(src, length)
        elif type == 'SSMA':
            return variant_supersmoother(src, length)
        elif type == 'ZEMA':
            return variant_zerolagema(src, length)
        elif type == 'TMA':
            return src.rolling(window=length).mean().rolling(window=length).mean()
        else:
            return src.rolling(window=length).mean()

    # Get source series
    if maSource == 'close':
        src_slow = df['close']
    elif maSource == 'open':
        src_slow = df['open']
    elif maSource == 'high':
        src_slow = df['high']
    elif maSource == 'low':
        src_slow = df['low']
    else:
        src_slow = df['close']

    if maSourceB == 'close':
        src_fast = df['close']
    elif maSourceB == 'open':
        src_fast = df['open']
    elif maSourceB == 'high':
        src_fast = df['high']
    elif maSourceB == 'low':
        src_fast = df['low']
    else:
        src_fast = df['close']

    # Moving Averages
    slowMA = variant(maType, src_slow, maLength)
    fastMA = variant(maTypeB, src_fast, maLengthB)

    # Trend Direction
    slowMATrend = pd.Series(0, index=df.index)
    fastMATrend = pd.Series(0, index=df.index)

    for i in range(maReaction, len(df)):
        if i >= maReaction:
            if df['close'].iloc[i] > df['close'].iloc[i-maReaction]:
                slowMATrend.iloc[i] = 1
            elif df['close'].iloc[i] < df['close'].iloc[i-maReaction]:
                slowMATrend.iloc[i] = -1
            else:
                slowMATrend.iloc[i] = slowMATrend.iloc[i-1] if i > 0 else 0

    for i in range(maReactionB, len(df)):
        if i >= maReactionB:
            if df['close'].iloc[i] > df['close'].iloc[i-maReactionB]:
                fastMATrend.iloc[i] = 1
            elif df['close'].iloc[i] < df['close'].iloc[i-maReactionB]:
                fastMATrend.iloc[i] = -1
            else:
                fastMATrend.iloc[i] = fastMATrend.iloc[i-1] if i > 0 else 0

    # Trend Conditions
    isFastMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA RAPIDA ALCISTA') & (fastMATrend > 0)
    isSlowMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA ALCISTA') & (slowMATrend > 0)
    isBothMATrendBullish = (bullishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (slowMATrend > 0) & (fastMATrend > 0)
    noBullishCondition = bullishTrendCondition == 'NINGUNA CONDICION'

    isFastMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA RAPIDA BAJISTA') & (fastMATrend < 0)
    isSlowMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA BAJISTA') & (slowMATrend < 0)
    isBothMATrendBearish = (bearishTrendCondition == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (slowMATrend < 0) & (fastMATrend < 0)
    noBearishCondition = bearishTrendCondition == 'NINGUNA CONDICION'

    # Final Elephant Candle Conditions
    finalGreenElephantCandle = isGreenElephantCandleStrong & (
        isFastMATrendBullish | isSlowMATrendBullish | isBothMATrendBullish | noBullishCondition
    )
    finalRedElephantCandle = isRedElephantCandleStrong & (
        isFastMATrendBearish | isSlowMATrendBearish | isBothMATrendBearish | noBearishCondition
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

    # Trendilo Indicator Calculation
    smooth = 1
    length_t = 50
    offset = 0.85
    sigma = 6

    pch = df['close'].diff(smooth) / df['close'] * 100

    def alma(arr, length, offset, sigma):
        m = np.arange(0, length)
        w = np.exp(-((m - length * offset) ** 2) / (2 * sigma ** 2))
        w = w / w.sum()
        result = pd.Series(np.convolve(arr, w, mode='same'), index=arr.index)
        return result

    avpch = alma(pch.values, length_t, offset, sigma)
    avpch = pd.Series(avpch, index=df.index)
    blength = length_t
    rms = np.sqrt((avpch ** 2).rolling(window=blength).mean()) * 1.0
    cdir = pd.Series(0, index=df.index)
    cdir[avpch > rms] = 1
    cdir[avpch < -rms] = -1

    # Entry signals
    long_signal = resultGreenElephantCandle & (cdir > 0)
    short_signal = resultRedElephantCandle & (cdir < 0)

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < previousBarsCount or pd.isna(slowMA.iloc[i]) or pd.isna(fastMA.iloc[i]):
            continue

        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_signal.iloc[i]:
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

        if short_signal.iloc[i]:
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