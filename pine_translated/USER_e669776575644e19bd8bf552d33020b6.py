import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters (use defaults from Pine Script)
    donchLength = 20
    previousBarsCount = 100
    minBodyPercentage = 70
    searchFactor = 1.3
    maType = 'SMA'
    maLength = 20
    maReaction = 1
    maTypeB = 'SMA'
    maLengthB = 8
    maReactionB = 1
    bullishTrendCondition = 'DIRECCION MEDIA RAPIDA ALCISTA'
    bearishTrendCondition = 'DIRECCION MEDIA RAPIDA BAJISTA'
    filterType = 'CON FILTRADO DE TENDENCIA'
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    
    smooth = 1
    length_trendilo = 50
    offset = 0.85
    sigma_trendilo = 6
    bmult = 1.0
    cblen = False
    blen = 20
    
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    
    # Donchian Channel
    highest_high = high_prices.rolling(donchLength).max()
    lowest_low = low_prices.rolling(donchLength).min()
    middle_band = (highest_high + lowest_low) / 2
    
    # ATR (Wilder's method)
    tr1 = high_prices - low_prices
    tr2 = (high_prices - close_prices.shift(1)).abs()
    tr3 = (low_prices - close_prices.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/previousBarsCount, min_periods=previousBarsCount).mean()
    
    # Elephant candle body percentage
    body = (close_prices - open_prices).abs()
    full_range = high_prices - low_prices
    body_percentage = body / full_range * 100
    
    is_green_candle = close_prices > open_prices
    is_red_candle = close_prices < open_prices
    
    is_green_valid = is_green_candle & (body_percentage >= minBodyPercentage)
    is_red_valid = is_red_candle & (body_percentage >= minBodyPercentage)
    
    is_green_strong = is_green_valid & (body >= atr.shift(1) * searchFactor)
    is_red_strong = is_red_valid & (body >= atr.shift(1) * searchFactor)
    
    # Moving averages
    if maType == 'SMA':
        slow_ma = close_prices.rolling(int(maLength)).mean()
    elif maType == 'EMA':
        slow_ma = close_prices.ewm(span=int(maLength), adjust=False).mean()
    elif maType == 'WMA':
        slow_ma = close_prices.rolling(int(maLength)).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif maType == 'VWMA':
        slow_ma = (close_prices * df['volume']).rolling(int(maLength)).sum() / df['volume'].rolling(int(maLength)).sum()
    elif maType == 'SMMA':
        smma_slow = pd.Series(index=close_prices.index, dtype=float)
        smma_slow.iloc[0] = close_prices.iloc[0]
        for i in range(1, len(close_prices)):
            smma_slow.iloc[i] = (smma_slow.iloc[i-1] * (int(maLength) - 1) + close_prices.iloc[i]) / int(maLength)
        slow_ma = smma_slow
    elif maType == 'DEMA':
        e1 = close_prices.ewm(span=int(maLength), adjust=False).mean()
        slow_ma = 2 * e1 - e1.ewm(span=int(maLength), adjust=False).mean()
    elif maType == 'TEMA':
        e1 = close_prices.ewm(span=int(maLength), adjust=False).mean()
        e2 = e1.ewm(span=int(maLength), adjust=False).mean()
        slow_ma = 3 * (e1 - e2) + e2.ewm(span=int(maLength), adjust=False).mean()
    elif maType == 'HullMA':
        half_len = int(maLength) // 2
        wma1 = close_prices.rolling(half_len).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        wma2 = close_prices.rolling(int(maLength)).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        hull = 2 * wma1 - wma2
        sqrt_len = int(np.sqrt(int(maLength)))
        slow_ma = hull.rolling(sqrt_len).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif maType == 'SSMA':
        len_ss = int(maLength)
        a1 = np.exp(-1.414 * 3.14159 / len_ss)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / len_ss)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        ssma_slow = pd.Series(index=close_prices.index, dtype=float)
        ssma_slow.iloc[0] = close_prices.iloc[0]
        for i in range(1, len(close_prices)):
            src_val = close_prices.iloc[i]
            src_prev = close_prices.iloc[i-1] if i > 0 else src_val
            v9_prev = ssma_slow.iloc[i-1] if i > 0 else 0
            v9_prev2 = ssma_slow.iloc[i-2] if i > 1 else v9_prev
            ssma_slow.iloc[i] = c1 * (src_val + src_prev) / 2 + c2 * v9_prev + c3 * v9_prev2
        slow_ma = ssma_slow
    elif maType == 'ZEMA':
        ema1 = close_prices.ewm(span=int(maLength), adjust=False).mean()
        ema2 = ema1.ewm(span=int(maLength), adjust=False).mean()
        slow_ma = 2 * ema1 - ema2
    elif maType == 'TMA':
        sma1 = close_prices.rolling(int(maLength)).mean()
        slow_ma = sma1.rolling(int(maLength)).mean()
    else:
        slow_ma = close_prices.rolling(int(maLength)).mean()
    
    if maTypeB == 'SMA':
        fast_ma = close_prices.rolling(int(maLengthB)).mean()
    elif maTypeB == 'EMA':
        fast_ma = close_prices.ewm(span=int(maLengthB), adjust=False).mean()
    elif maTypeB == 'WMA':
        fast_ma = close_prices.rolling(int(maLengthB)).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif maTypeB == 'VWMA':
        fast_ma = (close_prices * df['volume']).rolling(int(maLengthB)).sum() / df['volume'].rolling(int(maLengthB)).sum()
    elif maTypeB == 'SMMA':
        smma_fast = pd.Series(index=close_prices.index, dtype=float)
        smma_fast.iloc[0] = close_prices.iloc[0]
        for i in range(1, len(close_prices)):
            smma_fast.iloc[i] = (smma_fast.iloc[i-1] * (int(maLengthB) - 1) + close_prices.iloc[i]) / int(maLengthB)
        fast_ma = smma_fast
    elif maTypeB == 'DEMA':
        e1 = close_prices.ewm(span=int(maLengthB), adjust=False).mean()
        fast_ma = 2 * e1 - e1.ewm(span=int(maLengthB), adjust=False).mean()
    elif maTypeB == 'TEMA':
        e1 = close_prices.ewm(span=int(maLengthB), adjust=False).mean()
        e2 = e1.ewm(span=int(maLengthB), adjust=False).mean()
        fast_ma = 3 * (e1 - e2) + e2.ewm(span=int(maLengthB), adjust=False).mean()
    elif maTypeB == 'HullMA':
        half_len = int(maLengthB) // 2
        wma1 = close_prices.rolling(half_len).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        wma2 = close_prices.rolling(int(maLengthB)).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        hull = 2 * wma1 - wma2
        sqrt_len = int(np.sqrt(int(maLengthB)))
        fast_ma = hull.rolling(sqrt_len).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    elif maTypeB == 'SSMA':
        len_ss = int(maLengthB)
        a1 = np.exp(-1.414 * 3.14159 / len_ss)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / len_ss)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        ssma_fast = pd.Series(index=close_prices.index, dtype=float)
        ssma_fast.iloc[0] = close_prices.iloc[0]
        for i in range(1, len(close_prices)):
            src_val = close_prices.iloc[i]
            src_prev = close_prices.iloc[i-1] if i > 0 else src_val
            v9_prev = ssma_fast.iloc[i-1] if i > 0 else 0
            v9_prev2 = ssma_fast.iloc[i-2] if i > 1 else v9_prev
            ssma_fast.iloc[i] = c1 * (src_val + src_prev) / 2 + c2 * v9_prev + c3 * v9_prev2
        fast_ma = ssma_fast
    elif maTypeB == 'ZEMA':
        ema1 = close_prices.ewm(span=int(maLengthB), adjust=False).mean()
        ema2 = ema1.ewm(span=int(maLengthB), adjust=False).mean()
        fast_ma = 2 * ema1 - ema2
    elif maTypeB == 'TMA':
        sma1 = close_prices.rolling(int(maLengthB)).mean()
        fast_ma = sma1.rolling(int(maLengthB)).mean()
    else:
        fast_ma = close_prices.rolling(int(maLengthB)).mean()
    
    # Slow MA trend
    slow_ma_rising = slow_ma > slow_ma.shift(maReaction)
    slow_ma_falling = slow_ma < slow_ma.shift(maReaction)
    slow_mat_rend = pd.Series(0, index=close_prices.index)
    for i in range(1, len(slow_mat_rend)):
        if slow_ma_rising.iloc[i]:
            slow_mat_rend.iloc[i] = 1
        elif slow_ma_falling.iloc[i]:
            slow_mat_rend.iloc[i] = -1
        else:
            slow_mat_rend.iloc[i] = slow_mat_rend.iloc[i-1]
    
    # Fast MA trend
    fast_ma_rising = fast_ma > fast_ma.shift(maReactionB)
    fast_ma_falling = fast_ma < fast_ma.shift(maReactionB)
    fast_mat_rend = pd.Series(0, index=close_prices.index)
    for i in range(1, len(fast_mat_rend)):
        if fast_ma_rising.iloc[i]:
            fast_mat_rend.iloc[i] = 1
        elif fast_ma_falling.iloc[i]:
            fast_mat_rend.iloc[i] = -1
        else:
            fast_mat_rend.iloc[i] = fast_mat_rend.iloc[i-1]
    
    # Trend conditions
    is_price_above_fast_ma = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA') & (close_prices > fast_ma)
    is_price_above_slow_ma = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA') & (close_prices > slow_ma)
    is_price_above_both_ma = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close_prices > slow_ma) & (close_prices > fast_ma)
    is_price_above_slow_with_bull_trend = (bullishTrendCondition == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close_prices > slow_ma) & (slow_mat_rend > 0)
    is_price_above_fast_with