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
    
    # Default parameters from Pine Script inputs
    PDCM = 70  # body minimum percentage
    CDBA = 100  # ATR period
    FDB = 1.3  # search factor
    VVEV = True  # activate green elephant candles
    VVER = True  # activate red elephant candles
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    
    ma_type = 'SMA'
    ma_len = 20
    ma_src_col = 'close'
    reaction_ma_1 = 1
    
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_2 = 1
    
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    
    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']
    
    # Calculate Wilder ATR manually
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = pd.Series(tr.ewm(alpha=1/CDBA, adjust=False).mean())
    
    # Variant functions for MA calculation
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        result = pd.Series(np.zeros(len(src)), index=src.index)
        for i in range(len(src)):
            if i == 0:
                result.iloc[i] = c1 * (src.iloc[i] + src.iloc[i]) / 2
            elif i == 1:
                result.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * result.iloc[i-1]
            else:
                result.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * result.iloc[i-1] + c3 * result.iloc[i-2]
        return result
    
    def variant_smoothed(src, length):
        result = pd.Series(np.zeros(len(src)), index=src.index)
        for i in range(len(src)):
            if i == 0 or pd.isna(result.iloc[i-1]):
                result.iloc[i] = src.iloc[:i+1].mean()
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
        v2_ema = v2.ewm(span=length, adjust=False).mean()
        return 3 * (v2 - v2_ema) + v2_ema.ewm(span=length, adjust=False).mean()
    
    def variant(src, length, mtype):
        if mtype == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif mtype == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.sum(weights[:len(x)] * x) / np.sum(weights[:len(x)]), raw=True)
        elif mtype == 'VWMA':
            # Approximation for VWMA
            return src.ewm(span=length, adjust=False).mean()
        elif mtype == 'SMMA':
            return variant_smoothed(src, length)
        elif mtype == 'DEMA':
            return variant_doubleema(src, length)
        elif mtype == 'TEMA':
            return variant_tripleema(src, length)
        elif mtype == 'HullMA':
            wma_half = src.rolling(int(length/2)).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
            hull = 2 * wma_half - src.rolling(length).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
            return hull.rolling(int(np.sqrt(length))).mean()
        elif mtype == 'SSMA':
            return variant_supersmoother(src, length)
        elif mtype == 'ZEMA':
            return variant_zerolagema(src, length)
        elif mtype == 'TMA':
            return src.rolling(length).mean().rolling(length).mean()
        else:  # SMA
            return src.rolling(length).mean()
    
    ma_series = variant(df[ma_src_col], ma_len, ma_type)
    ma_series_b = variant(df[ma_src_b_col], ma_len_b, ma_type_b)
    
    # Calculate direction for slow MA
    direction = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if i < reaction_ma_1:
            continue
        rising = all(ma_series.iloc[i - reaction_ma_1 + 1:i + 1].diff().dropna() > 0)
        falling = all(ma_series.iloc[i - reaction_ma_1 + 1:i + 1].diff().dropna() < 0)
        if rising:
            direction.iloc[i] = 1
        elif falling:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1] if i > 0 else 0
    
    # Calculate direction for fast MA
    direction_b = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if i < reaction_ma_2:
            continue
        rising = all(ma_series_b.iloc[i - reaction_ma_2 + 1:i + 1].diff().dropna() > 0)
        falling = all(ma_series_b.iloc[i - reaction_ma_2 + 1:i + 1].diff().dropna() < 0)
        if rising:
            direction_b.iloc[i] = 1
        elif falling:
            direction_b.iloc[i] = -1
        else:
            direction_b.iloc[i] = direction_b.iloc[i - 1] if i > 0 else 0
    
    # Vela Elefante conditions
    body = np.abs(close - open_arr)
    total_range = high - low
    body_pct = np.where(total_range > 0, body * 100 / total_range, 0)
    body_pct_series = pd.Series(body_pct, index=df.index)
    
    VVE_0 = close > open_arr  # green candle
    VRE_0 = close < open_arr  # red candle
    
    VVE_1 = VVE_0 & (body_pct_series >= PDCM)
    VRE_1 = VRE_0 & (body_pct_series >= PDCM)
    
    # For VVE_2 and VRE_2, we need atr[1] which is shifted ATR
    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)
    
    # Trend conditions - bullish
    PMAMR = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA') & (close > ma_series_b)
    PMAML = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA') & (close > ma_series)
    PMAMRYL = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close > ma_series) & (close > ma_series_b)
    PMAMLYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close > ma_series) & (direction > 0)
    PMAMRYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close > ma_series_b) & (direction_b > 0)
    PMAMLRYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close > ma_series) & (close > ma_series_b) & (direction > 0) & (direction_b > 0)
    DMLA = (config_tend_alc == 'DIRECCION MEDIA LENTA ALCISTA') & (direction > 0)
    DMRA = (config_tend_alc == 'DIRECCION MEDIA RAPIDA ALCISTA') & (direction_b > 0)
    DMLRA = (config_tend_alc == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (direction > 0) & (direction_b > 0)
    NCA = config_tend_alc == 'NINGUNA CONDICION'
    
    # Trend conditions - bearish
    PMEAMR = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA') & (close < ma_series_b)
    PMEAML = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA') & (close < ma_series)
    PMEAMRYL = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close < ma_series) & (close < ma_series_b)
    PMEAMLYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close < ma_series) & (direction < 0)
    PMEAMRYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close < ma_series_b) & (direction_b < 0)
    PMEAMLRYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close < ma_series) & (close < ma_series_b) & (direction < 0) & (direction_b < 0)
    DMLB = (config_tend_baj == 'DIRECCION MEDIA LENTA BAJISTA') & (direction < 0)
    DMRB = (config_tend_baj == 'DIRECCION MEDIA RAPIDA BAJISTA') & (direction_b < 0)
    DMLRB = (config_tend_baj == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (direction < 0) & (direction_b < 0)
    NCB = config_tend_baj == 'NINGUNA CONDICION'
    
    # VVE_3 and VRE_3 with trend filter
    VVE_3 = (VVE_2 & PMAMR) | (VVE_2 & PMAML) | (VVE_2 & PMAMRYL) | (VVE_2 & PMAMLYDA) | (VVE_2 & PMAMRYDA) | (VVE_2 & PMAMLRYDA) | (VVE_2 & DMLA) | (VVE_2 & DMRA) | (VVE_2 & DMLRA) | (VVE_2 & NCA)
    VRE_3 = (VRE_2 & PMEAMR) | (VRE_2 & PMEAML) | (VRE_2 & PMEAMRYL) | (VRE_2 & PMEAMLYDB) | (VRE_2 & PMEAMRYDB) | (VRE_2 & PMEAMLRYDB) | (VRE_2 & DMLB) | (VRE_2 & DMRB) | (VRE_2 & DMLRB) | (VRE_2 & NCB)
    
    # Final entry signals
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:  # SIN FILTRADO DE TENDENCIA
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        long_signal = RES_VVE.iloc[i] if hasattr(RES_VVE.iloc[i], '__iter__') == False else bool(RES_VVE.iloc[i])
        short_signal = RES_VRE.iloc[i] if hasattr(RES_VRE.iloc[i], '__iter__') == False else bool(RES_VRE.iloc[i])
        
        if long_signal:
            entry_price = float(close.iloc[i])
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
        
        if short_signal:
            entry_price = float(close.iloc[i])
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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