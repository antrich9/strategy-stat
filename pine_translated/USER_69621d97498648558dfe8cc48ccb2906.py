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
    
    # Default input values from Pine Script
    PDCM = 70  # PORCENTAJE DE CUERPO MINIMO %
    CDBA = 100  # CANTIDAD DE BARRAS ANTERIORES
    FDB = 1.3  # FACTOR DE BUSQUEDA
    VVEV = True  # ACTIVAR VELAS ELEFANTE VERDES
    VVER = True  # ACTIVAR VELAS ELEFANTE ROJAS
    modo_tipo = 'CON FILTRADO DE TENDENCIA'  # Default mode
    
    # Moving average parameters
    ma_type = 'SMA'
    ma_len = 20
    ma_src_col = 'close'
    reaction_ma_1 = 1
    
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_2 = 1
    
    # Trend condition defaults
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    
    # Calculate ATR using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[CDBA - 1] = tr.iloc[:CDBA].mean()
    for i in range(CDBA, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (CDBA - 1) + tr.iloc[i]) / CDBA
    
    # Calculate variant moving average
    def variant_ma(ma_type, src, length):
        if ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'SMA':
            return src.rolling(length).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        elif ma_type == 'VWMA':
            volume_col = df['volume']
            return (src * volume_col).rolling(length).sum() / volume_col.rolling(length).sum()
        elif ma_type == 'SMMA':
            smma = pd.Series(index=df.index, dtype=float)
            smma.iloc[length - 1] = src.iloc[:length].mean()
            for i in range(length, len(df)):
                smma.iloc[i] = (smma.iloc[i - 1] * (length - 1) + src.iloc[i]) / length
            return smma
        elif ma_type == 'DEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == 'TEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        elif ma_type == 'HullMA':
            wma_half = src.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
            wma_full = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
            hull = 2 * wma_half - wma_full
            sqrt_len = int(np.sqrt(length))
            return hull.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
        elif ma_type == 'SSMA':
            a1 = np.exp(-1.414 * 3.14159 / length)
            b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            ssma = pd.Series(index=df.index, dtype=float)
            ssma.iloc[0] = src.iloc[0]
            for i in range(1, len(df)):
                if i == 1:
                    ssma.iloc[i] = c1 * (src.iloc[i] + src.iloc[i - 1]) / 2 + c2 * src.iloc[i - 1]
                else:
                    ssma.iloc[i] = c1 * (src.iloc[i] + src.iloc[i - 1]) / 2 + c2 * ssma.iloc[i - 1] + c3 * ssma.iloc[i - 2]
            return ssma
        elif ma_type == 'ZEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema1 + ema1 - ema2
        elif ma_type == 'TMA':
            sma1 = src.rolling(length).mean()
            return sma1.rolling(length).mean()
        else:
            return src.rolling(length).mean()
    
    ma_src = df[ma_src_col]
    ma_src_b = df[ma_src_b_col]
    
    ma_series = variant_ma(ma_type, ma_src, ma_len)
    ma_series_b = variant_ma(ma_type_b, ma_src_b, ma_len_b)
    
    # Calculate direction using Wilder's RSI style (rising/falling)
    def calculate_direction(ma_series, reaction_len):
        direction = pd.Series(0, index=df.index)
        for i in range(reaction_len, len(df)):
            is_rising = True
            for j in range(reaction_len):
                if ma_series.iloc[i - j] <= ma_series.iloc[i - j - 1]:
                    is_rising = False
                    break
            is_falling = True
            for j in range(reaction_len):
                if ma_series.iloc[i - j] >= ma_series.iloc[i - j - 1]:
                    is_falling = False
                    break
            if is_rising:
                direction.iloc[i] = 1
            elif is_falling:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
        return direction
    
    direction = calculate_direction(ma_series, reaction_ma_1)
    direction_b = calculate_direction(ma_series_b, reaction_ma_2)
    
    # Trend conditions
    close_price = df['close']
    
    PMAMR = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA') & (close_price > ma_series_b)
    PMAML = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA') & (close_price > ma_series)
    PMAMRYL = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close_price > ma_series) & (close_price > ma_series_b)
    PMAMLYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close_price > ma_series) & (direction > 0)
    PMAMRYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close_price > ma_series_b) & (direction_b > 0)
    PMAMLRYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close_price > ma_series) & (close_price > ma_series_b) & (direction > 0) & (direction_b > 0)
    DMLA = (config_tend_alc == 'DIRECCION MEDIA LENTA ALCISTA') & (direction > 0)
    DMRA = (config_tend_alc == 'DIRECCION MEDIA RAPIDA ALCISTA') & (direction_b > 0)
    DMLRA = (config_tend_alc == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (direction > 0) & (direction_b > 0)
    NCA = config_tend_alc == 'NINGUNA CONDICION'
    
    PMEAMR = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA') & (close_price < ma_series_b)
    PMEAML = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA') & (close_price < ma_series)
    PMEAMRYL = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close_price < ma_series) & (close_price < ma_series_b)
    PMEAMLYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close_price < ma_series) & (direction < 0)
    PMEAMRYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close_price < ma_series_b) & (direction_b < 0)
    PMEAMLRYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close_price < ma_series) & (close_price < ma_series_b) & (direction < 0) & (direction_b < 0)
    DMLB = (config_tend_baj == 'DIRECCION MEDIA LENTA BAJISTA') & (direction < 0)
    DMRB = (config_tend_baj == 'DIRECCION MEDIA RAPIDA BAJISTA') & (direction_b < 0)
    DMLRB = (config_tend_baj == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (direction < 0) & (direction_b < 0)
    NCB = config_tend_baj == 'NINGUNA CONDICION'
    
    # Elephant candle conditions
    open_price = df['open']
    high_price = df['high']
    low_price = df['low']
    
    VVE_0 = close_price > open_price
    VRE_0 = close_price < open_price
    
    body = abs(close_price - open_price)
    candle_range = high_price - low_price
    body_pct = (body / candle_range) * 100
    
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)
    
    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)
    
    # Trend filtered conditions
    VVE_3 = (
        (VVE_2 & PMAMR) |
        (VVE_2 & PMAML) |
        (VVE_2 & PMAMRYL) |
        (VVE_2 & PMAMLYDA) |
        (VVE_2 & PMAMRYDA) |
        (VVE_2 & PMAMLRYDA) |
        (VVE_2 & DMLA) |
        (VVE_2 & DMRA) |
        (VVE_2 & DMLRA) |
        (VVE_2 & NCA)
    )
    
    VRE_3 = (
        (VRE_2 & PMEAMR) |
        (VRE_2 & PMEAML) |
        (VRE_2 & PMEAMRYL) |
        (VRE_2 & PMEAMLYDB) |
        (VRE_2 & PMEAMRYDB) |
        (VRE_2 & PMEAMLRYDB) |
        (VRE_2 & DMLB) |
        (VRE_2 & DMRB) |
        (VRE_2 & DMLRB) |
        (VRE_2 & NCB)
    )
    
    # Final result conditions
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if RES_VVE.iloc[i]:
            entry_price = close_price.iloc[i]
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
        
        if RES_VRE.iloc[i]:
            entry_price = close_price.iloc[i]
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