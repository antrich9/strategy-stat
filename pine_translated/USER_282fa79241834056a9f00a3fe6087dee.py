import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']
    
    donchLength = 20
    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2
    
    CDBA = 100
    FDB = 1.3
    PDCM = 70
    VVEV = True
    VVER = True
    modo_tipo = "CON FILTRADO DE TENDENCIA"
    ma_type = "SMA"
    ma_len = 20
    ma_src = close
    ma_type_b = "SMA"
    ma_len_b = 8
    ma_src_b = close
    reaction_ma_1 = 1
    reaction_ma_2 = 1
    config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    config_tend_baj = "DIRECCION MEDIA RAPIDA BAJISTA"
    
    period = CDBA
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    alpha = 2.0 / (period + 1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = (-a1) * a1
        c1 = 1 - c2 - c3
        result = np.zeros_like(src)
        result[0] = src.iloc[0]
        for i in range(1, len(src)):
            result[i] = c1 * (src.iloc[i] + (src.iloc[i-1] if not np.isnan(src.iloc[i-1]) else result[i-1])) / 2 + c2 * result[i-1] + c3 * (result[i-2] if i >= 2 else result[i-1])
        return pd.Series(result, index=src.index)
    
    def variant_smoothed(src, length):
        result = pd.Series(np.nan, index=src.index)
        prev = src.iloc[0]
        result.iloc[0] = prev
        for i in range(1, len(src)):
            result.iloc[i] = (prev * (length - 1) + src.iloc[i]) / length if not np.isnan(prev) else src.iloc[i]
            prev = result.iloc[i]
        return result
    
    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + (ema1 - ema2)
    
    def variant_doubleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        return 2 * ema1 - ema1.ewm(span=length, adjust=False).mean()
    
    def variant_tripleema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return 3 * (ema1 - ema2) + ema2.ewm(span=length, adjust=False).mean()
    
    def variant(type, src, length):
        if type == "EMA":
            return src.ewm(span=length, adjust=False).mean()
        elif type == "WMA":
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
        elif type == "VWMA":
            return pd.Series(np.nan, index=src.index)
        elif type == "SMMA":
            return variant_smoothed(src, length)
        elif type == "DEMA":
            return variant_doubleema(src, length)
        elif type == "TEMA":
            return variant_tripleema(src, length)
        elif type == "HullMA":
            wma_half = src.rolling(length // 2).apply(lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True)
            hull = 2 * wma_half - src.rolling(length).apply(lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True)
            sqrt_len = int(np.sqrt(length))
            return hull.rolling(sqrt_len).apply(lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), raw=True)
        elif type == "SSMA":
            return variant_supersmoother(src, length)
        elif type == "ZEMA":
            return variant_zerolagema(src, length)
        elif type == "TMA":
            return src.rolling(length).mean().rolling(length).mean()
        else:
            return src.rolling(length).mean()
    
    ma_series = variant(ma_type, ma_src, ma_len)
    ma_series_b = variant(ma_type_b, ma_src_b, ma_len_b)
    
    shifted = ma_series.shift(1)
    rising = ma_series.rolling(reaction_ma_1).max() > shifted
    falling = ma_series.rolling(reaction_ma_1).min() < shifted
    direction = pd.Series(0, index=ma_series.index)
    for i in range(1, len(direction)):
        if rising.iloc[i]:
            direction.iloc[i] = 1
        elif falling.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1] if not np.isnan(direction.iloc[i-1]) else 0
    
    shifted_b = ma_series_b.shift(1)
    rising_b = ma_series_b.rolling(reaction_ma_2).max() > shifted_b
    falling_b = ma_series_b.rolling(reaction_ma_2).min() < shifted_b
    direction_b = pd.Series(0, index=ma_series_b.index)
    for i in range(1, len(direction_b)):
        if rising_b.iloc[i]:
            direction_b.iloc[i] = 1
        elif falling_b.iloc[i]:
            direction_b.iloc[i] = -1
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1] if not np.isnan(direction_b.iloc[i-1]) else 0
    
    PMAMR = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA") & (close > ma_series_b)
    PMAML = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA") & (close > ma_series)
    PMAMRYL = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y LENTA") & (close > ma_series) & (close > ma_series_b)
    PMAMLYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA") & (close > ma_series) & (direction > 0)
    PMAMRYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA") & (close > ma_series_b) & (direction_b > 0)
    PMAMLRYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA") & (close > ma_series) & (close > ma_series_b) & (direction > 0) & (direction_b > 0)
    DMLA = (config_tend_alc == "DIRECCION MEDIA LENTA ALCISTA") & (direction > 0)
    DMRA = (config_tend_alc == "DIRECCION MEDIA RAPIDA ALCISTA") & (direction_b > 0)
    DMLRA = (config_tend_alc == "DIRECCION MEDIA LENTA/RAPIDA ALCISTA") & (direction > 0) & (direction_b > 0)
    NCA = config_tend_alc == "NINGUNA CONDICION"
    
    PMEAMR = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA") & (close < ma_series_b)
    PMEAML = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA") & (close < ma_series)
    PMEAMRYL = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA Y LENTA") & (close < ma_series) & (close < ma_series_b)
    PMEAMLYDB = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA") & (close < ma_series) & (direction < 0)
    PMEAMRYDB = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA") & (close < ma_series_b) & (direction_b < 0)
    PMEAMLRYDB = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA") & (close < ma_series) & (close < ma_series_b) & (direction < 0) & (direction_b < 0)
    DMLB = (config_tend_baj == "DIRECCION MEDIA LENTA BAJISTA") & (direction < 0)
    DMRB = (config_tend_baj == "DIRECCION MEDIA RAPIDA BAJISTA") & (direction_b < 0)
    DMLRB = (config_tend_baj == "DIRECCION MEDIA LENTA/RAPIDA BAJISTA") & (direction < 0) & (direction_b < 0)
    NCB = config_tend_baj == "NINGUNA CONDICION"
    
    VVE_2 = (close > open_arr) & ((close - open_arr).abs() * 100 / (high - low).abs() >= PDCM) & ((close - open_arr).abs() >= atr.shift(1) * FDB)
    VRE_2 = (close < open_arr) & ((close - open_arr).abs() * 100 / (high - low).abs() >= PDCM) & ((close - open_arr).abs() >= atr.shift(1) * FDB)
    
    VVE_3 = VVE_2 & (PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA)
    VRE_3 = VRE_2 & (PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB)
    
    if modo_tipo == "CON FILTRADO DE TENDENCIA":
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER
    
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if RES_VVE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        elif RES_VRE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    return entries