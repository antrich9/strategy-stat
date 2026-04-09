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
    # Default parameters
    vixLength = 22
    vixMultiplier = 2.0
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    VVEV = True
    VVER = True
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    ma_type = 'SMA'
    ma_len = 20
    ma_src = df['close']
    reaction_ma_1 = 1
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b = df['close']
    reaction_ma_2 = 1

    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']

    # Calculate ATR (Wilder ATR)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/CDBA, adjust=False).mean()

    # Williams Vix Fix
    highestClose = close.rolling(vixLength).max()
    wvf = (highestClose - low) / highestClose * 100
    wvfSignal = wvf > vixMultiplier

    # Elephant candle conditions
    VVE_0 = close > open_arr
    VRE_0 = close < open_arr

    body_pct = np.abs(open_arr - close) * 100 / (high - low + 1e-10)
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    body_size = np.abs(open_arr - close)
    VVE_2 = VVE_1 & (body_size >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body_size >= atr.shift(1) * FDB)

    # Moving averages
    def variant(src, ma_type, len):
        if ma_type == 'EMA':
            return src.ewm(span=len, adjust=False).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, len + 1)
            return src.rolling(len).apply(lambda x: np.dot(x, weights[:len]) / weights.sum(), raw=True)
        elif ma_type == 'VWMA':
            return pd.Series(index=src.index, dtype=float)
        elif ma_type == 'SMMA':
            result = pd.Series(index=src.index, dtype=float)
            for i in range(len(src)):
                if i == 0:
                    result.iloc[i] = src.iloc[:len].mean()
                else:
                    result.iloc[i] = (result.iloc[i-1] * (len - 1) + src.iloc[i]) / len
            return result
        elif ma_type == 'DEMA':
            ema1 = src.ewm(span=len, adjust=False).mean()
            ema2 = ema1.ewm(span=len, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == 'TEMA':
            ema1 = src.ewm(span=len, adjust=False).mean()
            ema2 = ema1.ewm(span=len, adjust=False).mean()
            ema3 = ema2.ewm(span=len, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        elif ma_type == 'HullMA':
            half_len = int(len / 2)
            sqrt_len = int(np.sqrt(len))
            wma1 = src.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, half_len + 1)) / np.arange(1, half_len + 1).sum(), raw=True)
            wma2 = src.rolling(len).apply(lambda x: np.dot(x, np.arange(1, len + 1)) / np.arange(1, len + 1).sum(), raw=True)
            hull = 2 * wma1 - wma2
            return hull.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len + 1)) / np.arange(1, sqrt_len + 1).sum(), raw=True)
        elif ma_type == 'SSMA':
            result = pd.Series(index=src.index, dtype=float)
            a1 = np.exp(-1.414 * 3.14159 / len)
            b1 = 2 * a1 * np.cos(1.414 * 3.14159 / len)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            v9 = 0.0
            for i in range(len(src)):
                if i == 0:
                    result.iloc[i] = c1 * (src.iloc[i] + (src.iloc[i-1] if i > 0 else src.iloc[i])) / 2
                else:
                    prev_v9 = result.iloc[i-1] if i > 1 else 0
                    prev_v9_2 = result.iloc[i-2] if i > 2 else prev_v9
                    src_prev = src.iloc[i-1] if i > 0 else src.iloc[i]
                    result.iloc[i] = c1 * (src.iloc[i] + src_prev) / 2 + c2 * prev_v9 + c3 * prev_v9_2
            return result
        elif ma_type == 'ZEMA':
            ema1 = src.ewm(span=len, adjust=False).mean()
            ema2 = ema1.ewm(span=len, adjust=False).mean()
            return ema1 + ema1 - ema2
        elif ma_type == 'TMA':
            sma1 = src.rolling(len).mean()
            return sma1.rolling(len).mean()
        else:  # SMA
            return src.rolling(len).mean()

    ma_series = variant(ma_src, ma_type, ma_len)
    ma_series_b = variant(ma_src_b, ma_type_b, ma_len_b)

    # Direction calculation
    direction = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if i < reaction_ma_1:
            direction.iloc[i] = direction.iloc[i-1]
        elif ma_series.iloc[i] > ma_series.iloc[i - reaction_ma_1]:
            direction.iloc[i] = 1
        elif ma_series.iloc[i] < ma_series.iloc[i - reaction_ma_1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    direction_b = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if i < reaction_ma_2:
            direction_b.iloc[i] = direction_b.iloc[i-1]
        elif ma_series_b.iloc[i] > ma_series_b.iloc[i - reaction_ma_2]:
            direction_b.iloc[i] = 1
        elif ma_series_b.iloc[i] < ma_series_b.iloc[i - reaction_ma_2]:
            direction_b.iloc[i] = -1
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1]

    # Trend conditions
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

    VVE_3 = (VVE_2 & PMAMR) | (VVE_2 & PMAML) | (VVE_2 & PMAMRYL) | (VVE_2 & PMAMLYDA) | (VVE_2 & PMAMRYDA) | (VVE_2 & PMAMLRYDA) | (VVE_2 & DMLA) | (VVE_2 & DMRA) | (VVE_2 & DMLRA) | (VVE_2 & NCA)
    VRE_3 = (VRE_2 & PMEAMR) | (VRE_2 & PMEAML) | (VRE_2 & PMEAMRYL) | (VRE_2 & PMEAMLYDB) | (VRE_2 & PMEAMRYDB) | (VRE_2 & PMEAMLRYDB) | (VRE_2 & DMLB) | (VRE_2 & DMRB) | (VRE_2 & DMLRB) | (VRE_2 & NCB)

    RES_VVE = (VVE_3 & VVEV & (modo_tipo == 'CON FILTRADO DE TENDENCIA')) | (VVE_2 & VVEV & (modo_tipo == 'SIN FILTRADO DE TENDENCIA'))
    RES_VRE = (VRE_3 & VVER & (modo_tipo == 'CON FILTRADO DE TENDENCIA')) | (VRE_2 & VVER & (modo_tipo == 'SIN FILTRADO DE TENDENCIA'))

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < CDBA:
            continue
        if pd.isna(ma_series.iloc[i]) or pd.isna(ma_series_b.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        if RES_VVE.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_dict = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry_dict)
            trade_num += 1

        if RES_VRE.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_dict = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry_dict)
            trade_num += 1

    return entries