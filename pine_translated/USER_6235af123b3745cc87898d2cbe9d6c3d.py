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
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_type = 'SMA'
    ma_len = 20
    ma_src = df['close']
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b = df['close']
    reaction_ma_1 = 1
    reaction_ma_2 = 1
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    VVEV = True
    VVER = True

    close = df['close']
    open_arr = df['open']
    high = df['high']
    low = df['low']

    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2

    atr = np.zeros(len(df))
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr[i] = (atr[i-1] * (CDBA - 1) + tr.iloc[i]) / CDBA
    atr_series = pd.Series(atr, index=df.index)

    VVE_0 = close > open_arr
    VRE_0 = close < open_arr

    body = np.abs(open_arr - close)
    hl_range = high - low
    body_pct = np.where(hl_range != 0, body * 100 / hl_range, 0)
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    VVE_2 = VVE_1 & (body >= atr_series.shift(1) * FDB)
    VRE_2 = VRE_1 & (body >= atr_series.shift(1) * FDB)

    def calc_ma(src, length, mtype):
        if mtype == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif mtype == 'SMA':
            return src.rolling(length).mean()
        elif mtype == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[:len(x)].sum() if len(x) == length else np.nan, raw=True)
        elif mtype == 'VWMA':
            vol = df['volume']
            return (src * vol).rolling(length).sum() / vol.rolling(length).sum()
        elif mtype == 'SMMA':
            smma = np.zeros(len(src))
            smma[length-1] = src.iloc[:length].mean()
            for i in range(length, len(src)):
                smma[i] = (smma[i-1] * (length - 1) + src.iloc[i]) / length
            return pd.Series(smma, index=src.index)
        elif mtype == 'DEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif mtype == 'TEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            return 3 * (ema1 - ema2) + ema3
        elif mtype == 'HullMA':
            half_len = int(length / 2)
            sqrt_len = int(np.round(np.sqrt(length)))
            wma1 = src.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, half_len + 1)) / np.arange(1, half_len + 1).sum(), raw=True)
            wma2 = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(), raw=True)
            hull_src = 2 * wma1 - wma2
            return hull_src.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len + 1)) / np.arange(1, sqrt_len + 1).sum(), raw=True)
        elif mtype == 'SSMA':
            a1 = np.exp(-1.414 * 3.14159 / length)
            b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            ssma = np.zeros(len(src))
            for i in range(2, len(src)):
                src_val = src.iloc[i] if not np.isnan(src.iloc[i]) else 0
                src_prev = src.iloc[i-1] if not np.isnan(src.iloc[i-1]) else 0
                prev_ssma1 = ssma[i-1] if not np.isnan(ssma[i-1]) else 0
                prev_ssma2 = ssma[i-2] if not np.isnan(ssma[i-2]) else 0
                ssma[i] = c1 * (src_val + src_prev) / 2 + c2 * prev_ssma1 + c3 * prev_ssma2
            return pd.Series(ssma, index=src.index)
        elif mtype == 'ZEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema1 + ema1 - ema2
        elif mtype == 'TMA':
            sma1 = src.rolling(length).mean()
            return sma1.rolling(length).mean()
        else:
            return src.rolling(length).mean()

    ma_series = calc_ma(ma_src, ma_len, ma_type)
    ma_series_b = calc_ma(ma_src_b, ma_len_b, ma_type_b)

    direction = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if ma_series.iloc[i] > ma_series.iloc[i-1]:
            count = 1
            for j in range(2, reaction_ma_1 + 1):
                if i - j >= 0 and ma_series.iloc[i - j + 1] > ma_series.iloc[i - j]:
                    count += 1
                else:
                    break
            if count >= reaction_ma_1:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        elif ma_series.iloc[i] < ma_series.iloc[i-1]:
            count = 1
            for j in range(2, reaction_ma_1 + 1):
                if i - j >= 0 and ma_series.iloc[i - j + 1] < ma_series.iloc[i - j]:
                    count += 1
                else:
                    break
            if count >= reaction_ma_1:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        else:
            direction.iloc[i] = direction.iloc[i-1]

    direction_b = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if ma_series_b.iloc[i] > ma_series_b.iloc[i-1]:
            count = 1
            for j in range(2, reaction_ma_2 + 1):
                if i - j >= 0 and ma_series_b.iloc[i - j + 1] > ma_series_b.iloc[i - j]:
                    count += 1
                else:
                    break
            if count >= reaction_ma_2:
                direction_b.iloc[i] = 1
            else:
                direction_b.iloc[i] = direction_b.iloc[i-1]
        elif ma_series_b.iloc[i] < ma_series_b.iloc[i-1]:
            count = 1
            for j in range(2, reaction_ma_2 + 1):
                if i - j >= 0 and ma_series_b.iloc[i - j + 1] < ma_series_b.iloc[i - j]:
                    count += 1
                else:
                    break
            if count >= reaction_ma_2:
                direction_b.iloc[i] = -1
            else:
                direction_b.iloc[i] = direction_b.iloc[i-1]
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1]

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
    VRE_3 = VRE_2 & (close < ma_series) & (close < ma_series_b)

    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < CDBA or np.isnan(atr_series.iloc[i]):
            continue
        if RES_VVE.iloc[i] if hasattr(RES_VVE, 'iloc') else RES_VVE[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1
        if RES_VRE.iloc[i] if hasattr(RES_VRE, 'iloc') else RES_VRE[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1

    return entries