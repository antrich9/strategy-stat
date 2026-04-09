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
    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']
    volume_col = df['volume']

    n = len(df)
    entries = []
    trade_num = 1

    fiLength = 13
    emaLength = 2
    donchLength = 20

    forceIndex = (close_col - close_col.shift(1)) * volume_col
    fiEma = forceIndex.ewm(span=emaLength, adjust=False).mean()

    highestHigh = high_col.rolling(window=donchLength).max()
    lowestLow = low_col.rolling(window=donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2

    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_type = 'SMA'
    ma_len = 20
    ma_src = close_col
    reaction_ma_1 = 1
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b = close_col
    reaction_ma_2 = 1
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    VVEV = True
    VVER = True
    modo_tipo = 'CON FILTRADO DE TENDENCIA'

    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        result = np.zeros_like(src)
        result[0] = src.iloc[0] if isinstance(src.iloc[0], float) else src.iloc[0]
        for i in range(1, len(src)):
            val = src.iloc[i] if isinstance(src.iloc[i], float) else 0
            prev1 = result[i-1]
            prev2 = result[i-2] if i >= 2 else 0
            result[i] = c1 * (val + (result[i-1] if not np.isnan(result[i-1]) else val)) / 2 + c2 * prev1 + c3 * prev2
        return pd.Series(result, index=src.index)

    def variant_smoothed(src, length):
        result = pd.Series(np.nan, index=src.index)
        sma_val = src.iloc[:length].mean()
        if pd.notna(sma_val):
            result.iloc[length-1] = sma_val
        for i in range(length, len(src)):
            prev = result.iloc[i-1]
            if pd.notna(prev):
                result.iloc[i] = (prev * (length - 1) + src.iloc[i]) / length
            else:
                result.iloc[i] = src.iloc[i]
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
        v7 = 3 * (v2 - v2.ewm(span=length, adjust=False).mean()) + v2.ewm(span=length, adjust=False).mean()
        return v7

    def variant(type, src, length):
        if type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif type == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum() if len(x) == length else np.nan, raw=True)
        elif type == 'VWMA':
            return pd.Series([src.iloc[i] * volume_col.iloc[i] for i in range(length)]).rolling(length).sum() / volume_col.rolling(length).sum()
        elif type == 'SMMA':
            return variant_smoothed(src, length)
        elif type == 'DEMA':
            return variant_doubleema(src, length)
        elif type == 'TEMA':
            return variant_tripleema(src, length)
        elif type == 'HullMA':
            wma_half = src.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum() if len(x) == length // 2 else np.nan, raw=True)
            hull = 2 * wma_half - src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum() if len(x) == length else np.nan, raw=True)
            return hull.rolling(round(np.sqrt(length))).mean()
        elif type == 'SSMA':
            return variant_supersmoother(src, length)
        elif type == 'ZEMA':
            return variant_zerolagema(src, length)
        elif type == 'TMA':
            return src.rolling(length).mean().rolling(length).mean()
        else:
            return src.rolling(length).mean()

    ma_series = variant(ma_type, ma_src, ma_len)
    ma_series_b = variant(ma_type_b, ma_src_b, ma_len_b)

    direction = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(ma_series.iloc[i]) and pd.notna(ma_series.iloc[max(0, i-reaction_ma_1)]):
            if ma_series.iloc[i] > ma_series.iloc[i-reaction_ma_1]:
                direction.iloc[i] = 1
            elif ma_series.iloc[i] < ma_series.iloc[i-reaction_ma_1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if pd.notna(direction.iloc[i-1]) else 0
        else:
            direction.iloc[i] = direction.iloc[i-1] if pd.notna(direction.iloc[i-1]) else 0

    direction_b = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(ma_series_b.iloc[i]) and pd.notna(ma_series_b.iloc[max(0, i-reaction_ma_2)]):
            if ma_series_b.iloc[i] > ma_series_b.iloc[i-reaction_ma_2]:
                direction_b.iloc[i] = 1
            elif ma_series_b.iloc[i] < ma_series_b.iloc[i-reaction_ma_2]:
                direction_b.iloc[i] = -1
            else:
                direction_b.iloc[i] = direction_b.iloc[i-1] if pd.notna(direction_b.iloc[i-1]) else 0
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1] if pd.notna(direction_b.iloc[i-1]) else 0

    def calculate_wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    atr = calculate_wilder_atr(high_col, low_col, close_col, CDBA)

    VVE_0 = close_col > open_col
    VRE_0 = close_col < open_col

    body_size = abs(open_col - close_col)
    range_size = high_col - low_col
    body_pct = pd.Series(np.where(range_size != 0, body_size * 100 / range_size, 0), index=df.index)

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    VVE_2 = VVE_1 & (body_size >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body_size >= atr.shift(1) * FDB)

    PMAMR = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA') & (close_col > ma_series_b)
    PMAML = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA') & (close_col > ma_series)
    PMAMRYL = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA') & (close_col > ma_series) & (close_col > ma_series_b)
    PMAMLYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA') & (close_col > ma_series) & (direction > 0)
    PMAMRYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA') & (close_col > ma_series_b) & (direction_b > 0)
    PMAMLRYDA = (config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA') & (close_col > ma_series) & (close_col > ma_series_b) & (direction > 0) & (direction_b > 0)
    DMLA = (config_tend_alc == 'DIRECCION MEDIA LENTA ALCISTA') & (direction > 0)
    DMRA = (config_tend_alc == 'DIRECCION MEDIA RAPIDA ALCISTA') & (direction_b > 0)
    DMLRA = (config_tend_alc == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA') & (direction > 0) & (direction_b > 0)
    NCA = config_tend_alc == 'NINGUNA CONDICION'

    PMEAMR = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA') & (close_col < ma_series_b)
    PMEAML = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA') & (close_col < ma_series)
    PMEAMRYL = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA') & (close_col < ma_series) & (close_col < ma_series_b)
    PMEAMLYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA') & (close_col < ma_series) & (direction < 0)
    PMEAMRYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA') & (close_col < ma_series_b) & (direction_b < 0)
    PMEAMLRYDB = (config_tend_baj == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA') & (close_col < ma_series) & (close_col < ma_series_b) & (direction < 0) & (direction_b < 0)
    DMLB = (config_tend_baj == 'DIRECCION MEDIA LENTA BAJISTA') & (direction < 0)
    DMRB = (config_tend_baj == 'DIRECCION MEDIA RAPIDA BAJISTA') & (direction_b < 0)
    DMLRB = (config_tend_baj == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA') & (direction < 0) & (direction_b < 0)
    NCB = config_tend_baj == 'NINGUNA CONDICION'

    VVE_3 = (VVE_2 & PMAMR) | (VVE_2 & PMAML) | (VVE_2 & PMAMRYL) | (VVE_2 & PMAMLYDA) | (VVE_2 & PMAMRYDA) | (VVE_2 & PMAMLRYDA) | (VVE_2 & DMLA) | (VVE_2 & DMRA) | (VVE_2 & DMLRA) | (VVE_2 & NCA)
    VRE_3 = (VRE_2 & PMEAMR) | (VRE_2 & PMEAML) | (VRE_2 & PMEAMRYL) | (VRE_2 & PMEAMLYDB) | (VRE_2 & PMEAMRYDB) | (VRE_2 & PMEAMLRYDB) | (VRE_2 & DMLB) | (VRE_2 & DMRB) | (VRE_2 & DMLRB) | (VRE_2 & NCB)

    with_trend_filter = modo_tipo == 'CON FILTRADO DE TENDENCIA'
    without_trend_filter = modo_tipo == 'SIN FILTRADO DE TENDENCIA'

    if with_trend_filter:
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    elif without_trend_filter:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER
    else:
        RES_VVE = pd.Series(False, index=df.index)
        RES_VRE = pd.Series(False, index=df.index)

    for i in range(n):
        if pd.isna(atr.iloc[i]) or pd.isna(ma_series.iloc[i]) or pd.isna(ma_series_b.iloc[i]):
            continue

        if RES_VVE.iloc[i]:
            entry_price = close_col.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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

        if RES_VRE.iloc[i]:
            entry_price = close_col.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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