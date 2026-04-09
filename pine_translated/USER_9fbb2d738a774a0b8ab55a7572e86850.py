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
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    VVEV = True
    VVER = True
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    ma_type = 'SMA'
    ma_len = 20
    ma_src = 'close'
    reaction_ma_1 = 1
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b = 'close'
    reaction_ma_2 = 1
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'

    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']

    # Donchian Channel (though not used in entries, included for completeness)
    highestHigh = high_col.rolling(window=donchLength).max()
    lowestLow = low_col.rolling(window=donchLength).min()

    # ATR calculation (Wilder)
    tr = np.maximum(high_col - low_col, np.maximum(np.abs(high_col - close_col.shift(1)), np.abs(low_col - close_col.shift(1))))
    atr = pd.Series(tr).rolling(window=CDBA).mean()

    # Elephant candle conditions
    VVE_0 = close_col > open_col
    VRE_0 = close_col < open_col

    body = np.abs(open_col - close_col)
    body_pct = body * 100 / np.abs(high_col - low_col)
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)

    # Moving averages
    def variant_supersmoother(src, length):
        a1 = np.exp(-1.414 * 3.14159 / length)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        v9 = pd.Series(np.zeros(len(src)), index=src.index)
        for i in range(len(src)):
            if i == 0:
                v9.iloc[i] = c1 * src.iloc[i] / 2
            elif i == 1:
                v9.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * v9.iloc[i-1]
            else:
                v9.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * v9.iloc[i-1] + c3 * v9.iloc[i-2]
        return v9

    def variant_smoothed(src, length):
        v5 = pd.Series(np.zeros(len(src)), index=src.index)
        for i in range(len(src)):
            if pd.isna(v5.iloc[i-1]):
                v5.iloc[i] = src.iloc[:i+1].mean()
            else:
                v5.iloc[i] = (v5.iloc[i-1] * (length - 1) + src.iloc[i]) / length
        return v5

    def variant_zerolagema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 + ema1 - ema2

    def variant_doubleema(src, length):
        v2 = src.ewm(span=length, adjust=False).mean()
        return 2 * v2 - v2.ewm(span=length, adjust=False).mean()

    def variant_tripleema(src, length):
        v2 = src.ewm(span=length, adjust=False).mean()
        v7 = 3 * (v2 - v2.ewm(span=length, adjust=False).mean()) + v2.ewm(span=length, adjust=False).mean().ewm(span=length, adjust=False).mean()
        return v7

    def variant(type_name, src, length):
        if type_name == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif type_name == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)
        elif type_name == 'VWMA':
            return df['volume'].rolling(length).apply(lambda x: np.dot(x, close_col.iloc[x.index]) / x.sum(), raw=True)
        elif type_name == 'SMMA':
            return variant_smoothed(src, length)
        elif type_name == 'DEMA':
            return variant_doubleema(src, length)
        elif type_name == 'TEMA':
            return variant_tripleema(src, length)
        elif type_name == 'HullMA':
            half_len = int(length / 2)
            sqrt_len = int(np.round(np.sqrt(length)))
            wma_half = src.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, half_len + 1)[:len(x)]) / np.arange(1, half_len + 1)[:len(x)].sum(), raw=True)
            hull_src = 2 * wma_half - src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)[:len(x)]) / np.arange(1, length + 1)[:len(x)].sum(), raw=True)
            return hull_src.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len + 1)[:len(x)]) / np.arange(1, sqrt_len + 1)[:len(x)].sum(), raw=True)
        elif type_name == 'SSMA':
            return variant_supersmoother(src, length)
        elif type_name == 'ZEMA':
            return variant_zerolagema(src, length)
        elif type_name == 'TMA':
            return pd.concat([src.rolling(length).mean() for _ in range(2)], axis=1).mean(axis=1)
        else:
            return src.rolling(length).mean()

    src_input = close_col if ma_src == 'close' else df[ma_src] if ma_src in df.columns else close_col
    src_input_b = close_col if ma_src_b == 'close' else df[ma_src_b] if ma_src_b in df.columns else close_col

    ma_series = variant(ma_type, src_input, ma_len)
    ma_series_b = variant(ma_type_b, src_input_b, ma_len_b)

    # Direction calculation
    direction = pd.Series(np.zeros(len(df)), index=df.index)
    direction_b = pd.Series(np.zeros(len(df)), index=df.index)

    for i in range(len(df)):
        if i >= reaction_ma_1:
            rising_count = sum(ma_series.iloc[i-reaction_ma_1+1:i+1].diff().dropna() > 0)
            falling_count = sum(ma_series.iloc[i-reaction_ma_1+1:i+1].diff().dropna() < 0)
            if rising_count == reaction_ma_1:
                direction.iloc[i] = 1
            elif falling_count == reaction_ma_1:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if i > 0 else 0
        else:
            direction.iloc[i] = direction.iloc[i-1] if i > 0 else 0

    for i in range(len(df)):
        if i >= reaction_ma_2:
            rising_count_b = sum(ma_series_b.iloc[i-reaction_ma_2+1:i+1].diff().dropna() > 0)
            falling_count_b = sum(ma_series_b.iloc[i-reaction_ma_2+1:i+1].diff().dropna() < 0)
            if rising_count_b == reaction_ma_2:
                direction_b.iloc[i] = 1
            elif falling_count_b == reaction_ma_2:
                direction_b.iloc[i] = -1
            else:
                direction_b.iloc[i] = direction_b.iloc[i-1] if i > 0 else 0
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1] if i > 0 else 0

    # Trend conditions (bullish)
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

    # Trend conditions (bearish)
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

    # Combine trend with elephant candles
    VVE_3 = VVE_2 & (PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA)
    VRE_3 = VRE_2 & (PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB)

    # Final results
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
        if pd.isna(atr.iloc[i]) or pd.isna(ma_series.iloc[i]) or pd.isna(ma_series_b.iloc[i]):
            continue

        if RES_VVE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1

        if RES_VRE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1

    return entries