import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default parameter values from script inputs
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    VVEV = True
    VVER = True
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    ma_type = 'SMA'
    ma_len = 20
    ma_src_col = 'close'
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_1 = 1
    reaction_ma_2 = 1
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Wilder ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/CDBA, adjust=False).mean()

    # Body and body percentage
    body = (close - open_).abs()
    full_range = high - low
    body_pct = body / full_range * 100

    # Elephant candles conditions
    VVE_0 = close > open_
    VRE_0 = close < open_

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    body_prev = body.shift(1)
    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body_prev >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body_prev >= atr_prev * FDB)

    # Moving averages (SMA as default)
    ma_series = close.rolling(ma_len).mean()
    ma_series_b = close.rolling(ma_len_b).mean()

    # Direction indicators
    direction = pd.Series(0.0, index=df.index)
    direction_b = pd.Series(0.0, index=df.index)

    direction.iloc[0] = 0
    for i in range(1, len(df)):
        if close.iloc[i] > ma_series.iloc[i] and ma_series.iloc[i] > ma_series.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < ma_series.iloc[i] and ma_series.iloc[i] < ma_series.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    direction_b.iloc[0] = 0
    for i in range(1, len(df)):
        if close.iloc[i] > ma_series_b.iloc[i] and ma_series_b.iloc[i] > ma_series_b.iloc[i-1]:
            direction_b.iloc[i] = 1
        elif close.iloc[i] < ma_series_b.iloc[i] and ma_series_b.iloc[i] < ma_series_b.iloc[i-1]:
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

    # Apply trend filter
    VVE_3 = (
        (VVE_2 & PMAMR) | (VVE_2 & PMAML) | (VVE_2 & PMAMRYL) |
        (VVE_2 & PMAMLYDA) | (VVE_2 & PMAMRYDA) | (VVE_2 & PMAMLRYDA) |
        (VVE_2 & DMLA) | (VVE_2 & DMRA) | (VVE_2 & DMLRA) |
        (VVE_2 & NCA)
    )

    VRE_3 = (
        (VRE_2 & PMEAMR) | (VRE_2 & PMEAML) | (VRE_2 & PMEAMRYL) |
        (VRE_2 & PMEAMLYDB) | (VRE_2 & PMEAMRYDB) | (VRE_2 & PMEAMLRYDB) |
        (VRE_2 & DMLB) | (VRE_2 & DMRB) | (VRE_2 & DMLRB) |
        (VRE_2 & NCB)
    )

    # Final signals
    with_trend_filter = modo_tipo == 'CON FILTRADO DE TENDENCIA'
    without_trend_filter = modo_tipo == 'SIN FILTRADO DE TENDENCIA'

    RES_VVE = (VVE_3 & VVEV & with_trend_filter) | (VVE_2 & VVEV & without_trend_filter)
    RES_VRE = (VRE_3 & VVER & with_trend_filter) | (VRE_2 & VVER & without_trend_filter)

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(ma_series.iloc[i]) or pd.isna(ma_series_b.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if RES_VVE.iloc[i]:
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