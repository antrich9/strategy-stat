import pandas as pd
import numpy as np
from datetime import datetime, timezone

def compute_wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    alpha = (length - 1) / (length + 1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    # Input parameters from Pine Script
    donchLength = 20
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    VVER = True
    VVEV = True
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_type = 'SMA'
    ma_len = 20
    ma_type_b = 'SMA'
    ma_len_b = 8
    reaction_ma_1 = 1
    reaction_ma_2 = 1
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'

    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']

    # Donchian Channel
    highestHigh = high.rolling(donchLength).max()
    lowestLow = low.rolling(donchLength).min()
    middleBand = (highestHigh + lowestLow) / 2

    # ATR
    atr = compute_wilder_atr(high, low, close, CDBA)

    # Step 1: Green/Red candle
    VVE_0 = close > open_price
    VRE_0 = close < open_price

    # Step 2: Body percentage filter
    body = (close - open_price).abs()
    body_pct = body * 100 / (high - low)
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    # Step 3: ATR factor filter
    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)

    # Moving Averages
    ma_series = close.ewm(span=ma_len, adjust=False).mean()
    ma_series_b = close.ewm(span=ma_len_b, adjust=False).mean()

    # Direction slow MA
    window_1 = max(reaction_ma_1, 1)
    rising_1 = ma_series.rolling(window_1).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=True).fillna(0)
    falling_1 = ma_series.rolling(window_1).apply(lambda x: 1 if x.iloc[-1] < x.iloc[0] else 0, raw=True).fillna(0)
    raw_dir_1 = rising_1 - falling_1
    change_dir_1 = raw_dir_1 != raw_dir_1.shift(1)
    direction = pd.Series(0.0, index=close.index)
    for i in range(1, len(direction)):
        if pd.notna(raw_dir_1.iloc[i]):
            if raw_dir_1.iloc[i] != 0:
                direction.iloc[i] = raw_dir_1.iloc[i]
            elif change_dir_1.iloc[i]:
                direction.iloc[i] = 0
            else:
                direction.iloc[i] = direction.iloc[i-1] if pd.notna(direction.iloc[i-1]) else 0
    direction = direction.fillna(0)

    # Direction fast MA
    window_2 = max(reaction_ma_2, 1)
    rising_2 = ma_series_b.rolling(window_2).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=True).fillna(0)
    falling_2 = ma_series_b.rolling(window_2).apply(lambda x: 1 if x.iloc[-1] < x.iloc[0] else 0, raw=True).fillna(0)
    raw_dir_2 = rising_2 - falling_2
    change_dir_2 = raw_dir_2 != raw_dir_2.shift(1)
    direction_b = pd.Series(0.0, index=close.index)
    for i in range(1, len(direction_b)):
        if pd.notna(raw_dir_2.iloc[i]):
            if raw_dir_2.iloc[i] != 0:
                direction_b.iloc[i] = raw_dir_2.iloc[i]
            elif change_dir_2.iloc[i]:
                direction_b.iloc[i] = 0
            else:
                direction_b.iloc[i] = direction_b.iloc[i-1] if pd.notna(direction_b.iloc[i-1]) else 0
    direction_b = direction_b.fillna(0)

    # Trend conditions long
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

    # Trend conditions short
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

    # Combined trend
    VVE_3 = VVE_2 & (PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA)
    VRE_3 = VRE_2 & (PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB)

    # Final signals
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER

    # Build entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if RES_VVE.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
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
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries