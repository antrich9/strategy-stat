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
    # Default parameter values from Pine Script
    CDBA = 100  # bars for ATR
    PDCM = 70   # min body percentage
    FDB = 1.3   # search factor
    VVEV = True  # activate green elephant
    VVER = True  # activate red elephant
    modo_tipo = 'CON FILTRADO DE TENDENCIA'  # mode with trend filter
    
    # MA parameters
    ma_type = 'SMA'
    ma_len = 20
    ma_src_col = 'close'
    reaction_ma_1 = 1
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_2 = 1
    
    # Trend config defaults
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'

    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']

    # Wilder ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/CDBA, adjust=False).mean()

    # Green/Red candle flags
    VVE_0 = close > open_
    VRE_0 = close < open_

    # Body percentage condition (handle zero range)
    range_ = high - low
    body_pct = np.where(range_ != 0, np.abs(close - open_) / range_ * 100, 0)
    body_pct_series = pd.Series(body_pct, index=df.index)

    VVE_1 = VVE_0 & (body_pct_series >= PDCM)
    VRE_1 = VRE_0 & (body_pct_series >= PDCM)

    # Body size >= ATR[1] * FDB
    body_size = np.abs(close - open_)
    VVE_2 = VVE_1 & (body_size >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body_size >= atr.shift(1) * FDB)

    # Moving averages
    if ma_type == 'EMA':
        ma_series = close.ewm(span=ma_len, adjust=False).mean()
    elif ma_type == 'SMA':
        ma_series = close.rolling(ma_len).mean()
    else:
        ma_series = close.rolling(ma_len).mean()

    if ma_type_b == 'EMA':
        ma_series_b = close.ewm(span=ma_len_b, adjust=False).mean()
    elif ma_type_b == 'SMA':
        ma_series_b = close.rolling(ma_len_b).mean()
    else:
        ma_series_b = close.rolling(ma_len_b).mean()

    # Trend direction
    direction = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(ma_series.iloc[i]) and pd.notna(ma_series.iloc[i - reaction_ma_1]):
            vals = ma_series.iloc[max(0, i - reaction_ma_1 + 1):i + 1]
            if len(vals) >= 2 and all(pd.notna(vals)):
                if vals.iloc[-1] > vals.iloc[0]:
                    direction.iloc[i] = 1
                elif vals.iloc[-1] < vals.iloc[0]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i - 1]
        else:
            direction.iloc[i] = direction.iloc[i - 1] if pd.notna(direction.iloc[i - 1]) else 0

    direction_b = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(ma_series_b.iloc[i]) and pd.notna(ma_series_b.iloc[i - reaction_ma_2]):
            vals = ma_series_b.iloc[max(0, i - reaction_ma_2 + 1):i + 1]
            if len(vals) >= 2 and all(pd.notna(vals)):
                if vals.iloc[-1] > vals.iloc[0]:
                    direction_b.iloc[i] = 1
                elif vals.iloc[-1] < vals.iloc[0]:
                    direction_b.iloc[i] = -1
                else:
                    direction_b.iloc[i] = direction_b.iloc[i - 1]
        else:
            direction_b.iloc[i] = direction_b.iloc[i - 1] if pd.notna(direction_b.iloc[i - 1]) else 0

    # Trend filter conditions
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

    # Apply trend filters
    trend_filter_long = PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA
    trend_filter_short = PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB

    VVE_3 = VVE_2 & trend_filter_long
    VRE_3 = VRE_2 & trend_filter_short

    # Final entry conditions
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(ma_series.iloc[i]) or pd.isna(ma_series_b.iloc[i]):
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