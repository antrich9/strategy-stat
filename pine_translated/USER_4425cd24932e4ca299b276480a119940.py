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
    
    # Input parameters (default values from Pine Script)
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
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    VVEV = True
    VVER = True
    
    # Calculate ATR using Wilder's method
    tr = df['high'] - df['low']
    atr = pd.Series(index=df.index, dtype=float)
    if len(df) > CDBA:
        atr.iloc[CDBA] = tr.iloc[:CDBA+1].mean()
        alpha = 1.0 / CDBA
        for i in range(CDBA + 1, len(df)):
            atr.iloc[i] = (1 - alpha) * atr.iloc[i-1] + alpha * tr.iloc[i]
    
    # Calculate elephant candle conditions
    body = (df['close'] - df['open']).abs()
    body_pct = body / (df['high'] - df['low'] + 1e-10)
    
    VVE_1 = (df['close'] > df['open']) & (body_pct >= PDCM / 100.0)
    VRE_1 = (df['close'] < df['open']) & (body_pct >= PDCM / 100.0)
    
    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)
    
    # Calculate moving averages (SMA)
    ma_series = df['close'].rolling(ma_len).mean()
    ma_series_b = df['close'].rolling(ma_len_b).mean()
    
    # Calculate direction for fast MA
    direction_b = pd.Series(0, index=df.index, dtype=float)
    for i in range(ma_len_b + reaction_ma_2 - 1, len(df)):
        if pd.notna(ma_series_b.iloc[i]) and pd.notna(ma_series_b.iloc[i - reaction_ma_2]):
            if ma_series_b.iloc[i] > ma_series_b.iloc[i - reaction_ma_2]:
                direction_b.iloc[i] = 1
            elif ma_series_b.iloc[i] < ma_series_b.iloc[i - reaction_ma_2]:
                direction_b.iloc[i] = -1
    
    # Calculate trend conditions based on config
    PMAMR = config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA' and df['close'] > ma_series_b
    PMAML = config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA' and df['close'] > ma_series
    PMAMRYL = config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y LENTA' and df['close'] > ma_series and df['close'] > ma_series_b
    PMAMLYDA = config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA' and df['close'] > ma_series and direction_b > 0
    PMAMRYDA = config_tend_alc == 'PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA' and df['close'] > ma_series_b and direction_b > 0
    PMAMLRYDA = config_tend_alc == 'PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA' and df['close'] > ma_series and df['close'] > ma_series_b and direction_b > 0
    DMLA = config_tend_alc == 'DIRECCION MEDIA LENTA ALCISTA' and direction_b > 0
    DMRA = config_tend_alc == 'DIRECCION MEDIA RAPIDA ALCISTA' and direction_b > 0
    DMLRA = config_tend_alc == 'DIRECCION MEDIA LENTA/RAPIDA ALCISTA' and direction_b > 0
    NCA = config_tend_alc == 'NINGUNA CONDICION'
    
    PMEAMR = config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA' and df['close'] < ma_series_b
    PMEAML = config_tend_baj == 'PRECIO MENOR A MEDIA LENTA' and df['close'] < ma_series
    PMEAMRYL = config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y LENTA' and df['close'] < ma_series and df['close'] < ma_series_b
    PMEAMLYDB = config_tend_baj == 'PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA' and df['close'] < ma_series and direction_b < 0
    PMEAMRYDB = config_tend_baj == 'PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA' and df['close'] < ma_series_b and direction_b < 0
    PMEAMLRYDB = config_tend_baj == 'PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA' and df['close'] < ma_series and df['close'] < ma_series_b and direction_b < 0
    DMLB = config_tend_baj == 'DIRECCION MEDIA LENTA BAJISTA' and direction_b < 0
    DMRB = config_tend_baj == 'DIRECCION MEDIA RAPIDA BAJISTA' and direction_b < 0
    DMLRB = config_tend_baj == 'DIRECCION MEDIA LENTA/RAPIDA BAJISTA' and direction_b < 0
    NCB = config_tend_baj == 'NINGUNA CONDICION'
    
    VVE_3 = (VVE_2 & PMAMR) | (VVE_2 & PMAML) | (VVE_2 & PMAMRYL) | (VVE_2 & PMAMLYDA) | (VVE_2 & PMAMRYDA) | (VVE_2 & PMAMLRYDA) | (VVE_2 & DMLA) | (VVE_2 & DMRA) | (VVE_2 & DMLRA) | (VVE_2 & NCA)
    VRE_3 = (VRE_2 & PMEAMR) | (VRE_2 & PMEAML) | (VRE_2 & PMEAMRYL) | (VRE_2 & PMEAMLYDB) | (VRE_2 & PMEAMRYDB) | (VRE_2 & PMEAMLRYDB) | (VRE_2 & DMLB) | (VRE_2 & DMRB) | (VRE_2 & DMLRB) | (VRE_2 & NCB)
    
    # Calculate final entry signals
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
        if pd.isna(atr.iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
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