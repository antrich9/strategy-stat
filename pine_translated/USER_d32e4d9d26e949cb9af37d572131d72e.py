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
    
    # Strategy parameters (matching Pine Script defaults)
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_type = "SMA"
    ma_len = 20
    ma_src_col = 'close'
    reaction_ma_1 = 1
    ma_type_b = "SMA"
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_2 = 1
    ADVE = True
    VVER = True
    VVEV = True
    modo_tipo = "CON FILTRADO DE TENDENCIA"
    config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    config_tend_baj = "DIRECCION MEDIA RAPIDA BAJISTA"
    
    df = df.copy()
    
    # Handle missing columns gracefully
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate ATR (Wilder's method)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=CDBA).sum() / CDBA
    atr = atr.where(df.index >= CDBA - 1)  # Keep NaN for insufficient bars
    
    # Elephant candle body calculation
    body = np.abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    body_pct = (body * 100) / candle_range.replace(0, np.nan)
    
    # VVE_0: Green candle
    VVE_0 = (df['close'] > df['open'])
    
    # VRE_0: Red candle
    VRE_0 = (df['close'] < df['open'])
    
    # VVE_1: Green elephant candle (body% >= PDCM)
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    
    # VRE_1: Red elephant candle (body% >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)
    
    # VVE_2: VVE_1 AND body >= atr[1] * FDB
    atr_prev = atr.shift(1).fillna(0)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    
    # VRE_2: VRE_1 AND body >= atr[1] * FDB
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)
    
    # Moving average calculations
    src = df[ma_src_col] if ma_src_col in df.columns else df['close']
    src_b = df[ma_src_b_col] if ma_src_b_col in df.columns else df['close']
    
    if ma_type == "EMA":
        ma_series = src.ewm(span=ma_len, adjust=False).mean()
    elif ma_type == "SMA":
        ma_series = src.rolling(window=ma_len).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, ma_len + 1)
        ma_series = src.rolling(window=ma_len).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
    elif ma_type == "SMMA":
        ma_series = src.ewm(alpha=1.0/ma_len, adjust=False).mean()
    else:
        ma_series = src.rolling(window=ma_len).mean()
    
    if ma_type_b == "EMA":
        ma_series_b = src_b.ewm(span=ma_len_b, adjust=False).mean()
    elif ma_type_b == "SMA":
        ma_series_b = src_b.rolling(window=ma_len_b).mean()
    elif ma_type_b == "WMA":
        weights_b = np.arange(1, ma_len_b + 1)
        ma_series_b = src_b.rolling(window=ma_len_b).apply(lambda x: np.dot(x, weights_b[-len(x):]) / weights_b[-len(x):].sum(), raw=True)
    elif ma_type_b == "SMMA":
        ma_series_b = src_b.ewm(alpha=1.0/ma_len_b, adjust=False).mean()
    else:
        ma_series_b = src_b.rolling(window=ma_len_b).mean()
    
    # Rising function
    def rising(series, length):
        result = pd.Series(False, index=series.index)
        for i in range(length - 1, len(series)):
            if series.iloc[i-length+1:i+1].notna().all():
                if all(series.iloc[j] >= series.iloc[j-1] for j in range(i-length+2, i+1)):
                    result.iloc[i] = True
        return result
    
    # Falling function
    def falling(series, length):
        result = pd.Series(False, index=series.index)
        for i in range(length - 1, len(series)):
            if series.iloc[i-length+1:i+1].notna().all():
                if all(series.iloc[j] <= series.iloc[j-1] for j in range(i-length+2, i+1)):
                    result.iloc[i] = True
        return result
    
    rising_slow = rising(ma_series, reaction_ma_1)
    falling_slow = falling(ma_series, reaction_ma_1)
    rising_fast = rising(ma_series_b, reaction_ma_2)
    falling_fast = falling(ma_series_b, reaction_ma_2)
    
    # Direction calculation
    direction = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(rising_slow.iloc[i]) and rising_slow.iloc[i]:
            direction.iloc[i] = 1
        elif pd.notna(falling_slow.iloc[i]) and falling_slow.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    
    direction_b = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(rising_fast.iloc[i]) and rising_fast.iloc[i]:
            direction_b.iloc[i] = 1
        elif pd.notna(falling_fast.iloc[i]) and falling_fast.iloc[i]:
            direction_b.iloc[i] = -1
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1]
    
    close = df['close']
    ma_s = ma_series
    ma_f = ma_series_b
    dir_s = direction
    dir_f = direction_b
    
    # Bullish trend conditions
    PMAMR = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA") & (close > ma_f)
    PMAML = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA") & (close > ma_s)
    PMAMRYL = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y LENTA") & (close > ma_s) & (close > ma_f)
    PMAMLYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA") & (close > ma_s) & (dir_s > 0)
    PMAMRYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA") & (close > ma_f) & (dir_f > 0)
    PMAMLRYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA") & (close > ma_s) & (close > ma_f) & (dir_s > 0) & (dir_f > 0)
    DMLA = (config_tend_alc == "DIRECCION MEDIA LENTA ALCISTA") & (dir_s > 0)
    DMRA = (config_tend_alc == "DIRECCION MEDIA RAPIDA ALCISTA") & (dir_f > 0)
    DMLRA = (config_tend_alc == "DIRECCION MEDIA LENTA/RAPIDA ALCISTA") & (dir_s > 0) & (dir_f > 0)
    NCA = (config_tend_alc == "NINGUNA CONDICION")
    
    # Bearish trend conditions
    PMEAMR = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA") & (close < ma_f)
    PMEAML = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA") & (close < ma_s)
    PMEAMRYL = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA Y LENTA") & (close < ma_s) & (close < ma_f)
    PMEAMLYDB = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA") & (close < ma_s) & (dir_s < 0)
    PMEAMRYDB = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA") & (close < ma_f) & (dir_f < 0)
    PMEAMLRYDB = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA") & (close < ma_s) & (close < ma_f) & (dir_s < 0) & (dir_f < 0)
    DMLB = (config_tend_baj == "DIRECCION MEDIA LENTA BAJISTA") & (dir_s < 0)
    DMRB = (config_tend_baj == "DIRECCION MEDIA RAPIDA BAJISTA") & (dir_f < 0)
    DMLRB = (config_tend_baj == "DIRECCION MEDIA LENTA/RAPIDA BAJISTA") & (dir_s < 0) & (dir_f < 0)
    NCB = (config_tend_baj == "NINGUNA CONDICION")
    
    # VVE_3: VVE_2 with trend filter
    VVE_3 = VVE_2 & (PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA)
    
    # VRE_3: VRE_2 with trend filter
    VRE_3 = VRE_2 & (PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB)
    
    # Final entry signals
    if modo_tipo == "CON FILTRADO DE TENDENCIA":
        RES_VVE = VVE_3 & VVEV & ADVE
        RES_VRE = VRE_3 & VVER & ADVE
    else:
        RES_VVE = VVE_2 & VVEV & ADVE
        RES_VRE = VRE_2 & VVER & ADVE
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        entry_price_guess = df['close'].iloc[i]
        ts = df['time'].iloc[i]
        
        if RES_VVE.iloc[i]:
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price_guess),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price_guess),
                'raw_price_b': float(entry_price_guess)
            })
            trade_num += 1
        
        if RES_VRE.iloc[i]:
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price_guess),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price_guess),
                'raw_price_b': float(entry_price_guess)
            })
            trade_num += 1
    
    return entries