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
    # Parameters (from Pine Script defaults)
    CDBA = 100
    PDCM = 70
    FDB = 1.3
    VVEV = True
    VVER = True
    modo_tipo = "CON FILTRADO DE TENDENCIA"
    config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    config_tend_baj = "DIRECCION MEDIA RAPIDA BAJISTA"
    ma_type = "SMA"
    ma_len = 20
    ma_type_b = "SMA"
    ma_len_b = 8
    ma_src = df['close']
    ma_src_b = df['close']
    reaction_ma_1 = 1
    reaction_ma_2 = 1

    # Wilder ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/CDBA, adjust=False).mean()

    # Moving averages (SMA as default)
    ma_slow = ma_src.rolling(ma_len).mean()
    ma_fast = ma_src_b.rolling(ma_len_b).mean()

    # Elephant Candle detection
    body = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    body_pct = (body / candle_range) * 100

    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)

    # Trend direction
    rising_slow = ma_slow > ma_slow.shift(reaction_ma_1)
    falling_slow = ma_slow < ma_slow.shift(reaction_ma_1)
    direction = pd.Series(0, index=df.index)
    direction[rising_slow] = 1
    direction[falling_slow] = -1
    direction = direction.replace(0, np.nan).ffill().fillna(0)

    rising_fast = ma_fast > ma_fast.shift(reaction_ma_2)
    falling_fast = ma_fast < ma_fast.shift(reaction_ma_2)
    direction_b = pd.Series(0, index=df.index)
    direction_b[rising_fast] = 1
    direction_b[falling_fast] = -1
    direction_b = direction_b.replace(0, np.nan).ffill().fillna(0)

    # Trend conditions (using defaults)
    PMAMR = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA") & (df['close'] > ma_fast)
    PMAML = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA") & (df['close'] > ma_slow)
    PMAMRYL = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y LENTA") & (df['close'] > ma_slow) & (df['close'] > ma_fast)
    PMAMLYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA") & (df['close'] > ma_slow) & (direction > 0)
    PMAMRYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA") & (df['close'] > ma_fast) & (direction_b > 0)
    PMAMLRYDA = (config_tend_alc == "PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA") & (df['close'] > ma_slow) & (df['close'] > ma_fast) & (direction > 0) & (direction_b > 0)
    DMLA = (config_tend_alc == "DIRECCION MEDIA LENTA ALCISTA") & (direction > 0)
    DMRA = (config_tend_alc == "DIRECCION MEDIA RAPIDA ALCISTA") & (direction_b > 0)
    DMLRA = (config_tend_alc == "DIRECCION MEDIA LENTA/RAPIDA ALCISTA") & (direction > 0) & (direction_b > 0)
    NCA = config_tend_alc == "NINGUNA CONDICION"

    PMEAMR = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA") & (df['close'] < ma_fast)
    PMEAML = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA") & (df['close'] < ma_slow)
    PMEAMRYL = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA Y LENTA") & (df['close'] < ma_slow) & (df['close'] < ma_fast)
    PMEAMLYDB = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA Y DIRECCION BAJISTA") & (df['close'] < ma_slow) & (direction < 0)
    PMEAMRYDB = (config_tend_baj == "PRECIO MENOR A MEDIA RAPIDA Y DIRECCION BAJISTA") & (df['close'] < ma_fast) & (direction_b < 0)
    PMEAMLRYDB = (config_tend_baj == "PRECIO MENOR A MEDIA LENTA/RAPIDA Y DIRECCION BAJISTA") & (df['close'] < ma_slow) & (df['close'] < ma_fast) & (direction < 0) & (direction_b < 0)
    DMLB = (config_tend_baj == "DIRECCION MEDIA LENTA BAJISTA") & (direction < 0)
    DMRB = (config_tend_baj == "DIRECCION MEDIA RAPIDA BAJISTA") & (direction_b < 0)
    DMLRB = (config_tend_baj == "DIRECCION MEDIA LENTA/RAPIDA BAJISTA") & (direction < 0) & (direction_b < 0)
    NCB = config_tend_baj == "NINGUNA CONDICION"

    VVE_3 = VVE_2 & (PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA)
    VRE_3 = VRE_2 & (PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB)

    # Final results
    if modo_tipo == "CON FILTRADO DE TENDENCIA":
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(ma_slow.iloc[i]) or pd.isna(ma_fast.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        if RES_VVE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif RES_VRE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries