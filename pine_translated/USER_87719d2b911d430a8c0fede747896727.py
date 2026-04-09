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
    # Default parameters from Pine Script inputs
    donchLength = 20
    ADVE = True
    VVER = True
    VVEV = True
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    ma_type = 'SMA'
    ma_len = 20
    ma_src_col = 'close'
    reaction_ma_1 = 1
    ma_type_b = 'SMA'
    ma_len_b = 8
    ma_src_b_col = 'close'
    reaction_ma_2 = 1

    close = df['close']
    open_prices = df['open']
    high = df['high']
    low = df['low']

    # ATR using Wilder's method (CDBA = period)
    def wilder_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr

    atr = wilder_atr(df, CDBA)

    # Elephant candle conditions
    body = np.abs(close - open_prices)
    candle_range = high - low
    body_pct = np.where(candle_range > 0, body * 100 / candle_range, 0)

    VVE_0 = close > open_prices
    VRE_0 = close < open_prices

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    atr_prev = atr.shift(1)
    body_size = body
    VVE_2 = VVE_1 & (body_size >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body_size >= atr_prev * FDB)

    # Moving average variants
    src = df[ma_src_col]
    src_b = df[ma_src_b_col]

    def calc_ma(ma_type, src, length):
        if ma_type == 'EMA':
            return src.ewm(span=length, adjust=False).mean()
        elif ma_type == 'SMA':
            return src.rolling(length).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
        elif ma_type == 'VWMA':
            volume = df['volume']
            return (src * volume).rolling(length).sum() / volume.rolling(length).sum()
        elif ma_type == 'SMMA':
            smma = pd.Series(index=src.index, dtype=float)
            smma.iloc[length-1] = src.iloc[:length].mean()
            for i in range(length, len(src)):
                smma.iloc[i] = (smma.iloc[i-1] * (length - 1) + src.iloc[i]) / length
            return smma
        elif ma_type == 'DEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == 'TEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        elif ma_type == 'HullMA':
            wma_half = src.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
            wma_full = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
            hull = 2 * wma_half - wma_full
            sqrt_len = int(np.sqrt(length))
            return hull.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
        elif ma_type == 'SSMA':
            a1 = np.exp(-1.414 * 3.14159 / length)
            b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            ssma = pd.Series(0.0, index=src.index)
            for i in range(2, len(src)):
                if np.isnan(src.iloc[i-1]):
                    ssma.iloc[i] = src.iloc[:i+1].mean()
                else:
                    ssma.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * ssma.iloc[i-1] + c3 * ssma.iloc[i-2]
            return ssma
        elif ma_type == 'ZEMA':
            ema1 = src.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == 'TMA':
            sma1 = src.rolling(length).mean()
            return sma1.rolling(length).mean()
        else:
            return src.rolling(length).mean()

    ma_series = calc_ma(ma_type, src, ma_len)
    ma_series_b = calc_ma(ma_type_b, src_b, ma_len_b)

    # Direction calculation (rising/falling)
    def calc_direction(ma, reaction):
        direction = pd.Series(0, index=ma.index)
        for i in range(reaction, len(ma)):
            if ma.iloc[i] > ma.iloc[i-reaction]:
                direction.iloc[i] = 1
            elif ma.iloc[i] < ma.iloc[i-reaction]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if not pd.isna(direction.iloc[i-1]) else 0
        return direction

    direction = calc_direction(ma_series, reaction_ma_1)
    direction_b = calc_direction(ma_series_b, reaction_ma_2)

    # Trend conditions for long (bullish elephant)
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

    # Trend conditions for short (bearish elephant)
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

    # Combined trend conditions
    VVE_3 = VVE_2 & (PMAMR | PMAML | PMAMRYL | PMAMLYDA | PMAMRYDA | PMAMLRYDA | DMLA | DMRA | DMLRA | NCA)
    VRE_3 = VRE_2 & (PMEAMR | PMEAML | PMEAMRYL | PMEAMLYDB | PMEAMRYDB | PMEAMLRYDB | DMLB | DMRB | DMLRB | NCB)

    # Final entry conditions
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV
        RES_VRE = VRE_3 & VVER
    else:
        RES_VVE = VVE_2 & VVEV
        RES_VRE = VRE_2 & VVER

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(ma_series.iloc[i]) or pd.isna(ma_series_b.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        entry_price = close.iloc[i]

        if RES_VVE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if RES_VRE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries