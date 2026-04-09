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
    # Parameter defaults from Pine Script
    donchLength = 20
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_type = 'SMA'
    ma_len = 20
    ma_type_b = 'SMA'
    ma_len_b = 8
    config_tend_alc = 'DIRECCION MEDIA RAPIDA ALCISTA'
    config_tend_baj = 'DIRECCION MEDIA RAPIDA BAJISTA'
    modo_tipo = 'CON FILTRADO DE TENDENCIA'
    VVEV = True
    VVER = True
    ADVE = True
    reaction_ma_1 = 1
    reaction_ma_2 = 1

    # Helper: Wilder ATR
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        for i in range(period, len(tr)):
            if not np.isnan(atr.iloc[i]):
                atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
        return atr

    # Helper: Simple Moving Average
    def sma(src, length):
        return src.rolling(window=length).mean()

    # Helper: Rising/Falling direction
    def calc_direction(ma_series, reaction):
        direction = pd.Series(np.nan, index=ma_series.index)
        direction.iloc[0] = 0
        for i in range(1, len(ma_series)):
            if ma_series.iloc[i] > ma_series.iloc[i - reaction]:
                direction.iloc[i] = 1
            elif ma_series.iloc[i] < ma_series.iloc[i - reaction]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1] if not np.isnan(direction.iloc[i - 1]) else 0
        return direction

    # Calculate ATR
    atr = wilder_atr(df['high'], df['low'], df['close'], CDBA)

    # Calculate body percentage
    body = (df['close'] - df['open']).abs()
    range_val = df['high'] - df['low']
    body_pct = (body / range_val) * 100

    # Basic candle direction
    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']

    # Body percentage condition
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    # ATR factor condition
    atr_prev = atr.shift(1)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)

    # Calculate moving averages
    ma_series = sma(df['close'], ma_len)
    ma_series_b = sma(df['close'], ma_len_b)

    # Calculate direction
    direction = calc_direction(ma_series, reaction_ma_1)
    direction_b = calc_direction(ma_series_b, reaction_ma_2)

    # Trend conditions
    DMRA = direction_b > 0
    DMRB = direction_b < 0

    # Combine with trend filter
    VVE_3 = VVE_2 & DMRA
    VRE_3 = VRE_2 & DMRB

    # Final entry conditions
    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        RES_VVE = VVE_3 & VVEV & ADVE
        RES_VRE = VRE_3 & VVER & ADVE
    else:
        RES_VVE = VVE_2 & VVEV & ADVE
        RES_VRE = VRE_2 & VVER & ADVE

    # Build entries list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i == 0:
            continue

        ts = int(df['time'].iloc[i])
        entry_price = df['close'].iloc[i]

        if RES_VVE.iloc[i] if hasattr(RES_VVE, 'iloc') else RES_VVE[i]:
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

        if RES_VRE.iloc[i] if hasattr(RES_VRE, 'iloc') else RES_VRE[i]:
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