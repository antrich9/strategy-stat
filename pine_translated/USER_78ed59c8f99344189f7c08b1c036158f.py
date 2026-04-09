import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default parameters from the Pine Script inputs
    PDCM = 70                     # body minimum percentage
    CDBA = 100                    # ATR period (cantidad de barras anteriores)
    FDB = 1.3                     # factor de busqueda
    ma_len = 20                   # slow MA period
    ma_len_b = 8                  # fast MA period
    reaction_ma_1 = 1             # reaction length for slow MA (default)
    reaction_ma_2 = 1             # reaction length for fast MA (default)
    modo_tipo = "CON FILTRADO DE TENDENCIA"  # trend filter mode

    # Aliases
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']

    # -------------------------------------------------
    # 1. True Range and Wilder ATR (period = CDBA)
    # -------------------------------------------------
    tr = pd.Series(np.nan, index=df.index, dtype=float)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    for i in range(1, len(df)):
        hl = high.iloc[i] - low.iloc[i]
        hc = abs(high.iloc[i] - close.iloc[i - 1])
        lc = abs(low.iloc[i] - close.iloc[i - 1])
        tr.iloc[i] = max(hl, hc, lc)

    atr = tr.ewm(alpha=1.0 / CDBA, adjust=False).mean()

    # -------------------------------------------------
    # 2. Candle body conditions (VVE_0, VVE_1, VVE_2)
    # -------------------------------------------------
    body = (close - open_).abs()
    range_ = high - low
    range_ = range_.replace(0, np.nan)                 # avoid div/0
    body_pct = body * 100.0 / range_

    VVE_0 = close > open_
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VVE_2 = VVE_1 & (body >= atr.shift(1) * FDB)

    # -------------------------------------------------
    # 3. Moving averages and direction flags
    # -------------------------------------------------
    ma_slow = close.rolling(window=ma_len).mean()
    ma_fast = close.rolling(window=ma_len_b).mean()

    # Direction for slow MA (reaction length = reaction_ma_1)
    direction_slow = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if ma_slow.iloc[i] > ma_slow.iloc[i - reaction_ma_1]:
            direction_slow.iloc[i] = 1
        elif ma_slow.iloc[i] < ma_slow.iloc[i - reaction_ma_1]:
            direction_slow.iloc[i] = -1
        else:
            direction_slow.iloc[i] = direction_slow.iloc[i - 1]

    # Direction for fast MA (reaction length = reaction_ma_2)
    direction_fast = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if ma_fast.iloc[i] > ma_fast.iloc[i - reaction_ma_2]:
            direction_fast.iloc[i] = 1
        elif ma_fast.iloc[i] < ma_fast.iloc[i - reaction_ma_2]:
            direction_fast.iloc[i] = -1
        else:
            direction_fast.iloc[i] = direction_fast.iloc[i - 1]

    # -------------------------------------------------
    # 4. Apply trend filter (VVE_3)
    # -------------------------------------------------
    # The default config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    # => condition is direction_fast > 0
    VVE_3 = VVE_2 & (direction_fast > 0)

    # -------------------------------------------------
    # 5. Final entry condition (RES_VVE)
    # -------------------------------------------------
    if modo_tipo == "CON FILTRADO DE TENDENCIA":
        RES_VVE = VVE_3
    else:  # "SIN FILTRADO DE TENDENCIA"
        RES_VVE = VVE_2

    # -------------------------------------------------
    # 6. Build entry list
    # -------------------------------------------------
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if RES_VVE.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
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

    return entries