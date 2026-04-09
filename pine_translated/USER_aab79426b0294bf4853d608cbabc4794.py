import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default inputs from the Pine Script
    PDCM = 70                     # minimum body percent
    CDBA = 100                    # ATR length
    FDB = 1.3                     # body‑ATR factor
    ma_len_b = 8                  # fast MA period (SMA)

    # Trend‑filter mode (default)
    modo_tipo = "CON FILTRADO DE TENDENCIA"
    VVEV = True                   # activate green elephant candles

    # ---------- Wilder ATR(CDBA) ----------
    tr = pd.concat([
        df['high'] - df['low'],
        df['high'] - df['close'].shift(1),
        df['low'] - df['close'].shift(1)
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / CDBA, adjust=False).mean()
    atr_prev = atr.shift(1)

    # ---------- Candle body calculations ----------
    body = (df['open'] - df['close']).abs()
    hl = df['high'] - df['low']
    body_pct = pd.Series(
        np.where(hl != 0, body * 100 / hl, 0.0),
        index=df.index
    )

    # ---------- Elephant candle conditions ----------
    VVE_0 = df['close'] > df['open']
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VVE_2 = VVE_1 & (body >= atr_prev * FDB)

    # ---------- Fast moving average (SMA) ----------
    fast_ma = df['close'].rolling(window=ma_len_b).mean()

    # ---------- Direction of fast MA ----------
    direction_b = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if fast_ma.iloc[i] > fast_ma.iloc[i - 1]:
            direction_b.iloc[i] = 1
        elif fast_ma.iloc[i] < fast_ma.iloc[i - 1]:
            direction_b.iloc[i] = -1
        else:
            direction_b.iloc[i] = direction_b.iloc[i - 1]

    # Trend filter: DMRA (fast MA direction rising)
    trend_ok = direction_b > 0

    # ---------- Final entry condition ----------
    entry_cond = VVE_2 & trend_ok & VVEV

    # ---------- Build entry list ----------
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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