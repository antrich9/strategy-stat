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
    # ── Default parameters (mirroring the Pine Script inputs) ──────────────────
    CDBA = 100               # ATR length
    PDCM = 70                # minimum body % threshold
    FDB = 1.3                # body size factor vs ATR
    VVEV = True              # activate green elephant
    VVER = True              # activate red elephant
    modo_tipo = "CON FILTRADO DE TENDENCIA"   # with‑trend filter
    config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    config_tend_baj = "DIRECCION MEDIA RAPIDA BAJISTA"
    ma_len = 20              # slow MA period
    ma_len_b = 8             # fast MA period
    reaction_ma_1 = 1
    reaction_ma_2 = 1

    # ── Basic candle properties ────────────────────────────────────────────────
    body = (df['close'] - df['open']).abs()
    high_low = df['high'] - df['low']
    body_pct = np.where(high_low > 0, body * 100.0 / high_low, 0.0)

    # ── Elephant candle stage‑0 ────────────────────────────────────────────────
    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']

    # ── Elephant candle stage‑1 (body % threshold) ─────────────────────────────
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    # ── Wilder ATR (length = CDBA) ─────────────────────────────────────────────
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / CDBA, adjust=False).mean()

    # ── Elephant candle stage‑2 (body size vs ATR) ─────────────────────────────
    VVE_2 = VVE_1 & (body >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body >= atr.shift(1) * FDB)

    # ── Moving averages (SMA as default) ───────────────────────────────────────
    ma_slow = df['close'].rolling(window=ma_len).mean()
    ma_fast = df['close'].rolling(window=ma_len_b).mean()

    # ── Direction indicators ──────────────────────────────────────────────────
    diff_slow = ma_slow.diff()
    direction = pd.Series(
        np.where(diff_slow > 0, 1.0, np.where(diff_slow < 0, -1.0, np.nan)),
        index=df.index
    ).ffill().fillna(0.0)

    diff_fast = ma_fast.diff()
    direction_fast = pd.Series(
        np.where(diff_fast > 0, 1.0, np.where(diff_fast < 0, -1.0, np.nan)),
        index=df.index
    ).ffill().fillna(0.0)

    # ── Trend‑filter selection (bullish) ───────────────────────────────────────
    if config_tend_alc == "NINGUNA CONDICION":
        bull_filter = pd.Series(True, index=df.index)
    elif config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA":
        bull_filter = df['close'] > ma_fast
    elif config_tend_alc == "PRECIO MAYOR A MEDIA LENTA":
        bull_filter = df['close'] > ma_slow
    elif config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y LENTA":
        bull_filter = (df['close'] > ma_slow) & (df['close'] > ma_fast)
    elif config_tend_alc == "PRECIO MAYOR A MEDIA LENTA Y DIRECCION ALCISTA":
        bull_filter = (df['close'] > ma_slow) & (direction > 0)
    elif config_tend_alc == "PRECIO MAYOR A MEDIA RAPIDA Y DIRECCION ALCISTA":
        bull_filter = (df['close'] > ma_fast) & (direction_fast > 0)
    elif config_tend_alc == "PRECIO MAYOR A MEDIA LENTA/RAPIDA Y DIRECCION ALCISTA":
        bull_filter = (df['close'] > ma_slow) & (df['close'] > ma_fast) & (direction > 0) & (direction_fast > 0)
    elif config_tend_alc == "DIRECCION MEDIA LENTA ALCISTA":
        bull_filter = direction > 0
    elif config_tend_alc == "DIRECCION MEDIA RAPIDA ALCISTA":
        bull_filter = direction_fast > 0
    elif config_tend_alc == "DIRECCION MEDIA LENTA/RAPIDA ALCISTA":
        bull_filter = (direction > 0) & (direction_fast > 0)
    else:
        bull_filter = pd.Series(False, index=df.index)

    # ── Trend‑filter selection (bearish) ───────────────────────────────────────