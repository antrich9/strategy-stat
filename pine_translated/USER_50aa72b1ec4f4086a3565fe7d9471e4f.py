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
    # ------------------------------------------------------------------
    # Helper series
    # ------------------------------------------------------------------
    high      = df['high']
    low       = df['low']
    close     = df['close']
    open_     = df['open']          # 'open' conflicts with built‑in, rename

    # ------------------------------------------------------------------
    # 1. ATR (Wilder) – period = CDBA = 100
    # ------------------------------------------------------------------
    CDBA = 100
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / CDBA, adjust=False).mean()
    atr_prev = atr.shift(1)

    # ------------------------------------------------------------------
    # 2. Elephant candle detection (VVE / VRE)
    # ------------------------------------------------------------------
    PDCM = 70                     # minimal body percentage
    FDB  = 1.3                    # body‑vs‑ATR factor

    body   = (close - open_).abs()
    range_ = high - low
    body_pct = np.where(range_ > 0, body * 100.0 / range_, 0.0)

    VVE_0 = close > open_
    VRE_0 = close < open_

    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    VVE_2 = VVE_1 & (body >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body >= atr_prev * FDB)

    # ------------------------------------------------------------------
    # 3. Moving averages (slow & fast) – default SMA, periods 20 & 8
    # ------------------------------------------------------------------
    ma_len    = 20
    ma_len_b  = 8

    ma_slow = close.rolling(window=ma_len).mean()
    ma_fast = close.rolling(window=ma_len_b).mean()

    # direction: 1 = rising, -1 = falling, 0 = unchanged (forward‑filled)
    diff_slow = ma_slow.diff()
    direction_slow = pd.Series(
        np.where(diff_slow > 0, 1.0, np.where(diff_slow < 0, -1.0, np.nan)),
        index=close.index
    ).ffill().fillna(0.0)

    diff_fast = ma_fast.diff()
    direction_fast = pd.Series(
        np.where(diff_fast > 0, 1.0, np.where(diff_fast < 0, -1.0, np.nan)),
        index=close.index
    ).ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # 4. Trend filter (default: fast MA direction)
    # ------------------------------------------------------------------
    modo_tipo = 'CON FILTRADO DE TENDENCIA'      # default from script

    if modo_tipo == 'CON FILTRADO DE TENDENCIA':
        long_cond  = VVE_2 & (direction_fast > 0)
        short_cond = VRE_2 & (direction_fast < 0)
    else:   # SIN FILTRADO DE TENDENCIA
        long_cond  = VVE_2
        short_cond = VRE_2

    # ------------------------------------------------------------------
    # 5. Master toggles (default True)
    # ------------------------------------------------------------------
    VVEV = True   # activate green elephant candles
    VVER = True   # activate red elephant candles

    long_entry  = long_cond  & VVEV
    short_entry = short_cond & VVER

    # ------------------------------------------------------------------
    # 6. Build entry list
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i] or short_entry.iloc[i]:
            direction = 'long' if long_entry.iloc[i] else 'short'
            row = df.iloc[i]
            ts = int(row['time'])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1

    return entries