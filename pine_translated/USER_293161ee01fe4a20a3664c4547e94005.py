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
    # ---------- default Pine script input values ----------
    PDCM = 70                     # minimum body percentage
    CDBA = 100                    # ATR length
    FDB = 1.3                     # ATR factor
    ma_len = 20                   # slow MA period
    ma_len_b = 8                  # fast MA period
    reaction_ma_1 = 1             # reaction length for slow MA direction
    reaction_ma_2 = 1             # reaction length for fast MA direction

    # ---------- series shortcuts ----------
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # ---------- elephant candle helpers ----------
    body = (close - open_).abs()
    total_range = high - low
    # avoid division by zero
    body_pct = pd.Series(
        np.where(total_range > 0, body * 100.0 / total_range, 0.0),
        index=df.index
    )

    # stage 0 – colour
    VVE_0 = close > open_
    VRE_0 = close < open_

    # stage 1 – body % threshold
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    # ---------- ATR (Wilder) ----------
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / CDBA, adjust=False).mean()

    # stage 2 – body >= ATR[1] * factor
    prev_atr = atr.shift(1)
    body_atr_cond = body >= prev_atr * FDB
    VVE_2 = VVE_1 & body_atr_cond
    VRE_2 = VRE_1 & body_atr_cond

    # ---------- moving averages ----------
    ma_slow = close.rolling(window=ma_len).mean()
    ma_fast = close.rolling(window=ma_len_b).mean()

    # ---------- direction of MAs ----------
    diff_slow = ma_slow - ma_slow.shift(reaction_ma_1)
    direction = pd.Series(
        np.where(diff_slow > 0, 1,
                 np.where(diff_slow < 0, -1, np.nan)),
        index=df.index
    ).ffill().fillna(0).astype(int)

    diff_fast = ma_fast - ma_fast.shift(reaction_ma_2)
    direction_b = pd.Series(
        np.where(diff_fast > 0, 1,
                 np.where(diff_fast < 0, -1, np.nan)),
        index=df.index
    ).ffill().fillna(0).astype(int)

    # ---------- trend filters (default: fast MA direction) ----------
    bullish_filter = direction_b > 0
    bearish_filter = direction_b < 0

    # ---------- final entry signals ----------
    long_signal = VVE_2 & bullish_filter
    short_signal = VRE_2 & bearish_filter

    # ---------- build entry list ----------
    entries = []
    trade_num = 1
    for i in df.index:
        if long_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif short_signal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })