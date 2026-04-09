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
    # ---------- 1. EMA (200‑period) ----------
    ema = df['close'].ewm(span=200, adjust=False).mean()

    # ---------- 2. Supertrend (period=10, multiplier=3) ----------
    period = 10
    multiplier = 3

    # True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = df['high'].iloc[0] - df['low'].iloc[0]  # first bar

    # ATR – Wilder smoothing (alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    # Supertrend components
    hl2 = (df['high'] + df['low']) / 2.0
    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr

    # Direction: 1 = bullish, -1 = bearish
    direction = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > dn.iloc[i]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < up.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    # ---------- 3. Entry conditions ----------
    long_cond = (df['close'] > ema) & (direction == 1)
    short_cond = (df['close'] < ema) & (direction == -1)

    # ---------- 4. Detect transitions and build entry list ----------
    entries = []
    trade_num = 1

    prev_long = False
    prev_short = False

    for i in range(len(df)):
        cur_long = bool(long_cond.iloc[i])
        cur_short = bool(short_cond.iloc[i])

        if cur_long and not prev_long:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif cur_short and not prev_short:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        prev_long = cur_long
        prev_short = cur_short

    return entries