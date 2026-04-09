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
    # Default parameters (from Pine Script inputs)
    CDBA = 100          # lookback length for ATR
    PDCM = 70           # minimum body percent
    FDB = 1.3           # body factor
    VVEV = True         # enable green elephant
    VVER = True         # enable red elephant
    mode_filter = True  # 'CON FILTRADO DE TENDENCIA'
    ma_len = 20          # slow ma period
    ma_len_b = 8        # fast ma period

    # Compute true range and ATR (Wilder)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / CDBA, adjust=False).mean()

    # Body size
    body = (df['close'] - df['open']).abs()
    range_hl = df['high'] - df['low']
    body_pct = body * 100.0 / range_hl

    # Elephant detection stage 0
    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']

    # Stage 1: body percent threshold
    VVE_1 = VVE_0 & (body_pct >= PDCM)
    VRE_1 = VRE_0 & (body_pct >= PDCM)

    # Stage 2: ATR factor
    VVE_2 = VVE_1 & (body >= atr.shift(1) * FDB)
    VRE_2 = VRE_1 & (body >= atr.shift(1) * FDB)

    # Moving averages (simple SMA, matching default)
    ma_slow = df['close'].rolling(window=ma_len).mean()
    ma_fast = df['close'].rolling(window=ma_len_b).mean()

    # Direction of fast MA (1 = rising, -1 = falling)
    direction_fast = pd.Series(0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        if pd.notna(ma_fast.iloc[i]) and pd.notna(ma_fast.iloc[i - 1]):
            if ma_fast.iloc[i] > ma_fast.iloc[i - 1]:
                direction_fast.iloc[i] = 1
            elif ma_fast.iloc[i] < ma_fast.iloc[i - 1]:
                direction_fast.iloc[i] = -1
            else:
                direction_fast.iloc[i] = direction_fast.iloc[i - 1]
        else:
            direction_fast.iloc[i] = direction_fast.iloc[i - 1] if pd.notna(direction_fast.iloc[i - 1]) else 0.0

    # Stage 3: trend filter (using fast MA direction)
    VVE_3 = VVE_2 & (direction_fast == 1)
    VRE_3 = VRE_2 & (direction_fast == -1)

    # Entry signals based on mode
    if mode_filter:
        long_entry = VVE_3
        short_entry = VRE_3
    else:
        long_entry = VVE_2
        short_entry = VRE_2

    # Build result list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries