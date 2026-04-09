import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # ---- Time Window (Europe/London) ----
    dt = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    total_min = dt.dt.hour * 60 + dt.dt.minute

    morning_start   = 6 * 60 + 45   # 405
    morning_end     = 9 * 60 + 45   # 585
    afternoon_start = 14 * 60 + 45  # 885
    afternoon_end   = 16 * 60 + 45  # 1005

    in_time_window = (
        ((total_min >= morning_start) & (total_min < morning_end)) |
        ((total_min >= afternoon_start) & (total_min < afternoon_end))
    )

    # ---- ATR (Wilder smoothing, length 144) ----
    prev_close = df['close'].shift(1)
    tr = pd.concat(
        [
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1/144, adjust=False).mean().fillna(0)

    # Fair Value Gap width filter (default 0.5)
    fvgTH = 0.5
    atr_threshold = atr * fvgTH

    # ---- Helper shifted series ----
    high_1 = df['high'].shift(1)
    high_2 = df['high'].shift(2)
    low_1  = df['low'].shift(1)
    low_2  = df['low'].shift(2)
    close_1 = df['close'].shift(1)

    # ---- Bull / Bear detection ----
    bullG = df['low'] > high_1
    bearG = df['high'] < low_1

    bull = (df['low'] - high_2) > atr_threshold
    bull = bull & (df['low'] > high_2) & (close_1 > high_2) & ~(bullG | bullG.shift(1).fillna(False))

    bear = (low_2 - df['high']) > atr_threshold
    bear = bear & (df['high'] < low_2) & (close_1 < low_2) & ~(bearG | bearG.shift(1).fillna(False))

    # ---- Entry signals (first bar of each detection) ----
    prev_bull = bull.shift(1).fillna(False)
    prev_bear = bear.shift(1).fillna(False)

    long_entry = bull & ~prev_bull & in_time_window
    short_entry = bear & ~prev_bear & in_time_window

    # ---- Build entry list ----
    entries = []
    trade_num = 1

    for i in df.index:
        if long_entry.iloc[i]:
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
        elif short_entry.iloc[i]:
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