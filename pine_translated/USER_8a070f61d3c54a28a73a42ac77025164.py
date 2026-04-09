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
    df = df.copy()
    # ----- time window (Europe/London) -----
    dt = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    minutes_since_midnight = dt.dt.hour * 60 + dt.dt.minute
    morning_win = (minutes_since_midnight >= 6 * 60 + 45) & (minutes_since_midnight < 9 * 60 + 45)
    afternoon_win = (minutes_since_midnight >= 14 * 60 + 45) & (minutes_since_midnight < 16 * 60 + 45)
    in_window = morning_win | afternoon_win

    # ----- true range & Wilder ATR (length 144) -----
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=144, adjust=False).mean()
    fvgTH = 0.5
    atr_th = atr * fvgTH

    # ----- swing detection -----
    bullG = (df['low'] > df['high'].shift(1)).fillna(False)
    bearG = (df['high'] < df['low'].shift(1)).fillna(False)
    bullG_prev = bullG.shift(1).fillna(False)
    bearG_prev = bearG.shift(1).fillna(False)

    # ----- bullish FVG entry condition -----
    cond_bull = (
        (df['low'] - df['high'].shift(2) > atr_th) &
        (df['low'] > df['high'].shift(2)) &
        (df['close'].shift(1) > df['high'].shift(2)) &
        (~(bullG | bullG_prev)) &
        in_window &
        atr_th.notna()
    )

    # ----- bearish FVG entry condition -----
    cond_bear = (
        (df['low'].shift(2) - df['high'] > atr_th) &
        (df['high'] < df['low'].shift(2)) &
        (df['close'].shift(1) < df['low'].shift(2)) &
        (~(bearG | bearG_prev)) &
        in_window &
        atr_th.notna()
    )

    # ----- generate entry list -----
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if cond_bull.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif cond_bear.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries