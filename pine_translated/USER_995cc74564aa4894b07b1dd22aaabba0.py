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
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # ----- shift helpers -----
    high_1 = df['high'].shift(1)
    high_2 = df['high'].shift(2)
    low_1  = df['low'].shift(1)
    low_2  = df['low'].shift(2)
    close_1 = df['close'].shift(1)

    # ----- Wilder ATR (144) -----
    tr = np.maximum(
        df['high'] - df['low'],
        np.abs(df['high'] - close_1),
        np.abs(df['low'] - close_1)
    )
    atr = tr.ewm(span=144, adjust=False).mean()

    # Fair Value Gap width filter
    fvg_th   = 0.5
    atr_thresh = atr * fvg_th

    # ----- bull / bear gap detection -----
    bull_g = df['low'] > high_1
    bear_g = df['high'] < low_1

    # ----- bull entry condition -----
    bull = (
        (df['low'] - high_2) > atr_thresh) & (
        df['low'] > high_2) & (
        close_1 > high_2) & (
        ~(bull_g | bull_g.shift(1))
    )

    # ----- bear entry condition -----
    bear = (
        (low_2 - df['high']) > atr_thresh) & (
        df['high'] < low_2) & (
        close_1 < low_2) & (
        ~(bear_g | bear_g.shift(1))
    )

    # ----- time window (London) -----
    ts = pd.to_datetime(df['time'], unit='s')
    hour = ts.dt.hour
    minute = ts.dt.minute

    morning   = ((hour == 6) & (minute >= 45)) | ((hour == 7) | (hour == 8)) | ((hour == 9) & (minute <= 45))
    afternoon = ((hour == 14) & (minute >= 45)) | (hour == 15) | ((hour == 16) & (minute <= 45))
    in_window = morning | afternoon

    # ----- entry signals -----
    entry_long  = bull & in_window
    entry_short = bear & in_window

    # ----- generate entry list -----
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # skip bars where both signals are NaN
        if pd.isna(entry_long.iloc[i]) and pd.isna(entry_short.iloc[i]):
            continue

        # long entry on first true bar of the signal
        if entry_long.iloc[i] and (i == 0 or not entry_long.iloc[i - 1]):
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        # short entry on first true bar of the signal
        if entry_short.iloc[i] and (i == 0 or not entry_short.iloc[i - 1]):
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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