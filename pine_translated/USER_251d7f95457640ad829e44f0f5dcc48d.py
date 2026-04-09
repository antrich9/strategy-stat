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
    # === Pine script parameters ===
    atr_length = 200
    filter_width = 0.0  # default input.float(0., 'FVG Width Filter')

    # --- Wilder ATR (ATR length 200) ---
    prev_close = df['close'].shift(1)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - prev_close),
                               np.abs(df['low'] - prev_close)))
    tr = pd.Series(tr, index=df.index)

    atr = pd.Series(np.nan, index=df.index, dtype=float)
    if len(df) >= atr_length:
        # Initial simple average over the first `atr_length` bars
        atr.iloc[:atr_length] = tr.iloc[:atr_length].mean()
        # Wilder smoothing: new_ATR = (prev_ATR * (len-1) + TR) / len
        for i in range(atr_length, len(df)):
            atr.iloc[i] = (atr.iloc[i - 1] * (atr_length - 1) + tr.iloc[i]) / atr_length
    # If not enough bars, ATR stays NaN → filter conditions will be False

    # --- Bullish entry (long) conditions ---
    low_3   = df['low'].shift(3)
    high_1  = df['high'].shift(1)
    close_2 = df['close'].shift(2)
    close_cur = df['close']

    cond_bull_1 = low_3 > high_1
    cond_bull_2 = close_2 < low_3
    cond_bull_3 = close_cur > low_3
    filter_cond_bull = (low_3 - high_1) > atr * filter_width

    bull = cond_bull_1 & cond_bull_2 & cond_bull_3 & filter_cond_bull

    # --- Bearish entry (short) conditions ---
    low_1   = df['low'].shift(1)
    high_3  = df['high'].shift(3)

    cond_bear_1 = low_1 > high_3
    cond_bear_2 = close_2 > high_3
    cond_bear_3 = close_cur < high_3
    filter_cond_bear = (low_1 - high_3) > atr * filter_width

    bear = cond_bear_1 & cond_bear_2 & cond_bear_3 & filter_cond_bear

    # --- Build entry list ---
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bull.iloc[i]:
            direction = 'long'
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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
        elif bear.iloc[i]:
            direction = 'short'
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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